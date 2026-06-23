#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <dirent.h>
#include <vector>
#include <string.h>
#include <string>
#include <algorithm>

#define PI 3.14159265358979323846
#define TILE_W 16
#define TILE_H 8

typedef __align__(8) struct {
    float real;
    float imag;
} MyComplex;

typedef struct {
    int width;
    int height;
    int max_value;
    unsigned char *data;
} PGMImage;

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// # esempio di immagine PGM 4x3
// P5
// 4[width] 3[height]
// 255[max_value]
// 0  50 100 150 [row 1]
// 50 100 150 200 [row 2]
// 100 150 200 255 [row 3]
// con fscanf e fread si legge la parte testuale per i metadati
// poi i dati binari in un array di unsigned char (1 byte per pixel)

PGMImage readPGM(const char *filename) {
    PGMImage img;
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Errore nell'aprire il file");
        exit(1);
    }

    char format[3];
    fscanf(file, "%s", format);
    if (format[0] != 'P' || format[1] != '5') {
        fprintf(stderr, "Formato non supportato.\n");
        exit(1);
    }

    fscanf(file, "%d %d", &img.width, &img.height);
    fscanf(file, "%d", &img.max_value);
    fgetc(file); //scarta il newline dopo max_value

    img.data = (unsigned char *)malloc(img.width * img.height);
    fread(img.data, 1, img.width * img.height, file);
    fclose(file);

    return img;
}

void writePGM(const char *filename, PGMImage img) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Errore scrittura: %s\n", filename);
        perror("Dettaglio");
        exit(1);
    }

    fprintf(file, "P5\n%d %d\n%d\n", img.width, img.height, img.max_value);
    fwrite(img.data, 1, img.width * img.height, file);
    fclose(file);
}


__global__ void dft_opt_L1(const unsigned char* __restrict__ in, MyComplex* __restrict__ out, int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float tile[TILE_H][TILE_W];

    float sumReal = 0.0f;
    float sumImag = 0.0f;
    
    float angleX = -2.0f * PI * (float)x / width;
    float angleY = -2.0f * PI * (float)y / height;

    // NUOVO 1: Calcoliamo il "passo" di rotazione per l'asse X una sola volta per thread.
    float stepCosX, stepSinX;
    sincosf(angleX, &stepSinX, &stepCosX);

    for (int tileV = 0; tileV < height; tileV += TILE_H) {
        for (int tileU = 0; tileU < width; tileU += TILE_W) {

            // Carica tile collaborativamente dalla global memory
            int u = tileU + threadIdx.x;
            int v = tileV + threadIdx.y;
            tile[threadIdx.y][threadIdx.x] = (u < width && v < height)
                ? (float)in[v * width + u]
                : 0.0f;

            __syncthreads();

            if (x < width && y < height) {
                
                // NUOVO 2: Calcoliamo il punto di partenza (fase iniziale) per questo specifico Tile orizzontale
                float startCosU, startSinU;
                sincosf(angleX * tileU, &startSinU, &startCosU);

                for (int dv = 0; dv < TILE_H; dv++) {

                    // sin/cos verticale: costante per tutto il loop su du
                    float cosV, sinV;
                    sincosf(angleY * (tileV + dv), &sinV, &cosV);

                    // Ripristiniamo i fasori all'inizio della riga (du = 0)
                    float currentCosU = startCosU;
                    float currentSinU = startSinU;

                    #pragma unroll 16
                    for (int du = 0; du < TILE_W; du++) {

                        // NESSUN sincosf QUI DENTRO!

                        // Formula addizione con FMA usando i fasori correnti
                        float c = __fmaf_rn(currentCosU, cosV, -currentSinU * sinV);
                        float s = __fmaf_rn(currentSinU, cosV,  currentCosU * sinV);

                        float pixel = tile[dv][du];

                        // Accumulazione con FMA
                        sumReal = __fmaf_rn(pixel, c, sumReal);
                        sumImag = __fmaf_rn(pixel, s, sumImag);

                        // NUOVO 3: Ruotiamo il fasore per il prossimo step (du+1) con 2 FMA
                        float nextCosU = __fmaf_rn(currentCosU, stepCosX, -currentSinU * stepSinX);
                        float nextSinU = __fmaf_rn(currentSinU, stepCosX,  currentCosU * stepSinX);
                        currentCosU = nextCosU;
                        currentSinU = nextSinU;
                    }
                }
            }

            __syncthreads();
        }
    }

    if (x < width && y < height)
        out[y * width + x] = { sumReal, sumImag };

}


__global__ void idft_opt_L1(const MyComplex* __restrict__ in, unsigned char* __restrict__ out, int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Due array separati per evitare padding della struct
    __shared__ float tileReal[TILE_H][TILE_W];
    __shared__ float tileImag[TILE_H][TILE_W];

    float sum = 0.0f;

    float angleX = 2.0f * PI * (float)x / width;
    float angleY = 2.0f * PI * (float)y / height;

    // NUOVO 1: Passo di rotazione
    float stepCosX, stepSinX;
    sincosf(angleX, &stepSinX, &stepCosX);

    for (int tileV = 0; tileV < height; tileV += TILE_H) {
        for (int tileU = 0; tileU < width; tileU += TILE_W) {

            int u = tileU + threadIdx.x;
            int v = tileV + threadIdx.y;

            if (u < width && v < height) {
                MyComplex c = in[v * width + u];
                tileReal[threadIdx.y][threadIdx.x] = c.real;
                tileImag[threadIdx.y][threadIdx.x] = c.imag;
            } else {
                tileReal[threadIdx.y][threadIdx.x] = 0.0f;
                tileImag[threadIdx.y][threadIdx.x] = 0.0f;
            }

            __syncthreads();

            if (x < width && y < height) {
                
                // NUOVO 2: Punto di partenza per l'asse U
                float startCosU, startSinU;
                sincosf(angleX * tileU, &startSinU, &startCosU);

                for (int dv = 0; dv < TILE_H; dv++) {

                    // sin/cos verticale
                    float cosV, sinV;
                    sincosf(angleY * (tileV + dv), &sinV, &cosV);
                    
                    // Inizializziamo i fasori orizzontali
                    float currentCosU = startCosU;
                    float currentSinU = startSinU;

                    #pragma unroll 16
                    for (int du = 0; du < TILE_W; du++) {

                        // Formula addizione con FMA
                        float c = __fmaf_rn(currentCosU, cosV, -currentSinU * sinV);
                        float s = __fmaf_rn(currentSinU, cosV,  currentCosU * sinV);

                        // Accumulazione IDFT
                        float step1 = __fmaf_rn(-tileImag[dv][du], s, sum);
                        sum         = __fmaf_rn( tileReal[dv][du],  c, step1);

                        // NUOVO 3: Rotazione del fasore
                        float nextCosU = __fmaf_rn(currentCosU, stepCosX, -currentSinU * stepSinX);
                        float nextSinU = __fmaf_rn(currentSinU, stepCosX,  currentCosU * stepSinX);
                        currentCosU = nextCosU;
                        currentSinU = nextSinU;
                    }
                }
            }

            __syncthreads();
        }
    }

    if (x < width && y < height) {
        sum /= (float)(width * height);
        out[y * width + x] = (unsigned char)(sum < 0.0f ? 0 : (sum > 255.0f ? 255 : sum));
    }
}

__global__ void filtro(MyComplex *dft, int width, int height, float cutoff) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        
        // Calcoliamo la distanza dall'angolo più vicino (dove stanno le basse frequenze)
        float dx = (x < width / 2) ? x : (width - x);
        float dy = (y < height / 2) ? y : (height - y);
        float d = sqrtf(dx * dx + dy * dy);

        // EDGE DETECTION (Filtro Passa-Alto)
        if (d < cutoff) {
            dft[y * width + x].real = 0.0f;
            dft[y * width + x].imag = 0.0f;
        }
    }
}


int main(int argc, char *argv[]) {

    if (argc < 3) {
        fprintf(stderr, "Uso: %s <cartella_input_pgm> <cartella_output>\n", argv[0]);
        return 1;
    }

    // Crea output dir se non esiste (robusto su FUSE/Drive)
    {
        DIR *test = opendir(argv[2]);
        if (test) { closedir(test); }
        else if (mkdir(argv[2], 0755) != 0) {
            fprintf(stderr, "Impossibile creare cartella output: %s\n", argv[2]);
            perror("mkdir"); return 1;
        }
    }

    printf("[1/6] Input:  %s\n", argv[1]); fflush(stdout);
    printf("[1/6] Output: %s\n", argv[2]); fflush(stdout);

    // Raccogli e ordina i frame PGM
    DIR *dir = opendir(argv[1]);
    if (!dir) { perror("opendir input"); return 1; }
    struct dirent *entry;
    std::vector<std::string> files;
    while ((entry = readdir(dir)) != NULL) {
        std::string name(entry->d_name);
        if (name.find(".pgm") != std::string::npos)
            files.push_back(std::string(argv[1]) + "/" + name);
    }
    closedir(dir);
    std::sort(files.begin(), files.end());

    printf("[2/6] Frame trovati: %d\n", (int)files.size()); fflush(stdout);
    if (files.empty()) { fprintf(stderr, "Nessun .pgm trovato in %s\n", argv[1]); return 1; }

    // Dimensioni dal primo frame
    PGMImage first = readPGM(files[0].c_str());
    int W = first.width, H = first.height;
    free(first.data);
    printf("[3/6] Dimensioni: %dx%d\n", W, H); fflush(stdout);

    // Allocazione CUDA
    printf("[4/6] Allocazione memoria CUDA...\n"); fflush(stdout);
    unsigned char *h_in, *h_out;
    CHECK(cudaMallocHost(&h_in,  W * H));
    CHECK(cudaMallocHost(&h_out, W * H));
    unsigned char *d_in, *d_out;
    MyComplex *d_dft;
    CHECK(cudaMalloc(&d_in,  W * H));
    CHECK(cudaMalloc(&d_out, W * H));
    CHECK(cudaMalloc(&d_dft, W * H * sizeof(MyComplex)));
    printf("[4/6] Memoria allocata OK\n"); fflush(stdout);

    dim3 threadsDFT(TILE_W, TILE_H);
    dim3 threadsFilter(16, 16);
    dim3 blocksDFT((W+TILE_W-1)/TILE_W, (H+TILE_H-1)/TILE_H);
    dim3 blocksFilter((W+15)/16, (H+15)/16);
    float cutoff = 30.0f;

    // Timer totale
    cudaEvent_t t_start, t_stop;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_stop);
    cudaEventRecord(t_start);

    printf("[5/6] Avvio elaborazione %d frame...\n", (int)files.size()); fflush(stdout);

    for (int i = 0; i < (int)files.size(); i++) {

        // Stampa progresso e tempo stimato ogni frame
        if (i == 0 || i % 5 == 0) {
            printf("  Frame %d/%d\n", i + 1, (int)files.size());
            fflush(stdout);
        }

        PGMImage img = readPGM(files[i].c_str());
        memcpy(h_in, img.data, W * H);
        free(img.data);

        CHECK(cudaMemcpy(d_in, h_in, W*H, cudaMemcpyHostToDevice));

        dft_opt_L1  <<<blocksDFT,    threadsDFT   >>>(d_in, d_dft, W, H);
        filtro       <<<blocksFilter, threadsFilter>>>(d_dft, W, H, cutoff);
        idft_opt_L1 <<<blocksDFT,    threadsDFT   >>>(d_dft, d_out, W, H);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(h_out, d_out, W*H, cudaMemcpyDeviceToHost));

        char outpath[256];
        snprintf(outpath, sizeof(outpath), "%s/frame_%04d.pgm", argv[2], i);
        PGMImage out_img = {W, H, 255, h_out};
        writePGM(outpath, out_img);
    }

    cudaEventRecord(t_stop);
    cudaEventSynchronize(t_stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, t_start, t_stop);
    printf("[6/6] Completato! Tempo totale: %.2f s (%.2f ms/frame)\n",
           ms / 1000.0f, ms / files.size());
    fflush(stdout);

    cudaFreeHost(h_in); cudaFreeHost(h_out);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_dft);
    cudaEventDestroy(t_start); cudaEventDestroy(t_stop);
    return 0;
}