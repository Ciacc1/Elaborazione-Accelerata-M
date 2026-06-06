#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <cuda_runtime.h>
#include <dirent.h>
#include <sys/stat.h>
#include <vector>
#include <string.h>
#include <string>
#include <algorithm>

#define PI 3.14159265358979323846f
#define TILE_W 16
#define TILE_H 8

// ─────────────────────────────────────────────────────────────────────────────
// Tipi
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// I/O PGM
// ─────────────────────────────────────────────────────────────────────────────

PGMImage readPGM(const char *filename) {
    PGMImage img;
    FILE *file = fopen(filename, "rb");
    if (!file) { perror("Errore apertura file"); exit(1); }

    char format[3];
    (void)fscanf(file, "%s", format);
    if (format[0] != 'P' || format[1] != '5') {
        fprintf(stderr, "Formato non supportato.\n"); exit(1);
    }
    (void)fscanf(file, "%d %d", &img.width, &img.height);
    (void)fscanf(file, "%d",    &img.max_value);
    fgetc(file);

    img.data = (unsigned char *)malloc(img.width * img.height);
    (void)fread(img.data, 1, img.width * img.height, file);
    fclose(file);
    return img;
}

void writePGM(const char *filename, PGMImage img) {
    FILE *file = fopen(filename, "wb");
    if (!file) { perror("Errore apertura file scrittura"); exit(1); }
    fprintf(file, "P5\n%d %d\n%d\n", img.width, img.height, img.max_value);
    fwrite(img.data, 1, img.width * img.height, file);
    fclose(file);
}

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL 1 — DFT ottimizzata
// ─────────────────────────────────────────────────────────────────────────────

__global__ void dft_opt_L1(
    const unsigned char* __restrict__ in,
    MyComplex* __restrict__ out,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float tile[TILE_H][TILE_W];

    float sumReal = 0.0f, sumImag = 0.0f;
    float angleX  = -2.0f * PI * (float)x / width;
    float angleY  = -2.0f * PI * (float)y / height;

    float stepCosX, stepSinX;
    sincosf(angleX, &stepSinX, &stepCosX);

    for (int tileV = 0; tileV < height; tileV += TILE_H) {
        for (int tileU = 0; tileU < width; tileU += TILE_W) {

            int u = tileU + threadIdx.x;
            int v = tileV + threadIdx.y;
            tile[threadIdx.y][threadIdx.x] = (u < width && v < height)
                ? (float)__ldg(&in[v * width + u]) : 0.0f;
            __syncthreads();

            if (x < width && y < height) {
                float startCosU, startSinU;
                sincosf(angleX * tileU, &startSinU, &startCosU);

                for (int dv = 0; dv < TILE_H; dv++) {
                    float cosV, sinV;
                    sincosf(angleY * (tileV + dv), &sinV, &cosV);

                    float currentCosU = startCosU;
                    float currentSinU = startSinU;

                    #pragma unroll 16
                    for (int du = 0; du < TILE_W; du++) {
                        float c = __fmaf_rn(currentCosU, cosV, -currentSinU * sinV);
                        float s = __fmaf_rn(currentSinU, cosV,  currentCosU * sinV);
                        float pixel = tile[dv][du];
                        sumReal = __fmaf_rn(pixel, c, sumReal);
                        sumImag = __fmaf_rn(pixel, s, sumImag);

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

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL 2 — IDFT con filtro passa-alto FUSO (Edge Detection)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void idft_filtered(
    const MyComplex* __restrict__ in,
    unsigned char* __restrict__ out,
    int width, int height,
    float cutoff2,      
    float invN)         
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float tileReal[TILE_H][TILE_W];
    __shared__ float tileImag[TILE_H][TILE_W];

    float sum    = 0.0f;
    float angleX = 2.0f * PI * (float)x / width;
    float angleY = 2.0f * PI * (float)y / height;

    float stepCosX, stepSinX;
    sincosf(angleX, &stepSinX, &stepCosX);

    for (int tileV = 0; tileV < height; tileV += TILE_H) {
        for (int tileU = 0; tileU < width; tileU += TILE_W) {

            int u = tileU + threadIdx.x;
            int v = tileV + threadIdx.y;

            if (u < width && v < height) {
                float du = (u < width  / 2) ? (float)u : (float)(width  - u);
                float dv = (v < height / 2) ? (float)v : (float)(height - v);
                float d2 = du*du + dv*dv;

                if (d2 < cutoff2) {
                    tileReal[threadIdx.y][threadIdx.x] = 0.0f;
                    tileImag[threadIdx.y][threadIdx.x] = 0.0f;
                } else {
                    MyComplex coeff = in[v * width + u];
                    tileReal[threadIdx.y][threadIdx.x] = coeff.real;
                    tileImag[threadIdx.y][threadIdx.x] = coeff.imag;
                }
            } else {
                tileReal[threadIdx.y][threadIdx.x] = 0.0f;
                tileImag[threadIdx.y][threadIdx.x] = 0.0f;
            }

            __syncthreads();

            if (x < width && y < height) {
                float startCosU, startSinU;
                sincosf(angleX * tileU, &startSinU, &startCosU);

                for (int dv = 0; dv < TILE_H; dv++) {
                    float cosV, sinV;
                    sincosf(angleY * (tileV + dv), &sinV, &cosV);

                    float currentCosU = startCosU;
                    float currentSinU = startSinU;

                    #pragma unroll 16
                    for (int du = 0; du < TILE_W; du++) {
                        float c = __fmaf_rn(currentCosU, cosV, -currentSinU * sinV);
                        float s = __fmaf_rn(currentSinU, cosV,  currentCosU * sinV);

                        float step1 = __fmaf_rn(-tileImag[dv][du], s, sum);
                        sum         = __fmaf_rn( tileReal[dv][du],  c, step1);

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
        sum *= invN;
        out[y * width + x] = (unsigned char)(sum < 0.0f ? 0 : (sum > 255.0f ? 255 : sum));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN — Double buffering con 2 CUDA streams
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {

    if (argc < 3) {
        fprintf(stderr, "Uso: %s <cartella_input_pgm> <cartella_output>\n", argv[0]);
        return 1;
    }

    // Verifica se la cartella esiste già (aprendola come directory)
    // Questo evita problemi con errno su filesystem FUSE come Google Drive
    {
        DIR *test_dir = opendir(argv[2]);
        if (test_dir) {
            // Cartella esiste — ok
            closedir(test_dir);
        } else {
            // Non esiste — proviamo a crearla
            if (mkdir(argv[2], 0755) != 0) {
                fprintf(stderr, "Errore: impossibile aprire o creare la cartella output: %s\n", argv[2]);
                perror("Dettaglio");
                return 1;
            }
        }
    }
    printf("Output: %s\n", argv[2]);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    DIR *dir = opendir(argv[1]);
    if (!dir) { perror("Errore apertura cartella input"); return 1; }
    struct dirent *entry;
    std::vector<std::string> files;
    while ((entry = readdir(dir)) != NULL) {
        std::string name(entry->d_name);
        if (name.find(".pgm") != std::string::npos)
            files.push_back(std::string(argv[1]) + "/" + name);
    }
    closedir(dir);
    std::sort(files.begin(), files.end());

    if (files.empty()) { fprintf(stderr, "Nessun frame PGM trovato.\n"); return 1; }
    printf("Frame trovati: %d\n", (int)files.size());

    PGMImage first = readPGM(files[0].c_str());
    int W = first.width, H = first.height, N = W * H;
    free(first.data);
    printf("Dimensioni: %dx%d\n", W, H);

    dim3 threadsDFT(TILE_W, TILE_H);
    dim3 blocksDFT((W + TILE_W - 1) / TILE_W, (H + TILE_H - 1) / TILE_H);

    float cutoff    = 30.0f;
    float cutoff2   = cutoff * cutoff;   
    float invN      = 1.0f / (float)N;  

    cudaFuncSetCacheConfig(dft_opt_L1,    cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(idft_filtered, cudaFuncCachePreferShared);

    // ── Allocazione pinned host (double buffer) ──────────────────────────────
    unsigned char *h_in[2], *h_edge[2];
    for (int b = 0; b < 2; b++) {
        CHECK(cudaMallocHost(&h_in[b],   N));
        CHECK(cudaMallocHost(&h_edge[b], N));
    }

    // ── Allocazione device (double buffer) ───────────────────────────────────
    unsigned char *d_in[2], *d_edge[2];
    MyComplex     *d_dft[2];
    for (int b = 0; b < 2; b++) {
        CHECK(cudaMalloc(&d_in[b],   N));
        CHECK(cudaMalloc(&d_edge[b], N));
        CHECK(cudaMalloc(&d_dft[b],  N * sizeof(MyComplex)));
    }

    cudaStream_t stream[2];
    CHECK(cudaStreamCreate(&stream[0]));
    CHECK(cudaStreamCreate(&stream[1]));

    // Pre-caricamento frame 0 su stream 0
    {
        PGMImage img = readPGM(files[0].c_str());
        memcpy(h_in[0], img.data, N);
        free(img.data);
        CHECK(cudaMemcpyAsync(d_in[0], h_in[0], N, cudaMemcpyHostToDevice, stream[0]));
    }

    // ── Loop principale ───────────────────────────────────────────────────────
    for (int i = 0; i < (int)files.size(); i++) {

        int cur  = i % 2;       
        int next = (i + 1) % 2; 

        // 1. Esecuzione pipeline GPU (Stream Corrente)
        dft_opt_L1<<<blocksDFT, threadsDFT, 0, stream[cur]>>>(d_in[cur], d_dft[cur], W, H);
        idft_filtered<<<blocksDFT, threadsDFT, 0, stream[cur]>>>(d_dft[cur], d_edge[cur], W, H, cutoff2, invN);

        // 2. Prefetch frame successivo su stream alternato (Overlap CPU/GPU)
        if (i + 1 < (int)files.size()) {
            PGMImage nxt = readPGM(files[i + 1].c_str());
            memcpy(h_in[next], nxt.data, N);
            free(nxt.data);
            CHECK(cudaMemcpyAsync(d_in[next], h_in[next], N, cudaMemcpyHostToDevice, stream[next]));
        }

        // 3. Scaricamento dei risultati asincrono D→H
        CHECK(cudaMemcpyAsync(h_edge[cur], d_edge[cur], N, cudaMemcpyDeviceToHost, stream[cur]));

        // 4. Sincronizzazione dello stream corrente prima della scrittura I/O
        CHECK(cudaStreamSynchronize(stream[cur]));

        // 5. Scrittura output su disco
        char path_edge[512];
        snprintf(path_edge, sizeof(path_edge), "%s/edge_%04d.pgm", argv[2], i);
        printf("Scrivo: %s\n", path_edge);
        PGMImage out_edge = {W, H, 255, h_edge[cur]};
        writePGM(path_edge, out_edge);

        if (i % 10 == 0)
            printf("Frame %d/%d elaborato\n", i + 1, (int)files.size());
    }

    CHECK(cudaStreamSynchronize(stream[0]));
    CHECK(cudaStreamSynchronize(stream[1]));

    printf("Elaborazione completata.\n");
    printf("Output edge: %s/edge_XXXX.pgm\n", argv[2]);

    // ── Cleanup ────────────────────────────────────────────────────────────────
    for (int b = 0; b < 2; b++) {
        cudaFreeHost(h_in[b]);
        cudaFreeHost(h_edge[b]);
        cudaFree(d_in[b]);
        cudaFree(d_edge[b]);
        cudaFree(d_dft[b]);
    }
    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);

    return 0;
}