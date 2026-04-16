#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>


#define PI 3.14159265358979323846
#define TILE_SIZE 16

typedef struct {
    double real;
    double imag;
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
        perror("Errore nell'aprire il file per la scrittura");
        exit(1);
    }

    fprintf(file, "P5\n%d %d\n%d\n", img.width, img.height, img.max_value);
    fwrite(img.data, 1, img.width * img.height, file);
    fclose(file);
}

#define TILE_SIZE 16

#define TILE_SIZE 16
#define PI 3.14159265358979323846f

__global__ void dft2D_opt_shared(unsigned char *in, MyComplex *out, int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    float sumReal = 0.0f;
    float sumImag = 0.0f;

    float angleX = -2.0f * PI * (float)x / width;
    float angleY = -2.0f * PI * (float)y / height;

    for (int tileV = 0; tileV < height; tileV += TILE_SIZE) {
        for (int tileU = 0; tileU < width; tileU += TILE_SIZE) {

            // Carica tile collaborativamente dalla global memory
            int u = tileU + threadIdx.x;
            int v = tileV + threadIdx.y;
            tile[threadIdx.y][threadIdx.x] = (u < width && v < height)
                ? (float)in[v * width + u]
                : 0.0f;

            __syncthreads();

            if (x < width && y < height) {
                for (int dv = 0; dv < TILE_SIZE && (tileV + dv) < height; dv++) {

                    // sin/cos verticale: costante per tutto il loop su du
                    float cosV, sinV;
                    sincosf(angleY * (tileV + dv), &sinV, &cosV);
                    
                    #pragma unroll 8
                    for (int du = 0; du < TILE_SIZE && (tileU + du) < width; du++) {

                        float cosU, sinU;
                        sincosf(angleX * (tileU + du), &sinU, &cosU);

                        // Formula addizione con FMA
                        float c = __fmaf_rn(cosU, cosV, -sinU * sinV);
                        float s = __fmaf_rn(sinU, cosV,  cosU * sinV);

                        float pixel = tile[dv][du];

                        // Accumulazione con FMA
                        sumReal = __fmaf_rn(pixel, c, sumReal);
                        sumImag = __fmaf_rn(pixel, s, sumImag);
                    }
                }
            }

            __syncthreads();
        }
    }

    if (x < width && y < height)
        out[y * width + x] = { sumReal, sumImag };
}


__global__ void idft2D_opt_shared(MyComplex *in, unsigned char *out, int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Due array separati per evitare padding della struct
    __shared__ float tileReal[TILE_SIZE][TILE_SIZE];
    __shared__ float tileImag[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    float angleX = 2.0f * PI * (float)x / width;
    float angleY = 2.0f * PI * (float)y / height;

    for (int tileV = 0; tileV < height; tileV += TILE_SIZE) {
        for (int tileU = 0; tileU < width; tileU += TILE_SIZE) {

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
                for (int dv = 0; dv < TILE_SIZE && (tileV + dv) < height; dv++) {

                    // sin/cos verticale — costante per il loop su du
                    float cosV, sinV;
                    sincosf(angleY * (tileV + dv), &sinV, &cosV);
                    
                    #pragma unroll 8
                    for (int du = 0; du < TILE_SIZE && (tileU + du) < width; du++) {

                        float cosU, sinU;
                        sincosf(angleX * (tileU + du), &sinU, &cosU);
                        // Formula addizione con FMA
                        float c = __fmaf_rn(cosU, cosV, -sinU * sinV);
                        float s = __fmaf_rn(sinU, cosV,  cosU * sinV);

                        // Accumulazione IDFT con due FMA concatenate
                        float step1 = __fmaf_rn(-tileImag[dv][du], s, sum);
                        sum         = __fmaf_rn( tileReal[dv][du],  c, step1);
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

    int Xmezzi = width / 2;
    int Ymezzi = height / 2;
    float d;

    if (x < width && y < height) {
        d = sqrtf((x - Xmezzi) * (x - Xmezzi) + (y - Ymezzi) * (y - Ymezzi));

        if (d > cutoff) {
                dft[y * width + x].real = 0;
                dft[y * width + x].imag = 0;
        }
    }
    
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Uso: %s <percorso_immagine.pgm> \n", argv[0]);
        return 1;
    }
    const char *inputFile =argv[1];
    const char *outputFile = "output_cuda-512.pgm";
    float raggioFiltro = 260.0f; // 130 per 256 px    260 512 px

    
/*
    // --- CUDA events per timing ---
    cudaEvent_t start, stop;
    float ms = 0.0f;

    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
*/
    PGMImage img = readPGM(inputFile);

    printf("Immagine caricata di dimensioni H:%d W:%d\n", img.height, img.width);

    unsigned char *d_in, *d_out;
    MyComplex *d_dft;
    CHECK(cudaMalloc(&d_in, img.width * img.height));
    CHECK(cudaMalloc(&d_out, img.width * img.height));
    CHECK(cudaMalloc(&d_dft, img.width * img.height * sizeof(MyComplex)));

    cudaMemcpy(d_in, img.data, img.width * img.height, cudaMemcpyHostToDevice);

    dim3 threadsNum(16, 16);
    dim3 numBlocchi((img.width + threadsNum.x - 1) / threadsNum.x, (img.height + threadsNum.y - 1) / threadsNum.y);

    //-------------------------------------- DFT


  
    // start timer
    //cudaEventRecord(start);

    dft2D_opt_shared<<<numBlocchi, threadsNum>>>(d_in, d_dft, img.width, img.height);
    CHECK(cudaDeviceSynchronize());


    // stop timer
    //cudaEventRecord(stop);

    // attende fine kernel
    //cudaEventSynchronize(stop);
    // calcola tempo
    //cudaEventElapsedTime(&ms, start, stop);
    //printf("Tempo DFT: %f ms\n", ms);

    //-------------------------------------- 

    //-------------------------------------- FILTRO


    //reset timer
    //cudaEventRecord(start);
   
    filtro<<<numBlocchi, threadsNum>>>(d_dft, img.width, img.height, raggioFiltro);
    CHECK(cudaDeviceSynchronize());

    // stop timer
    //cudaEventRecord(stop);
    // attende fine kernel
    //cudaEventSynchronize(stop);
    // calcola tempo
    //cudaEventElapsedTime(&ms, start, stop);
    //printf("Tempo filtro: %f ms\n", ms);   
    
    //--------------------------------------

    //-------------------------------------- iDFT

    //reset timer
    //cudaEventRecord(start);

    idft2D_opt_shared<<<numBlocchi, threadsNum>>>(d_dft, d_out, img.width, img.height);
    CHECK(cudaDeviceSynchronize());

    // stop timer
    //cudaEventRecord(stop);
    // attende fine kernel
    //cudaEventSynchronize(stop);
    // calcola tempo
    //cudaEventElapsedTime(&ms, start, stop);
    //printf("Tempo iDFT: %f ms\n", ms);
    //-------------------------------------- 

    cudaMemcpy(img.data, d_out, img.width * img.height, cudaMemcpyDeviceToHost);

    writePGM(outputFile, img);


    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_dft);
    free(img.data);

    printf("Trasformata e antitrasformata completate. Risultato salvato in %s\n", outputFile);

    return 0;
}