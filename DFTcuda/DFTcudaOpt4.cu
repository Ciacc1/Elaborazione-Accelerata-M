#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>


#define PI 3.14159265358979323846
#define TILE_SIZE 16

typedef struct {
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

__global__ void dft_opt_shared(unsigned char *in, MyComplex *out, int width, int height) {

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
                for (int dv = 0; dv < TILE_SIZE < height; dv++) {

                    // sin/cos verticale: costante per tutto il loop su du
                    float cosV, sinV;
                    sincosf(angleY * (tileV + dv), &sinV, &cosV);
                    
                    #pragma unroll 8
                    for (int du = 0; du < TILE_SIZE < width; du++) {

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

__global__ void idft_opt(MyComplex *in, unsigned char *out, int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;

    float angleX = 2.0f * PI * (float)x / width;
    float angleY = 2.0f * PI * (float)y / height;


    for (int v = 0; v < height; v++) {
        float cosV, sinV;
        sincosf(angleY * v, &sinV, &cosV);

        // CONTROLLO SAFE PATH GLOBALE (Multiplo di 8)
        if (width % 8 == 0) {

            // SAFE PATH a blocchi di 8
            for (int u = 0; u < width; u += 8) {

                float cosU0, sinU0, cosU1, sinU1, cosU2, sinU2, cosU3, sinU3;
                float cosU4, sinU4, cosU5, sinU5, cosU6, sinU6, cosU7, sinU7;

                sincosf(angleX * (u + 0), &sinU0, &cosU0);
                sincosf(angleX * (u + 1), &sinU1, &cosU1);
                sincosf(angleX * (u + 2), &sinU2, &cosU2);
                sincosf(angleX * (u + 3), &sinU3, &cosU3);
                sincosf(angleX * (u + 4), &sinU4, &cosU4);
                sincosf(angleX * (u + 5), &sinU5, &cosU5);
                sincosf(angleX * (u + 6), &sinU6, &cosU6);
                sincosf(angleX * (u + 7), &sinU7, &cosU7);

                MyComplex coeff0 = in[v * width + (u + 0)];
                MyComplex coeff1 = in[v * width + (u + 1)];
                MyComplex coeff2 = in[v * width + (u + 2)];
                MyComplex coeff3 = in[v * width + (u + 3)];
                MyComplex coeff4 = in[v * width + (u + 4)];
                MyComplex coeff5 = in[v * width + (u + 5)];
                MyComplex coeff6 = in[v * width + (u + 6)];
                MyComplex coeff7 = in[v * width + (u + 7)];

                float c0 = __fmaf_rn(cosU0, cosV, -sinU0 * sinV);
                float s0 = __fmaf_rn(sinU0, cosV,  cosU0 * sinV);
                float st1_0 = __fmaf_rn(-coeff0.imag, s0, sum);
                sum = __fmaf_rn(coeff0.real, c0, st1_0);

                float c1 = __fmaf_rn(cosU1, cosV, -sinU1 * sinV);
                float s1 = __fmaf_rn(sinU1, cosV,  cosU1 * sinV);
                float st1_1 = __fmaf_rn(-coeff1.imag, s1, sum);
                sum = __fmaf_rn(coeff1.real, c1, st1_1);

                float c2 = __fmaf_rn(cosU2, cosV, -sinU2 * sinV);
                float s2 = __fmaf_rn(sinU2, cosV,  cosU2 * sinV);
                float st1_2 = __fmaf_rn(-coeff2.imag, s2, sum);
                sum = __fmaf_rn(coeff2.real, c2, st1_2);

                float c3 = __fmaf_rn(cosU3, cosV, -sinU3 * sinV);
                float s3 = __fmaf_rn(sinU3, cosV,  cosU3 * sinV);
                float st1_3 = __fmaf_rn(-coeff3.imag, s3, sum);
                sum = __fmaf_rn(coeff3.real, c3, st1_3);

                float c4 = __fmaf_rn(cosU4, cosV, -sinU4 * sinV);
                float s4 = __fmaf_rn(sinU4, cosV,  cosU4 * sinV);
                float st1_4 = __fmaf_rn(-coeff4.imag, s4, sum);
                sum = __fmaf_rn(coeff4.real, c4, st1_4);

                float c5 = __fmaf_rn(cosU5, cosV, -sinU5 * sinV);
                float s5 = __fmaf_rn(sinU5, cosV,  cosU5 * sinV);
                float st1_5 = __fmaf_rn(-coeff5.imag, s5, sum);
                sum = __fmaf_rn(coeff5.real, c5, st1_5);

                float c6 = __fmaf_rn(cosU6, cosV, -sinU6 * sinV);
                float s6 = __fmaf_rn(sinU6, cosV,  cosU6 * sinV);
                float st1_6 = __fmaf_rn(-coeff6.imag, s6, sum);
                sum = __fmaf_rn(coeff6.real, c6, st1_6);

                float c7 = __fmaf_rn(cosU7, cosV, -sinU7 * sinV);
                float s7 = __fmaf_rn(sinU7, cosV,  cosU7 * sinV);
                float st1_7 = __fmaf_rn(-coeff7.imag, s7, sum);
                sum = __fmaf_rn(coeff7.real, c7, st1_7);
            }

        } else {
            // EDGE PATH (Fallback normale)
            for (int u = 0; u < width; u++) {
                float cosU, sinU;
                sincosf(angleX * u, &sinU, &cosU);

                float c = __fmaf_rn(cosU, cosV, -sinU * sinV);
                float s = __fmaf_rn(sinU, cosV,  cosU * sinV);

                MyComplex coeff = in[v * width + u];
                float step1 = __fmaf_rn(-coeff.imag, s, sum);
                sum = __fmaf_rn( coeff.real,  c, step1);
            }
        }
    }

    sum /= (float)(width * height);
    out[y * width + x] = (unsigned char)(sum < 0.0f ? 0 : (sum > 255.0f ? 255 : sum));
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
    float raggioFiltro = 280.0f; // 130 per 256 px    260 512 px

    
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

    dim3 threadsDFT(16, 8);
    dim3 threadsFilter(16, 16);

    dim3 blocksDFT((img.width  + threadsDFT.x - 1) / threadsDFT.x, (img.height + threadsDFT.y - 1) / threadsDFT.y);
    dim3 blocksFilter((img.width  + threadsFilter.x - 1) / threadsFilter.x, (img.height + threadsFilter.y - 1) / threadsFilter.y);

    //-------------------------------------- DFT


  
    // start timer
    //cudaEventRecord(start);

    dft_opt_shared<<<blocksDFT, threadsDFT>>>(d_in, d_dft, img.width, img.height);
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
   
    filtro<<<blocksFilter, threadsFilter>>>(d_dft, img.width, img.height, raggioFiltro);
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

    idft_opt<<<blocksDFT, threadsDFT>>>(d_dft, d_out, img.width, img.height);
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