#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>


#define PI 3.14159265358979323846

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

__global__ void dft_opt1(unsigned char *in, MyComplex *out, int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sumReal = 0.0f;
    float sumImag = 0.0f;

    // Precalcola i fattori angolari per questo thread
    float angleX = -2.0f * PI * (float)x / width;
    float angleY = -2.0f * PI * (float)y / height;

    for (int v = 0; v < height; v++) {

        // sin/cos della componente verticale -> costanti per tutto il loop su u
        float cosV, sinV;
        sincosf(angleY * v, &sinV, &cosV);

        for (int u = 0; u < width; u++) {

            // sin/cos della componente orizzontale -> costanti su tutto il loop su v
            float cosU, sinU;
            sincosf(angleX * u, &sinU, &cosU);

            // cos(U+V) = cosU*cosV - sinU*sinV
            float c = __fmaf_rn(cosU, cosV, -sinU * sinV);
            // sin(U+V) = sinU*cosV + cosU*sinV
            float s = __fmaf_rn(sinU, cosV,  cosU * sinV);

            float pixel = (float)in[v * width + u];
            
            // sum = pixel*c + sumReal
            sumReal = __fmaf_rn(pixel, c, sumReal);
            sumImag = __fmaf_rn(pixel, s, sumImag);
        }
    }

    out[y * width + x] = { sumReal, sumImag };
}

__global__ void idft_opt1(MyComplex *in, unsigned char *out, int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;

    float angleX = 2.0f * PI * (float)x / width;
    float angleY = 2.0f * PI * (float)y / height;

    for (int v = 0; v < height; v++) {

        // sin/cos verticale — costante per il loop su u
        float cosV, sinV;
        sincosf(angleY * v, &sinV, &cosV);

        for (int u = 0; u < width; u++) {

            float cosU, sinU;
            sincosf(angleX * u, &sinU, &cosU);

            // Formula addizione con FMA
            float c = __fmaf_rn(cosU, cosV, -sinU * sinV);
            float s = __fmaf_rn(sinU, cosV,  cosU * sinV);

            MyComplex coeff = in[v * width + u];

            // IDFT: real*c - imag*s + sum
            // Decomposto in due FMA concatenate:
            // step1 = __fmaf_rn(-imag, s, sum)   → -imag*s + sum
            // step2 = __fmaf_rn( real, c, step1) →  real*c + (-imag*s + sum)
            float step1 = __fmaf_rn(-coeff.imag, s, sum);
            sum         = __fmaf_rn( coeff.real,  c, step1);
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
    const char *outputFile = "output_cuda_opt1-512.pgm";
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

    dim3 threadsNum(16, 8);
    dim3 numBlocchi((img.width + threadsNum.x - 1) / threadsNum.x, (img.height + threadsNum.y - 1) / threadsNum.y);

    //-------------------------------------- DFT


  
    // start timer
    //cudaEventRecord(start);

    dft_opt1<<<numBlocchi, threadsNum>>>(d_in, d_dft, img.width, img.height);
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

    idft_opt1<<<numBlocchi, threadsNum>>>(d_dft, d_out, img.width, img.height);
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