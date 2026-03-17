#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>


#define PI 3.14159265358979323846

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


__global__ void dft2D(unsigned char *in, MyComplex *out, int width, int height) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    MyComplex sum;
    sum.real = 0;
    sum.imag = 0;

    double angle;
    double pixel;

    if (x < width && y < height) {

        for (int v = 0; v < height; v++) {
            for (int u = 0; u < width; u++) {
                
                angle = -2.0f * PI * ((double)x * u / width + (double)y * v / height);
                pixel = in[v * width + u];
                sum.real += pixel * cos(angle);
                sum.imag += pixel * sin(angle);
            
            }
        }
        out[y * width + x] = sum;
    }

}

__global__ void idft2D(MyComplex *in, unsigned char *out, int width, int height) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    double sum = 0;

    double angle;
    MyComplex coeff;

    if (x < width && y < height) {

        for (int v = 0; v < height; v++) {
            for (int u = 0; u < width; u++) {

                angle = 2.0f * PI * ((double)x * u / width + (double)y * v / height);
                coeff = in[v * width + u];
                sum += coeff.real * cos(angle) - coeff.imag * sin(angle);

            }
        }

        sum /= (width * height);
        out[y * width + x] = (unsigned char)(sum < 0 ? 0 : (sum > 255 ? 255 : sum));
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

    

    // --- CUDA events per timing ---
    cudaEvent_t start, stop;
    float ms = 0.0f;

    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

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
    cudaEventRecord(start);

    dft2D<<<numBlocchi, threadsNum>>>(d_in, d_dft, img.width, img.height);
    CHECK(cudaDeviceSynchronize());


    // stop timer
    cudaEventRecord(stop);

    // attende fine kernel
    cudaEventSynchronize(stop);
    // calcola tempo
    cudaEventElapsedTime(&ms, start, stop);
    printf("Tempo DFT: %f ms\n", ms);

    //-------------------------------------- 

    //-------------------------------------- FILTRO


    //reset timer
    cudaEventRecord(start);
   
    filtro<<<numBlocchi, threadsNum>>>(d_dft, img.width, img.height, raggioFiltro);
    CHECK(cudaDeviceSynchronize());

    // stop timer
    cudaEventRecord(stop);
    // attende fine kernel
    cudaEventSynchronize(stop);
    // calcola tempo
    cudaEventElapsedTime(&ms, start, stop);
    printf("Tempo filtro: %f ms\n", ms);   
    
    //--------------------------------------

    //-------------------------------------- iDFT

    //reset timer
    cudaEventRecord(start);

    idft2D<<<numBlocchi, threadsNum>>>(d_dft, d_out, img.width, img.height);
    CHECK(cudaDeviceSynchronize());

    // stop timer
    cudaEventRecord(stop);
    // attende fine kernel
    cudaEventSynchronize(stop);
    // calcola tempo
    cudaEventElapsedTime(&ms, start, stop);
    printf("Tempo iDFT: %f ms\n", ms);
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