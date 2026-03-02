#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


//TODO - Aggiungere misurazioni sulle prestazioni

__global__ void rgbToGrayGPU(unsigned char *d_rgb, unsigned char *d_gray, int width, int height) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < width && iy < height) {
        int rgbOffset = (iy * width + ix) * 3;
        int grayOffset = iy * width + ix;
        unsigned char r = d_rgb[rgbOffset];
        unsigned char g = d_rgb[rgbOffset + 1];
        unsigned char b = d_rgb[rgbOffset + 2];

        d_gray[grayOffset] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

int main(int argc, char *argv[]) {
    // Controllo input
    if (argc < 2) {
        printf("Uso: %s <percorso_immagine> [output.jpg]\n", argv[0]);
        printf("Esempio: %s foto.jpg output.jpg\n", argv[0]);
        return 1;
    }

    const char *input_path = argv[1];
    const char *output_path = (argc > 2) ? argv[2] : "output_gray.jpg";

    // Carica immagine
    int width, height, channels;
    unsigned char *h_rgb = stbi_load(input_path, &width, &height, &channels, 3);

    if (!h_rgb) {
        printf("Errore: impossibile caricare l'immagine '%s'\n", input_path);
        return 1;
    }

    printf("Immagine caricata: %d x %d pixel, %d canali\n", width, height, channels);

    // Alloca memoria per output
    unsigned char *h_gray = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    if (!h_gray) {
        printf("Errore: impossibile allocare memoria host\n");
        stbi_image_free(h_rgb);
        return 1;
    }

    // Alloca memoria GPU
    unsigned char *d_rgb, *d_gray;
    cudaMalloc((void **)&d_rgb, width * height * 3 * sizeof(unsigned char));
    cudaMalloc((void **)&d_gray, width * height * sizeof(unsigned char));

    if (!d_rgb || !d_gray) {
        printf("Errore: impossibile allocare memoria GPU\n");
        free(h_gray);
        stbi_image_free(h_rgb);
        return 1;
    }

    // Copia RGB da CPU a GPU
    cudaMemcpy(d_rgb, h_rgb, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Configura blocchi e thread
    dim3 blockSize(16, 16);  // 256 thread per blocco
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    printf("Lancio kernel: Grid(%d, %d), Block(%d, %d)\n", 
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);

    // Lancia kernel
    rgbToGrayGPU<<<gridSize, blockSize>>>(d_rgb, d_gray, width, height);

    // Controlla errori
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Errore kernel: %s\n", cudaGetErrorString(err));
        cudaFree(d_rgb);
        cudaFree(d_gray);
        free(h_gray);
        stbi_image_free(h_rgb);
        return 1;
    }

    // Sincronizza GPU
    cudaDeviceSynchronize();

    // Copia risultato da GPU a CPU
    cudaMemcpy(h_gray, d_gray, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Salva immagine in scala di grigi
    if (stbi_write_jpg(output_path, width, height, 1, h_gray, 95)) {
        printf("Immagine salvata: %s\n", output_path);
    } else {
        printf("Errore: impossibile salvare l'immagine\n");
    }

    // Libera memoria
    cudaFree(d_rgb);
    cudaFree(d_gray);
    free(h_gray);
    stbi_image_free(h_rgb);

    return 0;
}