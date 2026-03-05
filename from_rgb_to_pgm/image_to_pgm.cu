#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_SIZE 16
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// Kernel CUDA per convertire RGB a scala di grigi (luminosity method)
__global__ void rgbToGrayKernel(unsigned char *input, unsigned char *output, 
                                 int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int pixel_idx = (y * width + x);
        int input_idx = pixel_idx * channels;
        
        if (channels == 1) {
            // Già in scala di grigi
            output[pixel_idx] = input[input_idx];
        } else if (channels == 3) {
            // RGB to Gray: 0.299*R + 0.587*G + 0.114*B
            unsigned char r = input[input_idx];
            unsigned char g = input[input_idx + 1];
            unsigned char b = input[input_idx + 2];
            
            float gray = 0.299f * r + 0.587f * g + 0.114f * b;
            output[pixel_idx] = (unsigned char)gray;
        } else if (channels == 4) {
            // RGBA to Gray
            unsigned char r = input[input_idx];
            unsigned char g = input[input_idx + 1];
            unsigned char b = input[input_idx + 2];
            
            float gray = 0.299f * r + 0.587f * g + 0.114f * b;
            output[pixel_idx] = (unsigned char)gray;
        }
    }
}

// Funzione per leggere un'immagine PPM (supporta JPEG, PNG tramite conversione)
int readImagePPM(const char *filename, unsigned char **image_data, 
                  int *width, int *height, int *channels) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Errore: impossibile aprire file %s\n", filename);
        return 0;
    }
    
    char magic[3];
    fscanf(file, "%2s", magic);
    
    if (magic[0] != 'P' || (magic[1] != '5' && magic[1] != '6')) {
        fprintf(stderr, "Errore: formato non supportato. Usa PPM (P5 o P6)\n");
        fprintf(stderr, "Converti prima l'immagine con: convert input.jpg -depth 8 output.ppm\n");
        fclose(file);
        return 0;
    }
    
    *channels = (magic[1] == '5') ? 1 : 3; // P5=grayscale, P6=RGB
    
    // Skip comments
    int c = fgetc(file);
    while (c == '#') {
        while (fgetc(file) != '\n');
        c = fgetc(file);
    }
    ungetc(c, file);
    
    // Leggi dimensioni
    fscanf(file, "%d %d", width, height);
    
    int maxval;
    fscanf(file, "%d", &maxval);
    fgetc(file); // skip whitespace
    
    int image_size = (*width) * (*height) * (*channels);
    *image_data = (unsigned char *)malloc(image_size);
    
    if (fread(*image_data, 1, image_size, file) != (size_t)image_size) {
        fprintf(stderr, "Errore: lettura incompleta dell'immagine\n");
        free(*image_data);
        fclose(file);
        return 0;
    }
    
    fclose(file);
    return 1;
}

// Funzione per scrivere immagine in formato PGM
int writePGM(const char *filename, unsigned char *image_data, 
              int width, int height) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Errore: impossibile creare file %s\n", filename);
        return 0;
    }
    
    // Header PGM
    fprintf(file, "P5\n");
    fprintf(file, "%d %d\n", width, height);
    fprintf(file, "255\n");
    
    // Scrivi dati
    if (fwrite(image_data, 1, width * height, file) != (size_t)(width * height)) {
        fprintf(stderr, "Errore: scrittura incompleta del file PGM\n");
        fclose(file);
        return 0;
    }
    
    fclose(file);
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Uso: %s <input.ppm> <output.pgm>\n", argv[0]);
        printf("\nNota: Converti l'immagine in PPM prima:\n");
        printf("  convert input.jpg -depth 8 input.ppm\n");
        printf("  convert input.png -depth 8 input.ppm\n");
        printf("\nSe non hai ImageMagick, usa questo script Python:\n");
        printf("  python3 convert_to_ppm.py input.jpg output.ppm\n");
        return 1;
    }
    
    cudaEvent_t start, stop;
    float ms = 0.0f;

    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const char *input_file = argv[1];
    const char *output_file = argv[2];
    
    printf("=== Conversione Immagine a PGM con CUDA ===\n");
    printf("Input: %s\n", input_file);
    printf("Output: %s\n\n", output_file);
    
    // Leggi immagine
    unsigned char *h_input = NULL;
    int width, height, channels;
    
    if (!readImagePPM(input_file, &h_input, &width, &height, &channels)) {
        return 1;
    }
    
    printf("Dimensioni immagine: %d x %d\n", width, height);
    printf("Canali: %d\n", channels);
    
    int input_size = width * height * channels;
    int output_size = width * height;
    
    // Alloca memoria sulla GPU
    unsigned char *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input_size));
    CHECK_CUDA(cudaMalloc(&d_output, output_size));
    
    // Copia input sulla GPU
    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    
    // Configura grid e block
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    printf("Grid: (%d, %d), Block: (%d, %d)\n\n", 
           grid.x, grid.y, block.x, block.y);
    
    // Esegui kernel
    printf("Esecuzione kernel CUDA...\n");

    cudaEventRecord(start);

    rgbToGrayKernel<<<grid, block>>>(d_input, d_output, width, height, channels);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("Kernel completato!\n\n");
    // stop timer
    cudaEventRecord(stop);

    // attende fine kernel
    cudaEventSynchronize(stop);
    // calcola tempo
    cudaEventElapsedTime(&ms, start, stop);
    printf("Tempo rgbtogray: %f ms\n", ms);
    
    // Copia risultato sulla CPU
    unsigned char *h_output = (unsigned char *)malloc(output_size);
    CHECK_CUDA(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));
    
    // Scrivi file PGM
    printf("Scrittura file PGM...\n");
    if (writePGM(output_file, h_output, width, height)) {
        printf("Conversione completata con successo!\n");
        printf("File salvato: %s\n", output_file);
    } else {
        fprintf(stderr, "Errore nella scrittura del file\n");
    }
    
    // Pulizia
    free(h_input);
    free(h_output);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    
    return 0;
}
