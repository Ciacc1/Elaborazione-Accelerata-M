#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "edge_detection.h"

// Struttura per immagine
typedef struct {
    unsigned char *data;
    int width;
    int height;
    int channels;
} Image;

// Leggi immagine PPM (formato semplice, non richiede librerie esterne)
Image* read_ppm(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return NULL;
    }
    
    char magic[3];
    fscanf(fp, "%s", magic);
    
    int width, height, maxval;
    fscanf(fp, "%d %d %d", &width, &height, &maxval);
    
    // Salta il whitespace
    fgetc(fp);
    
    Image *img = (Image*)malloc(sizeof(Image));
    img->width = width;
    img->height = height;
    img->channels = (strcmp(magic, "P6") == 0) ? 3 : 1;
    
    int size = width * height * img->channels;
    img->data = (unsigned char*)malloc(size);
    fread(img->data, 1, size, fp);
    
    fclose(fp);
    return img;
}

// Scrivi immagine PPM
void write_ppm(const char *filename, unsigned char *data, int width, int height, int channels) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot create file %s\n", filename);
        return;
    }
    
    fprintf(fp, "%s\n", (channels == 3) ? "P6" : "P5");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    
    fwrite(data, 1, width * height * channels, fp);
    fclose(fp);
}

// Converti immagine RGB a grayscale
float* rgb_to_grayscale(unsigned char *rgb_data, int width, int height) {
    float *gray = (float*)malloc(width * height * sizeof(float));
    
    for (int i = 0; i < width * height; i++) {
        // Formula standard: 0.299*R + 0.587*G + 0.114*B
        float r = rgb_data[i * 3 + 0] / 255.0f;
        float g = rgb_data[i * 3 + 1] / 255.0f;
        float b = rgb_data[i * 3 + 2] / 255.0f;
        gray[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
    
    return gray;
}

// Funzione principale
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_image> <output_image> [method]\n", argv[0]);
        printf("method: laplacian (default) or sobel\n");
        return 1;
    }
    
    const char *input_file = argv[1];
    const char *output_file = argv[2];
    const char *method = (argc > 3) ? argv[3] : "laplacian";
    
    printf("========== EDGE DETECTION VIA FFT ==========\n");
    printf("Input:  %s\n", input_file);
    printf("Output: %s\n", output_file);
    printf("Method: %s\n", method);
    printf("==========================================\n\n");
    
    // Leggi immagine
    Image *img = read_ppm(input_file);
    if (!img) return 1;
    
    printf("Image loaded: %d x %d\n", img->width, img->height);
    
    // Converti a grayscale se necessario
    float *h_gray = NULL;
    if (img->channels == 3) {
        printf("Converting RGB to grayscale...\n");
        h_gray = rgb_to_grayscale(img->data, img->width, img->height);
    } else {
        // Converti uint8 a float
        h_gray = (float*)malloc(img->width * img->height * sizeof(float));
        for (int i = 0; i < img->width * img->height; i++) {
            h_gray[i] = img->data[i] / 255.0f;
        }
    }
    
    // Alloca memoria GPU
    float *d_input;
    unsigned char *d_output;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, img->width * img->height * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, img->width * img->height * sizeof(unsigned char)));
    
    // Copia input su GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_gray, img->width * img->height * sizeof(float), cudaMemcpyHostToDevice));
    
    // Esegui edge detection
    PerformanceMetrics metrics = {0};
    
    printf("Running edge detection (%s)...\n", method);
    
    if (strcmp(method, "sobel") == 0) {
        edge_detection_sobel(d_input, d_output, img->width, img->height, &metrics);
    } else {
        // Per Laplacian, serve un cast diverso nel main
        float *d_output_float;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output_float, img->width * img->height * sizeof(float)));
        edge_detection_laplacian(d_input, d_output_float, img->width, img->height, &metrics);
        
        // Copia risultato float su GPU as uchar
        int num_threads = 256;
        int num_blocks = (img->width * img->height + num_threads - 1) / num_threads;
        
        // Temporary kernel per conversione
        extern __global__ void clip_to_byte_kernel(float *data, unsigned char *output, int width, int height);
        clip_to_byte_kernel<<<num_blocks, num_threads>>>(d_output_float, d_output, img->width, img->height);
        
        cudaFree(d_output_float);
    }
    
    // Copia risultato su host
    unsigned char *h_output = (unsigned char*)malloc(img->width * img->height);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, img->width * img->height, cudaMemcpyDeviceToHost));
    
    // Salva risultato
    write_ppm(output_file, h_output, img->width, img->height, 1);
    printf("Output saved: %s\n\n", output_file);
    
    // Stampa metriche
    print_metrics(&metrics, img->width, img->height);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_gray);
    free(h_output);
    free(img->data);
    free(img);
    
    return 0;
}
