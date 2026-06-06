#ifndef EDGE_DETECTION_H
#define EDGE_DETECTION_H

#include <cuda_runtime.h>
#include <cufft.h>

// Struttura per metriche di performance
typedef struct {
    float time_fft_forward;
    float time_kernel_multiply;
    float time_fft_inverse;
    float time_magnitude;
    float time_total;
    float bandwidth_gbps;
} PerformanceMetrics;

// Dichiarazioni funzioni kernel
__global__ void laplacian_filter_kernel(cufftComplex *spectrum, int width, int height, float scale);
__global__ void sobel_filter_kernel(cufftComplex *spectrum, int width, int height, float scale);
__global__ void extract_magnitude_kernel(cufftComplex *spectrum, float *magnitude, int width, int height);
__global__ void normalize_kernel(float *data, int width, int height, int num_pixels);
__global__ void clip_to_byte_kernel(float *data, unsigned char *output, int width, int height);

// Dichiarazioni wrapper
void edge_detection_laplacian(
    float *d_input,
    float *d_output,
    int width,
    int height,
    PerformanceMetrics *metrics
);

void edge_detection_sobel(
    float *d_input,
    unsigned char *d_output,
    int width,
    int height,
    PerformanceMetrics *metrics
);

// Utility
void print_metrics(PerformanceMetrics *metrics, int width, int height);

// Helper macro per controllo errori CUDA
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// Helper macro per controllo errori cuFFT
#define CHECK_CUFFT_ERROR(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            fprintf(stderr, "cuFFT Error: %d\n", err); \
            exit(1); \
        } \
    } while (0)

#endif
