#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>

// Struttura per memorizzare le metriche
typedef struct {
    float time_fft_forward;
    float time_kernel_multiply;
    float time_fft_inverse;
    float time_magnitude;
    float time_total;
    float bandwidth_gbps;
} PerformanceMetrics;

// Kernel per moltiplicare lo spettro FFT con il filtro Laplacian
__global__ void laplacian_filter_kernel(cufftComplex *spectrum, int width, int height, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        int pos = idy * width + idx;
        
        // Coordinate di frequenza normalizate
        float u = (idx < width / 2) ? (float)idx : (float)(idx - width);
        float v = (idy < height / 2) ? (float)idy : (float)(idy - height);
        
        // Normalizza rispetto alle dimensioni
        u /= width;
        v /= height;
        
        // Filtro Laplacian nel dominio Fourier: -4π²(u²+v²)
        float freq_magnitude_sq = u * u + v * v;
        float laplacian = -4.0f * M_PI * M_PI * freq_magnitude_sq;
        
        // Applica il filtro (moltiplicazione nello spazio complesso)
        float real = spectrum[pos].x;
        float imag = spectrum[pos].y;
        
        spectrum[pos].x = real * laplacian * scale;
        spectrum[pos].y = imag * laplacian * scale;
    }
}

// Kernel per moltiplicare lo spettro FFT con il filtro Sobel (approssimazione)
__global__ void sobel_filter_kernel(cufftComplex *spectrum, int width, int height, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        int pos = idy * width + idx;
        
        // Coordinate di frequenza normalizzate
        float u = (idx < width / 2) ? (float)idx : (float)(idx - width);
        float v = (idy < height / 2) ? (float)idy : (float)(idy - height);
        
        u /= width;
        v /= height;
        
        // Sobel approssimato: √(u² + v²)
        float sobel = sqrtf(u * u + v * v);
        
        // Applica il filtro
        float real = spectrum[pos].x;
        float imag = spectrum[pos].y;
        
        spectrum[pos].x = real * sobel * scale;
        spectrum[pos].y = imag * sobel * scale;
    }
}

// Kernel per estrarre la magnitudine dello spettro
__global__ void extract_magnitude_kernel(cufftComplex *spectrum, float *magnitude, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        int pos = idy * width + idx;
        float real = spectrum[pos].x;
        float imag = spectrum[pos].y;
        
        // Calcola magnitudine e normalizza con log per visualizzazione
        float mag = sqrtf(real * real + imag * imag);
        magnitude[pos] = logf(1.0f + mag);
    }
}

// Kernel per normalizzare l'output (IFFT richiede normalizzazione)
__global__ void normalize_kernel(float *data, int width, int height, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < width * height) {
        data[idx] /= num_pixels;
    }
}

// Kernel per normalizzazione min-max automatica
__global__ void auto_normalize_kernel(float *data, float min_val, float max_val, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < width * height) {
        float range = max_val - min_val;
        if (range < 1e-6f) range = 1.0f;
        
        // Normalizza a [0, 1]
        float normalized = (fabs(data[idx]) - min_val) / range;
        
        // Clipa e converte
        if (normalized > 1.0f) normalized = 1.0f;
        if (normalized < 0.0f) normalized = 0.0f;
        
        data[idx] = normalized * 255.0f;
    }
}

// Kernel per clippare valori tra 0 e 255 (con amplificazione)
__global__ void clip_to_byte_kernel(float *data, unsigned char *output, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < width * height) {
        float val = fabs(data[idx]);
        
        // Amplifica se i valori sono troppo piccoli
        val *= 50.0f;  // Amplificazione aggressiva
        
        if (val > 255.0f) val = 255.0f;
        if (val < 0.0f) val = 0.0f;
        output[idx] = (unsigned char)val;
    }
}

// Funzione wrapper per edge detection con Laplacian
void edge_detection_laplacian(
    float *d_input,
    float *d_output,
    int width,
    int height,
    PerformanceMetrics *metrics
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // Allocazione memoria per FFT (complessa)
    cufftComplex *d_spectrum;
    cudaMalloc((void**)&d_spectrum, width * height * sizeof(cufftComplex));
    
    // Crea piano FFT
    cufftHandle plan;
    cufftPlan2d(&plan, height, width, CUFFT_R2C);
    
    // FFT Forward (Real to Complex)
    cudaEventRecord(start);
    cufftExecR2C(plan, (cufftReal*)d_input, d_spectrum);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&metrics->time_fft_forward, start, stop);
    
    // Applica filtro Laplacian nel dominio Fourier
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    cudaEventRecord(start);
    laplacian_filter_kernel<<<grid, block>>>(d_spectrum, width, height, 1.0f);  // Aumenta da 0.1 a 1.0
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&metrics->time_kernel_multiply, start, stop);
    
    // Alloca memoria per output (reale dopo IFFT)
    float *d_temp;
    cudaMalloc((void**)&d_temp, width * height * sizeof(float));
    
    // FFT Inverse (Complex to Real)
    cufftHandle plan_inv;
    cufftPlan2d(&plan_inv, height, width, CUFFT_C2R);
    
    cudaEventRecord(start);
    cufftExecC2R(plan_inv, d_spectrum, (cufftReal*)d_temp);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&metrics->time_fft_inverse, start, stop);
    
    // Normalizza output
    int num_threads = 256;
    int num_blocks = (width * height + num_threads - 1) / num_threads;
    
    normalize_kernel<<<num_blocks, num_threads>>>(d_temp, width, height, width * height);
    cudaDeviceSynchronize();
    
    // Estrai magnitudine (per Laplacian, output è già reale)
    cudaEventRecord(start);
    clip_to_byte_kernel<<<num_blocks, num_threads>>>(d_temp, (unsigned char*)d_output, width, height);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&metrics->time_magnitude, start, stop);
    
    // Cleanup
    cudaFree(d_spectrum);
    cudaFree(d_temp);
    cufftDestroy(plan);
    cufftDestroy(plan_inv);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Funzione wrapper per edge detection con Sobel
void edge_detection_sobel(
    float *d_input,
    unsigned char *d_output,
    int width,
    int height,
    PerformanceMetrics *metrics
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Allocazione memoria per FFT
    cufftComplex *d_spectrum;
    cudaMalloc((void**)&d_spectrum, width * height * sizeof(cufftComplex));
    
    // Crea piano FFT
    cufftHandle plan;
    cufftPlan2d(&plan, height, width, CUFFT_R2C);
    
    // FFT Forward
    cudaEventRecord(start);
    cufftExecR2C(plan, (cufftReal*)d_input, d_spectrum);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&metrics->time_fft_forward, start, stop);
    
    // Applica filtro Sobel
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    cudaEventRecord(start);
    sobel_filter_kernel<<<grid, block>>>(d_spectrum, width, height, 0.5f);  // Aumenta da 0.05 a 0.5
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&metrics->time_kernel_multiply, start, stop);
    
    // FFT Inverse
    float *d_temp;
    cudaMalloc((void**)&d_temp, width * height * sizeof(float));
    
    cufftHandle plan_inv;
    cufftPlan2d(&plan_inv, height, width, CUFFT_C2R);
    
    cudaEventRecord(start);
    cufftExecC2R(plan_inv, d_spectrum, (cufftReal*)d_temp);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&metrics->time_fft_inverse, start, stop);
    
    // Normalizza e clippa
    int num_threads = 256;
    int num_blocks = (width * height + num_threads - 1) / num_threads;
    
    normalize_kernel<<<num_blocks, num_threads>>>(d_temp, width, height, width * height);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    clip_to_byte_kernel<<<num_blocks, num_threads>>>(d_temp, d_output, width, height);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&metrics->time_magnitude, start, stop);
    
    // Cleanup
    cudaFree(d_spectrum);
    cudaFree(d_temp);
    cufftDestroy(plan);
    cufftDestroy(plan_inv);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Funzione per calcolare e stampare le metriche
void print_metrics(PerformanceMetrics *metrics, int width, int height) {
    metrics->time_total = metrics->time_fft_forward + 
                         metrics->time_kernel_multiply + 
                         metrics->time_fft_inverse + 
                         metrics->time_magnitude;
    
    // Stima bandwidth (bytes trasferiti: FFT legge + scrive 2 volte)
    long long bytes = (long long)width * height * sizeof(float) * 4;
    metrics->bandwidth_gbps = (bytes / (1e9)) / (metrics->time_total / 1000.0f);
    
    printf("\n========== PERFORMANCE METRICS ==========\n");
    printf("Image size: %d x %d (%d pixels)\n", width, height, width * height);
    printf("FFT Forward:      %.3f ms\n", metrics->time_fft_forward);
    printf("Kernel Multiply:  %.3f ms\n", metrics->time_kernel_multiply);
    printf("FFT Inverse:      %.3f ms\n", metrics->time_fft_inverse);
    printf("Magnitude/Clip:   %.3f ms\n", metrics->time_magnitude);
    printf("----------------------------------------\n");
    printf("TOTAL TIME:       %.3f ms\n", metrics->time_total);
    printf("Estimated BW:     %.2f GB/s\n", metrics->bandwidth_gbps);
    printf("=========================================\n\n");
}
