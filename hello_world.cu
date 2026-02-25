#include <stdio.h>
#include <cuda_runtime.h>

// Kernel CUDA
__global__ void helloFromGPU()
{
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

int main()
{
    // --- CUDA events per timing ---
    cudaEvent_t start, stop;
    float ms = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start timer
    cudaEventRecord(start);

    // Launch kernel
    helloFromGPU<<<1, 10>>>();

    // stop timer
    cudaEventRecord(stop);

    // attende fine kernel
    cudaEventSynchronize(stop);

    // calcola tempo
    cudaEventElapsedTime(&ms, start, stop);

    printf("Kernel time: %f ms\n", ms);

    // cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();

    return 0;
}
