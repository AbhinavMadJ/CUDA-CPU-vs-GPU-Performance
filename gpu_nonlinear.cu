#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 10000000

__global__ void nonlinear_kernel(float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = sinf(x[idx]) + logf(x[idx] + 1.0f) + sqrtf(x[idx]);
    }
}

int main() {
    float *x_host, *y_host;
    float *x_dev, *y_dev;

    x_host = (float*)malloc(N * sizeof(float));
    y_host = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x_host[i] = ((float)rand() / RAND_MAX);
    }

    cudaMalloc(&x_dev, N * sizeof(float));
    cudaMalloc(&y_dev, N * sizeof(float));
    cudaMemcpy(x_dev, x_host, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Threads per block: %d\n", threadsPerBlock);
    printf("Blocks per grid:   %d\n", blocksPerGrid);
    printf("Total GPU threads launched: %d\n", blocksPerGrid * threadsPerBlock);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    nonlinear_kernel<<<blocksPerGrid, threadsPerBlock>>>(x_dev, y_dev, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU time: %.4f s\n", ms / 1000.0f);

    cudaMemcpy(y_host, y_dev, N * sizeof(float), cudaMemcpyDeviceToHost);

    free(x_host);
    free(y_host);
    cudaFree(x_dev);
    cudaFree(y_dev);
    return 0;
}