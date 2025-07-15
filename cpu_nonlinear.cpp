#include <stdio.h>
#include <math.h>
#include <chrono>

#define N 10000000

void nonlinear_cpu(float* x, float* y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = sinf(x[i]) + logf(x[i] + 1.0f) + sqrtf(x[i]);
    }
}

int main() {
    float *x, *y;
    x = new float[N];
    y = new float[N];

    // Initialize input
    for (int i = 0; i < N; i++) {
        x[i] = ((float)rand() / RAND_MAX);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    nonlinear_cpu(x, y, N);
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = t2 - t1;
    printf("CPU time: %.4f s\n", elapsed.count());

    delete[] x;
    delete[] y;
    return 0;
}
