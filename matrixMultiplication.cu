#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// Utility function to initialize matrices
void initializeMatrix(float* matrix, int size, bool randomize = true) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = randomize ? static_cast<float>(rand() % 100) / 10.0f : 0.0f;
    }
}

// CPU matrix multiplication
void matrixMultiplyCPU(float* A, float* B, float* C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
			C[i * size + j] = sum;
        }
    }
}

// GPU kernel for matrix multiplication
__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        float sum = 0.0f;
        for (int k = 0; k < size; k++) {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}

// GPU matrix multiplication
void matrixMultiplyGPU(float* A, float* B, float* C, int size, double* time) {
    float* d_A, * d_B, * d_C;
    size_t bytes = size * size * sizeof(float);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((size + block.x - 1) / block.x, (size + block.y - 1) / block.y);
    
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyKernel << <grid, block >> > (d_A, d_B, d_C, size);
	cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
	time[0] = std::chrono::duration<double, std::milli>(end - start).count();

    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Transpose matrix
void transposeMatrix(float* B, float* B_T, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            B_T[j * size + i] = B[i * size + j];
        }
    }
}

// Matrix multiplication with transposed matrix
void matrixMultiplyTransposed(float* A, float* B_T, float* C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i * size + j] = 0.0f;
            for (int k = 0; k < size; k++) {
                C[i * size + j] += A[i * size + k] * B_T[j * size + k];
            }
        }
    }
}

// Measure execution times and compare
void benchmark(int size) {
    std::cout << "Matrix size: " << size << "x" << size << std::endl;
	int matrixDimention = size * size;

    float* A = new float[matrixDimention];
    float* B = new float[matrixDimention];
    float* C = new float[matrixDimention];
    float* B_T = new float[matrixDimention];

    initializeMatrix(A, size);
    initializeMatrix(B, size);

    // CPU multiplication
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyCPU(A, B, C, size);
    auto end = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "CPU time: " << cpuTime << " ms" << std::endl;

    // GPU multiplication
    double* gpuTime = new double[1];
    matrixMultiplyGPU(A, B, C, size, gpuTime);
    std::cout << "GPU time: " << gpuTime[0] << " ms" << std::endl;
    
    // Transposed multiplication
    transposeMatrix(B, B_T, size);
    start = std::chrono::high_resolution_clock::now();
    matrixMultiplyTransposed(A, B_T, C, size);
    end = std::chrono::high_resolution_clock::now();
    double transposedTime = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Transposed CPU time: " << transposedTime << " ms" << std::endl;

    // Compute speedups
    double slowest = std::max({ cpuTime, gpuTime[0], transposedTime});
    std::cout << "Speedup (CPU vs GPU): " << cpuTime / gpuTime[0] << "x" << std::endl;
    std::cout << "Speedup (Transposed vs CPU): " << slowest / transposedTime << "x\n" << std::endl;
    

    // Free memory
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] B_T;
	delete[] gpuTime;
}

int main() {
    srand(0);
    benchmark(2);
    benchmark(512);
    benchmark(1024);
    benchmark(2048);
    return 0;
}
