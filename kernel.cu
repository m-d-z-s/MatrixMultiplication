#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

// Выделение памяти для одномерной матрицы
float* allocateMatrix(int size) {
    return new float[size * size];
}

// Освобождение памяти
void freeMatrix(float* matrix) {
    delete[] matrix;
}

// Заполнение матрицы случайными числами
void fillMatrix(float* matrix, int size) {
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
    }
}

// Печать матрицы
void printMatrix(float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << matrix[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Умножение матриц на CPU
void multiplyMatrixCPU(float* A, float* B, float* C, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            C[i * size + j] = 0.0f;
            for (int k = 0; k < size; ++k) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}

// Транспонирование матрицы на CPU
void transposeMatrix(float* B, float* B_T, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            B_T[j * size + i] = B[i * size + j];
        }
    }
}

// Умножение с транспонированной матрицей на CPU
void multiplyMatrixWithTranspose(float* A, float* B_T, float* C, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            C[i * size + j] = 0.0f;
            for (int k = 0; k < size; ++k) {
                C[i * size + j] += A[i * size + k] * B_T[j * size + k];
            }
        }
    }
}

// Ядро CUDA для обычного умножения
__global__ void multiplyMatrixGPUKernel(float* A, float* B, float* C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        float sum = 0.0f;
        for (int k = 0; k < size; ++k) {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}

// Ядро CUDA для транспонирования
__global__ void transposeMatrixGPUKernel(float* B, float* B_T, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        B_T[col * size + row] = B[row * size + col];
    }
}

// Вызов GPU для транспонирования матрицы
void transposeMatrixGPU(float* d_B, float* d_B_T, int size) {
    dim3 block(16, 16);
    dim3 grid((size + block.x - 1) / block.x, (size + block.y - 1) / block.y);
    transposeMatrixGPUKernel << <grid, block >> > (d_B, d_B_T, size);
}

// Вызов GPU для умножения матриц
void multiplyMatrixGPU(float* A, float* B, float* C, int size) {
    float* d_A, * d_B, * d_C;
    size_t bytes = size * size * sizeof(float);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((size + block.x - 1) / block.x, (size + block.y - 1) / block.y);

    multiplyMatrixGPUKernel << <grid, block >> > (d_A, d_B, d_C, size);

    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Умножение на транспонированной матрице на GPU
void multiplyMatrixWithTransposeGPU(float* A, float* B, float* C, int size) {
    float* d_A, * d_B, * d_B_T, * d_C;
    size_t bytes = size * size * sizeof(float);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_B_T, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    // Транспонирование на GPU
    transposeMatrixGPU(d_B, d_B_T, size);

    dim3 block(16, 16);
    dim3 grid((size + block.x - 1) / block.x, (size + block.y - 1) / block.y);

    multiplyMatrixGPUKernel << <grid, block >> > (d_A, d_B_T, d_C, size);

    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_B_T);
    cudaFree(d_C);
}

// Измерение времени выполнения
void measureTime() {
    int sizes[] = { 2, 512, 1024, 2048 };

    for (int size : sizes) {
        float* A = allocateMatrix(size);
        float* B = allocateMatrix(size);
        float* C = allocateMatrix(size);
        float* B_T = allocateMatrix(size);

        fillMatrix(A, size);
        fillMatrix(B, size);

        if (size == 2) {
            std::cout << "Matrix A:" << std::endl;
            printMatrix(A, size);
            std::cout << "Matrix B:" << std::endl;
            printMatrix(B, size);
        }

        // CPU
        clock_t start = clock();
        multiplyMatrixCPU(A, B, C, size);
        clock_t end = clock();
        double cpu_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
        if (size == 2) {
            std::cout << "Result (CPU):" << std::endl;
            printMatrix(C, size);
        }

        // Transposed CPU
        start = clock();
        transposeMatrix(B, B_T, size);
        multiplyMatrixWithTranspose(A, B_T, C, size);
        end = clock();
        double transposed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
        if (size == 2) {
            std::cout << "Result (Transposed CPU):" << std::endl;
            printMatrix(C, size);
        }

        // GPU
        start = clock();
        multiplyMatrixGPU(A, B, C, size);
        end = clock();
        double gpu_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
        if (size == 2) {
            std::cout << "Result (GPU):" << std::endl;
            printMatrix(C, size);
        }

        // Transposed GPU
        start = clock();
        multiplyMatrixWithTransposeGPU(A, B, C, size);
        end = clock();
        double transposed_gpu_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;

        std::cout << "Matrix size: " << size << "x" << size << std::endl;
        std::cout << "CPU time: " << cpu_time << " seconds" << std::endl;
        std::cout << "Transposed CPU time: " << transposed_time << " seconds" << std::endl;
        std::cout << "GPU time: " << gpu_time << " seconds" << std::endl;
        std::cout << "Transposed GPU time: " << transposed_gpu_time << " seconds" << std::endl;

        std::cout << "Speedup (CPU/GPU): " << cpu_time / gpu_time << std::endl;
        std::cout << "Speedup (Transposed CPU/GPU): " << transposed_time / gpu_time << std::endl;
        std::cout << "Speedup (Transposed GPU/GPU): " << transposed_gpu_time /gpu_time << std::endl;
        std::cout << std::endl;

        freeMatrix(A);
        freeMatrix(B);
        freeMatrix(C);
        freeMatrix(B_T);
    }
}

int main() {
    measureTime();
    return 0;
}
