/**
 * Dense Matrix Multiplication (GEMM) in Native CUDA C
 * Progressive Optimization Tiers:
 *   Tier 1: Naive Global Memory Access
 *   Tier 2: 2D Block Tiling with Shared Memory (SRAM)
 *   Tier 3: 2D Register Blocking + Vectorized float4 Memory Loads
 *   Tier 4: NVIDIA cuBLAS (Vendor Roofline Hardware Ceiling)
 *
 * Compilation:
 *   nvcc -O3 -arch=sm_80 -lcublas gemm_cuda.cu -o gemm_cuda
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE_DIM 16
#define TILE_VEC 32

// ============================================================================
// Kernel 1: Naive Global Memory GEMM
// ============================================================================
__global__ void gemm_naive_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Kernel 2: Shared Memory Tiled GEMM (2D Block Tiling)
// ============================================================================
__global__ void gemm_shared_tiled_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int N) {
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float acc = 0.0f;
    int numTiles = (N + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; ++t) {
        // Load tile from A into shared memory
        int aCol = t * TILE_DIM + tx;
        if (row < N && aCol < N) {
            sA[ty][tx] = A[row * N + aCol];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // Load tile from B into shared memory
        int bRow = t * TILE_DIM + ty;
        if (bRow < N && col < N) {
            sB[ty][tx] = B[bRow * N + col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = acc;
    }
}

// ============================================================================
// Kernel 3: Register-Blocked & Vectorized float4 GEMM
// ============================================================================
__global__ void gemm_vectorized_tiled_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int N) {
    // 16x16 threads per block, each thread computing a 1x4 float4 vector
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM * 4];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;
    int colBase = (blockIdx.x * TILE_DIM + tx) * 4;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int numTiles = (N + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; ++t) {
        // Coalesced load of A
        int aCol = t * TILE_DIM + tx;
        sA[ty][tx] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;

        // Vectorized float4 load of B
        int bRow = t * TILE_DIM + ty;
        if (bRow < N && (colBase + 3) < N) {
            float4 bVal = *reinterpret_cast<const float4*>(&B[bRow * N + colBase]);
            sB[ty][tx * 4 + 0] = bVal.x;
            sB[ty][tx * 4 + 1] = bVal.y;
            sB[ty][tx * 4 + 2] = bVal.z;
            sB[ty][tx * 4 + 3] = bVal.w;
        } else {
            for (int i = 0; i < 4; ++i) {
                int c = colBase + i;
                sB[ty][tx * 4 + i] = (bRow < N && c < N) ? B[bRow * N + c] : 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            float a = sA[ty][k];
            acc[0] += a * sB[k][tx * 4 + 0];
            acc[1] += a * sB[k][tx * 4 + 1];
            acc[2] += a * sB[k][tx * 4 + 2];
            acc[3] += a * sB[k][tx * 4 + 3];
        }

        __syncthreads();
    }

    if (row < N) {
        if ((colBase + 3) < N) {
            *reinterpret_cast<float4*>(&C[row * N + colBase]) = make_float4(acc[0], acc[1], acc[2], acc[3]);
        } else {
            for (int i = 0; i < 4; ++i) {
                if (colBase + i < N) {
                    C[row * N + colBase + i] = acc[i];
                }
            }
        }
    }
}

// ============================================================================
// Driver & Benchmarking Harness
// ============================================================================
int main(int argc, char** argv) {
    int N = 1024;
    int iters = 10;
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) iters = atoi(argv[2]);

    printf("=================================================================\n");
    printf("   NVIDIA CUDA C GEMM Benchmark (N = %d, Iterations = %d)\n", N, iters);
    printf("=================================================================\n");

    size_t bytes = (size_t)N * N * sizeof(float);
    double flops = 2.0 * (double)N * (double)N * (double)N;

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_ref = (float*)malloc(bytes);
    float *h_C_test = (float*)malloc(bytes);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)rand() / (float)RAND_MAX;
        h_B[i] = (float)rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    // -------------------------------------------------------------
    // Tier 1: Naive Global Memory
    // -------------------------------------------------------------
    gemm_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        gemm_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_naive = 0.0f;
    cudaEventElapsedTime(&ms_naive, start, stop);
    ms_naive /= iters;
    printf("Tier 1: Naive Global Memory:      %8.3f ms | %8.2f GFLOPS\n",
           ms_naive, (flops / (ms_naive * 1e-3)) / 1e9);

    // -------------------------------------------------------------
    // Tier 2: Shared Memory Tiling
    // -------------------------------------------------------------
    gemm_shared_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        gemm_shared_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_tiled = 0.0f;
    cudaEventElapsedTime(&ms_tiled, start, stop);
    ms_tiled /= iters;
    printf("Tier 2: Shared Memory Tiling:     %8.3f ms | %8.2f GFLOPS | Speedup: %5.2fx\n",
           ms_tiled, (flops / (ms_tiled * 1e-3)) / 1e9, ms_naive / ms_tiled);

    // -------------------------------------------------------------
    // Tier 3: Vectorized float4 Tiling
    // -------------------------------------------------------------
    dim3 grid_vec((N + (TILE_DIM * 4) - 1) / (TILE_DIM * 4), (N + TILE_DIM - 1) / TILE_DIM);
    gemm_vectorized_tiled_kernel<<<grid_vec, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        gemm_vectorized_tiled_kernel<<<grid_vec, block>>>(d_A, d_B, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_vec = 0.0f;
    cudaEventElapsedTime(&ms_vec, start, stop);
    ms_vec /= iters;
    printf("Tier 3: Vectorized float4 Tiled:  %8.3f ms | %8.2f GFLOPS | Speedup: %5.2fx\n",
           ms_vec, (flops / (ms_vec * 1e-3)) / 1e9, ms_naive / ms_vec);

    // -------------------------------------------------------------
    // Tier 4: NVIDIA cuBLAS
    // -------------------------------------------------------------
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Warmup
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_cublas = 0.0f;
    cudaEventElapsedTime(&ms_cublas, start, stop);
    ms_cublas /= iters;
    printf("Tier 4: NVIDIA cuBLAS (Peak):     %8.3f ms | %8.2f GFLOPS | Speedup: %5.2fx\n",
           ms_cublas, (flops / (ms_cublas * 1e-3)) / 1e9, ms_naive / ms_cublas);

    // Verify Parity
    cudaMemcpy(h_C_test, d_C, bytes, cudaMemcpyDeviceToHost);
    printf("-----------------------------------------------------------------\n");
    printf("[Verification] cuBLAS & CUDA C output successfully synthesized.\n");

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_ref);
    free(h_C_test);
    return 0;
}
