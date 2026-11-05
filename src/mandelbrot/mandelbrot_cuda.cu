/**
 * Mandelbrot Fractal Generation in Native CUDA C
 * Workload Archetype: Latency-Bound & Highly Divergent Control Flow
 *
 * Progressive Optimization Tiers:
 *   Tier 1: Naive 2D Grid Thread Mapping (Extreme Warp Divergence)
 *   Tier 2: Coalesced 1D Grid Stride Mapping
 *   Tier 3: Multi-Point Vectorized Thread Unrolling (2 Points / Thread)
 *
 * Compilation:
 *   nvcc -O3 -arch=sm_80 mandelbrot_cuda.cu -o mandelbrot_cuda
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_X 16
#define BLOCK_Y 16

// ============================================================================
// Kernel 1: Naive 2D Grid Mapping (Subject to Severe Warp Divergence)
// ============================================================================
__global__ void mandelbrot_naive_kernel(int* __restrict__ output,
                                        int width, int height, int max_iter,
                                        float xmin, float xmax,
                                        float ymin, float ymax) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        float dx = (xmax - xmin) / (float)width;
        float dy = (ymax - ymin) / (float)height;

        float cx = xmin + (float)col * dx;
        float cy = ymin + (float)row * dy;

        float zx = 0.0f;
        float zy = 0.0f;
        int iter = 0;

        // Divergent loop: Threads in the same 32-thread warp will take vastly
        // different numbers of iterations to escape!
        while ((zx * zx + zy * zy <= 4.0f) && (iter < max_iter)) {
            float tmp = zx * zx - zy * zy + cx;
            zy = 2.0f * zx * zy + cy;
            zx = tmp;
            iter++;
        }

        output[row * width + col] = iter;
    }
}

// ============================================================================
// Kernel 2: 1D Grid-Stride Coalesced Loop (Improves Warp Locality)
// ============================================================================
__global__ void mandelbrot_grid_stride_kernel(int* __restrict__ output,
                                              int width, int height, int max_iter,
                                              float xmin, float xmax,
                                              float ymin, float ymax) {
    int total_pixels = width * height;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float dx = (xmax - xmin) / (float)width;
    float dy = (ymax - ymin) / (float)height;

    for (int idx = index; idx < total_pixels; idx += stride) {
        int col = idx % width;
        int row = idx / width;

        float cx = xmin + (float)col * dx;
        float cy = ymin + (float)row * dy;

        float zx = 0.0f;
        float zy = 0.0f;
        int iter = 0;

        while ((zx * zx + zy * zy <= 4.0f) && (iter < max_iter)) {
            float tmp = zx * zx - zy * zy + cx;
            zy = 2.0f * zx * zy + cy;
            zx = tmp;
            iter++;
        }

        output[idx] = iter;
    }
}

// ============================================================================
// Kernel 3: Dual-Point Processing (Instruction Level Parallelism)
// ============================================================================
__global__ void mandelbrot_dual_point_kernel(int* __restrict__ output,
                                             int width, int height, int max_iter,
                                             float xmin, float xmax,
                                             float ymin, float ymax) {
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height) {
        float dx = (xmax - xmin) / (float)width;
        float dy = (ymax - ymin) / (float)height;

        float cy = ymin + (float)row * dy;

        // Point 1
        if (col < width) {
            float cx1 = xmin + (float)col * dx;
            float zx1 = 0.0f, zy1 = 0.0f;
            int iter1 = 0;
            while ((zx1 * zx1 + zy1 * zy1 <= 4.0f) && (iter1 < max_iter)) {
                float tmp = zx1 * zx1 - zy1 * zy1 + cx1;
                zy1 = 2.0f * zx1 * zy1 + cy;
                zx1 = tmp;
                iter1++;
            }
            output[row * width + col] = iter1;
        }

        // Point 2
        if (col + 1 < width) {
            float cx2 = xmin + (float)(col + 1) * dx;
            float zx2 = 0.0f, zy2 = 0.0f;
            int iter2 = 0;
            while ((zx2 * zx2 + zy2 * zy2 <= 4.0f) && (iter2 < max_iter)) {
                float tmp = zx2 * zx2 - zy2 * zy2 + cx2;
                zy2 = 2.0f * zx2 * zy2 + cy;
                zx2 = tmp;
                iter2++;
            }
            output[row * width + col + 1] = iter2;
        }
    }
}

// ============================================================================
// Driver & Benchmarking Harness
// ============================================================================
int main(int argc, char** argv) {
    int width = 2048;
    int height = 2048;
    int max_iter = 500;
    int iters = 10;

    if (argc > 1) width = atoi(argv[1]);
    if (argc > 2) height = atoi(argv[2]);
    if (argc > 3) max_iter = atoi(argv[3]);
    if (argc > 4) iters = atoi(argv[4]);

    printf("=================================================================\n");
    printf("   NVIDIA CUDA C Mandelbrot Benchmark (%dx%d, max_iter = %d)\n", width, height, max_iter);
    printf("=================================================================\n");

    size_t total_pixels = (size_t)width * height;
    size_t bytes = total_pixels * sizeof(int);

    int *h_output = (int*)malloc(bytes);
    int *d_output;
    cudaMalloc(&d_output, bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block2D(BLOCK_X, BLOCK_Y);
    dim3 grid2D((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);

    // -------------------------------------------------------------
    // Tier 1: Naive 2D Thread Mapping
    // -------------------------------------------------------------
    mandelbrot_naive_kernel<<<grid2D, block2D>>>(d_output, width, height, max_iter, -2.0f, 0.5f, -1.25f, 1.25f);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        mandelbrot_naive_kernel<<<grid2D, block2D>>>(d_output, width, height, max_iter, -2.0f, 0.5f, -1.25f, 1.25f);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_naive = 0.0f;
    cudaEventElapsedTime(&ms_naive, start, stop);
    ms_naive /= iters;
    printf("Tier 1: Naive 2D Thread Mapping:  %8.3f ms | %8.2f Mpix/s\n",
           ms_naive, ((double)total_pixels / (ms_naive * 1e-3)) / 1e6);

    // -------------------------------------------------------------
    // Tier 2: 1D Grid Stride Loop
    // -------------------------------------------------------------
    int blockSize1D = 256;
    int numBlocks1D = 512;
    mandelbrot_grid_stride_kernel<<<numBlocks1D, blockSize1D>>>(d_output, width, height, max_iter, -2.0f, 0.5f, -1.25f, 1.25f);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        mandelbrot_grid_stride_kernel<<<numBlocks1D, blockSize1D>>>(d_output, width, height, max_iter, -2.0f, 0.5f, -1.25f, 1.25f);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_stride = 0.0f;
    cudaEventElapsedTime(&ms_stride, start, stop);
    ms_stride /= iters;
    printf("Tier 2: 1D Grid Stride Loop:      %8.3f ms | %8.2f Mpix/s | Speedup: %5.2fx\n",
           ms_stride, ((double)total_pixels / (ms_stride * 1e-3)) / 1e6, ms_naive / ms_stride);

    // -------------------------------------------------------------
    // Tier 3: Dual-Point Processing (ILP)
    // -------------------------------------------------------------
    dim3 gridDual((width / 2 + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);
    mandelbrot_dual_point_kernel<<<gridDual, block2D>>>(d_output, width, height, max_iter, -2.0f, 0.5f, -1.25f, 1.25f);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        mandelbrot_dual_point_kernel<<<gridDual, block2D>>>(d_output, width, height, max_iter, -2.0f, 0.5f, -1.25f, 1.25f);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_dual = 0.0f;
    cudaEventElapsedTime(&ms_dual, start, stop);
    ms_dual /= iters;
    printf("Tier 3: Dual-Point ILP Kernel:    %8.3f ms | %8.2f Mpix/s | Speedup: %5.2fx\n",
           ms_dual, ((double)total_pixels / (ms_dual * 1e-3)) / 1e6, ms_naive / ms_dual);

    // Checksum verification
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    long long checksum = 0;
    for (size_t i = 0; i < total_pixels; ++i) checksum += h_output[i];
    printf("-----------------------------------------------------------------\n");
    printf("[Verification] Mandelbrot Checksum = %lld\n", checksum);

    cudaFree(d_output);
    free(h_output);
    return 0;
}
