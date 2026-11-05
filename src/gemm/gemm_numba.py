"""
Dense Matrix Multiplication (GEMM) - Numba CPU & GPU
Implementations:
  1. Numba CPU Serial (@njit)
  2. Numba CPU Parallel Multi-Threaded (@njit(parallel=True, fastmath=True))
  3. Numba GPU Naive Kernel (@cuda.jit)
  4. Numba GPU Shared-Memory Tiled Kernel (@cuda.jit)
"""

import time
import numpy as np
from numba import njit, prange, float32

# -----------------------------------------------------------------------------
# 1. Numba CPU Implementations
# -----------------------------------------------------------------------------
@njit(fastmath=True)
def gemm_numba_cpu_serial(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> None:
    """Numba CPU single-threaded with loop reordering (i-k-j) and fastmath."""
    N = A.shape[0]
    K = A.shape[1]
    M = B.shape[1]
    for i in range(N):
        for k in range(K):
            r = A[i, k]
            for j in range(M):
                C[i, j] += r * B[k, j]


@njit(parallel=True, fastmath=True)
def gemm_numba_cpu_parallel(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> None:
    """Numba CPU multi-threaded using OpenMP-style prange work-stealing across all CPU cores."""
    N = A.shape[0]
    K = A.shape[1]
    M = B.shape[1]
    for i in prange(N):
        for k in range(K):
            r = A[i, k]
            for j in range(M):
                C[i, j] += r * B[k, j]


# -----------------------------------------------------------------------------
# 2. Numba GPU Implementations (Guarded for portability)
# -----------------------------------------------------------------------------
try:
    from numba import cuda

    TPB = 16  # Threads per block (16x16 = 256 threads)

    @cuda.jit
    def gemm_numba_gpu_naive_kernel(A, B, C):
        """Naive global-memory Numba GPU kernel."""
        row, col = cuda.grid(2)
        if row < C.shape[0] and col < C.shape[1]:
            acc = 0.0
            for k in range(A.shape[1]):
                acc += A[row, k] * B[k, col]
            C[row, col] = acc

    @cuda.jit
    def gemm_numba_gpu_tiled_kernel(A, B, C):
        """2D Shared memory tiled Numba GPU kernel."""
        sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
        sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

        x, y = cuda.grid(2)
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y

        N = C.shape[0]
        acc = 0.0

        for tile in range(int((N + TPB - 1) / TPB)):
            # Load into shared memory with bounds checks
            a_col = tile * TPB + tx
            if y < N and a_col < N:
                sA[ty, tx] = A[y, a_col]
            else:
                sA[ty, tx] = 0.0

            b_row = tile * TPB + ty
            if b_row < N and x < N:
                sB[ty, tx] = B[b_row, x]
            else:
                sB[ty, tx] = 0.0

            cuda.syncthreads()

            for k in range(TPB):
                acc += sA[ty, k] * sB[k, tx]

            cuda.syncthreads()

        if y < N and x < N:
            C[y, x] = acc

    NUMBA_CUDA_SUPPORT = cuda.is_available()
except Exception:
    NUMBA_CUDA_SUPPORT = False


def benchmark_numba_gemm(N: int = 512, iters: int = 10) -> dict:
    """Benchmark all Numba GEMM backends."""
    rng = np.random.default_rng(42)
    A = rng.standard_normal((N, N), dtype=np.float32)
    B = rng.standard_normal((N, N), dtype=np.float32)
    C = np.zeros((N, N), dtype=np.float32)
    total_flops = 2.0 * (N ** 3)
    results = {}

    # --- Warmup & Benchmark CPU Serial ---
    C.fill(0)
    gemm_numba_cpu_serial(A, B, C)
    C.fill(0)
    t0 = time.perf_counter()
    for _ in range(iters):
        gemm_numba_cpu_serial(A, B, C)
        C.fill(0)
    t1 = time.perf_counter()
    dt_serial = (t1 - t0) / iters
    results["numba_cpu_serial"] = {
        "time_sec": dt_serial,
        "gflops": (total_flops / dt_serial) / 1e9,
    }

    # --- Warmup & Benchmark CPU Parallel ---
    C.fill(0)
    gemm_numba_cpu_parallel(A, B, C)
    C.fill(0)
    t0 = time.perf_counter()
    for _ in range(iters):
        gemm_numba_cpu_parallel(A, B, C)
        C.fill(0)
    t1 = time.perf_counter()
    dt_parallel = (t1 - t0) / iters
    results["numba_cpu_parallel"] = {
        "time_sec": dt_parallel,
        "gflops": (total_flops / dt_parallel) / 1e9,
        "speedup_vs_serial": dt_serial / dt_parallel,
    }

    # --- Benchmark GPU (if hardware present) ---
    if NUMBA_CUDA_SUPPORT:
        threads_per_block = (TPB, TPB)
        blocks_per_grid_x = int(np.ceil(N / threads_per_block[0]))
        blocks_per_grid_y = int(np.ceil(N / threads_per_block[1]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        d_A = cuda.to_device(A)
        d_B = cuda.to_device(B)
        d_C = cuda.device_array((N, N), dtype=np.float32)

        # Naive GPU
        gemm_numba_gpu_naive_kernel[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
        cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            gemm_numba_gpu_naive_kernel[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
            cuda.synchronize()
        t1 = time.perf_counter()
        dt_gpu_naive = (t1 - t0) / iters
        results["numba_gpu_naive"] = {
            "time_sec": dt_gpu_naive,
            "gflops": (total_flops / dt_gpu_naive) / 1e9,
        }

        # Tiled GPU
        gemm_numba_gpu_tiled_kernel[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
        cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            gemm_numba_gpu_tiled_kernel[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
            cuda.synchronize()
        t1 = time.perf_counter()
        dt_gpu_tiled = (t1 - t0) / iters
        results["numba_gpu_tiled"] = {
            "time_sec": dt_gpu_tiled,
            "gflops": (total_flops / dt_gpu_tiled) / 1e9,
            "speedup_vs_naive_gpu": dt_gpu_naive / dt_gpu_tiled,
        }

    return results


if __name__ == "__main__":
    res = benchmark_numba_gemm(N=512, iters=5)
    print(f"Numba GEMM (N=512):")
    for k, v in res.items():
        print(f"  {k:20s}: {v['time_sec']*1000:8.2f} ms | {v['gflops']:8.2f} GFLOPS")
