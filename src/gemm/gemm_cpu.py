"""
Dense Matrix Multiplication (GEMM) - CPU Baselines
Implementations:
  1. Pure Python 3-loop Naive (i-j-k) - Cache Unfriendly
  2. Pure Python 3-loop Optimized (i-k-j) - Cache Locality
  3. NumPy SIMD/BLAS Baseline
"""

import time
import numpy as np


def gemm_pure_python_ijk(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Naive 3-loop matrix multiplication with i-j-k loop ordering."""
    N = len(A)
    K = len(A[0])
    M = len(B[0])
    C = [[0.0 for _ in range(M)] for _ in range(N)]
    for i in range(N):
        for j in range(M):
            acc = 0.0
            for k in range(K):
                acc += A[i][k] * B[k][j]
            C[i][j] = acc
    return C


def gemm_pure_python_ikj(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Loop-interchanged matrix multiplication (i-k-j) for contiguous memory traversal."""
    N = len(A)
    K = len(A[0])
    M = len(B[0])
    C = [[0.0 for _ in range(M)] for _ in range(N)]
    for i in range(N):
        for k in range(K):
            r = A[i][k]
            for j in range(M):
                C[i][j] += r * B[k][j]
    return C


def gemm_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Hardware-accelerated CPU BLAS via NumPy."""
    return np.matmul(A, B)


def benchmark_cpu_gemm(N: int = 256, run_naive_python: bool = False) -> dict:
    """Benchmark CPU GEMM implementations."""
    rng = np.random.default_rng(42)
    A_np = rng.standard_normal((N, N), dtype=np.float32)
    B_np = rng.standard_normal((N, N), dtype=np.float32)
    total_flops = 2.0 * (N ** 3)

    results = {}

    # 1. NumPy BLAS baseline
    # Warmup
    _ = gemm_numpy(A_np, B_np)
    t0 = time.perf_counter()
    iters = 10 if N >= 512 else 50
    for _ in range(iters):
        C_np = gemm_numpy(A_np, B_np)
    t1 = time.perf_counter()
    np_time = (t1 - t0) / iters
    results["numpy_blas"] = {
        "time_sec": np_time,
        "gflops": (total_flops / np_time) / 1e9,
    }

    # 2. Pure Python (only for small N, e.g. <= 256)
    if run_naive_python and N <= 256:
        A_list = A_np.tolist()
        B_list = B_np.tolist()

        t0 = time.perf_counter()
        _ = gemm_pure_python_ikj(A_list, B_list)
        t1 = time.perf_counter()
        ikj_time = t1 - t0
        results["python_ikj"] = {
            "time_sec": ikj_time,
            "gflops": (total_flops / ikj_time) / 1e9,
        }

        t0 = time.perf_counter()
        C_py = gemm_pure_python_ijk(A_list, B_list)
        t1 = time.perf_counter()
        ijk_time = t1 - t0
        results["python_ijk"] = {
            "time_sec": ijk_time,
            "gflops": (total_flops / ijk_time) / 1e9,
            "max_error_vs_numpy": float(np.max(np.abs(np.array(C_py, dtype=np.float32) - C_np))),
        }

    return results


if __name__ == "__main__":
    res = benchmark_cpu_gemm(N=128, run_naive_python=True)
    print(f"GEMM (N=128):")
    for k, v in res.items():
        print(f"  {k:15s}: {v['time_sec']*1000:8.2f} ms | {v['gflops']:8.2f} GFLOPS")
