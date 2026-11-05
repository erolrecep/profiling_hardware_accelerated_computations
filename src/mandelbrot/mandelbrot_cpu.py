"""
Mandelbrot Set Generation - CPU Baselines
Implementations:
  1. Pure Python pixel-by-pixel escape loop (Embarrassingly Slow Baseline)
  2. NumPy Vectorized Masked Iterations (Active Array Compaction)
"""

import time
import numpy as np


def mandelbrot_pure_python(width: int, height: int, max_iter: int = 256,
                           xmin: float = -2.0, xmax: float = 0.5,
                           ymin: float = -1.25, ymax: float = 1.25) -> list[list[int]]:
    """Pure Python pixel-by-pixel Mandelbrot escape calculation."""
    dx = (xmax - xmin) / width
    dy = (ymax - ymin) / height
    output = [[0 for _ in range(width)] for _ in range(height)]

    for row in range(height):
        cy = ymin + row * dy
        for col in range(width):
            cx = xmin + col * dx
            zx = 0.0
            zy = 0.0
            iters = 0
            while zx * zx + zy * zy <= 4.0 and iters < max_iter:
                tmp = zx * zx - zy * zy + cx
                zy = 2.0 * zx * zy + cy
                zx = tmp
                iters += 1
            output[row][col] = iters

    return output


def mandelbrot_numpy_vectorized(width: int, height: int, max_iter: int = 256,
                                xmin: float = -2.0, xmax: float = 0.5,
                                ymin: float = -1.25, ymax: float = 1.25) -> np.ndarray:
    """NumPy SIMD vectorized complex plane computation with boolean active masking."""
    x = np.linspace(xmin, xmax, width, dtype=np.float32)
    y = np.linspace(ymin, ymax, height, dtype=np.float32)
    C = x[np.newaxis, :] + 1j * y[:, np.newaxis]
    Z = np.zeros_like(C)
    iterations = np.zeros((height, width), dtype=np.int32)
    mask = np.ones((height, width), dtype=bool)

    for i in range(max_iter):
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        escaped = np.abs(Z) > 2.0
        newly_escaped = escaped & mask
        iterations[newly_escaped] = i
        mask = mask & (~escaped)
        if not np.any(mask):
            break

    iterations[mask] = max_iter
    return iterations


def benchmark_cpu_mandelbrot(width: int = 512, height: int = 512, max_iter: int = 256,
                             run_naive_python: bool = False) -> dict:
    """Benchmark CPU Mandelbrot backends."""
    total_pixels = width * height
    results = {}

    # 1. NumPy Vectorized
    # Warmup
    _ = mandelbrot_numpy_vectorized(width, height, max_iter)
    t0 = time.perf_counter()
    iters_bench = 5
    for _ in range(iters_bench):
        res_np = mandelbrot_numpy_vectorized(width, height, max_iter)
    t1 = time.perf_counter()
    np_time = (t1 - t0) / iters_bench
    results["numpy_vectorized"] = {
        "time_sec": np_time,
        "mpix_per_sec": (total_pixels / np_time) / 1e6,
        "checksum": int(np.sum(res_np)),
    }

    # 2. Pure Python (only for small resolution, e.g. <= 256)
    if run_naive_python and (width * height <= 256 * 256):
        t0 = time.perf_counter()
        res_py = mandelbrot_pure_python(width, height, max_iter)
        t1 = time.perf_counter()
        py_time = t1 - t0
        results["pure_python"] = {
            "time_sec": py_time,
            "mpix_per_sec": (total_pixels / py_time) / 1e6,
            "checksum": int(np.sum(np.array(res_py, dtype=np.int32))),
        }

    return results


if __name__ == "__main__":
    res = benchmark_cpu_mandelbrot(width=256, height=256, max_iter=256, run_naive_python=True)
    print("Mandelbrot (256x256, max_iter=256):")
    for k, v in res.items():
        print(f"  {k:20s}: {v['time_sec']*1000:8.2f} ms | {v['mpix_per_sec']:8.2f} Mpix/s | Checksum: {v['checksum']}")
