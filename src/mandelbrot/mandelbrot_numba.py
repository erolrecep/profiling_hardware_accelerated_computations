"""
Mandelbrot Fractal - Numba CPU & GPU
Implementations:
  1. Numba CPU Serial (@njit(fastmath=True))
  2. Numba CPU Parallel Multi-Threaded (@njit(parallel=True, fastmath=True))
  3. Numba GPU (@cuda.jit)
"""

import time
import numpy as np
from numba import njit, prange

# -----------------------------------------------------------------------------
# 1. Numba CPU Implementations
# -----------------------------------------------------------------------------
@njit(fastmath=True)
def mandelbrot_numba_cpu_serial(output: np.ndarray, width: int, height: int, max_iter: int,
                                xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    """Numba CPU single-threaded with fastmath optimizations."""
    dx = (xmax - xmin) / width
    dy = (ymax - ymin) / height

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
            output[row, col] = iters


@njit(parallel=True, fastmath=True)
def mandelbrot_numba_cpu_parallel(output: np.ndarray, width: int, height: int, max_iter: int,
                                  xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    """Numba CPU multi-threaded parallelizing rows across all cores."""
    dx = (xmax - xmin) / width
    dy = (ymax - ymin) / height

    for row in prange(height):
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
            output[row, col] = iters


# -----------------------------------------------------------------------------
# 2. Numba GPU Implementations (Guarded for portability)
# -----------------------------------------------------------------------------
try:
    from numba import cuda

    @cuda.jit
    def mandelbrot_numba_gpu_kernel(output, width, height, max_iter, xmin, xmax, ymin, ymax):
        """Numba GPU 2D grid kernel."""
        col, row = cuda.grid(2)
        if col < width and row < height:
            dx = (xmax - xmin) / width
            dy = (ymax - ymin) / height
            cx = xmin + col * dx
            cy = ymin + row * dy

            zx = 0.0
            zy = 0.0
            iters = 0
            while zx * zx + zy * zy <= 4.0 and iters < max_iter:
                tmp = zx * zx - zy * zy + cx
                zy = 2.0 * zx * zy + cy
                zx = tmp
                iters += 1
            output[row, col] = iters

    NUMBA_CUDA_SUPPORT = cuda.is_available()
except Exception:
    NUMBA_CUDA_SUPPORT = False


def benchmark_numba_mandelbrot(width: int = 1024, height: int = 1024, max_iter: int = 500, iters: int = 5) -> dict:
    """Benchmark Numba CPU & GPU Mandelbrot backends."""
    output = np.zeros((height, width), dtype=np.int32)
    total_pixels = width * height
    results = {}

    xmin, xmax, ymin, ymax = -2.0, 0.5, -1.25, 1.25

    # 1. Warmup & Benchmark CPU Serial
    output.fill(0)
    mandelbrot_numba_cpu_serial(output, width, height, max_iter, xmin, xmax, ymin, ymax)
    output.fill(0)
    t0 = time.perf_counter()
    for _ in range(iters):
        mandelbrot_numba_cpu_serial(output, width, height, max_iter, xmin, xmax, ymin, ymax)
    t1 = time.perf_counter()
    dt_serial = (t1 - t0) / iters
    results["numba_cpu_serial"] = {
        "time_sec": dt_serial,
        "mpix_per_sec": (total_pixels / dt_serial) / 1e6,
        "checksum": int(np.sum(output)),
    }

    # 2. Warmup & Benchmark CPU Parallel
    output.fill(0)
    mandelbrot_numba_cpu_parallel(output, width, height, max_iter, xmin, xmax, ymin, ymax)
    output.fill(0)
    t0 = time.perf_counter()
    for _ in range(iters):
        mandelbrot_numba_cpu_parallel(output, width, height, max_iter, xmin, xmax, ymin, ymax)
    t1 = time.perf_counter()
    dt_parallel = (t1 - t0) / iters
    results["numba_cpu_parallel"] = {
        "time_sec": dt_parallel,
        "mpix_per_sec": (total_pixels / dt_parallel) / 1e6,
        "speedup_vs_serial": dt_serial / dt_parallel,
        "checksum": int(np.sum(output)),
    }

    # 3. GPU Benchmark (if hardware present)
    if NUMBA_CUDA_SUPPORT:
        threads_per_block = (16, 16)
        blocks_x = int(np.ceil(width / threads_per_block[0]))
        blocks_y = int(np.ceil(height / threads_per_block[1]))
        blocks = (blocks_x, blocks_y)

        d_out = cuda.device_array((height, width), dtype=np.int32)

        # Warmup
        mandelbrot_numba_gpu_kernel[blocks, threads_per_block](d_out, width, height, max_iter, xmin, xmax, ymin, ymax)
        cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            mandelbrot_numba_gpu_kernel[blocks, threads_per_block](d_out, width, height, max_iter, xmin, xmax, ymin, ymax)
            cuda.synchronize()
        t1 = time.perf_counter()
        dt_gpu = (t1 - t0) / iters
        h_res = d_out.copy_to_host()
        results["numba_gpu"] = {
            "time_sec": dt_gpu,
            "mpix_per_sec": (total_pixels / dt_gpu) / 1e6,
            "speedup_vs_serial": dt_serial / dt_gpu,
            "checksum": int(np.sum(h_res)),
        }

    return results


if __name__ == "__main__":
    res = benchmark_numba_mandelbrot(width=512, height=512, max_iter=256, iters=5)
    print("Numba Mandelbrot (512x512):")
    for k, v in res.items():
        print(f"  {k:20s}: {v['time_sec']*1000:8.2f} ms | {v['mpix_per_sec']:8.2f} Mpix/s | Checksum: {v['checksum']}")
