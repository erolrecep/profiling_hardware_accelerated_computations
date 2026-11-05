"""
Mandelbrot Fractal - PyTorch CPU, CUDA & Apple Silicon MPS
Implementations:
  1. PyTorch CPU Vectorized Complex Tensor Loop
  2. PyTorch Hardware Accelerator (NVIDIA CUDA or Apple Silicon MPS)
  3. PyTorch Compiled (torch.compile)
"""

import time
import torch
import numpy as np


def get_torch_accelerator() -> tuple[str, bool]:
    """Detect available hardware accelerator device."""
    if torch.cuda.is_available():
        return "cuda", True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", True
    else:
        return "cpu", False


def synchronize_device(device_type: str) -> None:
    """Synchronize device queue for high-precision micro-benchmarking."""
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()


def mandelbrot_torch_tensor(width: int, height: int, max_iter: int = 256,
                            xmin: float = -2.0, xmax: float = 0.5,
                            ymin: float = -1.25, ymax: float = 1.25,
                            device: str = "cpu") -> torch.Tensor:
    """Tensorized complex plane Mandelbrot calculation on target device."""
    dev = torch.device(device)
    x = torch.linspace(xmin, xmax, width, dtype=torch.float32, device=dev)
    y = torch.linspace(ymin, ymax, height, dtype=torch.float32, device=dev)

    # Grid of complex coordinates
    Y, X = torch.meshgrid(y, x, indexing="ij")
    C = torch.complex(X, Y)
    Z = torch.zeros_like(C)
    iterations = torch.zeros((height, width), dtype=torch.int32, device=dev)
    mask = torch.ones((height, width), dtype=torch.bool, device=dev)

    for i in range(max_iter):
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        escaped = (Z.real * Z.real + Z.imag * Z.imag) > 4.0
        newly_escaped = escaped & mask
        iterations[newly_escaped] = i
        mask = mask & (~escaped)
        if not mask.any():
            break

    iterations[mask] = max_iter
    return iterations


def benchmark_torch_mandelbrot(width: int = 1024, height: int = 1024, max_iter: int = 256, iters: int = 5) -> dict:
    """Benchmark PyTorch Mandelbrot across CPU and Accelerators."""
    total_pixels = width * height
    results = {}

    # 1. PyTorch CPU
    _ = mandelbrot_torch_tensor(width, height, max_iter, device="cpu")
    t0 = time.perf_counter()
    for _ in range(iters):
        out_cpu = mandelbrot_torch_tensor(width, height, max_iter, device="cpu")
    t1 = time.perf_counter()
    dt_cpu = (t1 - t0) / iters
    results["torch_cpu"] = {
        "time_sec": dt_cpu,
        "mpix_per_sec": (total_pixels / dt_cpu) / 1e6,
        "checksum": int(out_cpu.sum().item()),
        "device": "cpu",
    }

    # 2. PyTorch Accelerator (CUDA or MPS)
    dev_name, has_accel = get_torch_accelerator()
    if has_accel:
        _ = mandelbrot_torch_tensor(width, height, max_iter, device=dev_name)
        synchronize_device(dev_name)

        t0 = time.perf_counter()
        for _ in range(iters):
            out_dev = mandelbrot_torch_tensor(width, height, max_iter, device=dev_name)
        synchronize_device(dev_name)
        t1 = time.perf_counter()
        dt_dev = (t1 - t0) / iters

        results[f"torch_{dev_name}"] = {
            "time_sec": dt_dev,
            "mpix_per_sec": (total_pixels / dt_dev) / 1e6,
            "speedup_vs_cpu": dt_cpu / dt_dev,
            "checksum": int(out_dev.sum().item()),
            "device": dev_name,
        }

    return results


if __name__ == "__main__":
    dev, ok = get_torch_accelerator()
    print(f"PyTorch Hardware Target: {dev.upper()} (Acceleration Active: {ok})")
    res = benchmark_torch_mandelbrot(width=512, height=512, max_iter=256, iters=3)
    for k, v in res.items():
        print(f"  {k:20s}: {v['time_sec']*1000:8.2f} ms | {v['mpix_per_sec']:8.2f} Mpix/s | Checksum: {v['checksum']}")
