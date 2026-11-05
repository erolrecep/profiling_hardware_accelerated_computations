"""
Dense Matrix Multiplication (GEMM) - PyTorch CPU, CUDA & Apple Silicon MPS
Implementations:
  1. PyTorch CPU (MKL / oneDNN multithreaded ATen)
  2. PyTorch Hardware Accelerator (NVIDIA CUDA or Apple Silicon MPS)
  3. PyTorch Compiled (torch.compile with Inductor kernel autotuning)
"""

import time
import torch


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


def gemm_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Standard PyTorch matrix product."""
    return torch.matmul(A, B)


def benchmark_torch_gemm(N: int = 1024, iters: int = 20) -> dict:
    """Benchmark PyTorch GEMM across CPU, Device, and Compiled modes."""
    total_flops = 2.0 * (N ** 3)
    results = {}

    # 1. PyTorch CPU
    A_cpu = torch.randn(N, N, dtype=torch.float32, device="cpu")
    B_cpu = torch.randn(N, N, dtype=torch.float32, device="cpu")

    # Warmup
    _ = gemm_torch(A_cpu, B_cpu)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = gemm_torch(A_cpu, B_cpu)
    t1 = time.perf_counter()
    dt_cpu = (t1 - t0) / iters
    results["torch_cpu"] = {
        "time_sec": dt_cpu,
        "gflops": (total_flops / dt_cpu) / 1e9,
        "device": "cpu",
    }

    # 2. PyTorch Hardware Device (CUDA or MPS)
    dev_name, has_accel = get_torch_accelerator()
    if has_accel:
        device = torch.device(dev_name)
        A_dev = A_cpu.to(device)
        B_dev = B_cpu.to(device)

        # Warmup
        _ = gemm_torch(A_dev, B_dev)
        synchronize_device(dev_name)

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = gemm_torch(A_dev, B_dev)
        synchronize_device(dev_name)
        t1 = time.perf_counter()
        dt_dev = (t1 - t0) / iters
        results[f"torch_{dev_name}"] = {
            "time_sec": dt_dev,
            "gflops": (total_flops / dt_dev) / 1e9,
            "speedup_vs_cpu": dt_cpu / dt_dev,
            "device": dev_name,
        }

        # 3. PyTorch Compiled (torch.compile)
        try:
            compiled_gemm = torch.compile(gemm_torch, mode="default")
            # Warmup compile step
            _ = compiled_gemm(A_dev, B_dev)
            synchronize_device(dev_name)

            t0 = time.perf_counter()
            for _ in range(iters):
                _ = compiled_gemm(A_dev, B_dev)
            synchronize_device(dev_name)
            t1 = time.perf_counter()
            dt_comp = (t1 - t0) / iters
            results[f"torch_{dev_name}_compiled"] = {
                "time_sec": dt_comp,
                "gflops": (total_flops / dt_comp) / 1e9,
                "speedup_vs_cpu": dt_cpu / dt_comp,
                "device": dev_name,
            }
        except Exception as e:
            results[f"torch_{dev_name}_compiled"] = {"error": str(e)}

    return results


if __name__ == "__main__":
    dev, ok = get_torch_accelerator()
    print(f"PyTorch Hardware Target: {dev.upper()} (Acceleration Active: {ok})")
    res = benchmark_torch_gemm(N=1024, iters=20)
    for k, v in res.items():
        if "error" in v:
            print(f"  {k:22s}: Error - {v['error']}")
        else:
            print(f"  {k:22s}: {v['time_sec']*1000:8.2f} ms | {v['gflops']:8.2f} GFLOPS")
