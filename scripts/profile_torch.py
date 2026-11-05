"""
PyTorch Profiler Telemetry Script
Profiles GEMM and Mandelbrot operators using torch.profiler.
Exports:
  1. Terminal summary table (CPU time %, CUDA/MPS time %, Mem MB, Call counts)
  2. Chrome Tracing / Perfetto JSON timeline export to telemetry/
"""

import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/gemm")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/mandelbrot")))

from gemm_torch import gemm_torch, get_torch_accelerator, synchronize_device
from mandelbrot_torch import mandelbrot_torch_tensor

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../telemetry"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def profile_gemm(N: int = 1024, iters: int = 10):
    print("==========================================================================")
    print(f"  [PROFILING] PyTorch Profiler: Dense GEMM (N = {N})")
    print("==========================================================================")

    dev_name, has_accel = get_torch_accelerator()
    device = torch.device(dev_name)

    A = torch.randn(N, N, dtype=torch.float32, device=device)
    B = torch.randn(N, N, dtype=torch.float32, device=device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if dev_name == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    trace_path = os.path.join(OUTPUT_DIR, "torch_trace_gemm.json")

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=iters, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(OUTPUT_DIR) if dev_name == "cuda" else None,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(1 + 2 + iters):
            _ = gemm_torch(A, B)
            synchronize_device(dev_name)
            prof.step()

    prof.export_chrome_trace(trace_path)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
    print(f"✓ Chrome / Perfetto trace exported to: {trace_path}\n")


def profile_mandelbrot(width: int = 512, height: int = 512, max_iter: int = 256, iters: int = 5):
    print("==========================================================================")
    print(f"  [PROFILING] PyTorch Profiler: Mandelbrot ({width}x{height}, max_iter={max_iter})")
    print("==========================================================================")

    dev_name, has_accel = get_torch_accelerator()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if dev_name == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    trace_path = os.path.join(OUTPUT_DIR, "torch_trace_mandelbrot.json")

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=iters, repeat=1),
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(1 + 1 + iters):
            _ = mandelbrot_torch_tensor(width, height, max_iter, device=dev_name)
            synchronize_device(dev_name)
            prof.step()

    prof.export_chrome_trace(trace_path)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
    print(f"✓ Chrome / Perfetto trace exported to: {trace_path}\n")


if __name__ == "__main__":
    profile_gemm(N=512, iters=5)
    profile_mandelbrot(width=256, height=256, max_iter=128, iters=3)
