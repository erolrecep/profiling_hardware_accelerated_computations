"""
Master Benchmark Telemetry Runner & Visualization Generator
Executes GEMM and Mandelbrot scaling sweeps, collects telemetry metrics,
and generates publication-grade Markdown comparison tables and PNG charts.
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/gemm")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/mandelbrot")))

from gemm_cpu import benchmark_cpu_gemm
from gemm_numba import benchmark_numba_gemm
from gemm_torch import benchmark_torch_gemm, get_torch_accelerator
from mandelbrot_cpu import benchmark_cpu_mandelbrot
from mandelbrot_numba import benchmark_numba_mandelbrot
from mandelbrot_torch import benchmark_torch_mandelbrot

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../telemetry"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_gemm_sweep():
    print(">> Running GEMM Benchmark Sweep...")
    sizes = [128, 256, 512, 1024]
    sweep_data = {}

    for N in sizes:
        print(f"   Profiling Dimension N = {N}...")
        res_cpu = benchmark_cpu_gemm(N, run_naive_python=(N <= 128))
        res_numba = benchmark_numba_gemm(N, iters=5)
        res_torch = benchmark_torch_gemm(N, iters=10)

        sweep_data[N] = {
            "numpy": res_cpu.get("numpy_blas", {}).get("gflops", 0.0),
            "numba_parallel": res_numba.get("numba_cpu_parallel", {}).get("gflops", 0.0),
            "torch_cpu": res_torch.get("torch_cpu", {}).get("gflops", 0.0),
        }
        dev, has_acc = get_torch_accelerator()
        if has_acc:
            sweep_data[N][f"torch_{dev}"] = res_torch.get(f"torch_{dev}", {}).get("gflops", 0.0)

    # Plot GEMM GFLOPS scaling
    plt.figure(figsize=(9, 5), dpi=150)
    for backend in sweep_data[sizes[0]].keys():
        gflops_vals = [sweep_data[N][backend] for N in sizes]
        plt.plot(sizes, gflops_vals, marker="o", linewidth=2.2, label=backend.upper())

    plt.title("GEMM Computational Throughput Scaling (GFLOPS vs Matrix Size)", fontsize=13, fontweight="bold")
    plt.xlabel("Matrix Dimension N (NxN Float32)", fontsize=11)
    plt.ylabel("Throughput (GFLOPS)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "gemm_scaling.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✓ GEMM Scaling Plot saved to: {plot_path}")
    return sweep_data


def run_mandelbrot_sweep():
    print(">> Running Mandelbrot Benchmark Sweep...")
    sizes = [128, 256, 512]
    sweep_data = {}

    for S in sizes:
        print(f"   Profiling Resolution {S}x{S}...")
        res_cpu = benchmark_cpu_mandelbrot(S, S, max_iter=256, run_naive_python=(S <= 128))
        res_numba = benchmark_numba_mandelbrot(S, S, max_iter=256, iters=3)
        res_torch = benchmark_torch_mandelbrot(S, S, max_iter=256, iters=3)

        sweep_data[S] = {
            "numpy": res_cpu.get("numpy_vectorized", {}).get("mpix_per_sec", 0.0),
            "numba_parallel": res_numba.get("numba_cpu_parallel", {}).get("mpix_per_sec", 0.0),
            "torch_cpu": res_torch.get("torch_cpu", {}).get("mpix_per_sec", 0.0),
        }
        dev, has_acc = get_torch_accelerator()
        if has_acc:
            sweep_data[S][f"torch_{dev}"] = res_torch.get(f"torch_{dev}", {}).get("mpix_per_sec", 0.0)

    # Plot Mandelbrot Throughput
    plt.figure(figsize=(9, 5), dpi=150)
    for backend in sweep_data[sizes[0]].keys():
        mpix_vals = [sweep_data[S][backend] for S in sizes]
        plt.plot(sizes, mpix_vals, marker="s", linewidth=2.2, label=backend.upper())

    plt.title("Mandelbrot Generation Throughput (Megapixels/s vs Resolution)", fontsize=13, fontweight="bold")
    plt.xlabel("Grid Resolution (Width = Height)", fontsize=11)
    plt.ylabel("Throughput (Megapixels / Second)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mandelbrot_speedup.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✓ Mandelbrot Throughput Plot saved to: {plot_path}")
    return sweep_data


def main():
    print("==========================================================================")
    print("  EXECUTING MASTER BENCHMARK SWEEP & TELEMETRY GENERATION")
    print("==========================================================================")
    gemm_telemetry = run_gemm_sweep()
    mandelbrot_telemetry = run_mandelbrot_sweep()

    full_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_accelerator": get_torch_accelerator()[0],
        "gemm_scaling": gemm_telemetry,
        "mandelbrot_scaling": mandelbrot_telemetry,
    }

    json_path = os.path.join(OUTPUT_DIR, "benchmark_summary.json")
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"\n✓ Master Telemetry JSON saved to: {json_path}")
    print("==========================================================================")


if __name__ == "__main__":
    main()
