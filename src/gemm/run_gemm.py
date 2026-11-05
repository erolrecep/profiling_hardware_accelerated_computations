"""
Master CLI Runner: Dense Matrix Multiplication (GEMM) Benchmark Suite
Compares:
  - Python Naive i-j-k vs i-k-j
  - NumPy OpenBLAS / Accelerate
  - Numba CPU Serial vs Multi-Threaded Parallel
  - Numba GPU (if available)
  - PyTorch CPU vs GPU/MPS vs Compiled
"""

import argparse
import json
import os
import sys
from tabulate import tabulate

from gemm_cpu import benchmark_cpu_gemm
from gemm_numba import benchmark_numba_gemm
from gemm_torch import benchmark_torch_gemm, get_torch_accelerator


def main():
    parser = argparse.ArgumentParser(description="Dense GEMM Benchmark Across 6 Backends")
    parser.add_argument("--size", "-n", type=int, default=512, help="Matrix dimension N (NxN)")
    parser.add_argument("--iters", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--include-pure-python", action="store_true", help="Include slow pure Python (only for N <= 256)")
    parser.add_argument("--output-json", type=str, default="", help="Path to export JSON telemetry")
    args = parser.parse_args()

    N = args.size
    iters = args.iters
    dev, has_accel = get_torch_accelerator()

    print("==========================================================================")
    print(f"  PROFILING HARDWARE ACCELERATED COMPUTATIONS: GEMM BENCHMARK")
    print(f"  Matrix Dimension: {N} x {N} | Hardware Accel: {dev.upper()} ({'Active' if has_accel else 'Inactive'})")
    print("==========================================================================")

    records = []
    telemetry = {"dimension": N, "iterations": iters, "backends": {}}

    # 1. CPU Baselines
    print("[1/3] Benchmarking CPU Baselines (NumPy / Python)...")
    cpu_res = benchmark_cpu_gemm(N=N, run_naive_python=(args.include_pure_python or N <= 128))
    for k, v in cpu_res.items():
        telemetry["backends"][k] = v
        records.append([k, f"{v['time_sec']*1000:.3f}", f"{v['gflops']:.2f}", "1.00x (Baseline)"])

    base_time = cpu_res.get("numpy_blas", {}).get("time_sec", 1.0)

    # 2. Numba Baselines
    print("[2/3] Benchmarking Numba CPU / GPU...")
    numba_res = benchmark_numba_gemm(N=N, iters=iters)
    for k, v in numba_res.items():
        telemetry["backends"][k] = v
        spd = f"{base_time / v['time_sec']:.2f}x"
        records.append([k, f"{v['time_sec']*1000:.3f}", f"{v['gflops']:.2f}", spd])

    # 3. PyTorch Baselines
    print("[3/3] Benchmarking PyTorch CPU / Accelerators...")
    torch_res = benchmark_torch_gemm(N=N, iters=iters)
    for k, v in torch_res.items():
        telemetry["backends"][k] = v
        if "error" in v:
            records.append([k, "N/A", "N/A", v["error"]])
        else:
            spd = f"{base_time / v['time_sec']:.2f}x"
            records.append([k, f"{v['time_sec']*1000:.3f}", f"{v['gflops']:.2f}", spd])

    print("\n" + tabulate(records, headers=["Backend / Optimization Tier", "Latency (ms)", "Throughput (GFLOPS)", "Speedup vs NumPy"], tablefmt="github"))
    print("==========================================================================\n")

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(telemetry, f, indent=2)
        print(f"✓ Telemetry saved to: {args.output_json}")


if __name__ == "__main__":
    main()
