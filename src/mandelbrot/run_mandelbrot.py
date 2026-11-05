"""
Master CLI Runner: Mandelbrot Set Benchmark Suite Across 6 Backends
Compares:
  - Pure Python
  - NumPy Vectorized
  - Numba CPU Serial vs Multi-Threaded Parallel
  - Numba GPU (if available)
  - PyTorch CPU vs GPU/MPS
"""

import argparse
import json
import os
import sys
from tabulate import tabulate

from mandelbrot_cpu import benchmark_cpu_mandelbrot
from mandelbrot_numba import benchmark_numba_mandelbrot
from mandelbrot_torch import benchmark_torch_mandelbrot, get_torch_accelerator


def main():
    parser = argparse.ArgumentParser(description="Mandelbrot Benchmark Across 6 Backends")
    parser.add_argument("--width", "-w", type=int, default=512, help="Grid width")
    parser.add_argument("--height", "-H", type=int, default=512, help="Grid height")
    parser.add_argument("--max-iter", "-m", type=int, default=256, help="Maximum escape iterations")
    parser.add_argument("--iters", type=int, default=5, help="Benchmark repetitions")
    parser.add_argument("--include-pure-python", action="store_true", help="Include slow pure Python (width*height <= 256*256)")
    parser.add_argument("--output-json", type=str, default="", help="Path to export JSON telemetry")
    args = parser.parse_args()

    width = args.width
    height = args.height
    max_iter = args.max_iter
    iters = args.iters
    dev, has_accel = get_torch_accelerator()

    print("==========================================================================")
    print(f"  PROFILING HARDWARE ACCELERATED COMPUTATIONS: MANDELBROT BENCHMARK")
    print(f"  Resolution: {width} x {height} | Max Iter: {max_iter} | Accel: {dev.upper()} ({'Active' if has_accel else 'Inactive'})")
    print("==========================================================================")

    records = []
    telemetry = {"width": width, "height": height, "max_iter": max_iter, "backends": {}}

    # 1. CPU Baselines
    print("[1/3] Benchmarking CPU Baselines (NumPy / Python)...")
    cpu_res = benchmark_cpu_mandelbrot(width, height, max_iter, run_naive_python=args.include_pure_python)
    for k, v in cpu_res.items():
        telemetry["backends"][k] = v
        records.append([k, f"{v['time_sec']*1000:.3f}", f"{v['mpix_per_sec']:.2f}", str(v["checksum"]), "1.00x (Baseline)"])

    base_time = cpu_res.get("numpy_vectorized", {}).get("time_sec", 1.0)

    # 2. Numba Baselines
    print("[2/3] Benchmarking Numba CPU / GPU...")
    numba_res = benchmark_numba_mandelbrot(width, height, max_iter, iters=iters)
    for k, v in numba_res.items():
        telemetry["backends"][k] = v
        spd = f"{base_time / v['time_sec']:.2f}x"
        records.append([k, f"{v['time_sec']*1000:.3f}", f"{v['mpix_per_sec']:.2f}", str(v["checksum"]), spd])

    # 3. PyTorch Baselines
    print("[3/3] Benchmarking PyTorch CPU / Accelerators...")
    torch_res = benchmark_torch_mandelbrot(width, height, max_iter, iters=iters)
    for k, v in torch_res.items():
        telemetry["backends"][k] = v
        spd = f"{base_time / v['time_sec']:.2f}x"
        records.append([k, f"{v['time_sec']*1000:.3f}", f"{v['mpix_per_sec']:.2f}", str(v["checksum"]), spd])

    print("\n" + tabulate(records, headers=["Backend / Tier", "Latency (ms)", "Throughput (Mpix/s)", "Checksum", "Speedup vs NumPy"], tablefmt="github"))
    print("==========================================================================\n")

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(telemetry, f, indent=2)
        print(f"✓ Telemetry saved to: {args.output_json}")


if __name__ == "__main__":
    main()
