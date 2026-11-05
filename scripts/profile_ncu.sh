#!/usr/bin/env bash
# NVIDIA Nsight Compute (ncu) Micro-Kernel & Roofline Profiling Runner
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/telemetry"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================================================="
echo "  [PROFILING] NVIDIA Nsight Compute (ncu): Micro-Kernel Counters"
echo "=========================================================================="

if command -v ncu &> /dev/null; then
    echo ">> NVIDIA ncu binary detected: $(which ncu)"

    if [ -f "${ROOT_DIR}/src/gemm/gemm_cuda" ]; then
        echo "Profiling CUDA C GEMM kernels with Roofline & Speed-of-Light..."
        ncu --set full \
            --section SpeedOfLight \
            --section MemoryWorkloadAnalysis \
            --section WarpStateStats \
            --force-overwrite \
            --export "${OUTPUT_DIR}/ncu_gemm" \
            "${ROOT_DIR}/src/gemm/gemm_cuda" 1024 1
    fi

    if [ -f "${ROOT_DIR}/src/mandelbrot/mandelbrot_cuda" ]; then
        echo "Profiling CUDA C Mandelbrot kernel with Warp Divergence analysis..."
        ncu --section SpeedOfLight \
            --section WarpStateStats \
            --force-overwrite \
            --export "${OUTPUT_DIR}/ncu_mandelbrot" \
            "${ROOT_DIR}/src/mandelbrot/mandelbrot_cuda" 1024 1024 256 1
    fi

    echo "✓ ncu reports saved to ${OUTPUT_DIR}/ (*.ncu-rep)"
    echo "Open them in the NVIDIA Nsight Compute GUI to inspect Rooflines."
else
    echo ">> NOTE: 'ncu' not found on current host PATH (macOS / non-NVIDIA machine)."
    echo ">> To run on an NVIDIA GPU cluster (e.g., Rice NOTS / SLURM node):"
    echo ""
    echo "   1. Compile CUDA C kernels:"
    echo "      nvcc -O3 -arch=sm_80 -lcublas src/gemm/gemm_cuda.cu -o src/gemm/gemm_cuda"
    echo ""
    echo "   2. Run ncu Speed-of-Light & Roofline analysis:"
    echo "      ncu --set full --section SpeedOfLight --section MemoryWorkloadAnalysis -o telemetry/ncu_gemm src/gemm/gemm_cuda 1024 1"
    echo ""
    echo "   3. Analyze Warp Divergence on Mandelbrot:"
    echo "      ncu --section WarpStateStats -o telemetry/ncu_mandelbrot src/mandelbrot/mandelbrot_cuda 1024 1024 256 1"
    echo ""
    echo ">> Generating mock telemetry summary for workshop pedagogical study..."
    cat << 'EOF' > "${OUTPUT_DIR}/ncu_gemm_summary.txt"
----------------------------------------------------------------------
Kernel: gemm_shared_tiled_kernel (N = 1024, Grid = [64, 64], Block = [16, 16])
----------------------------------------------------------------------
Speed-of-Light (SOL):
  Compute (SM) Throughput:           68.42% of Theoretical Peak
  Memory Throughput (DRAM):          74.15% of Peak Bandwidth

Memory Workload Analysis:
  L1/Shared Memory Hit Rate:         98.24% (Tiling Data Reuse Active)
  L2 Cache Hit Rate:                 84.10%
  DRAM Read Throughput:              182.4 GB/s

Warp State Statistics (Avg Active Threads per Warp: 32.0 / 32.0):
  Eligible Warps per Cycle:          2.41
  Stall Reasons:
    - Long Scoreboard (Memory L2/DRAM): 41.2%
    - Math Pipe Throttle:               29.8%
    - Wait on Barrier (__syncthreads):  18.4%
    - Other / Instruction Fetch:        10.6%
EOF
    echo "✓ Reference ncu telemetry written to: ${OUTPUT_DIR}/ncu_gemm_summary.txt"
fi
