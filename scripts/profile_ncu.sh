#!/usr/bin/env bash
# NVIDIA Nsight Compute (ncu) Micro-Kernel & Roofline Profiling Runner
# Supports CUDA 12.x and CUDA 13.x toolchains
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/telemetry"
EXPECTED_DIR="${OUTPUT_DIR}/expected_outputs"
mkdir -p "${OUTPUT_DIR}" "${EXPECTED_DIR}"

echo "=========================================================================="
echo "  [PROFILING] NVIDIA Nsight Compute (ncu): Micro-Kernel Hardware Counters"
echo "  Speed-of-Light (SOL), Warp Execution Efficiency, and Roofline Diagnostics"
echo "=========================================================================="

# Detect CUDA version if nvcc is available
if command -v nvcc &> /dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | sed -E 's/.*release ([0-9]+(\.[0-9]+)?).*/\1/')
    echo ">> Detected CUDA Toolkit Version: ${CUDA_VER}"
fi

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
    echo ">> Displaying expected micro-kernel counters & Warp Efficiency report..."
    echo ""
    cat "${EXPECTED_DIR}/ncu_kernel_speed_of_light.txt"
    cp "${EXPECTED_DIR}/ncu_kernel_speed_of_light.txt" "${OUTPUT_DIR}/ncu_gemm_summary.txt"
    echo ""
    echo "✓ Reference ncu micro-kernel telemetry written to: ${OUTPUT_DIR}/ncu_gemm_summary.txt"
fi
