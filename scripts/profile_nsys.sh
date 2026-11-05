#!/usr/bin/env bash
# NVIDIA Nsight Systems (nsys) Macro-Timeline Profiling Runner
# Supports both CUDA 12.x and CUDA 13.x toolchains
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/telemetry"
EXPECTED_DIR="${OUTPUT_DIR}/expected_outputs"
mkdir -p "${OUTPUT_DIR}" "${EXPECTED_DIR}"

echo "=========================================================================="
echo "  [PROFILING] NVIDIA Nsight Systems (nsys): Macro-Timeline Profiler"
echo "  Supports CUDA 12.x (Ampere/Hopper) and CUDA 13.x (Blackwell) Environments"
echo "=========================================================================="

# Detect CUDA version if nvcc is available
if command -v nvcc &> /dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | sed -E 's/.*release ([0-9]+(\.[0-9]+)?).*/\1/')
    echo ">> Detected CUDA Toolkit Version: ${CUDA_VER}"
fi

if command -v nsys &> /dev/null; then
    echo ">> NVIDIA nsys binary detected: $(which nsys)"

    # Profile CUDA C GEMM (if compiled)
    if [ -f "${ROOT_DIR}/src/gemm/gemm_cuda" ]; then
        echo "Profiling CUDA C GEMM..."
        nsys profile \
            --trace=cuda,nvtx,osrt \
            --stats=true \
            --force-overwrite=true \
            --output="${OUTPUT_DIR}/nsys_gemm_cuda" \
            "${ROOT_DIR}/src/gemm/gemm_cuda" 1024 10
    fi

    # Profile PyTorch GPU GEMM
    echo "Profiling PyTorch GPU GEMM..."
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --stats=true \
        --force-overwrite=true \
        --output="${OUTPUT_DIR}/nsys_gemm_torch" \
        "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/src/gemm/run_gemm.py" --size 1024 --iters 5

    # Profile CUDA C Mandelbrot (if compiled)
    if [ -f "${ROOT_DIR}/src/mandelbrot/mandelbrot_cuda" ]; then
        echo "Profiling CUDA C Mandelbrot..."
        nsys profile \
            --trace=cuda,nvtx,osrt \
            --stats=true \
            --force-overwrite=true \
            --output="${OUTPUT_DIR}/nsys_mandelbrot_cuda" \
            "${ROOT_DIR}/src/mandelbrot/mandelbrot_cuda" 2048 2048 500 5
    fi

    echo "✓ nsys traces saved to ${OUTPUT_DIR}/ (*.nsys-rep)"
    echo "Open them in the NVIDIA Nsight Systems GUI to visualize timelines."
else
    echo ">> NOTE: 'nsys' not found on current host PATH (macOS / non-NVIDIA machine)."
    echo ">> Displaying expected side-by-side telemetry for CUDA 12 vs. CUDA 13..."
    echo ""
    cat "${EXPECTED_DIR}/nsys_cuda12_vs_cuda13.txt"
    cp "${EXPECTED_DIR}/nsys_cuda12_vs_cuda13.txt" "${OUTPUT_DIR}/nsys_gemm_summary.txt"
    echo ""
    echo "✓ Reference side-by-side nsys telemetry written to: ${OUTPUT_DIR}/nsys_gemm_summary.txt"
fi
