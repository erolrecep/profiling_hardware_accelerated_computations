#!/usr/bin/env bash
# Scalene Profiler Runner for Python CPU & Memory
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/telemetry"
mkdir -p "${OUTPUT_DIR}"

PYTHON="${ROOT_DIR}/.venv/bin/python"
SCALENE="${ROOT_DIR}/.venv/bin/scalene"

echo "=========================================================================="
echo "  [PROFILING] Scalene CPU/Memory: Dense GEMM (N=128)"
echo "=========================================================================="
${SCALENE} run --cpu-only -o "${OUTPUT_DIR}/scalene_gemm.json" \
    "${ROOT_DIR}/src/gemm/run_gemm.py" --- --size 128 --iters 3

echo ""
echo "=========================================================================="
echo "  [PROFILING] Scalene CPU/Memory: Mandelbrot (128x128)"
echo "=========================================================================="
${SCALENE} run --cpu-only -o "${OUTPUT_DIR}/scalene_mandelbrot.json" \
    "${ROOT_DIR}/src/mandelbrot/run_mandelbrot.py" --- --width 128 --height 128 --iters 2

echo ""
echo "✓ Scalene telemetry profiles generated:"
echo "   - ${OUTPUT_DIR}/scalene_gemm.json"
echo "   - ${OUTPUT_DIR}/scalene_mandelbrot.json"
echo ""
echo "To inspect terminal summary: ${SCALENE} view --cli ${OUTPUT_DIR}/scalene_gemm.json"
