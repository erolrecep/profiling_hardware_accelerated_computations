#!/usr/bin/env bash
# Memray Memory Profiler Runner & Flamegraph Generator
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/telemetry"
mkdir -p "${OUTPUT_DIR}"

PYTHON="${ROOT_DIR}/.venv/bin/python"
MEMRAY="${ROOT_DIR}/.venv/bin/memray"

echo "=========================================================================="
echo "  [PROFILING] Memray Memory Profiler: Dense GEMM (N=256)"
echo "=========================================================================="
${MEMRAY} run --force -o "${OUTPUT_DIR}/memray_gemm.bin" \
    "${ROOT_DIR}/src/gemm/run_gemm.py" --size 256 --iters 5

echo "--- Generating Flamegraph & Summary Table ---"
${MEMRAY} flamegraph --force -o "${OUTPUT_DIR}/memray_gemm_flamegraph.html" "${OUTPUT_DIR}/memray_gemm.bin"
${MEMRAY} summary "${OUTPUT_DIR}/memray_gemm.bin"

echo ""
echo "=========================================================================="
echo "  [PROFILING] Memray Memory Profiler: Mandelbrot (256x256)"
echo "=========================================================================="
${MEMRAY} run --force -o "${OUTPUT_DIR}/memray_mandelbrot.bin" \
    "${ROOT_DIR}/src/mandelbrot/run_mandelbrot.py" --width 256 --height 256 --iters 3

echo "--- Generating Flamegraph & Summary Table ---"
${MEMRAY} flamegraph --force -o "${OUTPUT_DIR}/memray_mandelbrot_flamegraph.html" "${OUTPUT_DIR}/memray_mandelbrot.bin"
${MEMRAY} summary "${OUTPUT_DIR}/memray_mandelbrot.bin"

echo ""
echo "✓ Memray interactive HTML flamegraphs generated:"
echo "   - ${OUTPUT_DIR}/memray_gemm_flamegraph.html"
echo "   - ${OUTPUT_DIR}/memray_mandelbrot_flamegraph.html"
