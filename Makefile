# ==============================================================================
# Makefile: Profiling Hardware Accelerated Computations Benchmark Suite
# Author: Recep Erol, PhD (Rice University Center for Research Computing)
# ==============================================================================

SHELL := /bin/bash
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PYTEST := $(VENV_DIR)/bin/pytest
NVCC := $(shell which nvcc 2>/dev/null)

.PHONY: help setup test benchmark_gemm benchmark_mandelbrot benchmark_all \
        profile_torch profile_scalene profile_memray profile_nsys profile_ncu \
        compile_cuda serve_presentation clean

help:
	@echo "=========================================================================="
	@echo "  Profiling Hardware Accelerated Computations - Build & Profiling Matrix"
	@echo "=========================================================================="
	@echo "  setup                 - Initialize Python virtual environment with uv and install dependencies"
	@echo "  test                  - Run PyTest parity suite across all 6 backends"
	@echo "  benchmark_gemm        - Run unified GEMM benchmark (Python, Numba, Torch)"
	@echo "  benchmark_mandelbrot  - Run unified Mandelbrot benchmark (Python, Numba, Torch)"
	@echo "  benchmark_all         - Execute master scaling sweep and generate telemetry charts"
	@echo "  profile_torch         - Profile operators with PyTorch Profiler (Perfetto JSON export)"
	@echo "  profile_scalene       - Profile CPU/Memory line-by-line with Scalene"
	@echo "  profile_memray        - Profile heap allocations and generate Memray HTML flamegraphs"
	@echo "  profile_nsys          - Run NVIDIA Nsight Systems system-wide timeline trace"
	@echo "  profile_ncu           - Run NVIDIA Nsight Compute micro-kernel & Roofline counters"
	@echo "  compile_cuda          - Compile native CUDA C kernels with nvcc (if available)"
	@echo "  serve_presentation    - Launch local HTTP server hosting Reveal.js presentation"
	@echo "  clean                 - Remove build artifacts, pycache, and temporary files"
	@echo "=========================================================================="

setup:
	@echo ">> Setting up Python virtual environment with uv..."
	uv venv --python 3.12 $(VENV_DIR)
	uv pip install -r requirements.txt
	@echo "✓ Environment successfully initialized."

test:
	@echo ">> Running Numerical Parity & Verification Tests..."
	$(PYTEST) tests/ -v

benchmark_gemm:
	@echo ">> Running Dense GEMM Benchmarks..."
	$(PYTHON) src/gemm/run_gemm.py --size 512 --iters 10

benchmark_mandelbrot:
	@echo ">> Running Mandelbrot Fractal Benchmarks..."
	$(PYTHON) src/mandelbrot/run_mandelbrot.py --width 512 --height 512 --max-iter 256 --iters 5

benchmark_all:
	@echo ">> Running Master Telemetry Sweep..."
	$(PYTHON) scripts/run_benchmarks.py

profile_torch:
	@echo ">> Profiling with PyTorch Profiler..."
	$(PYTHON) scripts/profile_torch.py

profile_scalene:
	@echo ">> Profiling with Scalene..."
	./scripts/profile_scalene.sh

profile_memray:
	@echo ">> Profiling with Memray..."
	./scripts/profile_memray.sh

profile_nsys:
	@echo ">> Running NVIDIA Nsight Systems Trace..."
	./scripts/profile_nsys.sh

profile_ncu:
	@echo ">> Running NVIDIA Nsight Compute Kernel Profiling..."
	./scripts/profile_ncu.sh

compile_cuda:
ifdef NVCC
	@echo ">> Compiling CUDA C kernels with $(NVCC)..."
	$(NVCC) -O3 -arch=sm_80 -lcublas src/gemm/gemm_cuda.cu -o src/gemm/gemm_cuda
	$(NVCC) -O3 -arch=sm_80 src/mandelbrot/mandelbrot_cuda.cu -o src/mandelbrot/mandelbrot_cuda
	@echo "✓ CUDA C binaries compiled successfully."
else
	@echo ">> Notice: nvcc compiler not found on PATH. CUDA C sources available in src/*/*.cu"
endif

serve_presentation:
	@echo ">> Serving Reveal.js Keynote Presentation on http://localhost:8000/presentation/ ..."
	$(PYTHON) -m http.server 8000

clean:
	@echo ">> Cleaning cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -f src/gemm/gemm_cuda src/mandelbrot/mandelbrot_cuda
	@echo "✓ Clean complete."
