#!/bin/bash
# ==============================================================================
# SLURM Job: Master Benchmark Sweep (GEMM & Mandelbrot)
# Project: Profiling Hardware Accelerated Computations
# ==============================================================================
#SBATCH --job-name=benchmark_all
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -e

echo "=========================================================================="
echo "  SLURM Job: Master Benchmark & Telemetry Sweep"
echo "  Job ID: ${SLURM_JOB_ID} | Host: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"
echo "=========================================================================="

# 1. Environment Module Loading
if command -v module &> /dev/null; then
    module purge || true
    module load cuda/12.2 || module load cuda/13.0 || module load cuda || true
    module load python/3.12 || module load python/3.11 || true
fi

# 2. Compile Native CUDA C Kernels
make compile_cuda

# 3. Run Benchmark Sweeps
echo ">> Running GEMM benchmarks..."
make benchmark_gemm

echo ">> Running Mandelbrot benchmarks..."
make benchmark_mandelbrot

echo ">> Running multi-size scaling sweep & generating telemetry plots..."
make benchmark_all

echo "=========================================================================="
echo "✓ Master benchmark complete. Results saved in telemetry/"
echo "=========================================================================="

