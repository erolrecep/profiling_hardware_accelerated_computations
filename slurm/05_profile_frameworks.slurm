#!/bin/bash
# ==============================================================================
# SLURM Job: Framework Diagnostics (PyTorch, Scalene, Memray)
# Project: Profiling Hardware Accelerated Computations
# ==============================================================================
#SBATCH --job-name=profile_frameworks
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
echo "  SLURM Job: Framework & Memory Profiling (PyTorch, Scalene, Memray)"
echo "  Job ID: ${SLURM_JOB_ID} | Host: $(hostname)"
echo "=========================================================================="

# 1. Environment Module Loading
if command -v module &> /dev/null; then
    module purge || true
    module load cuda/12.2 || module load cuda/13.0 || module load cuda || true
    module load python/3.12 || module load python/3.11 || true
fi

# 2. Run PyTorch Profiler
echo ">> Running PyTorch Profiler with GPU kernel attribution..."
make profile_torch

# 3. Run Scalene Line-by-Line Profiler
echo ">> Running Scalene Line-by-Line Profiler..."
make profile_scalene

# 4. Run Memray Memory Profiler
echo ">> Running Memray Memory Profiler & generating flamegraphs..."
make profile_memray

echo "=========================================================================="
echo "✓ Framework profiling complete."
echo "  Traces saved to telemetry/torch_trace_*.json and telemetry/*.html"
echo "=========================================================================="

