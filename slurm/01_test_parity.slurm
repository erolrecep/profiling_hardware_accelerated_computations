#!/bin/bash
# ==============================================================================
# SLURM Job: Numerical Parity & Verification Test Suite
# Project: Profiling Hardware Accelerated Computations
# ==============================================================================
#SBATCH --job-name=test_parity
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -e

echo "=========================================================================="
echo "  SLURM Job: Numerical Parity & Verification Suite"
echo "  Job ID: ${SLURM_JOB_ID} | Host: $(hostname)"
echo "  Allocated GPU(s): ${CUDA_VISIBLE_DEVICES:-None}"
echo "=========================================================================="

# 1. Environment Module Loading (Compatible with Rice NOTS / SLURM Clusters)
if command -v module &> /dev/null; then
    echo ">> Loading environment modules..."
    module purge || true
    module load cuda/12.2 || module load cuda/13.0 || module load cuda || true
    module load python/3.12 || module load python/3.11 || true
fi

# 2. Print Host Diagnostics
nvidia-smi || echo ">> Warning: nvidia-smi not available"
which python3
which nvcc || echo ">> Notice: nvcc not loaded via modules"

# 3. Setup Virtual Environment
if [ ! -d ".venv" ]; then
    echo ">> Initializing virtual environment..."
    make setup
fi

# 4. Run PyTest Parity Suite Across CPU, CUDA, Numba, and PyTorch
echo ">> Running automated verification tests..."
make test

echo "=========================================================================="
echo "✓ Verification suite completed successfully."
echo "=========================================================================="
