#!/bin/bash
# ==============================================================================
# SLURM Job: NVIDIA Nsight Compute (ncu) Micro-Kernel Profiling
# Project: Profiling Hardware Accelerated Computations
# ==============================================================================
#SBATCH --job-name=profile_ncu
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
echo "  SLURM Job: NVIDIA Nsight Compute (ncu) Hardware Counters"
echo "  Job ID: ${SLURM_JOB_ID} | Host: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"
echo "=========================================================================="

# 1. Environment Module Loading
if command -v module &> /dev/null; then
    module purge || true
    module load cuda/12.2 || module load cuda/13.0 || module load cuda || true
    module load python/3.12 || module load python/3.11 || true
fi

# 2. Check ncu Availability
if ! command -v ncu &> /dev/null; then
    echo ">> Error: 'ncu' binary not found on PATH. Ensure the CUDA toolkit is loaded."
    exit 1
fi

echo ">> ncu version: $(ncu --version | head -n 1)"

# 3. Compile CUDA C Kernels
make compile_cuda

# 4. Run ncu Micro-Kernel Profiling
echo ">> Running Nsight Compute kernel counters..."
./scripts/profile_ncu.sh

echo "=========================================================================="
echo "✓ ncu profiling complete. Reports saved to telemetry/*.ncu-rep"
echo "  Download to local workstation and open with: ncu-ui <report>.ncu-rep"
echo "=========================================================================="

