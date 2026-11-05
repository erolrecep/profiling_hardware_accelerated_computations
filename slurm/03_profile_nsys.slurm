#!/bin/bash
# ==============================================================================
# SLURM Job: NVIDIA Nsight Systems (nsys) Macro-Timeline Profiling
# Project: Profiling Hardware Accelerated Computations
# ==============================================================================
#SBATCH --job-name=profile_nsys
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
echo "  SLURM Job: NVIDIA Nsight Systems (nsys) Timeline Tracing"
echo "  Job ID: ${SLURM_JOB_ID} | Host: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"
echo "=========================================================================="

# 1. Environment Module Loading
if command -v module &> /dev/null; then
    module purge || true
    module load cuda/12.2 || module load cuda/13.0 || module load cuda || true
    module load python/3.12 || module load python/3.11 || true
fi

# 2. Check nsys Availability
if ! command -v nsys &> /dev/null; then
    echo ">> Error: 'nsys' binary not found on PATH. Ensure the CUDA toolkit is loaded."
    exit 1
fi

echo ">> nsys version: $(nsys --version | head -n 1)"

# 3. Compile CUDA C Kernels
make compile_cuda

# 4. Run nsys Timeline Profiling
echo ">> Running Nsight Systems trace..."
./scripts/profile_nsys.sh

echo "=========================================================================="
echo "✓ nsys profiling complete. Traces saved to telemetry/*.nsys-rep"
echo "  Download to local workstation and open with: nsys-ui <trace>.nsys-rep"
echo "=========================================================================="

