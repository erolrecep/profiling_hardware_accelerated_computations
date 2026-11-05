#!/usr/bin/env bash
# ==============================================================================
# Master SLURM Batch Submission Script
# Project: Profiling Hardware Accelerated Computations
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p slurm/logs telemetry

echo "=========================================================================="
echo "  Submitting SLURM Profiling Pipeline Jobs"
echo "=========================================================================="

if ! command -v sbatch &> /dev/null; then
    echo ">> Error: 'sbatch' command not found. This script must be run on a SLURM head node."
    echo ">> Usage on cluster:"
    echo "     ./slurm/submit_all.sh"
    exit 1
fi

# Submit Job 1: Verification Tests
JOB1=$(sbatch --parsable slurm/01_test_parity.slurm)
echo ">> [1/5] Submitted Test Parity Job:        ${JOB1}"

# Submit Job 2: Benchmarks (runs after tests succeed)
JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} slurm/02_benchmark_all.slurm)
echo ">> [2/5] Submitted Benchmark Sweep Job:    ${JOB2} (depends on ${JOB1})"

# Submit Job 3: NVIDIA Nsight Systems
JOB3=$(sbatch --parsable --dependency=afterok:${JOB2} slurm/03_profile_nsys.slurm)
echo ">> [3/5] Submitted Nsight Systems Job:     ${JOB3} (depends on ${JOB2})"

# Submit Job 4: NVIDIA Nsight Compute
JOB4=$(sbatch --parsable --dependency=afterok:${JOB2} slurm/04_profile_ncu.slurm)
echo ">> [4/5] Submitted Nsight Compute Job:     ${JOB4} (depends on ${JOB2})"

# Submit Job 5: Frameworks (PyTorch, Scalene, Memray)
JOB5=$(sbatch --parsable --dependency=afterok:${JOB2} slurm/05_profile_frameworks.slurm)
echo ">> [5/5] Submitted Framework Profiling Job: ${JOB5} (depends on ${JOB2})"

echo ""
echo "=========================================================================="
echo "All jobs submitted to SLURM queue!"
echo "Monitor queue with:  squeue -u \$USER"
echo "View job logs with:  tail -f slurm/logs/*.out"
echo "=========================================================================="
