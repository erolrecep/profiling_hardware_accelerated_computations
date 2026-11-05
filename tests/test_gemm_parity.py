"""
Unit & Parity Tests: Dense Matrix Multiplication (GEMM)
Verifies mathematical correctness and numerical parity across:
  - Pure Python (i-j-k & i-k-j)
  - NumPy OpenBLAS / Accelerate reference
  - Numba CPU Serial & Parallel
  - PyTorch CPU & Hardware Accelerator (MPS / CUDA)
"""

import sys
import os
import pytest
import numpy as np
import torch

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/gemm")))

from gemm_cpu import gemm_pure_python_ijk, gemm_pure_python_ikj, gemm_numpy
from gemm_numba import gemm_numba_cpu_serial, gemm_numba_cpu_parallel
from gemm_torch import gemm_torch, get_torch_accelerator, synchronize_device


@pytest.fixture
def test_matrices():
    N = 64
    rng = np.random.default_rng(1337)
    A = rng.standard_normal((N, N), dtype=np.float32)
    B = rng.standard_normal((N, N), dtype=np.float32)
    ref = np.matmul(A, B)
    return A, B, ref


def test_pure_python_ijk_parity(test_matrices):
    A, B, ref = test_matrices
    C_ijk = np.array(gemm_pure_python_ijk(A.tolist(), B.tolist()), dtype=np.float32)
    np.testing.assert_allclose(C_ijk, ref, rtol=1e-4, atol=1e-4)


def test_pure_python_ikj_parity(test_matrices):
    A, B, ref = test_matrices
    C_ikj = np.array(gemm_pure_python_ikj(A.tolist(), B.tolist()), dtype=np.float32)
    np.testing.assert_allclose(C_ikj, ref, rtol=1e-4, atol=1e-4)


def test_numba_cpu_serial_parity(test_matrices):
    A, B, ref = test_matrices
    C = np.zeros_like(ref)
    gemm_numba_cpu_serial(A, B, C)
    np.testing.assert_allclose(C, ref, rtol=1e-4, atol=1e-4)


def test_numba_cpu_parallel_parity(test_matrices):
    A, B, ref = test_matrices
    C = np.zeros_like(ref)
    gemm_numba_cpu_parallel(A, B, C)
    np.testing.assert_allclose(C, ref, rtol=1e-4, atol=1e-4)


def test_torch_cpu_parity(test_matrices):
    A, B, ref = test_matrices
    A_t = torch.from_numpy(A)
    B_t = torch.from_numpy(B)
    C_t = gemm_torch(A_t, B_t).numpy()
    np.testing.assert_allclose(C_t, ref, rtol=1e-4, atol=1e-4)


def test_torch_accelerator_parity(test_matrices):
    A, B, ref = test_matrices
    dev_name, has_accel = get_torch_accelerator()
    if not has_accel:
        pytest.skip("No hardware accelerator (CUDA/MPS) detected.")

    device = torch.device(dev_name)
    A_t = torch.from_numpy(A).to(device)
    B_t = torch.from_numpy(B).to(device)
    C_t = gemm_torch(A_t, B_t)
    synchronize_device(dev_name)

    C_np = C_t.cpu().numpy()
    np.testing.assert_allclose(C_np, ref, rtol=1e-4, atol=1e-4)
