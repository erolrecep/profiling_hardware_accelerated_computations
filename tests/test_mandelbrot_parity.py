"""
Unit & Parity Tests: Mandelbrot Set Generation
Verifies algorithmic correctness and escape-time parity across:
  - Pure Python
  - Numba CPU Serial
  - Numba CPU Parallel
  - NumPy Vectorized
  - PyTorch CPU & Hardware Accelerator (MPS / CUDA)
"""

import sys
import os
import pytest
import numpy as np
import torch

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/mandelbrot")))

from mandelbrot_cpu import mandelbrot_pure_python, mandelbrot_numpy_vectorized
from mandelbrot_numba import mandelbrot_numba_cpu_serial, mandelbrot_numba_cpu_parallel
from mandelbrot_torch import mandelbrot_torch_tensor, get_torch_accelerator, synchronize_device


@pytest.fixture
def mandelbrot_params():
    return {
        "width": 64,
        "height": 64,
        "max_iter": 128,
        "xmin": -2.0,
        "xmax": 0.5,
        "ymin": -1.25,
        "ymax": 1.25,
    }


def test_numba_serial_matches_pure_python(mandelbrot_params):
    p = mandelbrot_params
    ref_py = np.array(mandelbrot_pure_python(p["width"], p["height"], p["max_iter"], p["xmin"], p["xmax"], p["ymin"], p["ymax"]), dtype=np.int32)
    numba_out = np.zeros((p["height"], p["width"]), dtype=np.int32)
    mandelbrot_numba_cpu_serial(numba_out, p["width"], p["height"], p["max_iter"], p["xmin"], p["xmax"], p["ymin"], p["ymax"])
    np.testing.assert_array_equal(numba_out, ref_py)


def test_numba_parallel_matches_serial(mandelbrot_params):
    p = mandelbrot_params
    serial_out = np.zeros((p["height"], p["width"]), dtype=np.int32)
    parallel_out = np.zeros((p["height"], p["width"]), dtype=np.int32)

    mandelbrot_numba_cpu_serial(serial_out, p["width"], p["height"], p["max_iter"], p["xmin"], p["xmax"], p["ymin"], p["ymax"])
    mandelbrot_numba_cpu_parallel(parallel_out, p["width"], p["height"], p["max_iter"], p["xmin"], p["xmax"], p["ymin"], p["ymax"])

    np.testing.assert_array_equal(parallel_out, serial_out)


def test_torch_cpu_matches_numpy(mandelbrot_params):
    p = mandelbrot_params
    np_out = mandelbrot_numpy_vectorized(p["width"], p["height"], p["max_iter"], p["xmin"], p["xmax"], p["ymin"], p["ymax"])
    torch_out = mandelbrot_torch_tensor(p["width"], p["height"], p["max_iter"], p["xmin"], p["xmax"], p["ymin"], p["ymax"], device="cpu").numpy()

    # Compare pixel escape agreement
    agreement = np.mean(np_out == torch_out)
    assert agreement > 0.99, f"Vectorized NumPy and PyTorch CPU agreement was {agreement:.4f}, expected > 0.99"


def test_torch_accelerator_matches_cpu(mandelbrot_params):
    p = mandelbrot_params
    dev_name, has_accel = get_torch_accelerator()
    if not has_accel:
        pytest.skip("No hardware accelerator (CUDA/MPS) detected.")

    torch_cpu = mandelbrot_torch_tensor(p["width"], p["height"], p["max_iter"], p["xmin"], p["xmax"], p["ymin"], p["ymax"], device="cpu")
    torch_dev = mandelbrot_torch_tensor(p["width"], p["height"], p["max_iter"], p["xmin"], p["xmax"], p["ymin"], p["ymax"], device=dev_name)
    synchronize_device(dev_name)

    np.testing.assert_array_equal(torch_dev.cpu().numpy(), torch_cpu.numpy())
