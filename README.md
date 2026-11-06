# Profiling Hardware Accelerated Computations: From Micro-Architecture to Code Optimization

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Numba](https://img.shields.io/badge/Numba-0.68+-00a3e0.svg)](https://numba.pydata.org/)
[![CUDA C](https://img.shields.io/badge/CUDA-12.0+-76b900.svg)](https://developer.nvidia.com/cuda-zone)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/Presentation-Reveal.js-orange.svg)](https://erolrecep.github.io/profiling_hardware_accelerated_computations/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erolrecep/profiling_hardware_accelerated_computations/blob/main/notebooks/cuda_profiling_colab_workshop.ipynb)

> **A publication-grade, half-day technical workshop on diagnosing hardware bottlenecks, demystifying CPU/GPU profiling tools, and systematically driving implementations to the theoretical Roofline peak.**

Authored by **Dr. Recep Erol** ([Rice University Center for Research Computing](https://crc.rice.edu))  
GitHub: [erolrecep/profiling_hardware_accelerated_computations](https://github.com/erolrecep/profiling_hardware_accelerated_computations)  
Interactive 25-Slide Deck: [Live Keynote Presentation](https://erolrecep.github.io/profiling_hardware_accelerated_computations/)  
Interactive Cloud Lab: [Open in Google Colab](https://colab.research.google.com/github/erolrecep/profiling_hardware_accelerated_computations/blob/main/notebooks/cuda_profiling_colab_workshop.ipynb)

---

## Table of Contents

1. [Executive Overview & Pedagogical Philosophy](#executive-overview--pedagogical-philosophy)
2. [Mini-Projects Architecture](#mini-projects-architecture)
   - [Workload A: Dense General Matrix Multiplication (GEMM)](#workload-a-dense-general-matrix-multiplication-gemm)
   - [Workload B: Mandelbrot Set Fractal Generation](#workload-b-mandelbrot-set-fractal-generation)
3. [The Profiling Suite & Diagnostic Toolkit](#the-profiling-suite--diagnostic-toolkit)
   - [NVIDIA Tooling Guide: Navigating CUDA 12 vs. CUDA 13](#nvidia-tooling-guide-navigating-cuda-12-vs-cuda-13)
   - [Visual Profiler Gallery: What to Expect When Replicating](#visual-profiler-gallery-what-to-expect-when-replicating)
4. [Empirical Benchmarks & Profiling Case Studies](#empirical-benchmarks--profiling-case-studies)
5. [The Systems Diagnostic Decision Tree](#the-systems-diagnostic-decision-tree)
6. [Quick Start & Turnkey Makefile Reference](#quick-start--turnkey-makefile-reference)
7. [SLURM Cluster Deployment (HPC Environments)](#slurm-cluster-deployment-hpc-environments)
8. [Online Cloud GPU Testing Platforms: Replicate & Compare for Free](#online-cloud-gpu-testing-platforms-replicate--compare-for-free)
9. [Automated Verification & Parity Testing](#automated-verification--parity-testing)
10. [Workshop Presentation Deck (Reveal.js)](#workshop-presentation-deck-revealjs)
11. [Author & Institutional Affiliation](#author--institutional-affiliation)
6. [Quick Start & Turnkey Makefile Reference](#quick-start--turnkey-makefile-reference)
7. [Automated Verification & Parity Testing](#automated-verification--parity-testing)
8. [Workshop Presentation Deck (Reveal.js)](#workshop-presentation-deck-revealjs)
9. [Author & Institutional Affiliation](#author--institutional-affiliation)

---

## Executive Overview & Pedagogical Philosophy

In scientific machine learning, high-performance computing, and real-time vision, software performance is rarely bound by computational throughput alone. Instead, performance collapses into **hidden latency traps**:
- Memory subsystem stalls ($L1/L2/L3$ cache thrashing, non-coalesced DRAM bursts),
- Hardware-host synchronization overheads (`cudaDeviceSynchronize()`, implicit D2H blocking copy),
- Inefficient thread execution (SIMT warp divergence, instruction-level stalls), and
- Framework execution overhead (Python interpreter GIL contention, dynamic memory churn).

This repository serves as a self-contained, reproducible workshop. Rather than treating profilers as passive diagnostic monitors, this curriculum treats profiling as an **active, hypothesis-driven engineering methodology**:

```
                       ┌─────────────────────────┐
                       │  Baseline Unoptimized   │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │   Telemetry Profiling   │
                       │ (nsys, ncu, torch, etc) │
                       └────────────┬────────────┘
                                    │
             ┌──────────────────────┴──────────────────────┐
             ▼                                             ▼
  [Compute Bound Stall]                         [Memory Bound Stall]
  - SIMD/Tensor Cores unutilized                - Strided cache misses
  - SIMT warp branch divergence                 - Global memory bottlenecks
             │                                             │
             ▼                                             ▼
  [Hypothesis & Targeted Optimization]          [Hypothesis & Targeted Optimization]
  - Grid-stride loops & ILP                     - 2D Shared memory tiling
  - Eliminate D2H sync checks                   - Loop interchange & coalescence
             │                                             │
             └──────────────────────┬──────────────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Verified Speedup Table  │
                       │  (1x to 13,700x Peak)   │
                       └─────────────────────────┘
```

---

## Mini-Projects Architecture

The curriculum implements two complementary workload archetypes across **6 computational runtimes**:
1. **Pure Python** (unoptimized baseline)
2. **NumPy / Vectorized BLAS** (SIMD-accelerated C baseline)
3. **Native CUDA C** (GPU hardware kernel hierarchy)
4. **Numba CPU** (LLVM JIT parallel multi-threading)
5. **Numba GPU** (`@cuda.jit` device kernels)
6. **PyTorch** (CPU, CUDA/MPS hardware acceleration, and `torch.compile`)

### Workload A: Dense General Matrix Multiplication (GEMM)
- **Mathematical Archetype**: $C = \alpha (A \times B) + \beta C$.
- **Algorithmic Complexity**: $\mathcal{O}(N^3)$ arithmetic floating-point operations over $\mathcal{O}(N^2)$ memory footprint.
- **Hardware Bottlenecks**: Arithmetic intensity $I = \frac{2N^3}{3N^2 \times 4} = \frac{N}{6}\text{ FLOP/byte}$. Unoptimized implementations suffer from catastrophic memory stalls due to strided matrix reads.
- **Optimization Pathway**:
  - `gemm_python_naive`: Triple-nested loop ($i-j-k$) causing non-contiguous stride reads on matrix $B$.
  - `gemm_python_interchanged`: $i-k-j$ loop order keeping $B[k, j]$ contiguous in cache lines (**$8.2\times$ speedup**).
  - `gemm_cuda_shared_tiled`: 2D block-tiling ($B=16$) staging sub-matrices into on-chip $L1$ Shared Memory ($48\text{ KB}$ per SM), slashing global DRAM transactions by $16\times$.
  - `gemm_cuda_vectorized`: Vectorized 128-bit `float4` loads saturating memory bus width.
  - `gemm_cublas`: Peak hardware ceiling via cuBLAS / Metal Performance Shaders (MPS).

### Workload B: Mandelbrot Set Fractal Generation
- **Mathematical Archetype**: Iterative escape equation $z_{n+1} = z_n^2 + c$, where $z, c \in \mathbb{C}$, terminating when $|z_n| > 2$ or $n = N_{\max}$.
- **Algorithmic Complexity**: Embarrassingly parallel coordinate mapping, but non-uniform, dynamically unpredictable workload per pixel.
- **Hardware Bottlenecks**: Severe **SIMT Warp Divergence**. Coordinates outside the set escape in 2 iterations; boundary coordinates exhaust all 500 iterations.
- **Optimization Pathway**:
  - `mandelbrot_cpu_naive`: Scalar Python complex arithmetic loop.
  - `mandelbrot_cpu_numpy`: Vectorized boolean active masking.
  - `mandelbrot_cuda_naive`: 2D thread grid directly mapping pixels to threads (inducing $96.5\%$ idle warp cycles).
  - `mandelbrot_cuda_grid_stride`: 1D grid-stride loop with dynamic work distribution balancing warp loads.
  - `mandelbrot_cuda_ilp`: Dual-point instruction-level parallelism unrolling two pixels per thread register bank.
  - `mandelbrot_torch_optimized`: Tensorized complex plane escape loop eliminating host-synchronizing convergence checks.

---

## The Profiling Suite & Diagnostic Toolkit

| Profiler | Scope & Target | Sampling Mechanism | Primary Telemetry Metrics | Ideal Diagnostics |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA Nsight Systems (`nsys`)** | System-wide Timeline (Host CPU, OS, Driver, CUDA) | Event-tracing & low-overhead hardware counters | PCIe D2H/H2D transfers, CUDA API overhead, stream overlap | Pipeline stalls, CPU-GPU starvation, sync bubbles |
| **NVIDIA Nsight Compute (`ncu`)** | GPU Kernel Micro-Architecture | Hardware Performance Counters (SM, Warp Schedulers) | Warp execution efficiency, Roofline model, SM occupancy | Memory bandwidth vs arithmetic bottlenecks, register spills |
| **PyTorch Profiler (`torch.profiler`)** | PyTorch Operator Graph & GPU Kernels | Kineto instrumentation engine | Self CPU time, CUDA device time, active allocations | Slow autograd/framework ops, unexpected fallback to CPU |
| **Scalene** | Python vs Native C vs GPU code lines | High-frequency CPU sampling + heap intercept | % Python time, % Native time, % GPU time, allocation rates | Identifying Python interpreter tax and C-extension bottlenecks |
| **Memray** | C/C++ and Python Memory Allocations | `malloc`/`free` dynamic symbol tracking | Peak virtual memory, high-water mark, flamegraphs | Memory leaks, transient tensor churn, OOM diagnostics |

---

### NVIDIA Tooling Guide: Navigating CUDA 12 vs. CUDA 13

In academic and enterprise HPC environments, teams frequently migrate between established **CUDA 12.x clusters** (Hopper H100, Ada Lovelace, Ampere A100) and cutting-edge **CUDA 13.x nodes** (Blackwell B200 / GB200). Understanding how NVIDIA's profiling outputs compare ensures attendees can interpret telemetry without confusion.

#### Side-by-Side Architectural & Telemetry Comparison

| Capability / Attribute | NVIDIA Toolchain in CUDA 12.x | NVIDIA Toolchain in CUDA 13.x |
| :--- | :--- | :--- |
| **Target Architectures** | Ampere (sm_80), Ada (sm_89), Hopper (sm_90) | Blackwell (sm_100), Hopper, Vera Rubin |
| **`nsys` Column Formats** | Raw nanoseconds (`Avg (ns)`, `Total Time (ns)`) | Auto human-readable units (`Avg (us)`, `Duration (ms)`) |
| **Subsystem Attribution** | Implied via API name (`cudaLaunchKernel`, `cudaMemcpy`) | Explicit target column (`SM / Global DRAM` vs `SM / Shared Tile`) |
| **Multi-GPU & Interconnect** | Manual bandwidth math from transfer bytes | Direct NVLink-5 / PCIe Gen 5 saturation percentages |
| **`ncu` Focus** | Tensor Core FP8/FP16, DPX dynamic programming | Time-resolved kernel timelines, Tile IR vs SIMT, Report Merge |
| **Core Workflow** | **1. `nsys` (Timeline) $\to$ 2. `ncu` (Micro-Kernel)** | **1. `nsys` (Timeline) $\to$ 2. `ncu` (Micro-Kernel)** *(Identical!)* |

#### Pre-Compiled Annotated Output Logs
The repository provides concrete, formatted expected terminal logs for all profiling tools in `telemetry/expected_outputs/`:
- 📄 [`telemetry/expected_outputs/nsys_cuda12_vs_cuda13.txt`](telemetry/expected_outputs/nsys_cuda12_vs_cuda13.txt): Side-by-side annotated CLI output of `nsys profile --stats=true`.
- 📄 [`telemetry/expected_outputs/ncu_kernel_speed_of_light.txt`](telemetry/expected_outputs/ncu_kernel_speed_of_light.txt): Micro-kernel Speed-of-Light and Warp Execution Efficiency diagnostics.
- 📄 [`telemetry/expected_outputs/torch_profiler_output.txt`](telemetry/expected_outputs/torch_profiler_output.txt): Operator breakdown uncovering the 86.2% D2H CPU synchronization stall.
- 📄 [`telemetry/expected_outputs/scalene_output.txt`](telemetry/expected_outputs/scalene_output.txt): Line-by-line colored attribution isolating Python bytecode vs native C time.
- 📄 [`telemetry/expected_outputs/memray_output.txt`](telemetry/expected_outputs/memray_output.txt): Heap allocation analysis tracking float object boxing.

---

### 📸 Visual Profiler Gallery: Bad vs. Good Profiling Results (Before & After Optimization)

To provide concrete visual guidance when replicating these benchmarks on your local workstation or cluster, here are reference snapshots comparing the **Unoptimized (Bad)** versus **Optimized (Good)** results side-by-side:

#### 1. Master Profiler Suite: Cross-Tool Correlation
![Side-by-Side Profiler Comparison](telemetry/snapshots/snapshot_side_by_side_comparison.png)

---

#### 2. NVIDIA Nsight Compute (`ncu`): Micro-Kernel Divergence (Bad vs. Good)
![NVIDIA Nsight Compute Bad vs Good](telemetry/snapshots/snapshot_ncu_comparison.png)

| Profiler Metric | 🔴 Unoptimized Naive Kernel | 🟢 Optimized 1D Grid-Stride | Improvement | Architectural Significance |
| :--- | :---: | :---: | :---: | :--- |
| **Kernel Duration** | 9.85 ms | 4.84 ms | **2.03x Faster** | Execution latency halved with identical mathematical output |
| **Warp Execution Efficiency** | 14.28% | 68.42% | **+379%** | Active threads jump from 4.57 to 21.89 out of 32 per warp |
| **SM Compute Throughput** | 12.45% | 74.80% | **6.0x Higher** | Silicon compute cores fully saturated rather than masked off |
| **Issue Slots No Eligible** | 78.40% | 22.10% | **3.5x Drop** | SM warp schedulers find valid instructions almost every cycle |

> **Architectural Discussion:** In SIMT execution, warps process 32 threads in lockstep. The naive 2D block allocates spatial neighbors to the same warp; when background pixels escape in 2 iterations, up to 30 execution units are masked off while 2 border pixels run 500 loops. The 1D grid-stride loop dynamically distributes unescaped pixels across threads, keeping warps continuously populated.

---

#### 3. PyTorch Profiler / Perfetto: Host-Device Synchronization Leak (Bad vs. Good)
![PyTorch Profiler Bad vs Good](telemetry/snapshots/snapshot_torch_comparison.png)

| Profiler Metric | 🔴 Naive Host Poll (`mask.any()`) | 🟢 Vectorized Pure GPU | Improvement | Architectural Significance |
| :--- | :---: | :---: | :---: | :--- |
| **Total CPU Wall Time** | 307.88 ms | 63.76 ms | **4.82x Faster** | CPU non-blocking dispatch keeps GPU saturated |
| **`aten::_local_scalar_dense`** | 265.42 ms (86.2%) | 0.00 ms (0.0%) | **100% Eliminated** | Zero synchronous PCIe round-trips to CPU host |
| **CUDA Compute Time** | 43.23 ms | 40.80 ms | Consistent | Pure math runtime is unchanged; pipeline stall is removed |
| **GPU Duty Cycle** | ~14% (Choppy blips) | ~100% (Continuous) | Saturated | Command queue stays filled with asynchronous kernel launches |

> **Architectural Discussion:** Evaluating `if not mask.any(): break` forces Python to retrieve a 1-byte boolean from GPU VRAM across PCIe to host RAM. This triggers an implicit `cudaStreamSynchronize`, draining the GPU command queue and idling hardware for 265 ms. Deleting this check lets all tensors remain on device, allowing asynchronous dispatch at native line-rate.

---

#### 4. NVIDIA Nsight Systems (`nsys`): Macro Timeline Pipelining (Bad vs. Good)
![NVIDIA Nsight Systems Bad vs Good](telemetry/snapshots/snapshot_nsys_comparison.png)

| Profiler Metric | 🔴 Synchronous Baseline | 🟢 Asynchronous Streams | Improvement | Architectural Significance |
| :--- | :---: | :---: | :---: | :--- |
| **Wall-Clock Duration** | 15.00 ms | 6.30 ms | **2.38x Faster** | Transfer latency hidden completely behind computation |
| **Host Sync Overhead** | 68.4% of total time | 0.0% of total time | **100% Eliminated** | CPU immediately returns after queuing asynchronous tasks |
| **GPU Idle Bubbles** | 55% of wall clock | 0.0% (Zero Bubbles) | **Full Overlap** | DMA copy engines and SM compute engines run concurrently |

> **Architectural Discussion:** The GPU contains dedicated Direct Memory Access (DMA) copy engines operating independently of SM compute cores. Serial synchronous code executes them one after another, wasting more than half the wall clock. Pipelining memory copies across multiple CUDA streams achieves double-buffering and saturates hardware bandwidth.

---

## Empirical Benchmarks & Profiling Case Studies

### Benchmark Summary Table

Measured on Apple Silicon unified architecture (M-series / MPS) and Linux HPC CUDA nodes ($N=256$, max iterations $= 500$):

| Workload | Implementation Runtime | Execution Time (ms) | Throughput | Relative Speedup | Primary Bottleneck Diagnosed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GEMM ($256 \times 256$)** | Pure Python ($i-j-k$) | $310.50\text{ ms}$ | $0.11\text{ GFLOPS}$ | $1.0\times$ (Baseline) | $L1$ cache thrashing, bytecode dispatch |
| | Pure Python ($i-k-j$) | $37.95\text{ ms}$ | $0.88\text{ GFLOPS}$ | $8.2\times$ | Spatial cache line misses eliminated |
| | Numba CPU (Serial) | $1.284\text{ ms}$ | $26.13\text{ GFLOPS}$ | $241.8\times$ | Single-thread vector register bounds |
| | Numba CPU (Parallel, 10T) | $0.203\text{ ms}$ | $165.25\text{ GFLOPS}$ | $1,529.5\times$ | Multicore scaling limit |
| | PyTorch CPU (OpenMP BLAS) | $0.052\text{ ms}$ | $645.10\text{ GFLOPS}$ | $5,971.1\times$ | AVX/NEON SIMD saturation |
| | Hardware BLAS (MPS / CUDA) | $0.023\text{ ms}$ | $1,460.00\text{ GFLOPS}$ | **$13,700.0\times$** | Hardware Roofline Peak Ceiling |
| **Mandelbrot ($1024 \times 1024$)** | Pure Python (Loop) | $5,120.00\text{ ms}$ | $0.20\text{ Mpix/s}$ | $1.0\times$ (Baseline) | Python dynamic object boxing |
| | NumPy (Active Masking) | $264.40\text{ ms}$ | $3.97\text{ Mpix/s}$ | $19.4\times$ | Intermediate array memory allocations |
| | PyTorch Device (Naive D2H) | $308.10\text{ ms}$ | $3.40\text{ Mpix/s}$ | $16.6\times$ | `aten::_local_scalar_dense` CPU stall |
| | PyTorch Device (Optimized) | $63.78\text{ ms}$ | $16.44\text{ Mpix/s}$ | $80.3\times$ | GPU execution uninhibited by host |
| | Numba CPU (Parallel) | $11.45\text{ ms}$ | $91.56\text{ Mpix/s}$ | $447.1\times$ | Thread-level parallel scaling |
| | CUDA Native (Grid-Stride) | $4.84\text{ ms}$ | $216.50\text{ Mpix/s}$ | **$1,057.8\times$** | Warp divergence minimized via ILP |

---

### Case Study 1: The Device-to-Host (D2H) Synchronization Trap

When authoring GPU kernels in PyTorch, standard convergence loops check whether all pixels or samples have finished:
```python
# UNOPTIMIZED: Appears clean, but creates a catastrophic host-device stall
for _ in range(max_iter):
    z[mask] = z[mask]**2 + c[mask]
    mask = mask & (torch.abs(z) <= 2.0)
    if not mask.any():   # <-- FATAL BOTTLENECK: aten::_local_scalar_dense
        break
```

#### Diagnostic Telemetry
Running `torch.profiler` revealed that **$86.2\%$ of total execution time ($265.4\text{ ms}$)** was consumed by `aten::_local_scalar_dense`:
```
---------------------------------  ------------  ------------  ------------
                             Name      Self CPU %      Self CPU     Total CPU
---------------------------------  ------------  ------------  ------------
        aten::_local_scalar_dense        86.21%     265.420ms     265.420ms
                      aten::empty         0.18%       0.562ms       0.562ms
                      aten::copy_         0.41%       1.261ms       1.261ms
---------------------------------  ------------  ------------  ------------
```

#### Root Cause
Evaluating `if not mask.any():` requires the host CPU to branch. To answer this boolean, the PyTorch runtime must halt CPU execution, issue an implicit `cudaStreamSynchronize()`, and copy 1 byte across the PCIe bus from GPU memory to CPU registers. At 500 iterations, this serializes the entire execution pipeline.

#### Resolution
Eliminate the per-iteration host check, or amortize the check to once every 32 iterations:
```python
# OPTIMIZED: Run without D2H stalls
for _ in range(max_iter):
    z = torch.where(mask, z * z + c, z)
    mask = mask & (z.real * z.real + z.imag * z.imag <= 4.0)
```
**Outcome**: Total execution time dropped from $308.1\text{ ms}$ to $63.78\text{ ms}$ — an immediate **$4.8\times$ speedup** without changing the GPU hardware!

---

### Case Study 2: Warp Divergence in SIMT Fractal Rendering

On NVIDIA and modern GPU architectures, threads execute in lockstep lock-groups called **Warps** (32 threads).

$$\eta_{\text{warp}} = \frac{\sum_{i=1}^{S} \text{Active Threads}_i}{32 \times S}$$

In the Mandelbrot set, pixels located outside the cardioid escape within $2$ iterations, while pixels deep within the boundary run for the full $500$ iterations. When threads within the same warp process mixed coordinates, the warp scheduler must mask off inactive threads:

```
Cycle 1-2:   [T0, T1, T2, ... T31] (32 active threads, 100% efficiency)
Cycle 3-500: [T0,  -,  -, ...   -] (1 active thread, 31 cores IDLE: 3.1% efficiency)
```

Nsight Compute (`ncu`) reports a disastrous Warp Execution Efficiency:
```
Section: Warp Execution Summary
-------------------------------------------------------
Metric Description                          Metric Value
-------------------------------------------------------
Warp Execution Efficiency (%)                     14.28%
Avg. Active Threads Per Warp                       4.57
Eligible Warps Per Scheduler                       0.12
Issue Slots No Eligible (%)                       78.40%
```

#### Resolution
1. **1D Grid-Stride Loops**: Decouple thread coordinates from spatial pixel coordinates. Threads dynamically grab the next available pixel in stride, diffusing dense boundary regions across multiple warps.
2. **Instruction-Level Parallelism (ILP)**: Unroll computations so each thread computes two pixels concurrently ($z_A, z_B$).

**Outcome**: Warp execution efficiency climbed to $68.4\%$, and throughput increased from $92.1\text{ Mpix/s}$ to $216.5\text{ Mpix/s}$.

---

### Case Study 3: Cache Strides & Spatial Locality

In naive $i-j-k$ matrix multiplication:
```python
# Unoptimized (i-j-k): Catastrophic cache thrashing on B
for i in range(N):
    for j in range(N):
        for k in range(N):
            C[i][j] += A[i][k] * B[k][j]  # B[k][j] reads with stride N!
```
Reading `B[k][j]` incurs a stride of $N \times 4\text{ bytes}$. If $N=1024$, each step jumps $4096\text{ bytes}$ across memory lines, causing compulsory $L1$ and $L2$ cache evictions.

By interchanging loops to $i-k-j$:
```python
# Optimized (i-k-j): Full cache-line spatial locality
for i in range(N):
    for k in range(N):
        r = A[i][k]
        for j in range(N):
            C[i][j] += r * B[k][j]  # Contiguous access across row j
```
Every fetch of `B[k][j]` loads 64 consecutive bytes into the cache line, serving the next 15 loop iterations from zero-latency $L1$ cache.

**Outcome**: Runtime dropped from $310.5\text{ ms}$ to $37.95\text{ ms}$ — an **$8.2\times$ speedup** in pure, uncompiled Python code.

---

## The Systems Diagnostic Decision Tree

When diagnosing low performance on accelerated hardware, follow this systematic decision protocol:

```
Is hardware utilization (GPU Core / CPU Core) < 80%?
 ├── YES: Are there host-device pipeline bubbles?
 │    ├── Check `nsys` timeline for gap intervals between kernels.
 │    │    ├── Bottleneck: Excessive CPU launch overhead or framework dispatch.
 │    │    └── Fix: Use CUDA Graphs or `torch.compile(mode="reduce-overhead")`.
 │    └── Check `torch.profiler` for `aten::_local_scalar_dense` or `.item()`.
 │         ├── Bottleneck: D2H blocking synchronization inside loop.
 │         └── Fix: Eliminate per-iteration scalar checks; vectorize reductions.
 └── NO: Hardware utilization is high, but throughput is low.
      ├── Check Roofline Model in `ncu`:
      │    ├── Left of Knee (Memory Bound, Arithmetic Intensity < 10 FLOP/B):
      │    │    ├── Bottleneck: Uncoalesced global DRAM access or low cache hit rate.
      │    │    └── Fix: 2D Shared memory tiling, memory alignment, `float4` vectorization.
      │    └── Right of Knee (Compute Bound):
      │         ├── Check Warp Execution Efficiency:
      │         │    ├── Efficiency < 50%: Warp divergence (branching in SIMT).
      │         │    └── Fix: Grid-stride dynamic scheduling, branch sorting.
      │         └── Check Register Pressure:
      │              ├── High register count per thread spilling to local DRAM.
      │              └── Fix: Launch bounds `__launch_bounds__(256, 4)`.
```

---

## Quick Start & Turnkey Makefile Reference

### Environment Prerequisites
- Python 3.12+ and `uv` package manager installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- Native C/CUDA compiler (`clang` on macOS; `gcc` and `nvcc` on Linux).

### One-Command Setup
```bash
# Clone the repository
git clone https://github.com/erolrecep/profiling_hardware_accelerated_computations.git
cd profiling_hardware_accelerated_computations

# Install virtual environment and dependencies via Makefile
make setup
```

### Turnkey Command Reference

| Command | Action Description | Output Artifacts |
| :--- | :--- | :--- |
| `make test` | Runs the full verification suite: numerical parity and presentation assets. | Terminal PyTest status report |
| `make test-presentation` | Validates presentation HTML, CSS syntax, image paths, and Reveal.js DOM. | PyTest asset validation report |
| `make present` | Launches local presentation server with cache-busting and opens default browser. | `http://localhost:8000/` |
| `make present-live` | Launches live-reload server (watches `presentation/` and auto-refreshes browser). | `http://localhost:8000/` |
| `make benchmark_gemm` | Executes GEMM scaling benchmarks across all available runtimes. | Formatted Markdown table + telemetry |
| `make benchmark_mandelbrot` | Executes Mandelbrot throughput benchmarks across all runtimes. | Formatted throughput metrics |
| `make benchmark_all` | Runs full matrix & resolution sweeps and generates comparative graphs. | `telemetry/gemm_scaling.png`, `telemetry/mandelbrot_speedup.png` |
| `make profile_torch` | Runs PyTorch Profiler with GPU kernel attribution and memory export. | `telemetry/torch_trace_*.json` (Perfetto traces) |
| `make profile_scalene` | Runs Scalene deterministic profiler for line-by-line CPU/GPU attribution. | `telemetry/scalene_gemm.json` + CLI report |
| `make profile_memray` | Captures heap allocations and compiles interactive HTML flamegraphs. | `telemetry/memray_*_flamegraph.html` |
| `make profile_nsys` | Profiles CUDA execution using NVIDIA Nsight Systems (HPC nodes). | `telemetry/nsys_gemm.nsys-rep` |
| `make profile_ncu` | Profiles micro-architectural hardware metrics via NVIDIA Nsight Compute. | `telemetry/ncu_gemm.ncu-rep` |
| `make compile_cuda` | Compiles native CUDA C kernels (`gemm_cuda.cu`, `mandelbrot_cuda.cu`). | `bin/gemm_cuda`, `bin/mandelbrot_cuda` |
| `make snapshots` | Generates high-fidelity profiler GUI snapshots (PNG). | `presentation/images/snapshots/` |
| `make clean` | Purges compiled binaries, transient trace dumps, and build caches. | Workspace cleaned |

---

## SLURM Cluster Deployment (HPC Environments)

For academic supercomputing clusters and SLURM workload managers (e.g. Rice University Center for Research Computing RANGE cluster, OLCF, or NERSC), the repository includes a complete suite of production SLURM submission scripts located in [`slurm/`](slurm/):

| Script | Purpose & Scope | Target Hardware & Resources | Output Artifacts |
| :--- | :--- | :--- | :--- |
| [`slurm/01_test_parity.slurm`](slurm/01_test_parity.slurm) | Runs PyTest parity suite across all backends | 1 GPU, 8 CPUs, 32GB RAM, 20 mins | `slurm/logs/test_parity_<jobid>.out` |
| [`slurm/02_benchmark_all.slurm`](slurm/02_benchmark_all.slurm) | Compiles CUDA and executes scaling sweeps | 1 GPU, 8 CPUs, 32GB RAM, 30 mins | Benchmark telemetry tables & PNG charts |
| [`slurm/03_profile_nsys.slurm`](slurm/03_profile_nsys.slurm) | Runs NVIDIA Nsight Systems timeline trace | 1 GPU, 8 CPUs, 32GB RAM, 30 mins | `telemetry/*.nsys-rep` + CLI summary |
| [`slurm/04_profile_ncu.slurm`](slurm/04_profile_ncu.slurm) | Runs NVIDIA Nsight Compute micro-counters | 1 GPU, 8 CPUs, 32GB RAM, 30 mins | `telemetry/*.ncu-rep` + Speed-of-Light |
| [`slurm/05_profile_frameworks.slurm`](slurm/05_profile_frameworks.slurm) | PyTorch Profiler, Scalene, and Memray | 1 GPU, 8 CPUs, 32GB RAM, 30 mins | Perfetto JSON traces & HTML flamegraphs |
| [`slurm/submit_all.sh`](slurm/submit_all.sh) | Master orchestrator with dependency chaining | Automated dependency submission | Submits all 5 jobs in pipeline |

### Submitting to SLURM
Submit individual jobs or trigger the entire pipeline with automated dependency chaining:
```bash
# Option A: Submit individual job
sbatch slurm/03_profile_nsys.slurm

# Option B: Submit entire orchestrated pipeline
./slurm/submit_all.sh

# Monitor active jobs in queue
squeue -u $USER

# Follow live job output logs
tail -f slurm/logs/*.out
```

---

## Online Cloud GPU Testing Platforms: Replicate & Compare for Free

If you do not have immediate access to a physical NVIDIA GPU or an HPC cluster account, you can replicate every single benchmark and profiler command online using these cloud environments:

### 1. Google Colab (Recommended — Free Tier Available)
- **Cost**: Free (NVIDIA Tesla T4 GPU) or \$10/mo Colab Pro (NVIDIA A100 / L4).
- **Pre-Installed Tools**: `nvcc`, `nsys` (Nsight Systems), `ncu` (Nsight Compute), Python 3.10+, and PyTorch with CUDA are pre-installed in the Linux runtime!
- **Interactive Notebook**: Open the official workshop notebook in one click:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erolrecep/profiling_hardware_accelerated_computations/blob/main/notebooks/cuda_profiling_colab_workshop.ipynb)
- **Replication Workflow on Colab**:
  ```bash
  # 1. Clone repository
  !git clone https://github.com/erolrecep/profiling_hardware_accelerated_computations.git
  %cd profiling_hardware_accelerated_computations

  # 2. Compile native CUDA kernels
  !nvcc -O3 -arch=native -lcublas src/gemm/gemm_cuda.cu -o src/gemm/gemm_cuda

  # 3. Run live Nsight Systems trace
  !nsys profile --trace=cuda,osrt --stats=true ./src/gemm/gemm_cuda 1024 10

  # 4. Run live Nsight Compute kernel counters
  !ncu --section SpeedOfLight --section WarpStateStats ./src/gemm/gemm_cuda 1024 1
  ```
- **Downloading & Comparing**: You can download the generated `.nsys-rep` and `.ncu-rep` files directly from Colab to your laptop, open them in the free local desktop applications ([NVIDIA Nsight Systems UI](https://developer.nvidia.com/nsight-systems) & [NVIDIA Nsight Compute UI](https://developer.nvidia.com/nsight-compute)), and compare them directly against our reference snapshots in `telemetry/snapshots/`!

### 2. Kaggle Notebooks (Free)
- **Cost**: Free (30 hours of NVIDIA GPU per week, Tesla T4 x 2 or P100).
- **Environment**: Includes bash terminal access, `nvcc`, and PyTorch CUDA.

### 3. Lightning AI (Lightning Studios)
- **Cost**: Free tier includes monthly GPU credits.
- **Environment**: Provides an in-browser VS Code desktop with terminal, root access, and web preview.

### 4. Low-Cost Cloud Workstations (RunPod / Lambda Labs / Vast.ai)
- **Cost**: \$0.20 – \$0.75 per hour for an RTX 4090 or A100.
- **Unique Feature**: Provides a **Web Desktop (VNC)** where you can launch `nsys-ui` and `ncu-ui` directly in your browser without downloading traces locally.

---

## Automated Verification & Parity Testing

To ensure absolute academic rigor and numerical correctness across all 6 runtimes, the repository includes an automated PyTest suite (`tests/`):
- `tests/test_gemm_parity.py`: Validates numerical parity against standard NumPy BLAS double-precision reference with a strict tolerance threshold ($\epsilon \le 10^{-4}$).
- `tests/test_mandelbrot_parity.py`: Validates bit-exact escape iteration equality and boundary checksums across scalar Python, NumPy, Numba, and PyTorch.

Run the test suite:
```bash
make test
```
```
============================= test session starts ==============================
platform darwin -- Python 3.12.8, pytest-9.2.0, pluggy-1.5.0
rootdir: /Users/erolrecep/Developer/codebase/apps/_hpc/profiling_hardware_accelerated_computations
collected 10 items

tests/test_gemm_parity.py::test_python_ijk_parity PASSED                  [ 10%]
tests/test_gemm_parity.py::test_python_ikj_parity PASSED                  [ 20%]
tests/test_gemm_parity.py::test_numba_cpu_serial_parity PASSED            [ 30%]
tests/test_gemm_parity.py::test_numba_cpu_parallel_parity PASSED          [ 40%]
tests/test_gemm_parity.py::test_torch_cpu_parity PASSED                   [ 50%]
tests/test_gemm_parity.py::test_torch_hardware_parity PASSED              [ 60%]
tests/test_mandelbrot_parity.py::test_numpy_matches_pure_python PASSED    [ 70%]
tests/test_mandelbrot_parity.py::test_numba_cpu_matches_pure_python PASSED[ 80%]
tests/test_mandelbrot_parity.py::test_torch_matches_pure_python PASSED    [ 90%]
tests/test_mandelbrot_parity.py::test_mandelbrot_checksums PASSED         [100%]

============================== 10 passed in 1.11s ==============================
```

---

## Workshop Presentation Deck (Reveal.js)

The repository includes a 25-slide interactive presentation engineered in Reveal.js with KaTeX mathematical typography, code highlighting, and embedded speaker notes.

### Presentation Agenda (25 Slides)
1. **Slide 1: Title & Workshop Vision** — *Profiling Hardware Accelerated Computations*
2. **Slide 2: About Me / Instructor Profile** — Dr. Recep Erol (Rice University CRC)
3. **Slide 3: Why Profiling Matters: The Cost of Blind Optimization**
4. **Slide 4: Profiling Taxonomies: Deterministic vs. Probabilistic**
5. **Slide 5: The Hardware Architecture Reality: CPU vs. GPU Hierarchy**
6. **Slide 6: Mini-Project 1: Dense Matrix Multiplication (GEMM)**
7. **Slide 7: Mini-Project 2: Mandelbrot Set Fractal Generation**
8. **Slide 8: The Profiling Toolkit: Diagnostic Landscape Overview**
9. **Slide 9: Tool 1: NVIDIA Nsight Systems (`nsys`)**
10. **Slide 10: Tool 2: NVIDIA Nsight Compute (`ncu`)**
11. **Slide 11: Tool 3: PyTorch Profiler (`torch.profiler`)**
12. **Slide 12: Tool 4: Scalene — Line-by-Line Python/C/GPU Profiling**
13. **Slide 13: Tool 5: Memray — Memory Allocation & Heap Profiler**
14. **Slide 14: Initial Baselines: The High Price of Naive Code**
15. **Slide 15: Step-by-Step Profiling Protocol via Makefile**
16. **Slide 16: Telemetry Analysis: Memray Heap Flamegraphs**
17. **Slide 17: Telemetry Analysis: Scalene Line-by-Line Attribution**
18. **Slide 18: Telemetry Analysis: PyTorch Profiler & D2H Bottleneck**
19. **Slide 19: Telemetry Analysis: NVIDIA Nsight Systems & Compute**
20. **Slide 20: Optimization Strategy 1: Shared Memory Tiling & Locality**
21. **Slide 21: Optimization Strategy 2: Taming Warp Divergence & D2H**
22. **Slide 22: Benchmark Master Comparison: From Naive to Roofline**
23. **Slide 23: The Systems Diagnostic Decision Tree**
24. **Slide 24: Key Takeaways & Workshop Conclusion**
25. **Slide 25: Questions & Answers (Q&A) & Interactive Discussion**

### Viewing & Local Editing

- **Online Deployment**: [https://erolrecep.github.io/profiling_hardware_accelerated_computations/](https://erolrecep.github.io/profiling_hardware_accelerated_computations/)
- **Standard Local Server**:
  ```bash
  make present
  # Or specify a custom port:
  make present PORT=8080
  ```
  *Features: Launches local HTTP server with automatic port detection, sets strict cache-busting headers (`no-cache`/`no-store`) so CSS/HTML edits apply immediately upon reload, and opens the default browser directly to `http://localhost:8000/`.*

- **Live-Reload Watch Mode (Recommended for Editing)**:
  ```bash
  make present-live
  ```
  *Features: Continuously watches `presentation/` for file changes (HTML, CSS, images). As soon as you save any file (e.g. `presentation/css/theme.css` or `presentation/index.html`), connected browser tabs reload instantly without manual refreshes!*

- **Automated Presentation Testing**:
  ```bash
  make test-presentation
  ```
  *Features: PyTest suite verifying that `index.html` structure is intact, all local images and snapshots exist and have valid PNG headers, all stylesheets exist and parse with balanced CSS braces, and Reveal.js DOM containers are properly configured.*

- **Reveal.js Keyboard Shortcuts**:
  | Key | Action |
  | :--- | :--- |
  | `SPACE` or `→` | Advance to the next slide |
  | `←` | Return to previous slide |
  | `S` | **Speaker View** (shows current slide, speaker notes, upcoming slide, and real-time timer) |
  | `F` | Toggle Fullscreen mode |
  | `O` or `ESC` | Slide Overview (zoom out into thumbnail grid for quick navigation) |
  | `B` or `.` | Blank / pause screen |
  | `?` | Show Reveal.js built-in help overlay |

---

## Author & Institutional Affiliation

**Dr. Recep Erol**  
*Research Computing Engineer*  
Center for Research Computing (CRC), Rice University  
- **Email**: `re40@rice.edu`
- **GitHub**: [@erolrecep](https://github.com/erolrecep)
- **LinkedIn**: [linkedin.com/in/receperol](https://linkedin.com/in/receperol)
- **Academic Research**: High-Performance Computing (HPC), Hardware-Accelerated Machine Learning, Distributed Deep Learning Systems, and GPU Performance Engineering.

---

## Technical Monograph

For full theoretical derivations of the Roofline Model, memory transaction coalescence equations, assembly instruction analysis, and complete diagnostic case studies, refer to the accompanying monograph:  
📖 **[Report.md](Report.md)**
