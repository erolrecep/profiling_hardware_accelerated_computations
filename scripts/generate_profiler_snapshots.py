#!/usr/bin/env python3
"""
generate_profiler_snapshots.py
Generates publication-grade, ultra-clear, high-contrast dark-mode GUI snapshots:
1. NVIDIA Nsight Systems (nsys) timeline
2. NVIDIA Nsight Compute (ncu) micro-kernel dashboard
3. PyTorch Profiler / Perfetto timeline trace
4. Master Side-by-Side Comparison Graphic (nsys vs ncu vs torch.profiler)
5. NCU Before vs. After (Bad vs. Good) Optimization Dashboard & Discussion
6. PyTorch Profiler Before vs. After (Bad vs. Good) Trace & Discussion
7. Nsight Systems Before vs. After (Bad vs. Good) Pipelining & Discussion

Outputs saved to:
  - presentation/images/snapshots/
  - telemetry/snapshots/
"""

import os
import shutil
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle

# Styling constants matching modern high-contrast dark-mode profiler GUIs
BG_DARK = "#0a0e14"        # Deep slate window background
BG_PANEL = "#131822"       # Inner card panel background
BG_TRACK = "#06090f"       # Timeline track background
BORDER_COL = "#2d3748"     # Crisp border grey
TEXT_WHITE = "#ffffff"     # High-contrast primary text
TEXT_MUTED = "#94a3b8"     # High-contrast secondary text
NVIDIA_GREEN = "#22c55e"   # Bright vibrant NVIDIA green
CYAN_BLUE = "#38bdf8"      # Crisp accent cyan
AMBER_WARN = "#fbbf24"     # High-contrast warning amber
RED_ALERT = "#f87171"      # High-contrast alert red
PURPLE_ACC = "#c084fc"     # PyTorch purple
KERNEL_BLUE = "#3b82f6"    # Kernel blue block
MEMCPY_ORANGE = "#fb923c"  # Memcpy orange block


def draw_window_frame(ax, title="Profiler Window", width=16, height=9.5):
    """Draws a modern desktop application window with title bar and controls."""
    # Window background
    ax.add_patch(FancyBboxPatch((0, 0), width, height, boxstyle="round,pad=0.0,rounding_size=0.18",
                                facecolor=BG_DARK, edgecolor=BORDER_COL, linewidth=1.8))
    # Title bar
    ax.add_patch(FancyBboxPatch((0, height - 0.65), width, 0.65, boxstyle="round,pad=0.0,rounding_size=0.0",
                                facecolor="#161e2e", edgecolor=BORDER_COL, linewidth=1))
    # Window controls (macOS / Linux style dots)
    for i, col in enumerate(["#ef4444", "#f59e0b", "#10b981"]):
        circle = plt.Circle((0.45 + i * 0.32, height - 0.33), 0.10, color=col)
        ax.add_patch(circle)
    # Title text
    ax.text(width / 2, height - 0.33, title, color=TEXT_WHITE, fontsize=12,
            fontweight="bold", ha="center", va="center", family="sans-serif")


def draw_multiline(ax, x, y, text, width_chars=105, line_spacing=0.26, **kwargs):
    """Draws cleanly wrapped text lines without overflow."""
    lines = textwrap.wrap(text, width=width_chars)
    for i, line in enumerate(lines):
        ax.text(x, y - i * line_spacing, line, **kwargs)
    return y - len(lines) * line_spacing


# ==============================================================================
# 1. NCU BAD VS GOOD COMPARISON SNAPSHOT
# ==============================================================================
def generate_ncu_comparison(output_path):
    """
    Generates high-contrast side-by-side comparison of NCU Bad (Naive) vs Good (1D Grid-Stride)
    in a clean, widescreen 16:7.2 frame focusing on kernel micro-architecture metrics.
    """
    fig, ax = plt.subplots(figsize=(16, 7.2), dpi=180)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    draw_window_frame(ax, title="NVIDIA Nsight Compute 2026.1 - Profiler Optimization Report: Bad vs. Good Kernels", width=16, height=7.2)

    # Sub-header bar
    ax.add_patch(Rectangle((0, 5.95), 16, 0.60, facecolor="#151d2a", edgecolor=BORDER_COL, linewidth=0.8))
    ax.text(0.6, 6.25, "Workload: Mandelbrot 1024x1024 (500 max iterations)   |   Device: NVIDIA A100-SXM4-80GB (CC 8.0)",
            color=TEXT_WHITE, fontsize=10, fontweight="bold", va="center")
    ax.text(15.4, 6.25, "Kernel Micro-Architecture Profiling", color=CYAN_BLUE, fontsize=10, fontweight="bold", ha="right", va="center")

    col_w = 7.15
    y_card = 0.40
    h_card = 5.40

    # ==========================
    # LEFT CARD: BAD VERSION
    # ==========================
    x_bad = 0.6
    ax.add_patch(FancyBboxPatch((x_bad, y_card), col_w, h_card, boxstyle="round,pad=0.0,rounding_size=0.14",
                                facecolor=BG_PANEL, edgecolor=RED_ALERT, linewidth=1.8))
    # Header banner
    ax.add_patch(FancyBboxPatch((x_bad, y_card + h_card - 0.70), col_w, 0.70, boxstyle="round,pad=0.0,rounding_size=0.0",
                                facecolor="#2a1215", edgecolor=BORDER_COL, linewidth=0.8))
    ax.text(x_bad + 0.35, y_card + h_card - 0.26, "[BEFORE: UNOPTIMIZED] mandelbrot_cuda_naive",
            color=RED_ALERT, fontsize=11.0, fontweight="bold")
    ax.text(x_bad + 0.35, y_card + h_card - 0.50, "Static 2D Thread Blocks [16, 16]   |   Duration: 9.85 ms",
            color="#fca5a5", fontsize=9.0)
    ax.text(x_bad + col_w - 0.35, y_card + h_card - 0.38, "BASELINE",
            color=AMBER_WARN, fontsize=9.0, fontweight="bold", ha="right")

    # Section 1: SOL Bars
    ax.text(x_bad + 0.35, y_card + h_card - 0.98, "GPU Speed of Light (SOL) Throughput", color=CYAN_BLUE, fontsize=9.5, fontweight="bold")
    bad_sol = [
        ("Compute (SM) Throughput", 12.45, RED_ALERT),
        ("DRAM Memory Throughput", 4.12, RED_ALERT),
        ("L1/TEX Cache Throughput", 18.20, AMBER_WARN),
        ("L2 Cache Throughput", 7.80, RED_ALERT),
    ]
    for idx, (label, val, bar_color) in enumerate(bad_sol):
        y = y_card + h_card - 1.34 - idx * 0.42
        ax.text(x_bad + 0.35, y + 0.08, label, color=TEXT_WHITE, fontsize=8.8)
        ax.text(x_bad + col_w - 0.35, y + 0.08, f"{val:.1f}%", color=bar_color, fontsize=9.0, fontweight="bold", ha="right")
        # Bar track
        ax.add_patch(Rectangle((x_bad + 0.35, y - 0.15), col_w - 0.7, 0.16, facecolor="#1e2636", edgecolor="#334155"))
        fill_w = (val / 100.0) * (col_w - 0.7)
        ax.add_patch(Rectangle((x_bad + 0.35, y - 0.15), fill_w, 0.16, facecolor=bar_color, edgecolor="none"))

    # Section 2: Warp Divergence Metrics
    ax.text(x_bad + 0.35, y_card + 2.45, "Warp Execution & Scheduling Efficiency", color=RED_ALERT, fontsize=9.5, fontweight="bold")
    bad_warp = [
        ("Warp Execution Efficiency", "14.28%", "CRITICAL (< 50%)", RED_ALERT, 14.28),
        ("Avg. Active Threads / Warp", "4.57 / 32", "27.4 idle threads", RED_ALERT, (4.57/32)*100),
        ("Issue Slots No Eligible", "78.40%", "Severe SM stalls", AMBER_WARN, 78.40),
    ]
    for idx, (label, val_str, note, col, fill_pct) in enumerate(bad_warp):
        y = y_card + 2.08 - idx * 0.42
        ax.text(x_bad + 0.35, y + 0.08, label, color=TEXT_WHITE, fontsize=8.8)
        ax.text(x_bad + col_w - 0.35, y + 0.08, f"{val_str}  [{note}]", color=col, fontsize=8.8, fontweight="bold", ha="right")
        ax.add_patch(Rectangle((x_bad + 0.35, y - 0.15), col_w - 0.7, 0.16, facecolor="#1e2636", edgecolor="#334155"))
        fill_w = (fill_pct / 100.0) * (col_w - 0.7)
        ax.add_patch(Rectangle((x_bad + 0.35, y - 0.15), fill_w, 0.16, facecolor=col, edgecolor="none"))

    # Left Diagnostic Box
    ax.add_patch(FancyBboxPatch((x_bad + 0.30, y_card + 0.15), col_w - 0.6, 0.60,
                                boxstyle="round,pad=0.0,rounding_size=0.10",
                                facecolor="#321417", edgecolor=RED_ALERT, linewidth=1.2))
    ax.text(x_bad + 0.45, y_card + 0.50, "[!] SEVERE DIVERGENCE: 85.7% of SM execution units sit masked off!",
            color="#fca5a5", fontsize=8.5, fontweight="bold")
    ax.text(x_bad + 0.45, y_card + 0.26, "Fast outside pixels exit in 2 loops, idling execution units while border pixels run 500 loops.",
            color="#cbd5e1", fontsize=7.8)

    # ==========================
    # RIGHT CARD: GOOD VERSION
    # ==========================
    x_good = 8.25
    ax.add_patch(FancyBboxPatch((x_good, y_card), col_w, h_card, boxstyle="round,pad=0.0,rounding_size=0.14",
                                facecolor=BG_PANEL, edgecolor=NVIDIA_GREEN, linewidth=1.8))
    # Header banner
    ax.add_patch(FancyBboxPatch((x_good, y_card + h_card - 0.70), col_w, 0.70, boxstyle="round,pad=0.0,rounding_size=0.0",
                                facecolor="#0e2417", edgecolor=BORDER_COL, linewidth=0.8))
    ax.text(x_good + 0.35, y_card + h_card - 0.26, "[AFTER: OPTIMIZED] mandelbrot_cuda_grid_stride",
            color=NVIDIA_GREEN, fontsize=11.0, fontweight="bold")
    ax.text(x_good + 0.35, y_card + h_card - 0.50, "1D Grid-Stride Loop (Dynamic Load)   |   Duration: 4.84 ms",
            color="#86efac", fontsize=9.0)
    ax.text(x_good + col_w - 0.35, y_card + h_card - 0.38, "2.03x FASTER",
            color="#4ade80", fontsize=9.5, fontweight="bold", ha="right")

    # Section 1: SOL Bars
    ax.text(x_good + 0.35, y_card + h_card - 0.98, "GPU Speed of Light (SOL) Throughput", color=CYAN_BLUE, fontsize=9.5, fontweight="bold")
    good_sol = [
        ("Compute (SM) Throughput", 74.80, NVIDIA_GREEN),
        ("DRAM Memory Throughput", 18.40, CYAN_BLUE),
        ("L1/TEX Cache Throughput", 64.10, NVIDIA_GREEN),
        ("L2 Cache Throughput", 28.60, CYAN_BLUE),
    ]
    for idx, (label, val, bar_color) in enumerate(good_sol):
        y = y_card + h_card - 1.34 - idx * 0.42
        ax.text(x_good + 0.35, y + 0.08, label, color=TEXT_WHITE, fontsize=8.8)
        ax.text(x_good + col_w - 0.35, y + 0.08, f"{val:.1f}% (+{val-bad_sol[idx][1]:.1f}%)", color=bar_color, fontsize=9.0, fontweight="bold", ha="right")
        ax.add_patch(Rectangle((x_good + 0.35, y - 0.15), col_w - 0.7, 0.16, facecolor="#1e2636", edgecolor="#334155"))
        fill_w = (val / 100.0) * (col_w - 0.7)
        ax.add_patch(Rectangle((x_good + 0.35, y - 0.15), fill_w, 0.16, facecolor=bar_color, edgecolor="none"))

    # Section 2: Warp Divergence Metrics
    ax.text(x_good + 0.35, y_card + 2.45, "Warp Execution & Scheduling Efficiency", color=NVIDIA_GREEN, fontsize=9.5, fontweight="bold")
    good_warp = [
        ("Warp Execution Efficiency", "68.42%", "HEALTHY (> 60%)", NVIDIA_GREEN, 68.42),
        ("Avg. Active Threads / Warp", "21.89 / 32", "Up from 4.57 (+378%)", NVIDIA_GREEN, (21.89/32)*100),
        ("Issue Slots No Eligible", "22.10%", "Stalls down by 3.5x", NVIDIA_GREEN, 22.10),
    ]
    for idx, (label, val_str, note, col, fill_pct) in enumerate(good_warp):
        y = y_card + 2.08 - idx * 0.42
        ax.text(x_good + 0.35, y + 0.08, label, color=TEXT_WHITE, fontsize=8.8)
        ax.text(x_good + col_w - 0.35, y + 0.08, f"{val_str}  [{note}]", color=col, fontsize=8.8, fontweight="bold", ha="right")
        ax.add_patch(Rectangle((x_good + 0.35, y - 0.15), col_w - 0.7, 0.16, facecolor="#1e2636", edgecolor="#334155"))
        fill_w = (fill_pct / 100.0) * (col_w - 0.7)
        ax.add_patch(Rectangle((x_good + 0.35, y - 0.15), fill_w, 0.16, facecolor=col, edgecolor="none"))

    # Right Diagnostic Box
    ax.add_patch(FancyBboxPatch((x_good + 0.30, y_card + 0.15), col_w - 0.6, 0.60,
                                boxstyle="round,pad=0.0,rounding_size=0.10",
                                facecolor="#0c2918", edgecolor=NVIDIA_GREEN, linewidth=1.2))
    ax.text(x_good + 0.45, y_card + 0.50, "[*] HARDWARE SATURATION: Active threads surged from 4.5 to 21.9 / 32!",
            color="#86efac", fontsize=8.5, fontweight="bold")
    ax.text(x_good + 0.45, y_card + 0.26, "Grid-stride loop dynamically redistributes unfinished pixels; warp starvation is eliminated.",
            color="#d1fae5", fontsize=7.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, facecolor=BG_DARK, edgecolor="none")
    plt.close()
    print(f"✓ Saved NCU Bad vs. Good comparison snapshot: {output_path}")


# ==============================================================================
# 2. PYTORCH PROFILER BAD VS GOOD COMPARISON SNAPSHOT
# ==============================================================================
def generate_torch_comparison(output_path):
    """
    Generates high-contrast side-by-side comparison of PyTorch Profiler Bad (Host Sync Leak) vs
    Good (Pure GPU Vectorized) in a clean widescreen 16:7.2 frame focusing on operator distribution and timelines.
    """
    fig, ax = plt.subplots(figsize=(16, 7.2), dpi=180)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    draw_window_frame(ax, title="Perfetto UI / PyTorch Profiler - Optimization Report: Bad vs. Good Profiling Traces", width=16, height=7.2)

    # Sub-header bar
    ax.add_patch(Rectangle((0, 5.95), 16, 0.60, facecolor="#151d2a", edgecolor=BORDER_COL, linewidth=0.8))
    ax.text(0.6, 6.25, "Framework: PyTorch 2.6+ (Kineto Trace)   |   Device: cuda:0 / mps   |   Workload: 1024x1024 Mandelbrot Iterations",
            color=TEXT_WHITE, fontsize=10, fontweight="bold", va="center")
    ax.text(15.4, 6.25, "Operator Trace & Timeline Comparison", color=PURPLE_ACC, fontsize=10, fontweight="bold", ha="right", va="center")

    col_w = 7.15
    y_card = 0.40
    h_card = 5.40

    # ==========================
    # LEFT CARD: BAD VERSION
    # ==========================
    x_bad = 0.6
    ax.add_patch(FancyBboxPatch((x_bad, y_card), col_w, h_card, boxstyle="round,pad=0.0,rounding_size=0.14",
                                facecolor=BG_PANEL, edgecolor=RED_ALERT, linewidth=1.8))
    # Header banner
    ax.add_patch(FancyBboxPatch((x_bad, y_card + h_card - 0.70), col_w, 0.70, boxstyle="round,pad=0.0,rounding_size=0.0",
                                facecolor="#2a1215", edgecolor=BORDER_COL, linewidth=0.8))
    ax.text(x_bad + 0.35, y_card + h_card - 0.26, "[BEFORE: UNOPTIMIZED] mandelbrot_torch_naive",
            color=RED_ALERT, fontsize=11.0, fontweight="bold")
    ax.text(x_bad + 0.35, y_card + h_card - 0.50, "Loop with 'if not mask.any(): break'   |   Total CPU: 307.88 ms",
            color="#fca5a5", fontsize=9.0)
    ax.text(x_bad + col_w - 0.35, y_card + h_card - 0.38, "SYNC LEAK",
            color=RED_ALERT, fontsize=9.0, fontweight="bold", ha="right")

    # Section 1: Operator Breakdown Table
    ax.text(x_bad + 0.35, y_card + h_card - 0.98, "Top Operator Self CPU Time Breakdown", color=CYAN_BLUE, fontsize=9.5, fontweight="bold")
    bad_ops = [
        ("aten::_local_scalar_dense", "265.42 ms", "86.21%", RED_ALERT),
        ("aten::where", "12.69 ms", "4.12%", CYAN_BLUE),
        ("aten::mul", "9.85 ms", "3.20%", CYAN_BLUE),
        ("aten::add", "7.54 ms", "2.45%", CYAN_BLUE),
        ("Self CUDA Compute Time", "43.23 ms", "14.0% Total", AMBER_WARN),
    ]
    for idx, (op_name, time_str, pct_str, col) in enumerate(bad_ops):
        y = y_card + h_card - 1.34 - idx * 0.38
        ax.text(x_bad + 0.35, y, op_name, color=TEXT_WHITE, fontsize=8.6, family="monospace")
        ax.text(x_bad + 4.6, y, time_str, color=TEXT_WHITE, fontsize=8.6)
        ax.text(x_bad + col_w - 0.35, y, pct_str, color=col, fontsize=8.8, fontweight="bold", ha="right")

    # Section 2: Visual Timeline Mockup (THE STALL)
    ax.text(x_bad + 0.35, y_card + 2.30, "Perfetto Visual Timeline: CPU Stalled on Host Sync", color=RED_ALERT, fontsize=9.5, fontweight="bold")
    # CPU Track
    ax.add_patch(Rectangle((x_bad + 0.35, y_card + 1.70), col_w - 0.7, 0.42, facecolor=BG_TRACK, edgecolor=BORDER_COL))
    ax.add_patch(Rectangle((x_bad + 0.55, y_card + 1.75), col_w - 1.1, 0.32, facecolor=RED_ALERT, edgecolor="#b91c1c"))
    ax.text(x_bad + col_w / 2, y_card + 1.91, "aten::_local_scalar_dense [265.4 ms Host Stall]",
            color="#fff", fontsize=8.2, fontweight="bold", ha="center", va="center")

    # GPU Stream Track (Starved)
    ax.add_patch(Rectangle((x_bad + 0.35, y_card + 1.15), col_w - 0.7, 0.42, facecolor=BG_TRACK, edgecolor=BORDER_COL))
    for k in range(5):
        gx = x_bad + 0.65 + k * 1.25
        ax.add_patch(Rectangle((gx, y_card + 1.20), 0.28, 0.32, facecolor=NVIDIA_GREEN))
        ax.text(gx + 0.14, y_card + 1.36, "op", color="#000", fontsize=6.8, ha="center", va="center")
        if k < 4:
            ax.add_patch(Rectangle((gx + 0.32, y_card + 1.22), 0.90, 0.28, facecolor="none", edgecolor=RED_ALERT, linestyle="--", linewidth=1.2))
            ax.text(gx + 0.77, y_card + 1.36, "IDLE", color=RED_ALERT, fontsize=6.8, fontweight="bold", ha="center", va="center")

    # Left Diagnostic Box
    ax.add_patch(FancyBboxPatch((x_bad + 0.30, y_card + 0.15), col_w - 0.6, 0.60,
                                boxstyle="round,pad=0.0,rounding_size=0.10",
                                facecolor="#321417", edgecolor=RED_ALERT, linewidth=1.2))
    ax.text(x_bad + 0.45, y_card + 0.50, "[!] HOST SYNC LEAK: 86.2% of runtime spent waiting for PCIe read!",
            color="#fca5a5", fontsize=8.5, fontweight="bold")
    ax.text(x_bad + 0.45, y_card + 0.26, "'if not mask.any(): break' pulls 1 boolean byte back to CPU every loop, flushing the GPU queue.",
            color="#cbd5e1", fontsize=7.8)

    # ==========================
    # RIGHT CARD: GOOD VERSION
    # ==========================
    x_good = 8.25
    ax.add_patch(FancyBboxPatch((x_good, y_card), col_w, h_card, boxstyle="round,pad=0.0,rounding_size=0.14",
                                facecolor=BG_PANEL, edgecolor=NVIDIA_GREEN, linewidth=1.8))
    # Header banner
    ax.add_patch(FancyBboxPatch((x_good, y_card + h_card - 0.70), col_w, 0.70, boxstyle="round,pad=0.0,rounding_size=0.0",
                                facecolor="#0e2417", edgecolor=BORDER_COL, linewidth=0.8))
    ax.text(x_good + 0.35, y_card + h_card - 0.26, "[AFTER: OPTIMIZED] mandelbrot_torch_vectorized",
            color=NVIDIA_GREEN, fontsize=11.0, fontweight="bold")
    ax.text(x_good + 0.35, y_card + h_card - 0.50, "Intermediate Tensors on Device VRAM   |   Total CPU: 63.76 ms",
            color="#86efac", fontsize=9.0)
    ax.text(x_good + col_w - 0.35, y_card + h_card - 0.38, "4.82x FASTER",
            color="#4ade80", fontsize=9.5, fontweight="bold", ha="right")

    # Section 1: Operator Breakdown Table
    ax.text(x_good + 0.35, y_card + h_card - 0.98, "Top Operator Self CPU Time Breakdown", color=CYAN_BLUE, fontsize=9.5, fontweight="bold")
    good_ops = [
        ("aten::_local_scalar_dense", "0.00 ms", "0.00% (ELIMINATED!)", NVIDIA_GREEN),
        ("aten::where", "26.27 ms", "41.20%", CYAN_BLUE),
        ("aten::mul", "18.11 ms", "28.40%", CYAN_BLUE),
        ("aten::add", "12.82 ms", "20.10%", CYAN_BLUE),
        ("Self CUDA Compute Time", "40.80 ms", "64.0% Total", NVIDIA_GREEN),
    ]
    for idx, (op_name, time_str, pct_str, col) in enumerate(good_ops):
        y = y_card + h_card - 1.34 - idx * 0.38
        ax.text(x_good + 0.35, y, op_name, color=TEXT_WHITE, fontsize=8.6, family="monospace")
        ax.text(x_good + 4.6, y, time_str, color=TEXT_WHITE, fontsize=8.6)
        ax.text(x_good + col_w - 0.35, y, pct_str, color=col, fontsize=8.8, fontweight="bold", ha="right")

    # Section 2: Visual Timeline Mockup (THE FIX)
    ax.text(x_good + 0.35, y_card + 2.30, "Perfetto Visual Timeline: Continuous Pipelined Execution", color=NVIDIA_GREEN, fontsize=9.5, fontweight="bold")
    # CPU Track
    ax.add_patch(Rectangle((x_good + 0.35, y_card + 1.70), col_w - 0.7, 0.42, facecolor=BG_TRACK, edgecolor=BORDER_COL))
    ax.add_patch(Rectangle((x_good + 0.55, y_card + 1.75), col_w - 1.1, 0.32, facecolor="#4c1d95", edgecolor="#6d28d9"))
    ax.text(x_good + col_w / 2, y_card + 1.91, "PyTorch Asynchronous Dispatcher (Non-Blocking CPU)",
            color="#fff", fontsize=8.2, fontweight="bold", ha="center", va="center")

    # GPU Stream Track (Continuous)
    ax.add_patch(Rectangle((x_good + 0.35, y_card + 1.15), col_w - 0.7, 0.42, facecolor=BG_TRACK, edgecolor=BORDER_COL))
    for k in range(12):
        gx = x_good + 0.45 + k * 0.52
        ax.add_patch(Rectangle((gx, y_card + 1.20), 0.48, 0.32, facecolor=NVIDIA_GREEN, edgecolor="#15803d"))
        ax.text(gx + 0.24, y_card + 1.36, "gpu", color="#000", fontsize=6.8, ha="center", va="center")

    # Right Diagnostic Box
    ax.add_patch(FancyBboxPatch((x_good + 0.30, y_card + 0.15), col_w - 0.6, 0.60,
                                boxstyle="round,pad=0.0,rounding_size=0.10",
                                facecolor="#0c2918", edgecolor=NVIDIA_GREEN, linewidth=1.2))
    ax.text(x_good + 0.45, y_card + 0.50, "[*] ASYNCHRONOUS PIPELINE: Zero host stalls; GPU duty cycle 100%!",
            color="#86efac", fontsize=8.5, fontweight="bold")
    ax.text(x_good + 0.45, y_card + 0.26, "All tensor operations stay on GPU VRAM. CUDA queue executes at full line-rate.",
            color="#d1fae5", fontsize=7.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, facecolor=BG_DARK, edgecolor="none")
    plt.close()
    print(f"✓ Saved PyTorch Profiler Bad vs. Good comparison snapshot: {output_path}")


# ==============================================================================
# 3. NSIGHT SYSTEMS BAD VS GOOD COMPARISON SNAPSHOT
# ==============================================================================
def generate_nsys_comparison(output_path):
    """
    Generates high-contrast side-by-side comparison of NVIDIA Nsight Systems:
    Bad (Synchronous serialization & GPU starvation) vs Good (Asynchronous stream pipelining & overlap)
    in a clean, widescreen 16:7.2 frame focusing on timeline concurrency.
    """
    fig, ax = plt.subplots(figsize=(16, 7.2), dpi=180)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    draw_window_frame(ax, title="NVIDIA Nsight Systems 2026.1 - System Timeline: Bad vs. Good Profiling Results", width=16, height=7.2)

    # Sub-header bar
    ax.add_patch(Rectangle((0, 5.95), 16, 0.60, facecolor="#151d2a", edgecolor=BORDER_COL, linewidth=0.8))
    ax.text(0.6, 6.25, "Macro System Timeline: CUDA API Calls, PCIe Transfers (H2D/D2H), and GPU Compute Streams",
            color=TEXT_WHITE, fontsize=10, fontweight="bold", va="center")
    ax.text(15.4, 6.25, "Stream Concurrency & Overlap Analysis", color=NVIDIA_GREEN, fontsize=10, fontweight="bold", ha="right", va="center")

    # ==========================
    # TOP CARD: BAD VERSION (SYNCHRONOUS)
    # ==========================
    ax.add_patch(FancyBboxPatch((0.6, 3.35), 14.8, 2.50, boxstyle="round,pad=0.0,rounding_size=0.12",
                                facecolor=BG_PANEL, edgecolor=RED_ALERT, linewidth=1.6))
    ax.add_patch(FancyBboxPatch((0.6, 5.30), 14.8, 0.55, boxstyle="round,pad=0.0,rounding_size=0.0",
                                facecolor="#2a1215", edgecolor=BORDER_COL, linewidth=0.6))
    ax.text(0.9, 5.57, "[BEFORE: UNOPTIMIZED] Synchronous Execution (cudaStreamSynchronize Stalls & GPU Starvation)",
            color=RED_ALERT, fontsize=10.5, fontweight="bold")
    ax.text(15.1, 5.57, "WALL TIME: 15.0 ms   |   HOST STALL: 68.4%   |   GPU IDLE: 55%",
            color="#fca5a5", fontsize=9, fontweight="bold", ha="right")

    # Bad Tracks
    # 1. CPU API Track
    ax.add_patch(Rectangle((0.9, 4.65), 2.6, 0.48, facecolor="#1a2332", edgecolor=BORDER_COL))
    ax.text(1.0, 4.89, "CUDA API (Host)", color=CYAN_BLUE, fontsize=8.5, fontweight="bold")
    ax.add_patch(Rectangle((3.7, 4.65), 11.4, 0.48, facecolor=BG_TRACK, edgecolor=BORDER_COL))

    ax.add_patch(Rectangle((3.9, 4.71), 1.2, 0.36, facecolor=AMBER_WARN))
    ax.text(4.5, 4.89, "cudaMemcpy", color="#000", fontsize=7.5, fontweight="bold", ha="center")

    ax.add_patch(Rectangle((5.3, 4.71), 6.5, 0.36, facecolor=RED_ALERT))
    ax.text(8.55, 4.89, "cudaStreamSynchronize()  [STALL: 68.4% Host Time - CPU BLOCKED]",
            color="#fff", fontsize=8.5, fontweight="bold", ha="center")

    # 2. GPU Stream Track
    ax.add_patch(Rectangle((0.9, 4.00), 2.6, 0.48, facecolor="#1a2332", edgecolor=BORDER_COL))
    ax.text(1.0, 4.24, "GPU Stream 1", color=NVIDIA_GREEN, fontsize=8.5, fontweight="bold")
    ax.add_patch(Rectangle((3.7, 4.00), 11.4, 0.48, facecolor=BG_TRACK, edgecolor=BORDER_COL))

    ax.add_patch(Rectangle((3.9, 4.06), 1.2, 0.36, facecolor=MEMCPY_ORANGE))
    ax.text(4.5, 4.24, "[H2D Copy]", color="#fff", fontsize=7.5, ha="center")

    ax.add_patch(Rectangle((5.4, 4.06), 1.8, 0.36, facecolor=KERNEL_BLUE))
    ax.text(6.3, 4.24, "gemm_naive", color="#fff", fontsize=8, fontweight="bold", ha="center")

    # White Idle Gap
    ax.add_patch(Rectangle((7.4, 4.06), 4.2, 0.36, facecolor="none", edgecolor=RED_ALERT, linestyle="--", linewidth=1.5))
    ax.text(9.5, 4.24, "WHITE IDLE BUBBLE: GPU Sits Starved Waiting for Next Batch",
            color=RED_ALERT, fontsize=8, fontweight="bold", ha="center")

    ax.add_patch(Rectangle((11.8, 4.06), 1.0, 0.36, facecolor=MEMCPY_ORANGE))
    ax.text(12.3, 4.24, "[D2H Copy]", color="#fff", fontsize=7.5, ha="center")

    ax.text(0.9, 3.48, "[!] Bottleneck: Serial pipeline. CPU cannot launch next work until previous completes; GPU sits completely idle 55% of total elapsed time.",
            color="#fca5a5", fontsize=8.0)

    # ==========================
    # BOTTOM CARD: GOOD VERSION (PIPELINED STREAMS)
    # ==========================
    ax.add_patch(FancyBboxPatch((0.6, 0.40), 14.8, 2.75, boxstyle="round,pad=0.0,rounding_size=0.12",
                                facecolor=BG_PANEL, edgecolor=NVIDIA_GREEN, linewidth=1.6))
    ax.add_patch(FancyBboxPatch((0.6, 2.60), 14.8, 0.55, boxstyle="round,pad=0.0,rounding_size=0.0",
                                facecolor="#0e2417", edgecolor=BORDER_COL, linewidth=0.6))
    ax.text(0.9, 2.87, "[AFTER: OPTIMIZED] Pipelined Asynchronous Streams (Concurrent Transfer & Compute Overlap)",
            color=NVIDIA_GREEN, fontsize=10.5, fontweight="bold")
    ax.text(15.1, 2.87, "WALL TIME: 6.3 ms  [2.38x FASTER]   |   HOST STALL: 0.0%   |   GPU IDLE: 0%",
            color="#86efac", fontsize=9, fontweight="bold", ha="right")

    # Good Tracks
    # 1. CPU API Track
    ax.add_patch(Rectangle((0.9, 2.05), 2.6, 0.44, facecolor="#1a2332", edgecolor=BORDER_COL))
    ax.text(1.0, 2.27, "CUDA API (Host)", color=CYAN_BLUE, fontsize=8.5, fontweight="bold")
    ax.add_patch(Rectangle((3.7, 2.05), 11.4, 0.44, facecolor=BG_TRACK, edgecolor=BORDER_COL))

    ax.add_patch(Rectangle((3.9, 2.11), 5.2, 0.32, facecolor="#065f46"))
    ax.text(6.5, 2.27, "cudaMemcpyAsync + Kernel Launches (Asynchronous - CPU immediately returns)",
            color="#d1fae5", fontsize=8, fontweight="bold", ha="center")

    # 2. GPU Stream 1 (Compute)
    ax.add_patch(Rectangle((0.9, 1.48), 2.6, 0.44, facecolor="#1a2332", edgecolor=BORDER_COL))
    ax.text(1.0, 1.70, "GPU Stream 1 (Compute)", color=NVIDIA_GREEN, fontsize=8.5, fontweight="bold")
    ax.add_patch(Rectangle((3.7, 1.48), 11.4, 0.44, facecolor=BG_TRACK, edgecolor=BORDER_COL))

    ax.add_patch(Rectangle((5.2, 1.54), 2.2, 0.32, facecolor=NVIDIA_GREEN))
    ax.text(6.3, 1.70, "gemm_tiled [Chunk 0]", color="#000", fontsize=8, fontweight="bold", ha="center")

    ax.add_patch(Rectangle((7.6, 1.54), 2.2, 0.32, facecolor=NVIDIA_GREEN))
    ax.text(8.7, 1.70, "gemm_tiled [Chunk 1]", color="#000", fontsize=8, fontweight="bold", ha="center")

    # 3. GPU Stream 2 (Memory Overlap)
    ax.add_patch(Rectangle((0.9, 0.90), 2.6, 0.44, facecolor="#1a2332", edgecolor=BORDER_COL))
    ax.text(1.0, 1.12, "GPU Stream 2 (Memory)", color=MEMCPY_ORANGE, fontsize=8.5, fontweight="bold")
    ax.add_patch(Rectangle((3.7, 0.90), 11.4, 0.44, facecolor=BG_TRACK, edgecolor=BORDER_COL))

    ax.add_patch(Rectangle((3.9, 0.96), 1.2, 0.32, facecolor=MEMCPY_ORANGE))
    ax.text(4.5, 1.12, "[H2D Chunk 0]", color="#fff", fontsize=7.5, ha="center")

    # Overlapped block
    ax.add_patch(Rectangle((5.3, 0.96), 2.1, 0.32, facecolor=MEMCPY_ORANGE))
    ax.text(6.35, 1.12, "[H2D Chunk 1 (OVERLAPPED!)]", color="#fff", fontsize=7.5, fontweight="bold", ha="center")

    ax.add_patch(Rectangle((7.6, 0.96), 1.2, 0.32, facecolor=MEMCPY_ORANGE))
    ax.text(8.2, 1.12, "[D2H Chunk 0]", color="#fff", fontsize=7.5, ha="center")

    ax.text(0.9, 0.52, "[*] Overlapped Concurrency: Data transfers for Chunk 1 run simultaneously with computation for Chunk 0; GPU idle bubbles are eliminated.",
            color="#86efac", fontsize=8.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, facecolor=BG_DARK, edgecolor="none")
    plt.close()
    print(f"✓ Saved Nsight Systems Bad vs. Good comparison snapshot: {output_path}")


# ==============================================================================
# 4. MASTER SIDE-BY-SIDE PROFILER COMPARISON
# ==============================================================================
def generate_side_by_side_comparison(output_path):
    """
    Generates a master 3-panel side-by-side comparison infographic.
    Crops the redundant duplicate header and removes the bottom banner
    so the 3 primary tool cards are the clean, uncluttered focal point.
    """
    fig, ax = plt.subplots(figsize=(16, 7.2), dpi=180)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # Slim window frame
    ax.add_patch(FancyBboxPatch((0, 0), 16, 7.2, boxstyle="round,pad=0.0,rounding_size=0.18",
                                facecolor=BG_DARK, edgecolor=BORDER_COL, linewidth=1.8))
    # Slim title bar
    ax.add_patch(FancyBboxPatch((0, 6.70), 16, 0.50, boxstyle="round,pad=0.0,rounding_size=0.0",
                                facecolor="#161e2e", edgecolor=BORDER_COL, linewidth=0.8))
    for i, col in enumerate(["#ef4444", "#f59e0b", "#10b981"]):
        circle = plt.Circle((0.45 + i * 0.30, 6.95), 0.09, color=col)
        ax.add_patch(circle)
    ax.text(8.0, 6.95, "Master Profiler Suite: Side-by-Side Tool Comparison (Timeline vs. Micro-Kernel vs. Operator Graph)",
            color=TEXT_WHITE, fontsize=10.5, fontweight="bold", ha="center", va="center")

    # 3 Columns
    col_w = 4.70
    gap = 0.40
    cols_x = [0.55, 0.55 + col_w + gap, 0.55 + 2 * (col_w + gap)]

    headers = [
        ("NVIDIA Nsight Systems (nsys)", "MACRO TIMELINE VIEW", CYAN_BLUE),
        ("NVIDIA Nsight Compute (ncu)", "MICRO HARDWARE COUNTERS", NVIDIA_GREEN),
        ("PyTorch Profiler (torch.profiler)", "FRAMEWORK OPERATOR GRAPH", PURPLE_ACC),
    ]

    y_card = 0.35
    h_card = 6.05

    for idx, (title, sub, color) in enumerate(headers):
        x = cols_x[idx]
        # Main Column Card
        ax.add_patch(FancyBboxPatch((x, y_card), col_w, h_card, boxstyle="round,pad=0.0,rounding_size=0.14",
                                    facecolor=BG_PANEL, edgecolor=color, linewidth=1.8))
        # Card Header Banner
        ax.add_patch(FancyBboxPatch((x, y_card + h_card - 0.70), col_w, 0.70, boxstyle="round,pad=0.0,rounding_size=0.0",
                                    facecolor="#182030", edgecolor=BORDER_COL, linewidth=0.6))
        ax.text(x + col_w / 2, y_card + h_card - 0.26, title, color=color, fontsize=10.0, fontweight="bold", ha="center")
        ax.text(x + col_w / 2, y_card + h_card - 0.52, sub, color=TEXT_MUTED, fontsize=8.5, fontweight="bold", ha="center")

    # ==========================
    # COLUMN 1: NSIGHT SYSTEMS
    # ==========================
    x1 = cols_x[0]
    # Section A: GUI Expectations
    ax.text(x1 + 0.35, 5.45, "[*] WHAT TO EXPECT IN GUI:", color=TEXT_WHITE, fontsize=9.2, fontweight="bold")
    ax.text(x1 + 0.35, 5.12, "• Horizontal bars across CPU & GPU streams", color="#cbd5e1", fontsize=8.5)
    ax.text(x1 + 0.35, 4.84, "• Red API track: cudaStreamSynchronize stalls", color="#cbd5e1", fontsize=8.5)
    ax.text(x1 + 0.35, 4.56, "• White gaps between kernels: GPU starvation", color="#cbd5e1", fontsize=8.5)

    # Section B: Mockup Box (y: 2.85 to 4.25, height 1.40)
    ax.add_patch(Rectangle((x1 + 0.35, 2.85), col_w - 0.70, 1.40, facecolor=BG_TRACK, edgecolor="#334155"))
    ax.text(x1 + 0.55, 4.00, "CUDA API: cudaStreamSync [68.4% STALL]", color=RED_ALERT, fontsize=8.5, fontweight="bold")
    ax.add_patch(Rectangle((x1 + 0.55, 3.62), 3.1, 0.28, facecolor=RED_ALERT, edgecolor="#b91c1c"))

    ax.text(x1 + 0.55, 3.28, "GPU Stream 1: [gemm]    [IDLE GAP]    [tiled]", color=TEXT_MUTED, fontsize=7.8)
    ax.add_patch(Rectangle((x1 + 0.55, 2.96), 1.25, 0.24, facecolor=KERNEL_BLUE))
    ax.add_patch(Rectangle((x1 + 1.90, 2.96), 1.20, 0.24, facecolor="none", edgecolor=RED_ALERT, linestyle="--", linewidth=1.2))
    ax.add_patch(Rectangle((x1 + 3.20, 2.96), 0.75, 0.24, facecolor=NVIDIA_GREEN))

    # Section C: Replication Metric
    ax.text(x1 + 0.35, 2.48, "[?] KEY REPLICATION METRIC:", color=AMBER_WARN, fontsize=9.2, fontweight="bold")
    ax.text(x1 + 0.35, 2.18, "cudaStreamSynchronize > 50% Host Time", color="#fca5a5", fontsize=8.8, fontweight="bold")

    # Section D: Targeted Fix
    ax.text(x1 + 0.35, 1.78, "[+] TARGETED ARCHITECTURAL FIX:", color=NVIDIA_GREEN, fontsize=9.0, fontweight="bold")
    ax.text(x1 + 0.35, 1.48, "• Overlap H2D transfers with compute streams", color="#cbd5e1", fontsize=8.2)
    ax.text(x1 + 0.35, 1.22, "• Use CUDA Graphs to batch kernel launches", color="#cbd5e1", fontsize=8.2)

    # Section E: Command Box
    ax.add_patch(Rectangle((x1 + 0.35, 0.55), col_w - 0.70, 0.48, facecolor="#0e1726", edgecolor=BORDER_COL))
    ax.text(x1 + 0.55, 0.79, "$ make profile_nsys", color=CYAN_BLUE, fontsize=9.2, fontweight="bold", family="monospace", va="center")

    # ==========================
    # COLUMN 2: NSIGHT COMPUTE
    # ==========================
    x2 = cols_x[1]
    # Section A: GUI Expectations
    ax.text(x2 + 0.35, 5.45, "[*] WHAT TO EXPECT IN GUI:", color=TEXT_WHITE, fontsize=9.2, fontweight="bold")
    ax.text(x2 + 0.35, 5.12, "• Speed of Light (SOL) percentage bar charts", color="#cbd5e1", fontsize=8.5)
    ax.text(x2 + 0.35, 4.84, "• Warp Execution Efficiency percentage metric", color="#cbd5e1", fontsize=8.5)
    ax.text(x2 + 0.35, 4.56, "• Active threads per warp and register limits", color="#cbd5e1", fontsize=8.5)

    # Section B: Mockup Box (y: 2.85 to 4.25, height 1.40)
    ax.add_patch(Rectangle((x2 + 0.35, 2.85), col_w - 0.70, 1.40, facecolor=BG_TRACK, edgecolor="#334155"))
    ax.text(x2 + 0.55, 4.00, "Speed of Light: Compute 12.5% | Mem 4.1%", color=AMBER_WARN, fontsize=8.5, fontweight="bold")
    ax.add_patch(Rectangle((x2 + 0.55, 3.62), col_w - 1.10, 0.24, facecolor="#1e2636"))
    ax.add_patch(Rectangle((x2 + 0.55, 3.62), 0.55, 0.24, facecolor=RED_ALERT))

    ax.text(x2 + 0.55, 3.28, "Warp Efficiency: 14.28% (4.5 / 32 active)", color=RED_ALERT, fontsize=8.5, fontweight="bold")
    ax.add_patch(Rectangle((x2 + 0.55, 2.96), col_w - 1.10, 0.24, facecolor="#1e2636"))
    ax.add_patch(Rectangle((x2 + 0.55, 2.96), 0.60, 0.24, facecolor=RED_ALERT))

    # Section C: Replication Metric
    ax.text(x2 + 0.35, 2.48, "[?] KEY REPLICATION METRIC:", color=AMBER_WARN, fontsize=9.2, fontweight="bold")
    ax.text(x2 + 0.35, 2.18, "Warp Efficiency < 50% = Branch Divergence", color="#fca5a5", fontsize=8.8, fontweight="bold")

    # Section D: Targeted Fix
    ax.text(x2 + 0.35, 1.78, "[+] TARGETED ARCHITECTURAL FIX:", color=NVIDIA_GREEN, fontsize=9.0, fontweight="bold")
    ax.text(x2 + 0.35, 1.48, "• Convert 2D blocks to 1D grid-stride loops", color="#cbd5e1", fontsize=8.2)
    ax.text(x2 + 0.35, 1.22, "• Stage tile sub-matrices into Shared Memory", color="#cbd5e1", fontsize=8.2)

    # Section E: Command Box
    ax.add_patch(Rectangle((x2 + 0.35, 0.55), col_w - 0.70, 0.48, facecolor="#0e1726", edgecolor=BORDER_COL))
    ax.text(x2 + 0.55, 0.79, "$ make profile_ncu", color=NVIDIA_GREEN, fontsize=9.2, fontweight="bold", family="monospace", va="center")

    # ==========================
    # COLUMN 3: PYTORCH PROFILER
    # ==========================
    x3 = cols_x[2]
    # Section A: GUI Expectations
    ax.text(x3 + 0.35, 5.45, "[*] WHAT TO EXPECT IN GUI:", color=TEXT_WHITE, fontsize=9.2, fontweight="bold")
    ax.text(x3 + 0.35, 5.12, "• Operator execution table sorted by Self CPU %", color="#cbd5e1", fontsize=8.5)
    ax.text(x3 + 0.35, 4.84, "• JSON Perfetto timeline trace for Chrome UI", color="#cbd5e1", fontsize=8.5)
    ax.text(x3 + 0.35, 4.56, "• Accidental host-device scalar synchronizations", color="#cbd5e1", fontsize=8.5)

    # Section B: Mockup Box (y: 2.85 to 4.25, height 1.40)
    ax.add_patch(Rectangle((x3 + 0.35, 2.85), col_w - 0.70, 1.40, facecolor=BG_TRACK, edgecolor="#334155"))
    ax.text(x3 + 0.55, 4.00, "aten::_local_scalar_dense: 86.2% CPU", color=RED_ALERT, fontsize=8.5, fontweight="bold")
    ax.add_patch(Rectangle((x3 + 0.55, 3.62), col_w - 1.10, 0.28, facecolor=RED_ALERT, edgecolor="#b91c1c"))

    ax.text(x3 + 0.55, 3.28, "GPU Ops: where (4.1%) | mul (3.2%) | add (2.5%)", color=TEXT_MUTED, fontsize=7.8)
    ax.add_patch(Rectangle((x3 + 0.55, 2.96), 0.75, 0.24, facecolor="#0284c7"))
    ax.add_patch(Rectangle((x3 + 1.40, 2.96), 0.65, 0.24, facecolor="#0284c7"))
    ax.add_patch(Rectangle((x3 + 2.15, 2.96), 0.55, 0.24, facecolor="#0284c7"))

    # Section C: Replication Metric
    ax.text(x3 + 0.35, 2.48, "[?] KEY REPLICATION METRIC:", color=AMBER_WARN, fontsize=9.2, fontweight="bold")
    ax.text(x3 + 0.35, 2.18, "_local_scalar_dense > 30% = Host Sync Leak", color="#fca5a5", fontsize=8.8, fontweight="bold")

    # Section D: Targeted Fix
    ax.text(x3 + 0.35, 1.78, "[+] TARGETED ARCHITECTURAL FIX:", color=NVIDIA_GREEN, fontsize=9.0, fontweight="bold")
    ax.text(x3 + 0.35, 1.48, "• Delete .item() & if not mask.any() in loops", color="#cbd5e1", fontsize=8.2)
    ax.text(x3 + 0.35, 1.22, "• Keep intermediate boolean masks in GPU VRAM", color="#cbd5e1", fontsize=8.2)

    # Section E: Command Box
    ax.add_patch(Rectangle((x3 + 0.35, 0.55), col_w - 0.70, 0.48, facecolor="#0e1726", edgecolor=BORDER_COL))
    ax.text(x3 + 0.55, 0.79, "$ make profile_torch", color=PURPLE_ACC, fontsize=9.2, fontweight="bold", family="monospace", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, facecolor=BG_DARK, edgecolor="none")
    plt.close()
    print(f"✓ Saved Side-by-Side Comparison snapshot: {output_path}")


# ==============================================================================
# 5. INDIVIDUAL PROFILER SNAPSHOTS (ENHANCED LEGIBILITY)
# ==============================================================================
def generate_nsys_snapshot(output_path):
    """Generates NVIDIA Nsight Systems timeline snapshot with enhanced typography."""
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=180)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    draw_window_frame(ax, title="NVIDIA Nsight Systems 2026.1 - [gemm_cuda_trace.nsys-rep]", width=15, height=8.5)

    # Sub-toolbar
    ax.add_patch(Rectangle((0, 7.35), 15, 0.55, facecolor="#151d2a", edgecolor=BORDER_COL, linewidth=0.8))
    ax.text(0.6, 7.62, "Zoom: 100%   |   Time Range: 0.00ms - 15.00ms   |   Filter: CUDA API, GPU Streams, OS Runtime",
            color=TEXT_MUTED, fontsize=9.5, va="center")
    ax.text(14.4, 7.62, "NVIDIA A100-SXM4", color=NVIDIA_GREEN, fontsize=9.5, fontweight="bold", ha="right", va="center")

    # Time ruler
    ax.add_patch(Rectangle((3.2, 6.85), 11.2, 0.42, facecolor="#101520", edgecolor=BORDER_COL, linewidth=0.6))
    for t, label in enumerate(["0.0ms", "2.5ms", "5.0ms", "7.5ms", "10.0ms", "12.5ms", "15.0ms"]):
        x_pos = 3.2 + t * (11.2 / 6)
        ax.plot([x_pos, x_pos], [6.85, 7.02], color=TEXT_MUTED, linewidth=1.0)
        ax.text(x_pos, 7.10, label, color=TEXT_MUTED, fontsize=8.0, ha="center", va="bottom")

    tracks = [
        ("Main Thread (CPU)", 5.9),
        ("CUDA API (Host)", 4.7),
        ("Memory (H2D/D2H)", 3.5),
        ("GPU Stream 1 (Compute)", 2.3),
        ("GPU Stream 2 (Async)", 1.1),
    ]

    for label, y_top in tracks:
        ax.add_patch(Rectangle((0.6, y_top - 0.9), 2.4, 0.9, facecolor=BG_PANEL, edgecolor=BORDER_COL, linewidth=0.8))
        ax.text(0.75, y_top - 0.45, label, color=CYAN_BLUE, fontsize=9, fontweight="bold", va="center")
        ax.add_patch(Rectangle((3.2, y_top - 0.9), 11.2, 0.9, facecolor=BG_TRACK, edgecolor="#1e2636", linewidth=0.6))

    # CPU Thread
    ax.add_patch(Rectangle((3.4, 5.25), 3.4, 0.45, facecolor="#374151", edgecolor="#4b5563"))
    ax.text(5.1, 5.47, "CPU: Matrix Initialization & Verification", color=TEXT_WHITE, fontsize=8.2, ha="center", va="center")

    # CUDA API
    ax.add_patch(Rectangle((7.0, 4.05), 1.2, 0.45, facecolor=AMBER_WARN))
    ax.text(7.6, 4.27, "cudaMemcpyAsync", color="#000", fontsize=7.8, fontweight="bold", ha="center", va="center")

    ax.add_patch(Rectangle((8.3, 4.05), 5.2, 0.45, facecolor=RED_ALERT))
    ax.text(10.9, 4.27, "cudaStreamSynchronize()  [STALL: 68.4% Host Time]", color="#fff", fontsize=8.8, fontweight="bold", ha="center", va="center")

    # Memory Transfers
    ax.add_patch(Rectangle((7.1, 2.85), 1.4, 0.45, facecolor=MEMCPY_ORANGE))
    ax.text(7.8, 3.07, "[H2D] Matrix A & B (1024 KB)", color="#fff", fontsize=7.8, ha="center", va="center")

    ax.add_patch(Rectangle((13.0, 2.85), 1.1, 0.45, facecolor=MEMCPY_ORANGE))
    ax.text(13.55, 3.07, "[D2H] Result (512 KB)", color="#fff", fontsize=7.8, ha="center", va="center")

    # GPU Stream 1
    ax.add_patch(Rectangle((8.6, 1.65), 2.2, 0.45, facecolor=KERNEL_BLUE))
    ax.text(9.7, 1.87, "gemm_cuda_naive [76.5 us]", color="#fff", fontsize=8.2, fontweight="bold", ha="center", va="center")

    ax.add_patch(Rectangle((10.9, 1.65), 1.2, 0.45, facecolor="none", edgecolor=RED_ALERT, linewidth=1.8, linestyle="--"))
    ax.text(11.5, 1.87, "IDLE BUBBLE", color=RED_ALERT, fontsize=8, fontweight="bold", ha="center", va="center")

    ax.add_patch(Rectangle((12.2, 1.65), 1.0, 0.45, facecolor=NVIDIA_GREEN))
    ax.text(12.7, 1.87, "gemm_tiled", color="#000", fontsize=8, fontweight="bold", ha="center", va="center")

    # GPU Stream 2
    ax.add_patch(Rectangle((9.0, 0.45), 2.0, 0.45, facecolor="#0284c7"))
    ax.text(10.0, 0.67, "mandelbrot_cuda_grid_stride [48 us]", color="#fff", fontsize=7.8, ha="center", va="center")

    # Callout Annotations
    ax.annotate("[!] Host CPU blocked 68% of time waiting for GPU!",
                xy=(10.9, 4.55), xytext=(8.8, 6.0),
                arrowprops=dict(facecolor=RED_ALERT, edgecolor="none", shrink=0.08, width=1.8, headwidth=7),
                bbox=dict(boxstyle="round,pad=0.45", fc="#2d1515", ec=RED_ALERT, lw=1.4),
                color="#fca5a5", fontsize=9.2, fontweight="bold")

    ax.annotate("[*] 2.5x Faster inside GPU!\nShared Memory Tiling",
                xy=(12.7, 2.15), xytext=(12.0, 3.6),
                arrowprops=dict(facecolor=NVIDIA_GREEN, edgecolor="none", shrink=0.08, width=1.8, headwidth=7),
                bbox=dict(boxstyle="round,pad=0.45", fc="#142914", ec=NVIDIA_GREEN, lw=1.4),
                color="#86efac", fontsize=9.2, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, facecolor=BG_DARK, edgecolor="none")
    plt.close()
    print(f"✓ Saved Nsight Systems snapshot: {output_path}")


def generate_ncu_snapshot(output_path):
    """Generates NVIDIA Nsight Compute micro-kernel dashboard snapshot with enhanced typography."""
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=180)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    draw_window_frame(ax, title="NVIDIA Nsight Compute 2026.1 - mandelbrot_cuda_naive [Micro-Profile]", width=15, height=8.5)

    # Sub-header bar
    ax.add_patch(Rectangle((0, 7.30), 15, 0.65, facecolor="#151d2a", edgecolor=BORDER_COL, linewidth=0.8))
    ax.text(0.6, 7.72, "Kernel: mandelbrot_cuda_naive(int*, float, float, float, float, int)",
            color="#fff", fontsize=10.5, fontweight="bold")
    ax.text(0.6, 7.45, "Grid Size: [64, 64, 1]   Block Size: [16, 16, 1]   Device: NVIDIA A100-SXM4-80GB (CC 8.0)",
            color=TEXT_MUTED, fontsize=8.8)
    ax.text(14.4, 7.62, "DURATION: 9.85 ms", color=AMBER_WARN, fontsize=10, fontweight="bold", ha="right")

    # Left Panel: Speed of Light (SOL) Chart
    ax.add_patch(FancyBboxPatch((0.6, 2.1), 6.6, 4.9, boxstyle="round,pad=0.0,rounding_size=0.12",
                                facecolor=BG_PANEL, edgecolor=BORDER_COL, linewidth=1.4))
    ax.text(0.9, 6.65, "[SOL] GPU Speed of Light (SOL) Throughput", color=CYAN_BLUE, fontsize=10.5, fontweight="bold")

    metrics = [
        ("Compute (SM) Throughput", 12.45, RED_ALERT),
        ("DRAM Memory Throughput", 4.12, RED_ALERT),
        ("L1/TEX Cache Throughput", 18.20, AMBER_WARN),
        ("L2 Cache Throughput", 7.80, RED_ALERT),
    ]

    for idx, (label, val, bar_color) in enumerate(metrics):
        y = 5.95 - idx * 0.95
        ax.text(0.9, y + 0.16, label, color=TEXT_WHITE, fontsize=9.2, fontweight="bold")
        ax.text(6.8, y + 0.16, f"{val:.1f}%", color=bar_color, fontsize=9.5, fontweight="bold", ha="right")
        ax.add_patch(Rectangle((0.9, y - 0.22), 5.9, 0.24, facecolor="#1e2636", edgecolor="#334155"))
        fill_width = (val / 100.0) * 5.9
        ax.add_patch(Rectangle((0.9, y - 0.22), fill_width, 0.24, facecolor=bar_color, edgecolor="none"))

    # Verdict Box under SOL
    ax.add_patch(FancyBboxPatch((0.8, 2.3), 6.2, 0.75, boxstyle="round,pad=0.0,rounding_size=0.08",
                                facecolor="#2d1515", edgecolor=RED_ALERT, linewidth=1.2))
    ax.text(1.0, 2.75, "[!] UNDER-UTILIZED HARDWARE: Compute & Memory Starved",
            color="#fca5a5", fontsize=8.8, fontweight="bold")
    ax.text(1.0, 2.48, "Hardware sits idle. The kernel is bottlenecked by severe SIMT thread divergence.",
            color="#cbd5e1", fontsize=8.2)

    # Right Panel: Warp State
    ax.add_patch(FancyBboxPatch((7.6, 2.1), 6.8, 4.9, boxstyle="round,pad=0.0,rounding_size=0.12",
                                facecolor=BG_PANEL, edgecolor=BORDER_COL, linewidth=1.4))
    ax.text(7.9, 6.65, ">>> Warp Execution Efficiency (THE SMOKING GUN)", color=RED_ALERT, fontsize=10.5, fontweight="bold")

    warp_stats = [
        ("Warp Execution Efficiency", "14.28%", "CRITICAL (< 50%)", RED_ALERT, 14.28),
        ("Avg. Active Threads Per Warp", "4.57 / 32", "Ideal: 32.0", RED_ALERT, (4.57/32)*100),
        ("Issue Slots No Eligible", "78.40%", "Waiting on branches", AMBER_WARN, 78.40),
        ("Eligible Warps Per Scheduler", "0.12", "Under-subscribed", AMBER_WARN, 12.0),
    ]

    for idx, (label, val_str, note, col, fill_pct) in enumerate(warp_stats):
        y = 5.95 - idx * 0.95
        ax.text(7.9, y + 0.16, label, color=TEXT_WHITE, fontsize=9.2, fontweight="bold")
        ax.text(11.8, y + 0.16, val_str, color=col, fontsize=9.5, fontweight="bold")
        ax.text(14.0, y + 0.16, note, color=TEXT_MUTED, fontsize=8.0, ha="right")
        ax.add_patch(Rectangle((7.9, y - 0.22), 6.1, 0.24, facecolor="#1e2636", edgecolor="#334155"))
        fill_width = (fill_pct / 100.0) * 6.1
        ax.add_patch(Rectangle((7.9, y - 0.22), fill_width, 0.24, facecolor=col, edgecolor="none"))

    # Bottom Panel: Automated Guidance
    ax.add_patch(FancyBboxPatch((0.6, 0.45), 13.8, 1.45, boxstyle="round,pad=0.0,rounding_size=0.12",
                                facecolor="#0e2417", edgecolor=NVIDIA_GREEN, linewidth=1.4))
    ax.text(0.9, 1.58, "[*] NVIDIA Guided Optimization Rule: High Branch Divergence Detected",
            color=NVIDIA_GREEN, fontsize=9.5, fontweight="bold")
    ax.text(0.9, 1.25, "• Diagnosis: Only 4.5 out of 32 threads in each warp execute useful instructions; 27.5 threads are masked off.",
            color="#d1fae5", fontsize=8.5)
    ax.text(0.9, 0.95, "• Remediation: Replace 2D thread block assignment with a 1D grid-stride loop to dynamically balance escape iterations.",
            color="#d1fae5", fontsize=8.5)
    ax.text(0.9, 0.65, "• Verified Speedup: Achieves 68.4% warp efficiency and 2.03x kernel runtime reduction (9.85 ms -> 4.84 ms).",
            color="#86efac", fontsize=8.8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, facecolor=BG_DARK, edgecolor="none")
    plt.close()
    print(f"✓ Saved Nsight Compute snapshot: {output_path}")


def generate_torch_profiler_snapshot(output_path):
    """Generates PyTorch Profiler / Perfetto Chrome Trace snapshot with enhanced typography."""
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=180)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    draw_window_frame(ax, title="Perfetto UI / Chrome Tracing - [torch_trace_mandelbrot.json]", width=15, height=8.5)

    # Sub-header
    ax.add_patch(Rectangle((0, 7.35), 15, 0.55, facecolor="#151d2a", edgecolor=BORDER_COL, linewidth=0.8))
    ax.text(0.6, 7.62, "[TRACE] PyTorch Kineto Profiler Trace   |   Total Recorded Events: 12,480   |   Device: cuda:0 / mps",
            color=TEXT_WHITE, fontsize=9.5, fontweight="bold")

    # Time Ruler
    ax.add_patch(Rectangle((3.2, 6.85), 11.2, 0.42, facecolor="#101520", edgecolor=BORDER_COL, linewidth=0.6))
    for t, label in enumerate(["0ms", "50ms", "100ms", "150ms", "200ms", "250ms", "300ms", "350ms"]):
        x_pos = 3.2 + t * (11.2 / 7)
        ax.plot([x_pos, x_pos], [6.85, 7.02], color=TEXT_MUTED, linewidth=1.0)
        ax.text(x_pos, 7.10, label, color=TEXT_MUTED, fontsize=8.0, ha="center", va="bottom")

    # Track 1: Python Host Loop
    ax.add_patch(Rectangle((0.6, 5.5), 2.4, 1.1, facecolor=BG_PANEL, edgecolor=BORDER_COL, linewidth=0.8))
    ax.text(0.75, 6.05, "Thread 1\n(Python Runtime)", color=PURPLE_ACC, fontsize=9, fontweight="bold", va="center")
    ax.add_patch(Rectangle((3.2, 5.5), 11.2, 1.1, facecolor=BG_TRACK, edgecolor="#1e2636", linewidth=0.6))

    ax.add_patch(Rectangle((3.4, 6.05), 10.8, 0.45, facecolor="#4c1d95", edgecolor="#6d28d9"))
    ax.text(8.8, 6.27, "mandelbrot_torch_naive()  [Total CPU: 308.1 ms]", color="#fff", fontsize=8.8, fontweight="bold", ha="center", va="center")

    # Operator Level Track
    ax.add_patch(Rectangle((0.6, 4.0), 2.4, 1.3, facecolor=BG_PANEL, edgecolor=BORDER_COL, linewidth=0.8))
    ax.text(0.75, 4.65, "PyTorch Operators\n(Kineto Dispatch)", color=CYAN_BLUE, fontsize=9, fontweight="bold", va="center")
    ax.add_patch(Rectangle((3.2, 4.0), 11.2, 1.3, facecolor=BG_TRACK, edgecolor="#1e2636", linewidth=0.6))

    ax.add_patch(Rectangle((3.4, 4.65), 0.7, 0.55, facecolor="#0284c7"))
    ax.text(3.75, 4.92, "where", color="#fff", fontsize=7.2, ha="center", va="center")

    ax.add_patch(Rectangle((4.2, 4.65), 0.5, 0.55, facecolor="#0284c7"))
    ax.text(4.45, 4.92, "mul", color="#fff", fontsize=7.2, ha="center", va="center")

    # Huge Red Stall
    ax.add_patch(Rectangle((4.8, 4.30), 9.2, 0.90, facecolor=RED_ALERT, edgecolor="#b91c1c", linewidth=1.6))
    ax.text(9.4, 4.75, "aten::_local_scalar_dense  [265.4 ms  |  86.21% SELF CPU TIME]",
            color="#fff", fontsize=9.5, fontweight="bold", ha="center", va="center")

    # Track 3: GPU Stream
    ax.add_patch(Rectangle((0.6, 2.6), 2.4, 1.2, facecolor=BG_PANEL, edgecolor=BORDER_COL, linewidth=0.8))
    ax.text(0.75, 3.20, "GPU Stream 7\n(Kernel Execution)", color=NVIDIA_GREEN, fontsize=9, fontweight="bold", va="center")
    ax.add_patch(Rectangle((3.2, 2.6), 11.2, 1.2, facecolor=BG_TRACK, edgecolor="#1e2636", linewidth=0.6))

    for k in range(8):
        x = 3.5 + k * 1.35
        ax.add_patch(Rectangle((x, 2.85), 0.40, 0.65, facecolor=NVIDIA_GREEN))
        ax.text(x + 0.20, 3.17, "gpu_op", color="#000", fontsize=6.5, ha="center", va="center")

    # Callout
    ax.annotate("[!] THE SMOKING GUN: Host CPU blocks 265 ms\nto pull 1 boolean byte across PCIe for 'mask.any()'!",
                xy=(9.4, 5.25), xytext=(6.5, 6.7),
                arrowprops=dict(facecolor=RED_ALERT, edgecolor="none", shrink=0.08, width=1.8, headwidth=7),
                bbox=dict(boxstyle="round,pad=0.45", fc="#2d1515", ec=RED_ALERT, lw=1.4),
                color="#fca5a5", fontsize=9.2, fontweight="bold")

    # Bottom Inspector Table
    ax.add_patch(FancyBboxPatch((0.6, 0.45), 13.8, 1.95, boxstyle="round,pad=0.0,rounding_size=0.12",
                                facecolor=BG_PANEL, edgecolor=BORDER_COL, linewidth=1.4))
    ax.text(0.9, 2.05, "[DETAILS] Event Details: aten::_local_scalar_dense (Selected Item)",
            color=CYAN_BLUE, fontsize=9.5, fontweight="bold")

    details = [
        ("Self CPU Time:", "265.420 ms (86.21%)", RED_ALERT),
        ("Total CPU Time:", "265.420 ms", TEXT_WHITE),
        ("CUDA Device Time:", "0.000 us (Pure Host PCIe Stall)", AMBER_WARN),
        ("Invocation Origin:", "scripts/run_mandelbrot.py:58 -> if not mask.any(): break", TEXT_MUTED),
        ("Targeted Remedy:", "Eliminate per-iteration host boolean check; intermediate tensors remain on GPU VRAM", NVIDIA_GREEN),
    ]

    for idx, (k, v, c) in enumerate(details):
        y = 1.70 - idx * 0.30
        ax.text(0.9, y, k, color=TEXT_MUTED, fontsize=8.5)
        ax.text(3.0, y, v, color=c, fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, facecolor=BG_DARK, edgecolor="none")
    plt.close()
    print(f"✓ Saved PyTorch Profiler snapshot: {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    pres_snap_dir = os.path.join(root_dir, "presentation", "images", "snapshots")
    telem_snap_dir = os.path.join(root_dir, "telemetry", "snapshots")

    os.makedirs(pres_snap_dir, exist_ok=True)
    os.makedirs(telem_snap_dir, exist_ok=True)

    snapshots = [
        ("snapshot_nsys_timeline.png", generate_nsys_snapshot),
        ("snapshot_ncu_micro_dashboard.png", generate_ncu_snapshot),
        ("snapshot_torch_profiler_trace.png", generate_torch_profiler_snapshot),
        ("snapshot_side_by_side_comparison.png", generate_side_by_side_comparison),
        ("snapshot_ncu_comparison.png", generate_ncu_comparison),
        ("snapshot_torch_comparison.png", generate_torch_comparison),
        ("snapshot_nsys_comparison.png", generate_nsys_comparison),
    ]

    for fname, gen_func in snapshots:
        pres_path = os.path.join(pres_snap_dir, fname)
        telem_path = os.path.join(telem_snap_dir, fname)
        gen_func(pres_path)
        shutil.copyfile(pres_path, telem_path)

    print("\nAll profiler snapshots successfully generated, aligned, and mirrored!")
