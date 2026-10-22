#!/usr/bin/env python3
"""
generate_benchmark_charts.py
============================
Generates crisp, high-resolution benchmark visualization charts using Pillow
for inclusion in README.md, GitHub documentation, and research logs.
"""

import os
from PIL import Image, ImageDraw, ImageFont

os.makedirs("assets", exist_ok=True)

# Helper function to get default font
def get_font(size=14):
    try:
        # Standard DejaVu or FreeSans on Linux
        candidates = [
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/gnu-free/FreeSansBold.ttf",
        ]
        for c in candidates:
            if os.path.exists(c):
                return ImageFont.truetype(c, size)
    except Exception:
        pass
    return ImageFont.load_default()

def get_font_regular(size=12):
    try:
        candidates = [
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/gnu-free/FreeSans.ttf",
        ]
        for c in candidates:
            if os.path.exists(c):
                return ImageFont.truetype(c, size)
    except Exception:
        pass
    return ImageFont.load_default()

# -------------------------------------------------------------
# Chart 1: Throughput & Inference Speed Comparison
# -------------------------------------------------------------
def create_throughput_chart():
    width, height = 900, 480
    im = Image.new("RGB", (width, height), color="#0F172A")  # Slate 900
    draw = ImageDraw.Draw(im)

    font_title = get_font(22)
    font_sub = get_font_regular(13)
    font_label = get_font(13)
    font_val = get_font(14)
    font_axis = get_font_regular(11)

    # Title Banner
    draw.text((40, 25), "INFERENCE THROUGHPUT & HARDWARE TELEMETRY", fill="#F8FAFC", font=font_title)
    draw.text((40, 55), "Target: RTX 3070 Ti (8GB VRAM) + 16GB Host RAM | Bounded Context (2048)", fill="#94A3B8", font=font_sub)

    # Bounding Box for chart
    cx, cy, cw, ch = 80, 110, 740, 310
    draw.rectangle([cx, cy, cx + cw, cy + ch], outline="#334155", width=1, fill="#1E293B")

    # Gridlines
    max_val = 60
    for v in [0, 15, 30, 45, 60]:
        y = cy + ch - int((v / max_val) * ch)
        draw.line([(cx, y), (cx + cw, y)], fill="#334155", width=1)
        draw.text((cx - 35, y - 7), f"{v}", fill="#64748B", font=font_axis)

    # Data Points
    # (Model name, Prompt Proc Speed, Token Gen Speed)
    data = [
        ("DeepSeek 16B\n(MoE Q3_K_M)", 33.31, 16.15, "#10B981", "#34D399"),
        ("Qwen 32B CPU\n(Dense Q3_K_M)", 1.55, 0.12, "#EF4444", "#F87171"),
        ("Qwen 32B GPU Offload\n(24 Layers VRAM)", 52.4, 24.8, "#3B82F6", "#60A5FA"),
    ]

    group_w = cw // len(data)
    bar_w = 42

    for i, (name, pp, tg, col_pp, col_tg) in enumerate(data):
        gx = cx + i * group_w + (group_w - (bar_w * 2 + 15)) // 2

        # Bar 1: Prompt Processing
        h_pp = int((pp / max_val) * ch)
        y_pp = cy + ch - h_pp
        draw.rectangle([gx, y_pp, gx + bar_w, cy + ch], fill=col_pp)
        draw.text((gx + 2, y_pp - 20), f"{pp:.1f}", fill="#F8FAFC", font=font_val)

        # Bar 2: Token Generation
        h_tg = max(int((tg / max_val) * ch), 4)
        y_tg = cy + ch - h_tg
        draw.rectangle([gx + bar_w + 12, y_tg, gx + bar_w * 2 + 12, cy + ch], fill=col_tg)
        draw.text((gx + bar_w + 14, y_tg - 20), f"{tg:.1f}", fill="#F8FAFC", font=font_val)

        # Labels
        lines = name.split("\n")
        draw.text((gx - 5, cy + ch + 12), lines[0], fill="#E2E8F0", font=font_label)
        draw.text((gx + 5, cy + ch + 30), lines[1], fill="#94A3B8", font=font_axis)

    # Legend
    lx = cx + cw - 250
    ly = cy + 18
    draw.rectangle([lx, ly, lx + 18, ly + 14], fill="#10B981")
    draw.text((lx + 26, ly), "Prompt Prefill (t/s)", fill="#E2E8F0", font=font_axis)
    draw.rectangle([lx, ly + 22, lx + 18, ly + 36], fill="#34D399")
    draw.text((lx + 26, ly + 22), "Token Decode (t/s)", fill="#E2E8F0", font=font_axis)

    im.save("assets/benchmark_throughput_comparison.png")
    print("[✓] Saved assets/benchmark_throughput_comparison.png")

# -------------------------------------------------------------
# Chart 2: Intelligence Retention vs Quantization Formats
# -------------------------------------------------------------
def create_retention_chart():
    width, height = 900, 480
    im = Image.new("RGB", (width, height), color="#0F172A")
    draw = ImageDraw.Draw(im)

    font_title = get_font(22)
    font_sub = get_font_regular(13)
    font_label = get_font(13)
    font_val = get_font(14)
    font_axis = get_font_regular(11)

    draw.text((40, 25), "INTELLIGENCE RETENTION & QUALITY COMPARISON", fill="#F8FAFC", font=font_title)
    draw.text((40, 55), "Fidelity Retention vs Quantization Scheme (BF16 Baseline vs TQ1_0 vs BitNet Q3_K_M)", fill="#94A3B8", font=font_sub)

    cx, cy, cw, ch = 80, 110, 740, 310
    draw.rectangle([cx, cy, cx + cw, cy + ch], outline="#334155", width=1, fill="#1E293B")

    # Gridlines (0% to 100%)
    for v in [0, 25, 50, 75, 100]:
        y = cy + ch - int((v / 100.0) * ch)
        draw.line([(cx, y), (cx + cw, y)], fill="#334155", width=1)
        draw.text((cx - 45, y - 7), f"{v}%", fill="#64748B", font=font_axis)

    # Models: (Name, Weight Cosine Sim %, Benchmark Pass Rate %, Color, Subtitle)
    schemes = [
        ("Original Donor\n(Uncompressed)", 100.0, 100.0, "#38BDF8", "16.0 BPW (BF16)"),
        ("Naive TQ1_0\n(Outlier Trap)", 1.5, 0.0, "#EF4444", "1.69 BPW (Collapses)"),
        ("Q2_K Uniform\n(Double Quant)", 76.4, 33.3, "#F59E0B", "2.56 BPW (Degraded)"),
        ("BitNet Q3_K_M\n(Protected Attn)", 93.8, 100.0, "#10B981", "3.89 BPW (Preserved)"),
    ]

    group_w = cw // len(schemes)
    bar_w = 38

    for i, (name, cos_sim, pass_rate, col, bpw_str) in enumerate(schemes):
        gx = cx + i * group_w + (group_w - (bar_w * 2 + 12)) // 2

        # Bar 1: Cosine Similarity
        h_cos = max(int((cos_sim / 100.0) * ch), 4)
        y_cos = cy + ch - h_cos
        draw.rectangle([gx, y_cos, gx + bar_w, cy + ch], fill=col)
        draw.text((gx - 2, y_cos - 18), f"{cos_sim:.1f}%", fill="#F8FAFC", font=font_val)

        # Bar 2: Pass Rate
        h_pass = max(int((pass_rate / 100.0) * ch), 4)
        y_pass = cy + ch - h_pass
        draw.rectangle([gx + bar_w + 10, y_pass, gx + bar_w * 2 + 10, cy + ch], fill="#818CF8")
        draw.text((gx + bar_w + 8, y_pass - 18), f"{pass_rate:.0f}%", fill="#F8FAFC", font=font_val)

        # Name & Subtitle
        lines = name.split("\n")
        draw.text((gx - 10, cy + ch + 12), lines[0], fill="#E2E8F0", font=font_label)
        draw.text((gx - 5, cy + ch + 28), lines[1], fill="#94A3B8", font=font_axis)
        draw.text((gx - 5, cy + ch + 42), bpw_str, fill="#64748B", font=font_axis)

    # Legend
    lx = cx + 20
    ly = cy + 18
    draw.rectangle([lx, ly, lx + 18, ly + 14], fill="#10B981")
    draw.text((lx + 26, ly), "Weight Cosine Similarity (%)", fill="#E2E8F0", font=font_axis)
    draw.rectangle([lx, ly + 22, lx + 18, ly + 36], fill="#818CF8")
    draw.text((lx + 26, ly + 22), "Ruthless Benchmark Pass Rate (%)", fill="#E2E8F0", font=font_axis)

    im.save("assets/retention_fidelity_chart.png")
    print("[✓] Saved assets/retention_fidelity_chart.png")

# -------------------------------------------------------------
# Chart 3: Memory Footprint & Swap Dynamics
# -------------------------------------------------------------
def create_memory_chart():
    width, height = 900, 480
    im = Image.new("RGB", (width, height), color="#0F172A")
    draw = ImageDraw.Draw(im)

    font_title = get_font(22)
    font_sub = get_font_regular(13)
    font_label = get_font(13)
    font_val = get_font(14)
    font_axis = get_font_regular(11)

    draw.text((40, 25), "MEMORY ALLOCATION & DISK SWAP DYNAMICS", fill="#F8FAFC", font=font_title)
    draw.text((40, 55), "Physical RAM (16GB) vs GPU VRAM (8GB) vs NVMe Swap Allocation", fill="#94A3B8", font=font_sub)

    cx, cy, cw, ch = 80, 110, 740, 310
    draw.rectangle([cx, cy, cx + cw, cy + ch], outline="#334155", width=1, fill="#1E293B")

    # Gridlines (0 GB to 24 GB)
    max_gb = 24.0
    for v in [0, 6, 12, 16, 20, 24]:
        y = cy + ch - int((v / max_gb) * ch)
        col = "#DC2626" if v == 16 else "#334155"  # Highlight 16GB threshold in red
        draw.line([(cx, y), (cx + cw, y)], fill=col, width=1 if v != 16 else 2)
        label = f"{v} GB" + (" (RAM Limit)" if v == 16 else "")
        draw.text((cx - 75 if v == 16 else cx - 45, y - 7), label, fill="#EF4444" if v == 16 else "#64748B", font=font_axis)

    # Scenarios: (Name, Host RAM GB, VRAM GB, Swap GB)
    scenarios = [
        ("DeepSeek 16B\n(CPU Only)", 7.6, 0.0, 0.0, "Zero Swap Paging"),
        ("Qwen 32B\n(CPU Only)", 14.8, 0.0, 8.2, "CRITICAL: Swap Locked"),
        ("Qwen 32B Hybrid\n(24 Layers VRAM)", 9.2, 5.7, 0.0, "Smooth Acceleration"),
    ]

    group_w = cw // len(scenarios)
    bar_w = 40

    for i, (name, host_ram, vram, swap, note) in enumerate(scenarios):
        gx = cx + i * group_w + (group_w - (bar_w * 3 + 16)) // 2

        # Bar 1: Host RAM
        h_ram = int((host_ram / max_gb) * ch)
        y_ram = cy + ch - h_ram
        draw.rectangle([gx, y_ram, gx + bar_w, cy + ch], fill="#3B82F6")
        draw.text((gx + 2, y_ram - 18), f"{host_ram}G", fill="#F8FAFC", font=font_val)

        # Bar 2: GPU VRAM
        h_vram = max(int((vram / max_gb) * ch), 3) if vram > 0 else 0
        y_vram = cy + ch - h_vram
        if vram > 0:
            draw.rectangle([gx + bar_w + 8, y_vram, gx + bar_w * 2 + 8, cy + ch], fill="#10B981")
            draw.text((gx + bar_w + 10, y_vram - 18), f"{vram}G", fill="#F8FAFC", font=font_val)

        # Bar 3: NVMe Swap
        h_swap = max(int((swap / max_gb) * ch), 3) if swap > 0 else 0
        y_swap = cy + ch - h_swap
        if swap > 0:
            draw.rectangle([gx + bar_w * 2 + 16, y_swap, gx + bar_w * 3 + 16, cy + ch], fill="#EF4444")
            draw.text((gx + bar_w * 2 + 18, y_swap - 18), f"{swap}G", fill="#FCA5A5", font=font_val)

        # Labels
        lines = name.split("\n")
        draw.text((gx - 5, cy + ch + 12), lines[0], fill="#E2E8F0", font=font_label)
        draw.text((gx - 5, cy + ch + 28), lines[1], fill="#94A3B8", font=font_axis)
        draw.text((gx - 5, cy + ch + 42), note, fill="#10B981" if "Zero" in note or "Smooth" in note else "#EF4444", font=font_axis)

    # Legend
    lx = cx + cw - 260
    ly = cy + 18
    draw.rectangle([lx, ly, lx + 16, ly + 12], fill="#3B82F6")
    draw.text((lx + 24, ly - 1), "Host RAM Resident", fill="#E2E8F0", font=font_axis)
    draw.rectangle([lx, ly + 18, lx + 16, ly + 30], fill="#10B981")
    draw.text((lx + 24, ly + 17), "RTX 3070 Ti VRAM", fill="#E2E8F0", font=font_axis)
    draw.rectangle([lx, ly + 36, lx + 16, ly + 48], fill="#EF4444")
    draw.text((lx + 24, ly + 35), "NVMe Disk Swap (Thrash)", fill="#E2E8F0", font=font_axis)

    im.save("assets/memory_swap_envelope.png")
    print("[✓] Saved assets/memory_swap_envelope.png")

if __name__ == "__main__":
    create_throughput_chart()
    create_retention_chart()
    create_memory_chart()
