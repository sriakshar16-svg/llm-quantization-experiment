import time
import torch
import torch.nn.functional as F

device = torch.device("cuda:0")
print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

# Benchmark Base-3 MLP forward on GPU
in_f = 5120
out_f = 27648
num_groups = (out_f * in_f + 127) // 128
packed_size = (out_f * in_f + 4) // 5

packed = torch.randint(0, 243, (packed_size,), dtype=torch.uint8, device=device)
scales = torch.ones(num_groups, dtype=torch.float16, device=device)
x = torch.randn(1, 1, in_f, dtype=torch.float16, device=device)

# Build LUT
vals = torch.arange(256, dtype=torch.int32, device=device)
u0 = (vals % 3) - 1
u1 = ((vals // 3) % 3) - 1
u2 = ((vals // 9) % 3) - 1
u3 = ((vals // 27) % 3) - 1
u4 = ((vals // 81) % 3) - 1
lut = torch.stack([u0, u1, u2, u3, u4], dim=1).to(torch.int8)

# Warmup
torch.cuda.synchronize()
for _ in range(3):
    trits = lut[packed.long()].reshape(-1)[:in_f * out_f]
    w_grouped = trits.view(num_groups, 128).to(torch.float16) * scales.unsqueeze(-1)
    w = w_grouped.view(out_f, in_f)
    out = F.linear(x, w)
    torch.cuda.synchronize()

torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
for _ in range(10):
    trits = lut[packed.long()].reshape(-1)[:in_f * out_f]
    w_grouped = trits.view(num_groups, 128).to(torch.float16) * scales.unsqueeze(-1)
    w = w_grouped.view(out_f, in_f)
    out = F.linear(x, w)
    torch.cuda.synchronize()
t1 = time.perf_counter()
peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
print(f"[GPU Native] 1 MLP projection time: {(t1 - t0) / 10 * 1000:.2f} ms | Peak alloc: {peak:.1f} MB")

# Benchmark CPU-resident streaming
packed_cpu = packed.cpu()
scales_cpu = scales.cpu()

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    p_gpu = packed_cpu.to(device, non_blocking=False)
    s_gpu = scales_cpu.to(device, non_blocking=False)
    trits = lut[p_gpu.long()].reshape(-1)[:in_f * out_f]
    w_grouped = trits.view(num_groups, 128).to(torch.float16) * s_gpu.unsqueeze(-1)
    w = w_grouped.view(out_f, in_f)
    out = F.linear(x, w)
    torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"[CPU Unpinned Stream] 1 MLP projection time: {(t1 - t0) / 10 * 1000:.2f} ms")

# Benchmark CPU Pinned streaming
packed_pinned = packed_cpu.pin_memory()
scales_pinned = scales_cpu.pin_memory()

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    p_gpu = packed_pinned.to(device, non_blocking=True)
    s_gpu = scales_pinned.to(device, non_blocking=True)
    trits = lut[p_gpu.long()].reshape(-1)[:in_f * out_f]
    w_grouped = trits.view(num_groups, 128).to(torch.float16) * s_gpu.unsqueeze(-1)
    w = w_grouped.view(out_f, in_f)
    out = F.linear(x, w)
    torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"[CPU Pinned Stream] 1 MLP projection time: {(t1 - t0) / 10 * 1000:.2f} ms")
