import torch
import triton
import triton.language as tl
import argparse
import matplotlib.pyplot as plt

@triton.jit 
def conv1d_kernel(x_ptr, w_ptr, out_ptr, batch, in_len, out_len, k, BLOCK_SIZE: tl.constexpr): 
    pid = tl.program_id(0)
    batch_idx = pid // out_len 
    out_idx = pid % out_len

    k_offsets = tl.arange(0, BLOCK_SIZE)
    k_mask = k_offsets < k 
    weights = tl.load(w_ptr + k_offsets, mask=k_mask, other=0.)
    
    x_offsets = out_idx + k_offsets
    x_window = tl.load(x_ptr + batch_idx * in_len + x_offsets, mask=k_mask, other=0.) 

    acc = tl.sum(x_window * weights, axis=0)
    tl.store(out_ptr + batch_idx * out_len + out_idx, acc)

def conv1d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda 
    batch, in_len = x.shape 
    k = w.shape[0]
    out_len = in_len - k + 1
    out = torch.empty((batch, out_len), device=x.device, dtype=x.dtype)

    grid = (batch * out_len,) 
    BLOCK_SIZE = triton.next_power_of_2(k)
    conv1d_kernel[grid](x, w, out, batch, in_len, out_len, k, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def conv1d_tiled_kernel(
    x_ptr, w_ptr, out_ptr,
    batch, in_len, out_len, kernel_size,
    BLOCK_OUT: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_out_tiles = tl.cdiv(out_len, BLOCK_OUT)
    batch_idx = pid // num_out_tiles
    tile_idx = pid % num_out_tiles
    
    out_start = tile_idx * BLOCK_OUT
    out_offsets = tl.arange(0, BLOCK_OUT)
    out_mask = (out_start + out_offsets) < out_len
    
    acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
    
    for k in tl.static_range(BLOCK_K):
        if k < kernel_size:
            w_k = tl.load(w_ptr + k)
            x_offs = out_start + out_offsets + k
            x_mask = x_offs < in_len
            x_vals = tl.load(x_ptr + batch_idx * in_len + x_offs, mask=x_mask, other=0.0)
            acc += x_vals * w_k
    
    tl.store(out_ptr + batch_idx * out_len + out_start + out_offsets, acc, mask=out_mask)


def conv1d_tiled(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda
    batch, in_len = x.shape
    k = w.shape[0]
    out_len = in_len - k + 1
    out = torch.empty((batch, out_len), device=x.device, dtype=x.dtype)
    
    BLOCK_OUT = 128
    BLOCK_K = triton.next_power_of_2(k)
    num_out_tiles = triton.cdiv(out_len, BLOCK_OUT)
    grid = (batch * num_out_tiles,)
    conv1d_tiled_kernel[grid](x, w, out, batch, in_len, out_len, k, BLOCK_OUT=BLOCK_OUT, BLOCK_K=BLOCK_K)
    return out 


def conv1d_torch(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.conv1d(
        x.unsqueeze(1), weight.view(1, 1, -1)
    ).squeeze(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conv1D kernel')
    parser.add_argument('--bench', action='store_true', help='Run benchmark')
    args = parser.parse_args()
    
    torch.manual_seed(0)
    batch, in_len, kernel_size = 64, 4096, 32
    x = torch.rand(batch, in_len, device='cuda')
    weight = torch.rand(kernel_size, device='cuda')
    
    out_torch = conv1d_torch(x, weight)
    
    out_naive = conv1d(x, weight)
    assert torch.allclose(out_naive, out_torch, atol=1e-5), f"Naive max diff: {(out_naive - out_torch).abs().max()}"
    print(f"Naive correctness: PASSED (max diff: {(out_naive - out_torch).abs().max():.2e})")
    
    out_tiled = conv1d_tiled(x, weight)
    assert torch.allclose(out_tiled, out_torch, atol=1e-5), f"Tiled max diff: {(out_tiled - out_torch).abs().max()}"
    print(f"Tiled correctness: PASSED (max diff: {(out_tiled - out_torch).abs().max():.2e})")
    
    ms_naive = triton.testing.do_bench(lambda: conv1d(x, weight))
    ms_tiled = triton.testing.do_bench(lambda: conv1d_tiled(x, weight))
    ms_torch = triton.testing.do_bench(lambda: conv1d_torch(x, weight))
    
    print(f"\n--- Benchmark ---")
    print(f"Naive:  {ms_naive:.4f} ms")
    print(f"Tiled:  {ms_tiled:.4f} ms")
    print(f"PyTorch: {ms_torch:.4f} ms")
    
    out_len = in_len - kernel_size + 1
    flops = batch * out_len * kernel_size * 2
    print(f"\n--- TFLOPS ---")
    print(f"Naive:  {(flops / (ms_naive * 1e-3)) / 1e12:.4f}")
    print(f"Tiled:  {(flops / (ms_tiled * 1e-3)) / 1e12:.4f}")
    print(f"PyTorch: {(flops / (ms_torch * 1e-3)) / 1e12:.4f}")
    
    bytes_moved = (batch * in_len + kernel_size + batch * out_len) * 4
    print(f"\n--- Memory BW (GB/s) ---")
    print(f"Naive:  {(bytes_moved / (ms_naive * 1e-3)) / 1e9:.2f}")
    print(f"Tiled:  {(bytes_moved / (ms_tiled * 1e-3)) / 1e9:.2f}")
    print(f"PyTorch: {(bytes_moved / (ms_torch * 1e-3)) / 1e9:.2f}")
    
    if args.bench:
        test_configs = [
            (16, 1024, 8),
            (32, 2048, 16),
            (64, 4096, 32),
            (128, 8192, 64),
            (256, 16384, 128),
        ]
        naive_times, tiled_times, torch_times = [], [], []
        
        print("\nBenchmarking across configs...")
        for b, l, k in test_configs:
            x = torch.rand(b, l, device='cuda')
            w = torch.rand(k, device='cuda')
            
            ms_n = triton.testing.do_bench(lambda: conv1d(x, w))
            ms_t = triton.testing.do_bench(lambda: conv1d_tiled(x, w))
            ms_p = triton.testing.do_bench(lambda: conv1d_torch(x, w))
            naive_times.append(ms_n)
            tiled_times.append(ms_t)
            torch_times.append(ms_p)
            print(f"(B={b}, L={l}, K={k}): Naive {ms_n:.4f}, Tiled {ms_t:.4f}, PyTorch {ms_p:.4f} ms")
        
        labels = [f"B={b},L={l},K={k}" for b, l, k in test_configs]
        x_pos = range(len(test_configs))
        
        plt.figure(figsize=(12, 6))
        width = 0.25
        plt.bar([p - width for p in x_pos], torch_times, width, label='PyTorch', alpha=0.8)
        plt.bar([p for p in x_pos], naive_times, width, label='Triton Naive', alpha=0.8)
        plt.bar([p + width for p in x_pos], tiled_times, width, label='Triton Tiled', alpha=0.8)
        plt.xlabel('Configuration')
        plt.ylabel('Time (ms)')
        plt.title('Conv1D Benchmark: PyTorch vs Triton')
        plt.xticks(x_pos, labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('conv1d_bench.png', dpi=300, bbox_inches='tight')
        print("Benchmark plot saved as 'conv1d_bench.png'")
