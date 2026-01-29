import torch
import triton
import triton.language as tl
import argparse
import matplotlib.pyplot as plt


@triton.jit 
def ln_fwd_kernel(x_ptr, out_ptr, gamma_ptr, beta_ptr, eps, nrows, ncols, BLOCK_SIZE: tl.constexpr) -> torch.Tensor: 
    pid = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    mask = col_offsets < ncols
    x_row = tl.load(x_ptr + pid * ncols + col_offsets, mask=mask, other=0.0)
    
    mean = tl.sum(x_row, axis=0) / ncols
    x_ctr = x_row - mean 
    var = tl.sum(x_ctr * x_ctr, axis=0) / ncols 
    rstd = 1.0 / tl.sqrt(var + eps)

    x_norm = x_ctr * rstd 

    gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + col_offsets, mask=mask, other=0.0)

    out = gamma * x_norm + beta 
    tl.store(out_ptr + pid * ncols + col_offsets, out, mask=mask)

def ln_fwd(
    x: torch.Tensor, 
    gamma: torch.Tensor, 
    beta: torch.Tensor,
    eps: float = 1e-5, 
) -> torch.Tensor: 
    assert len(x.shape) == 2, "ERROR: Please reshape to 2D before passing into LN fwd..."
    assert x.is_cuda 

    nrows, ncols = x.shape 
    out = torch.empty_like(x)
    grid = (nrows,)

    BLOCK_SIZE = triton.next_power_of_2(ncols)
    ln_fwd_kernel[grid](x, out, gamma, beta, eps, nrows, ncols, BLOCK_SIZE=BLOCK_SIZE)
    return out 


def layernorm_torch(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True, correction=0.)
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LayerNorm kernel')
    parser.add_argument('--bench', action='store_true', help='Run benchmark')
    args = parser.parse_args()
    
    torch.manual_seed(0)
    batch, hidden = 1024, 2048
    x = torch.rand(batch, hidden, device='cuda')
    gamma = torch.ones(hidden, device='cuda')
    beta = torch.zeros(hidden, device='cuda')
    
    out_triton = ln_fwd(x, gamma, beta)
    out_torch = layernorm_torch(x, gamma, beta)
    assert torch.allclose(out_triton, out_torch, atol=1e-5), f"Max diff: {(out_triton - out_torch).abs().max()}"
    print(f"Correctness test passed! Max diff: {(out_triton - out_torch).abs().max():.2e}")
    
    ms_triton = triton.testing.do_bench(lambda: ln_fwd(x, gamma, beta))
    ms_torch = triton.testing.do_bench(lambda: layernorm_torch(x, gamma, beta))
    print(f"Triton: {ms_triton:.4f} ms, PyTorch: {ms_torch:.4f} ms")
    
    flops = batch * hidden * 5
    tflops_triton = (flops / (ms_triton * 1e-3)) / 1e12
    tflops_torch = (flops / (ms_torch * 1e-3)) / 1e12
    print(f"Peak TFLOPS - Triton: {tflops_triton:.4f}, PyTorch: {tflops_torch:.4f}")
    
    bytes_moved = (batch * hidden * 2 + hidden * 2) * 4
    gbps_triton = (bytes_moved / (ms_triton * 1e-3)) / 1e9
    gbps_torch = (bytes_moved / (ms_torch * 1e-3)) / 1e9
    print(f"Memory BW - Triton: {gbps_triton:.2f} GB/s, PyTorch: {gbps_torch:.2f} GB/s")
    
    if args.bench:
        test_configs = [
            (256, 512),
            (512, 1024),
            (1024, 2048),
            (2048, 4096),
            (4096, 8192),
        ]
        triton_times, torch_times = [], []
        
        print("\nBenchmarking across configs...")
        for b, h in test_configs:
            x = torch.rand(b, h, device='cuda')
            g = torch.ones(h, device='cuda')
            bt = torch.zeros(h, device='cuda')
            
            ms_t = triton.testing.do_bench(lambda: ln_fwd(x, g, bt))
            ms_p = triton.testing.do_bench(lambda: layernorm_torch(x, g, bt))
            triton_times.append(ms_t)
            torch_times.append(ms_p)
            print(f"(B={b}, H={h}): Triton {ms_t:.4f} ms, PyTorch {ms_p:.4f} ms")
        
        labels = [f"{b}x{h}" for b, h in test_configs]
        x_pos = range(len(test_configs))
        
        plt.figure(figsize=(10, 6))
        plt.bar([p - 0.2 for p in x_pos], torch_times, 0.4, label='PyTorch', alpha=0.8)
        plt.bar([p + 0.2 for p in x_pos], triton_times, 0.4, label='Triton', alpha=0.8)
        plt.xlabel('Matrix Size (batch x hidden)')
        plt.ylabel('Time (ms)')
        plt.title('LayerNorm Benchmark: PyTorch vs Triton')
        plt.xticks(x_pos, labels)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('layernorm_bench.png', dpi=300, bbox_inches='tight')
        print("Benchmark plot saved as 'layernorm_bench.png'")
