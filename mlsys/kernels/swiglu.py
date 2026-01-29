import torch
import triton
import triton.language as tl
import argparse
import matplotlib.pyplot as plt


@triton.jit
def swiglu_kernel(
    x_ptr, gate_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements 
    
    x_block = tl.load(x_ptr + offsets, mask=mask)
    gate_block = tl.load(gate_ptr + offsets, mask=mask)

    sig_block = 1.0 / (1.0 + tl.exp(-gate_block))
    tl.store(out_ptr + offsets, x_block * gate_block * sig_block, mask=mask)


def swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and gate.is_cuda
    assert x.shape == gate.shape
    
    n_elements = x.numel()
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    swiglu_kernel[grid](x, gate, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def swiglu_torch(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return x * torch.nn.functional.silu(gate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SwiGLU kernel')
    parser.add_argument('--bench', action='store_true', help='Run benchmark')
    args = parser.parse_args()
    
    torch.manual_seed(42)
    
    test_shapes = [
        (128,),
        (1024, 1024),
        (4, 16, 512),
    ]
    
    for shape in test_shapes:
        x = torch.randn(shape, device='cuda')
        gate = torch.randn(shape, device='cuda')
        
        out_triton = swiglu(x, gate)
        out_torch = swiglu_torch(x, gate)
        
        if torch.allclose(out_triton, out_torch, atol=1e-5):
            print(f"shape={shape}: PASSED")
        else:
            print(f"shape={shape}: FAILED - max diff: {(out_triton - out_torch).abs().max():.2e}")
    
    print("\n--- Benchmark (batch=4, seq=512, hidden=1024) ---")
    x = torch.randn(4, 512, 1024, device='cuda')
    gate = torch.randn(4, 512, 1024, device='cuda')
    
    ms_triton = triton.testing.do_bench(lambda: swiglu(x, gate))
    ms_torch = triton.testing.do_bench(lambda: swiglu_torch(x, gate))
    print(f"Triton: {ms_triton:.4f} ms, PyTorch: {ms_torch:.4f} ms")
    
    n_elements = x.numel()
    flops = n_elements * 12
    tflops_triton = (flops / (ms_triton * 1e-3)) / 1e12
    tflops_torch = (flops / (ms_torch * 1e-3)) / 1e12
    print(f"TFLOPS - Triton: {tflops_triton:.4f}, PyTorch: {tflops_torch:.4f}")
    
    bytes_moved = n_elements * 4 * 3
    gbps_triton = (bytes_moved / (ms_triton * 1e-3)) / 1e9
    gbps_torch = (bytes_moved / (ms_torch * 1e-3)) / 1e9
    print(f"Memory BW - Triton: {gbps_triton:.2f} GB/s, PyTorch: {gbps_torch:.2f} GB/s")
    
    if args.bench:
        test_configs = [
            (2, 256, 512),
            (4, 512, 1024),
            (8, 1024, 2048),
            (16, 2048, 4096),
        ]
        triton_times, torch_times = [], []
        
        print("\nBenchmarking across configs...")
        for b, s, h in test_configs:
            x = torch.randn(b, s, h, device='cuda')
            gate = torch.randn(b, s, h, device='cuda')
            
            ms_t = triton.testing.do_bench(lambda: swiglu(x, gate))
            ms_p = triton.testing.do_bench(lambda: swiglu_torch(x, gate))
            triton_times.append(ms_t)
            torch_times.append(ms_p)
            print(f"(B={b}, S={s}, H={h}): Triton {ms_t:.4f} ms, PyTorch {ms_p:.4f} ms")
        
        labels = [f"B={b},S={s},H={h}" for b, s, h in test_configs]
        x_pos = range(len(test_configs))
        
        plt.figure(figsize=(10, 6))
        plt.bar([p - 0.2 for p in x_pos], torch_times, 0.4, label='PyTorch', alpha=0.8)
        plt.bar([p + 0.2 for p in x_pos], triton_times, 0.4, label='Triton', alpha=0.8)
        plt.xlabel('Configuration (batch, seq, hidden)')
        plt.ylabel('Time (ms)')
        plt.title('SwiGLU Benchmark: PyTorch vs Triton')
        plt.xticks(x_pos, labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('swiglu_bench.png', dpi=300, bbox_inches='tight')
        print("Benchmark plot saved as 'swiglu_bench.png'")
