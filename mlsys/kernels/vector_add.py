import torch
import triton
import triton.language as tl
import argparse
import matplotlib.pyplot as plt

@triton.jit 
def vadd_kernel(
    x_ptr, y_ptr, out_ptr, 
    n_elements, BLOCK_SIZE: tl.constexpr
): 
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements 
    x_chunk = tl.load(x_ptr + offsets, mask=mask); x_chunk.to(tl.float16)
    y_chunk = tl.load(y_ptr + offsets, mask=mask); y_chunk.to(tl.float16)
    tl.store(out_ptr + offsets, x_chunk + y_chunk, mask=mask)

def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: 
    assert x.is_cuda and y.is_cuda 
    assert x.shape == y.shape 
    out = torch.empty_like(x)
    nel = x.numel()
    grid = lambda meta: (triton.cdiv(nel, meta['BLOCK_SIZE']),)
    vadd_kernel[grid](x, y, out, nel, BLOCK_SIZE=512)
    return out 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vector addition kernel')
    parser.add_argument('--bench', action='store_true', help='Run benchmark')
    args = parser.parse_args()
    
    torch.manual_seed(0)
    n = 10_000_000
    x = torch.rand(n, device='cuda')
    y = torch.rand(n, device='cuda')
    
    out_triton = vector_add(x, y)
    out_torch = x + y
    assert torch.allclose(out_triton, out_torch), "Results don't match!"
    print(f"Correctness test passed! Max diff: {(out_triton - out_torch).abs().max():.2e}")
    
    ms_triton = triton.testing.do_bench(lambda: vector_add(x, y))
    ms_torch = triton.testing.do_bench(lambda: x + y)
    print(f"Triton: {ms_triton:.4f} ms, PyTorch: {ms_torch:.4f} ms")
    
    flops = n
    tflops_triton = (flops / (ms_triton * 1e-3)) / 1e12
    tflops_torch = (flops / (ms_torch * 1e-3)) / 1e12
    print(f"Peak TFLOPS - Triton: {tflops_triton:.4f}, PyTorch: {tflops_torch:.4f}")
    
    bytes_moved = n * 4 * 3
    gbps_triton = (bytes_moved / (ms_triton * 1e-3)) / 1e9
    gbps_torch = (bytes_moved / (ms_torch * 1e-3)) / 1e9
    print(f"Memory BW - Triton: {gbps_triton:.2f} GB/s, PyTorch: {gbps_torch:.2f} GB/s")
    
    if args.bench:
        vec_sizes = [2**i for i in range(10, 28, 2)]
        triton_times = []
        torch_times = []
        
        print("\nBenchmarking across sizes...")
        for size in vec_sizes:
            x = torch.rand(size, device='cuda')
            y = torch.rand(size, device='cuda')
            
            ms_t = triton.testing.do_bench(lambda: vector_add(x, y))
            ms_p = triton.testing.do_bench(lambda: x + y)
            triton_times.append(ms_t)
            torch_times.append(ms_p)
            print(f"Size {size:10d}: Triton {ms_t:.4f} ms, PyTorch {ms_p:.4f} ms")
        
        plt.figure(figsize=(10, 6))
        plt.loglog(vec_sizes, torch_times, 'o-', label='PyTorch', linewidth=2, markersize=6)
        plt.loglog(vec_sizes, triton_times, 's-', label='Triton', linewidth=2, markersize=6)
        plt.xlabel('Vector Size')
        plt.ylabel('Time (ms)')
        plt.title('Vector Addition Benchmark: PyTorch vs Triton')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('vector_add_bench.png', dpi=300, bbox_inches='tight')
        print("Benchmark plot saved as 'vector_add_bench.png'")
