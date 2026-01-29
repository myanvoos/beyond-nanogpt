import torch
import triton
import triton.language as tl
import argparse
import matplotlib.pyplot as plt


@triton.jit
def reverse_array_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    el = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + (n_elements - 1 - offsets), el, mask=mask)


def reverse_array(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.ndim == 1, "Expected 1D tensor"
    
    n_elements = x.numel()
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    reverse_array_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def reverse_array_torch(x: torch.Tensor) -> torch.Tensor:
    return x.flip(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reverse array kernel')
    parser.add_argument('--bench', action='store_true', help='Run benchmark')
    args = parser.parse_args()
    
    torch.manual_seed(42)
    
    for n in [128, 1000, 10_000, 1_000_000]:
        x = torch.rand(n, device='cuda')
        
        out_triton = reverse_array(x)
        out_torch = reverse_array_torch(x)
        
        if torch.allclose(out_triton, out_torch):
            print(f"n={n}: PASSED")
        else:
            print(f"n={n}: FAILED - max diff: {(out_triton - out_torch).abs().max():.2e}")
    
    print("\n--- Benchmark (n=10M) ---")
    n = 10_000_000
    x = torch.rand(n, device='cuda')
    
    ms_triton = triton.testing.do_bench(lambda: reverse_array(x))
    ms_torch = triton.testing.do_bench(lambda: reverse_array_torch(x))
    print(f"Triton: {ms_triton:.4f} ms, PyTorch: {ms_torch:.4f} ms")
    
    bytes_moved = n * 4 * 2
    gbps_triton = (bytes_moved / (ms_triton * 1e-3)) / 1e9
    gbps_torch = (bytes_moved / (ms_torch * 1e-3)) / 1e9
    print(f"Memory BW - Triton: {gbps_triton:.2f} GB/s, PyTorch: {gbps_torch:.2f} GB/s")
    
    if args.bench:
        sizes = [2**i for i in range(10, 26, 2)]
        triton_times, torch_times = [], []
        
        print("\nBenchmarking across sizes...")
        for size in sizes:
            x = torch.rand(size, device='cuda')
            ms_t = triton.testing.do_bench(lambda: reverse_array(x))
            ms_p = triton.testing.do_bench(lambda: reverse_array_torch(x))
            triton_times.append(ms_t)
            torch_times.append(ms_p)
            print(f"n={size:10d}: Triton {ms_t:.4f} ms, PyTorch {ms_p:.4f} ms")
        
        plt.figure(figsize=(10, 6))
        plt.loglog(sizes, torch_times, 'o-', label='PyTorch', linewidth=2, markersize=6)
        plt.loglog(sizes, triton_times, 's-', label='Triton', linewidth=2, markersize=6)
        plt.xlabel('Array Size')
        plt.ylabel('Time (ms)')
        plt.title('Reverse Array Benchmark: PyTorch vs Triton')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('reverse_array_bench.png', dpi=300, bbox_inches='tight')
        print("Benchmark plot saved as 'reverse_array_bench.png'")
