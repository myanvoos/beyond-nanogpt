import torch
import triton
import triton.language as tl
import argparse
import matplotlib.pyplot as plt


@triton.jit
def copy2d_kernel(
    a_ptr, b_ptr,
    stride_a_row, stride_a_col,
    stride_b_row, stride_b_col,
    BLOCK_SIZE: tl.constexpr,
):
    offsets_a = tl.arange(0, 32)[:, None] * stride_a_row + tl.arange(0, 32)[None, :] * stride_a_col
    offsets_b = tl.arange(0, 32)[:, None] * stride_b_row + tl.arange(0, 32)[None, :] * stride_b_col

    vals = tl.load(a_ptr + offsets_a)
    tl.store(b_ptr + offsets_b, vals)


def copy2d(a: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and a.ndim == 2
    assert a.shape[0] >= 32 and a.shape[1] >= 32
    
    b = torch.empty(32, 32, device=a.device, dtype=a.dtype)
    
    BLOCK_SIZE = 32
    grid = (1,)
    
    copy2d_kernel[grid](
        a, b,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return b


def copy2d_torch(a: torch.Tensor) -> torch.Tensor:
    return a[:32, :32].clone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='2D copy kernel')
    parser.add_argument('--bench', action='store_true', help='Run benchmark')
    args = parser.parse_args()
    
    torch.manual_seed(42)
    
    test_shapes = [
        (32, 32),
        (64, 64),
        (128, 256),
        (100, 100),
    ]
    
    for shape in test_shapes:
        a = torch.randn(shape, device='cuda')
        
        b_triton = copy2d(a)
        b_torch = copy2d_torch(a)
        
        if torch.allclose(b_triton, b_torch, atol=1e-6):
            print(f"src shape={shape}: PASSED")
        else:
            print(f"src shape={shape}: FAILED - max diff: {(b_triton - b_torch).abs().max():.2e}")
    
    print("\n--- Benchmark (copy 32x32 from 1024x1024) ---")
    a = torch.randn(1024, 1024, device='cuda')
    
    ms_triton = triton.testing.do_bench(lambda: copy2d(a))
    ms_torch = triton.testing.do_bench(lambda: copy2d_torch(a))
    print(f"Triton: {ms_triton:.4f} ms, PyTorch: {ms_torch:.4f} ms")
    
    n_elements = 32 * 32
    bytes_moved = n_elements * 4 * 2
    gbps_triton = (bytes_moved / (ms_triton * 1e-3)) / 1e9
    gbps_torch = (bytes_moved / (ms_torch * 1e-3)) / 1e9
    print(f"Memory BW - Triton: {gbps_triton:.2f} GB/s, PyTorch: {gbps_torch:.2f} GB/s")
    
    if args.bench:
        src_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
        triton_times, torch_times = [], []
        
        print("\nBenchmarking across source sizes...")
        for size in src_sizes:
            a = torch.randn(size, size, device='cuda')
            ms_t = triton.testing.do_bench(lambda: copy2d(a))
            ms_p = triton.testing.do_bench(lambda: copy2d_torch(a))
            triton_times.append(ms_t)
            torch_times.append(ms_p)
            print(f"Src {size}x{size}: Triton {ms_t:.4f} ms, PyTorch {ms_p:.4f} ms")
        
        plt.figure(figsize=(10, 6))
        plt.plot(src_sizes, torch_times, 'o-', label='PyTorch', linewidth=2, markersize=6)
        plt.plot(src_sizes, triton_times, 's-', label='Triton', linewidth=2, markersize=6)
        plt.xlabel('Source Matrix Size (NxN)')
        plt.ylabel('Time (ms)')
        plt.title('2D Copy (32x32 block) Benchmark: PyTorch vs Triton')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('copy2d_bench.png', dpi=300, bbox_inches='tight')
        print("Benchmark plot saved as 'copy2d_bench.png'")
