import torch
import triton
import triton.language as tl
import argparse
import matplotlib.pyplot as plt


@triton.jit
def tiled_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_a_row, stride_a_col,
    stride_b_row, stride_b_col,
    stride_c_row, stride_c_col,
    BLOCK_SIZE: tl.constexpr,
):
    n_tiles = tl.cdiv(K, BLOCK_SIZE)
    pid = tl.program_id(0)

    pid_m = pid // tl.cdiv(N, BLOCK_SIZE)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE) 

    a_ptr += pid_m * BLOCK_SIZE * stride_a_row 
    b_ptr += pid_n * BLOCK_SIZE * stride_b_col 
    c_ptr += pid_m * BLOCK_SIZE * stride_c_row + pid_n * BLOCK_SIZE * stride_c_col

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for i in range(n_tiles): 
        b_row_mask = (tl.arange(0, BLOCK_SIZE) + (i * BLOCK_SIZE))[:, None] < K 
        b_col_mask = (pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))[None, :] < N 
        mask_b = b_row_mask & b_col_mask
        b_rows_indices = tl.arange(0, BLOCK_SIZE)[:, None] + (i * BLOCK_SIZE)
        b_cols_indices = tl.arange(0, BLOCK_SIZE)[None, :]
        offsets_b = b_rows_indices * stride_b_row + b_cols_indices * stride_b_col

        a_row_mask = (pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))[:, None] < M 
        a_col_mask = ((i * BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE))[None, :] < K
        mask_a = a_row_mask & a_col_mask
        a_cols_indices = tl.arange(0, BLOCK_SIZE)[None, :] + (i * BLOCK_SIZE)
        a_rows_indices = tl.arange(0, BLOCK_SIZE)[:, None]
        offsets_a = a_cols_indices * stride_a_col + a_rows_indices * stride_a_row

        a_tile = tl.load(a_ptr + offsets_a, mask=mask_a, other=0.)
        b_tile = tl.load(b_ptr + offsets_b, mask=mask_b, other=0.)
        acc += tl.dot(a_tile, b_tile)
    
    offsets_c = tl.arange(0, BLOCK_SIZE)[:, None] * stride_c_row + tl.arange(0, BLOCK_SIZE)[None, :] * stride_c_col
    c_row_mask = (pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))[:, None] < M 
    c_col_mask = (pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))[None, :] < N 
    mask_c = c_row_mask & c_col_mask
    tl.store(c_ptr + offsets_c, acc.to(tl.bfloat16), mask=mask_c)

def tiled_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K_check, N = b.shape
    assert K == K_check
    
    BLOCK_SIZE = 128 
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    
    num_tiles_m = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_tiles_n = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_tiles_m * num_tiles_n,)
    
    tiled_gemm_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return c


def tiled_gemm_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.to(torch.bfloat16)
    b = b.to(torch.bfloat16)
    return (a @ b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tiled GEMM kernel')
    parser.add_argument('--bench', action='store_true', help='Run benchmark')
    args = parser.parse_args()
    
    torch.manual_seed(42)
    
    test_cases = [
        (32, 32, 32),
        (256, 256, 256),
        (2048, 2048, 2048),
        (8192, 8192, 8192),
    ]
    
    for M, K, N in test_cases:
        a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
        b = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)
        
        c_triton = tiled_gemm(a, b)
        c_torch = tiled_gemm_torch(a, b)
        
        if torch.allclose(c_triton, c_torch, atol=1e-1):
            print(f"(M={M}, K={K}, N={N}): PASSED")
        else:
            print(f"(M={M}, K={K}, N={N}): FAILED - max diff: {(c_triton - c_torch).abs().max():.2e}")
    
    print("\n--- Benchmark (last test case) ---")
    M, K, N = test_cases[-1]
    a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)
    
    ms_triton = triton.testing.do_bench(lambda: tiled_gemm(a, b))
    ms_torch = triton.testing.do_bench(lambda: tiled_gemm_torch(a, b))
    print(f"Triton: {ms_triton:.4f} ms, PyTorch: {ms_torch:.4f} ms")
    
    flops = 2 * M * N * K
    bytes_hbm = (
        a.numel() * a.element_size()
        + b.numel() * b.element_size()
        + (M * N) * 2
    )

    tflops_triton = (flops / (ms_triton * 1e-3)) / 1e12
    tflops_torch = (flops / (ms_torch * 1e-3)) / 1e12
    gbps_triton = (bytes_hbm / (ms_triton * 1e-3)) / 1e9
    gbps_torch = (bytes_hbm / (ms_torch * 1e-3)) / 1e9
    print(f"TFLOPS - Triton: {tflops_triton:.2f}, PyTorch: {tflops_torch:.2f}")
    print(f"GB/s (min HBM) - Triton: {gbps_triton:.2f}, PyTorch: {gbps_torch:.2f}")
    
    if args.bench:
        test_sizes = [
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
        ]
        triton_times, torch_times = [], []
        
        print("\nBenchmarking across sizes...")
        for M, K, N in test_sizes:
            a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
            b = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)
            
            ms_t = triton.testing.do_bench(lambda: tiled_gemm(a, b))
            ms_p = triton.testing.do_bench(lambda: tiled_gemm_torch(a, b))
            triton_times.append(ms_t)
            torch_times.append(ms_p)
            
            flops = 2 * M * N * K
            tflops_t = (flops / (ms_t * 1e-3)) / 1e12
            tflops_p = (flops / (ms_p * 1e-3)) / 1e12
            print(f"({M}x{K}x{N}): Triton {ms_t:.4f} ms ({tflops_t:.1f} TFLOPS), PyTorch {ms_p:.4f} ms ({tflops_p:.1f} TFLOPS)")
        
        labels = [f"{M}x{K}" for M, K, N in test_sizes]
        x_pos = range(len(test_sizes))
        
        plt.figure(figsize=(10, 6))
        plt.bar([p - 0.2 for p in x_pos], torch_times, 0.4, label='PyTorch', alpha=0.8)
        plt.bar([p + 0.2 for p in x_pos], triton_times, 0.4, label='Triton', alpha=0.8)
        plt.xlabel('Matrix Size (MxKxN)')
        plt.ylabel('Time (ms)')
        plt.title('Tiled GEMM Benchmark: PyTorch vs Triton')
        plt.xticks(x_pos, labels)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('tiled_gemm_bench.png', dpi=300, bbox_inches='tight')
        print("Benchmark plot saved as 'tiled_gemm_bench.png'")
