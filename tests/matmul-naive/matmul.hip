#include <kittens.hpp>

using namespace kittens;

using T = bf16;

constexpr int NUM_THREADS = 512;
constexpr int BLOCK_SIZE_M = 128;
constexpr int BLOCK_SIZE_N = 128;
constexpr int BLOCK_SIZE_K = 16;

__global__ __launch_bounds__(NUM_THREADS) void gpu_matmul_ABt_ker(T *A, T *B, T *C, int M, int N, int K)
{
    int tile_m = blockIdx.x;
    int tile_n = blockIdx.y;

    for (int elem_idx = threadIdx.x; elem_idx < BLOCK_SIZE_M * BLOCK_SIZE_N; elem_idx += NUM_THREADS)
    {
        int elem_n = elem_idx % BLOCK_SIZE_N;
        int elem_m = elem_idx / BLOCK_SIZE_N;

        int real_m = tile_m * BLOCK_SIZE_M + elem_m;
        int real_n = tile_n * BLOCK_SIZE_N + elem_n;

        if (real_m >= M || real_n >= N)
            continue;

        float sum = 0;
        for (int k = 0; k < K; k++)
        {
            T &a_elem = A[real_m * K + k];
            // Since B is column-major, we need to stride by K. row-major would be B[k * N + real_n]
            T &b_elem = B[real_n * K + k];
            sum += a_elem * b_elem;
        }
        T &c_elem = C[real_m * N + real_n];
        c_elem = static_cast<T>(sum);
    }
}

void gpu_matmul_ABt(T *A, T *B, T *C, int M, int N, int K, std::vector<T> &h_C)
{
    dim3 block(NUM_THREADS);
    dim3 grid((M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N);
    float ms = 0;

    // warmup kernel
    gpu_matmul_ABt_ker<<<grid, block>>>(A, B, C, M, N, K);

    constexpr int num_iters = 5;
    for (int i = 0; i < num_iters; i++)
    {
        kernel_timer t(&ms, 1.0f / num_iters);
        gpu_matmul_ABt_ker<<<grid, block>>>(A, B, C, M, N, K);
    }

    int flops = 2 * M * N * K;
    float gflops = flops / (ms * 1e3);
    std::cout << "GFLOPS: " << gflops << std::endl;

    hipCheck(hipMemcpy(h_C.data(), C, M * N * sizeof(T), hipMemcpyDeviceToHost));
}

int main()
{
    int M = 102;
    int N = 1024;
    int K = 10244;

    auto [h_A, d_A] = init_random<T>(M * K);
    auto [h_B, d_B] = init_random<T>(K * N);
    auto [h_C, d_C] = init_random<T>(M * N);

    auto h_C_ref = h_C;
    cpu_matmul<T, /* A */ false, /* B.T */ true>(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);
    gpu_matmul_ABt(d_A, d_B, d_C, M, N, K, h_C);

    assert_equal(h_C_ref, h_C);

    hipCheck(hipFree(d_A));
    hipCheck(hipFree(d_B));
    hipCheck(hipFree(d_C));

    return 0;
}