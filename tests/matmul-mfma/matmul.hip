#include <kittens.hpp>
#include <array>
#include <rocwmma/rocwmma.hpp>

using namespace kittens;

using T = bf16;
using mfma_float2 = __attribute__((__vector_size__(4 * sizeof(short)))) short;
using mfma_float4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

constexpr int NUM_WAVES = 1;

constexpr int NUM_THREADS_PER_WAVE = 64;
constexpr int NUM_THREADS = NUM_WAVES * NUM_THREADS_PER_WAVE;
constexpr int BLOCK_SIZE_M = 16;
constexpr int BLOCK_SIZE_N = 16;
constexpr int BLOCK_SIZE_K = 16;

using bf16_4 = std::array<bf16, 4>;

__device__ inline void load(T *global, int global_rows, int global_cols, T *reg, int tile_row_idx, int tile_col_idx)
{
    using T2 = mfma_float2;

    constexpr int tile_size = 16;
    constexpr int bytes_to_load = sizeof(T2);

    int stride_bw_rows = global_cols;
    constexpr int contiguous_elements_to_load = bytes_to_load / sizeof(T);

    int laneid = threadIdx.x % NUM_THREADS_PER_WAVE;

    int row_win_tile = laneid % tile_size;
    int col_win_tile = (laneid / tile_size) * contiguous_elements_to_load;

    // Assume both global and reg are row-major
    *reinterpret_cast<T2 *>(reg) = *reinterpret_cast<T2 *>(&global[
        // Row
        (tile_row_idx * tile_size + row_win_tile) * stride_bw_rows +
        // Col
        (tile_col_idx * tile_size + col_win_tile)]);
}

__device__ inline void store(T *global, int global_rows, int global_cols, T *reg, int tile_row_idx, int tile_col_idx)
{
    using T2 = mfma_float2;
    int laneid = threadIdx.x % NUM_THREADS_PER_WAVE;

    constexpr int tile_size = 16;
    constexpr int bytes_to_store = sizeof(T2);
    constexpr int contiguous_elements_to_store = bytes_to_store / sizeof(T);

    int row_win_tile = laneid % tile_size;
    int col_win_tile = (laneid / tile_size) * contiguous_elements_to_store;

    // Assume RT is col-major and global is row-major -- need if condition to swap strides otherwise
    global[row_win_tile + (col_win_tile + 0) * global_rows] = reg[0];
    global[row_win_tile + (col_win_tile + 1) * global_rows] = reg[1];
    global[row_win_tile + (col_win_tile + 2) * global_rows] = reg[2];
    global[row_win_tile + (col_win_tile + 3) * global_rows] = reg[3];
}

__device__ inline void mma(mfma_float4 &c_reg, bf16_4 const &a_reg, bf16_4 const &b_reg)
{

    c_reg = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
        *reinterpret_cast<mfma_float2 const *>(&a_reg[0]),
        *reinterpret_cast<mfma_float2 const *>(&b_reg[0]),
        *reinterpret_cast<mfma_float4 *>(&c_reg), 0, 0, 0);
}

__global__ __launch_bounds__(NUM_THREADS) void gpu_matmul_ABt_ker(T *A, T *B, T *C, int M, int N, int K)
{
    int laneid = threadIdx.x % NUM_THREADS_PER_WAVE;
    int waveid = threadIdx.x / NUM_THREADS_PER_WAVE;

    bf16_4 a_reg;
    bf16_4 b_reg;
    mfma_float4 c_reg{0};

    load(A, M, K, &a_reg[0], 0, 0);
    load(B, K, N, &b_reg[0], 0, 0);
    mma(c_reg, a_reg, b_reg);

    bf16_4 c_reg_half{static_cast<bf16>(c_reg[0]), static_cast<bf16>(c_reg[1]), static_cast<bf16>(c_reg[2]), static_cast<bf16>(c_reg[3])};
    store(C, M, N, &c_reg_half[0], 0, 0);
}

void gpu_matmul_ABt(T *A, T *B, T *C, int M, int N, int K, std::vector<T> &h_C)
{
    dim3 block(NUM_THREADS);
    dim3 grid((M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N);
    float ms = 0;

    // warmup kernel
    gpu_matmul_ABt_ker<<<grid, block>>>(A, B, C, M, N, K);

    constexpr int num_iters = 0;
    for (int i = 0; i < num_iters; i++)
    {
        kernel_timer t(&ms, 1.0f / num_iters);
        gpu_matmul_ABt_ker<<<grid, block>>>(A, B, C, M, N, K);
    }

    int flops = 2 * M * N * K;
    float gflops = flops / (ms * 1e3);
    // std::cout << "GFLOPS: " << gflops << std::endl;

    hipCheck(hipMemcpy(h_C.data(), C, M * N * sizeof(T), hipMemcpyDeviceToHost));
}

int main()
{
    int M = 16;
    int N = 16;
    int K = 16;

    auto [h_A, d_A] = init<fill_random, T>(M * K);
    auto [h_B, d_B] = init<fill_random, T>(K * N);
    auto [h_C, d_C] = init<fill_zeros, T>(M * N);

    auto h_C_ref = h_C;
    cpu_matmul<T, /* A */ false, /* B.T */ true>(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);
    gpu_matmul_ABt(d_A, d_B, d_C, M, N, K, h_C);

    assert_equal(h_C_ref, h_C);

    print_tensor_to_file<bf16>("matmul.csv", {{"A", h_A.data(), M, K}, {"B", h_B.data(), K, N}, {"C_ref", h_C_ref.data(), M, N}, {"C", h_C.data(), M, N}});

    hipCheck(hipFree(d_A));
    hipCheck(hipFree(d_B));
    hipCheck(hipFree(d_C));

    return 0;
}