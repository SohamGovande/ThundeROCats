#include <kittens.hpp>
#include <array>
#include <rocwmma/rocwmma.hpp>

using namespace kittens;

using T = bf16;
using mfma_float2 = __attribute__((__vector_size__(4 * sizeof(short)))) short;
using mfma_bfloat8 = __attribute__((__vector_size__(8 * sizeof(short)))) short;
using mfma_float8 = __attribute__((__vector_size__(8 * sizeof(float)))) float;
using mfma_float16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using bf16_2 = std::array<bf16, 2>;
using bf16_4 = std::array<bf16, 4>;
using bf16_8 = std::array<bf16, 8>;
using bf16_16 = std::array<bf16, 16>;

constexpr int NUM_WAVES = 1;

constexpr int NUM_THREADS_PER_WAVE = 64;
constexpr int NUM_THREADS = NUM_WAVES * NUM_THREADS_PER_WAVE;

constexpr int BLOCK_SIZE_M = 64;
constexpr int BLOCK_SIZE_N = 64;
constexpr int BLOCK_SIZE_K = 8;

constexpr int MMA_ATOM_SIZE_M = 32;
constexpr int MMA_ATOM_SIZE_N = 32;
constexpr int MMA_ATOM_SIZE_K = 8;

constexpr int WAVE_TILE_SIZE_M = BLOCK_SIZE_M / MMA_ATOM_SIZE_M;
constexpr int WAVE_TILE_SIZE_N = BLOCK_SIZE_N / MMA_ATOM_SIZE_N;
constexpr int WAVE_TILE_SIZE_K = BLOCK_SIZE_K / MMA_ATOM_SIZE_K;

__device__ inline void load(T *global, int global_rows, int global_cols, T *reg, int tile_row_idx, int tile_col_idx)
{
    constexpr int tile_size_rows = 32;
    constexpr int tile_size_cols = 8;

    static_assert(NUM_THREADS_PER_WAVE % tile_size_rows == 0, "NUM_THREADS_PER_WAVE must be divisible by tile_size_rows");
    constexpr int contiguous_elements_to_load = tile_size_cols / (NUM_THREADS_PER_WAVE / tile_size_rows);

    static_assert(contiguous_elements_to_load == 4, "TEST FAILED: contiguous_elements_to_load != 4");

    int stride_bw_rows = global_cols;
    using T2 = std::array<T, contiguous_elements_to_load>;

    int laneid = threadIdx.x % NUM_THREADS_PER_WAVE;

    int row_win_tile = laneid % tile_size_rows;
    int col_win_tile = (laneid / tile_size_rows) * contiguous_elements_to_load;

    auto global_ptr = &global[
        // Row
        (tile_row_idx * tile_size_rows + row_win_tile) * stride_bw_rows +
        // Col
        (tile_col_idx * tile_size_cols + col_win_tile)];

    // Assume both global and reg are row-major
    *reinterpret_cast<T2 *>(reg) = *reinterpret_cast<T2 *>(global_ptr);
}

__device__ inline void store(T *global, int global_rows, int global_cols, T *reg, int tile_row_idx, int tile_col_idx)
{
    constexpr int tile_size_rows = 32;
    constexpr int tile_size_cols = 32;

    static_assert(NUM_THREADS_PER_WAVE % tile_size_rows == 0, "NUM_THREADS_PER_WAVE must be divisible by tile_size_rows");
    constexpr int contiguous_elements_to_store = tile_size_cols / (NUM_THREADS_PER_WAVE / tile_size_rows);
    constexpr bool should_swizzle_4_slices = true;

    int laneid = threadIdx.x % NUM_THREADS_PER_WAVE;

    int row_win_tile = laneid % tile_size_rows;
    int col_win_tile = (laneid / tile_size_rows) * contiguous_elements_to_store;

    int global_row_start = tile_row_idx * tile_size_rows;
    int global_col_start = tile_col_idx * tile_size_cols;

#pragma unroll
    for (int i = 0; i < contiguous_elements_to_store; i++)
    {
        // Assume RT is col-major and global is row-major -- here, we use the (col, row) coordinates from within the tile instead of (row, col)
        int global_row_offset = col_win_tile + i;
        int global_col_offset = row_win_tile;

        if (should_swizzle_4_slices)
        {
            int row_group_idx = global_row_offset / 4;
            int row_win_tile_group = global_row_offset % 4;
            // Express the following transformation: Let i = row_tile_group_idx. We need to remap i to i' such that:
            // 0 -> 0, 1 -> 4, 2 -> 1, 3 -> 5, 4 -> 2, 5 -> 6, 6 -> 3, 7 -> 7
            int row_win_tile_swizzled = (2 * (row_group_idx % 4) + (row_group_idx / 4)) * 4 + row_win_tile_group;
            global_row_offset = row_win_tile_swizzled;
        }
        global[(global_row_start + global_row_offset) * global_cols + (global_col_start + global_col_offset)] = reg[i];
    }
}

__device__ inline void mma(mfma_float16 &c_reg, bf16_4 const &a_reg, bf16_4 const &b_reg)
{

    c_reg = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(
        *reinterpret_cast<__attribute__((__vector_size__(4 * sizeof(short)))) short const *>(&a_reg[0]),
        *reinterpret_cast<__attribute__((__vector_size__(4 * sizeof(short)))) short const *>(&b_reg[0]),
        *reinterpret_cast<__attribute__((__vector_size__(16 * sizeof(float)))) float *>(&c_reg), 0, 0, 0);
}

__global__ __launch_bounds__(NUM_THREADS) void gpu_matmul_ABt_ker(T *A, T *B, T *C, int M, int N, int K)
{

    mfma_float16 c_reg[WAVE_TILE_SIZE_M][WAVE_TILE_SIZE_N];

    // Zero the accumulators
    for (int i = 0; i < WAVE_TILE_SIZE_M; i++)
        for (int j = 0; j < WAVE_TILE_SIZE_N; j++)
            c_reg[i][j] = {0};

    // GEMM mainloop
    for (int k_block = 0; k_block < K; k_block += BLOCK_SIZE_K)
    {
        bf16_4 a_reg[WAVE_TILE_SIZE_M], b_reg[WAVE_TILE_SIZE_N];

        for (int i = 0; i < WAVE_TILE_SIZE_M; i++)
            load(A, M, K, &a_reg[i][0], i, k_block / BLOCK_SIZE_K);
        for (int i = 0; i < WAVE_TILE_SIZE_N; i++)
            load(B, N, K, &b_reg[i][0], i, k_block / BLOCK_SIZE_K);

        for (int i = 0; i < WAVE_TILE_SIZE_M; i++)
            for (int j = 0; j < WAVE_TILE_SIZE_N; j++)
                mma(c_reg[i][j], a_reg[i], b_reg[j]);
    }

    // Convert the accumulators to bf16
    bf16_16 c_reg_half[WAVE_TILE_SIZE_M][WAVE_TILE_SIZE_N];

    for (int i = 0; i < WAVE_TILE_SIZE_M; i++)
        for (int j = 0; j < WAVE_TILE_SIZE_N; j++)
            for (int k = 0; k < c_reg_half[i][j].size(); k++)
                c_reg_half[i][j][k] = static_cast<bf16>(c_reg[i][j][k]);

    // Store the accumulators to global memory
    for (int i = 0; i < WAVE_TILE_SIZE_M; i++)
        for (int j = 0; j < WAVE_TILE_SIZE_N; j++)
            store(C, M, N, &c_reg_half[i][j][0], i, j);
}

void gpu_matmul_ABt(T *A, T *B, T *C, int M, int N, int K, std::vector<T> &h_C)
{
    dim3 block(NUM_THREADS);
    dim3 grid((M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N);
    std::cout << "Problem Shape: (" << M << ", " << N << ", " << K << ")" << std::endl;
    std::cout << "Launching with grid (" << grid.x << ", " << grid.y << ", " << grid.z << ") with block (" << block.x << ", " << block.y << ", " << block.z << ")" << std::endl;
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
    int M = BLOCK_SIZE_M;
    int N = BLOCK_SIZE_N;
    int K = BLOCK_SIZE_K * 8;

    auto [h_A, d_A] = init<fill_random, T>(M * K);
    auto [h_B, d_B] = init<fill_random, T>(K * N);
    auto [h_C, d_C] = init<fill_zeros, T>(M * N);

    auto h_C_ref = h_C;
    cpu_matmul<T, /* A */ false, /* B.T */ true>(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);
    gpu_matmul_ABt(d_A, d_B, d_C, M, N, K, h_C);

    assert_equal(h_C_ref, h_C);

    print_tensor_to_file<bf16>("matmul.csv", {{"A", h_A.data(), M, K}, {"B", h_B.data(), N, K}, {"C_ref", h_C_ref.data(), M, N}, {"C", h_C.data(), M, N}});

    hipCheck(hipFree(d_A));
    hipCheck(hipFree(d_B));
    hipCheck(hipFree(d_C));

    return 0;
}