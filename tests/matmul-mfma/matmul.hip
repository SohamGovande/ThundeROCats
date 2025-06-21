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
constexpr int BLOCK_SIZE_M = 32;
constexpr int BLOCK_SIZE_N = 32;
constexpr int BLOCK_SIZE_K = 8;

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

    // for (int i = 0; i < NUM_THREADS_PER_WAVE; i++)
    // {
    //     if (false)
    //         if (threadIdx.x == i)
    //         {
    //             printf("\nlaneid: %d, row_win_tile: %d, col_win_tile: %d, contiguous_elements_to_load: %d, global_start: %.3f", threadIdx.x, row_win_tile, col_win_tile, contiguous_elements_to_load, static_cast<float>(*global_ptr));
    //         }
    //     __syncthreads();
    // }

    // Assume both global and reg are row-major
    *reinterpret_cast<T2 *>(reg) = *reinterpret_cast<T2 *>(global_ptr);
}

__device__ inline void store(T *global, int global_rows, int global_cols, T *reg, int tile_row_idx, int tile_col_idx)
{
    constexpr int tile_size_rows = 32;
    constexpr int tile_size_cols = 32;

    static_assert(NUM_THREADS_PER_WAVE % tile_size_rows == 0, "NUM_THREADS_PER_WAVE must be divisible by tile_size_rows");
    constexpr int contiguous_elements_to_store = tile_size_cols / (NUM_THREADS_PER_WAVE / tile_size_rows);

    int laneid = threadIdx.x % NUM_THREADS_PER_WAVE;

    int row_win_tile = laneid % tile_size_rows;
    int col_win_tile = (laneid / tile_size_rows) * contiguous_elements_to_store;

#pragma unroll
    for (int i = 0; i < contiguous_elements_to_store; i++)
    {
        // Assume RT is col-major and global is row-major -- need if condition to swap strides otherwise
        int element_col = (col_win_tile + i);
        int element_row = row_win_tile;

        int col_tile_group_idx = element_col / 4;
        int col_win_tile_group = element_col % 4;
        // Express the following transformation: Let i = col_tile_group_idx. We need to remap i to i' such that:
        // 0 -> 0, 1 -> 4, 2 -> 1, 3 -> 5, 4 -> 2, 5 -> 6, 6 -> 3, 7 -> 7
        int col_win_tile_swizzled = (2 * (col_tile_group_idx % 4) + (col_tile_group_idx / 4)) * 4 + col_win_tile_group;
        global[element_row + col_win_tile_swizzled * global_rows] = reg[i];
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
    mfma_float16 c_reg{0};
    auto fragC = rocwmma::fragment<rocwmma::accumulator, 32, 32, 8, rocwmma::bfloat16_t>();
    auto fragAcc = rocwmma::fragment<rocwmma::accumulator, 32, 32, 8, rocwmma::float32_t>();
    rocwmma::fill_fragment(fragAcc, 0.0f);

    for (int k_block = 0; k_block < K; k_block += BLOCK_SIZE_K)
    {
        bf16_4 a_reg, b_reg;

        load(A, M, K, &a_reg[0], 0, k_block / BLOCK_SIZE_K);
        load(B, N, K, &b_reg[0], 0, k_block / BLOCK_SIZE_K);

        // auto fragA = rocwmma::fragment<rocwmma::matrix_a, 32, 32, 8, rocwmma::bfloat16_t, rocwmma::row_major>();
        // auto fragB = rocwmma::fragment<rocwmma::matrix_b, 32, 32, 8, rocwmma::bfloat16_t, rocwmma::col_major>();
        // rocwmma::load_matrix_sync(fragA, A + (0 * K + k_block), K);
        // rocwmma::load_matrix_sync(fragB, B + (0 * K + k_block), K);

        for (int i = 0; i < NUM_THREADS_PER_WAVE; i++)
        {
            if (false)
                // if (threadIdx.x == i)
                // {
                //     printf("\nlaneid: %d, a_reg: %.3f %.3f %.3f %.3f, b_reg: %.3f %.3f %.3f %.3f, a_acc: %.3f %.3f %.3f %.3f, b_acc: %.3f %.3f %.3f %.3f",
                //            threadIdx.x,
                //            static_cast<float>(a_reg[0]), static_cast<float>(a_reg[1]), static_cast<float>(a_reg[2]), static_cast<float>(a_reg[3]),
                //            static_cast<float>(b_reg[0]), static_cast<float>(b_reg[1]), static_cast<float>(b_reg[2]), static_cast<float>(b_reg[3]),
                //            static_cast<float>(fragA.x[0]), static_cast<float>(fragA.x[1]), static_cast<float>(fragA.x[2]), static_cast<float>(fragA.x[3]),
                //            static_cast<float>(fragB.x[0]), static_cast<float>(fragB.x[1]), static_cast<float>(fragB.x[2]), static_cast<float>(fragB.x[3]));
                // }
                __syncthreads();
        }

        mma(c_reg, a_reg, b_reg);
        // rocwmma::mma_sync(fragAcc, fragA, fragB, fragAcc);
    }

    bf16_16 c_reg_half;

    for (int i = 0; i < c_reg_half.size(); i++)
        c_reg_half[i] = static_cast<bf16>(c_reg[i]);

    // for (int i = 0; i < NUM_THREADS_PER_WAVE; i++)
    // {
    // if (threadIdx.x == i)
    // {
    //     printf("\nlaneid: %d, c_reg: %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f",
    //            threadIdx.x,
    //            static_cast<float>(c_reg_half[0]), static_cast<float>(c_reg_half[1]),
    //            static_cast<float>(c_reg_half[2]), static_cast<float>(c_reg_half[3]),
    //            static_cast<float>(c_reg_half[4]), static_cast<float>(c_reg_half[5]),
    //            static_cast<float>(c_reg_half[6]), static_cast<float>(c_reg_half[7]),
    //            static_cast<float>(c_reg_half[8]), static_cast<float>(c_reg_half[9]),
    //            static_cast<float>(c_reg_half[10]), static_cast<float>(c_reg_half[11]),
    //            static_cast<float>(c_reg_half[12]), static_cast<float>(c_reg_half[13]),
    //            static_cast<float>(c_reg_half[14]), static_cast<float>(c_reg_half[15]));
    // }
    // __syncthreads();
    // }
    store(C, M, N, &c_reg_half[0], 0, 0);

    // rocwmma::load_matrix_sync(fragC, c + (cRow * ldc + cCol), ldc, rocwmma::mem_row_major);

    // D = alpha * A x B + beta * C
    // for (int i = 0; i < fragC.num_elements; ++i)
    // {
    //     fragC.x[i] = fragAcc.x[i];
    // }

    // Store to D
    // rocwmma::store_matrix_sync(C, fragC, N, rocwmma::mem_row_major);
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
    int M = 32;
    int N = 32;
    int K = 32;

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