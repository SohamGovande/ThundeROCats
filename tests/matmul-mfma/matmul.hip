#include <array>
#include <kittens.hpp>
#include <rocwmma/rocwmma.hpp>

using namespace kittens;

struct globals {
  using abc_t = gl<bf16, -1, -1, -1, -1>;

  abc_t A, B, C;
};

using T = bf16;
using rt_fl_32x32 = std::array<float, 16>;
using rt_bf_32x16 = std::array<bf16, 8>;
using rt_bf_32x32 = std::array<bf16, 16>;

constexpr int MMA_ATOM_SIZE_M = 32;
constexpr int MMA_ATOM_SIZE_N = 32;
constexpr int MMA_ATOM_SIZE_K = 16;

constexpr int WAVE_TILE_COUNT_M = 1;
constexpr int WAVE_TILE_COUNT_N = 1;
constexpr int WAVE_TILE_COUNT_K = 1;

constexpr int WAVE_SIZE_M = MMA_ATOM_SIZE_M * WAVE_TILE_COUNT_M;
constexpr int WAVE_SIZE_N = MMA_ATOM_SIZE_N * WAVE_TILE_COUNT_N;
constexpr int WAVE_SIZE_K = MMA_ATOM_SIZE_K * WAVE_TILE_COUNT_K;

constexpr int BLOCK_WAVE_COUNT_M = 1;
constexpr int BLOCK_WAVE_COUNT_N = 1;
constexpr int NUM_WAVES = BLOCK_WAVE_COUNT_M * BLOCK_WAVE_COUNT_N;
constexpr int NUM_THREADS_PER_WAVE = warpSize;
static_assert(NUM_THREADS_PER_WAVE == 64, "NUM_THREADS_PER_WAVE must be equal to 64. This was what the code was written for.");
constexpr int NUM_THREADS = BLOCK_WAVE_COUNT_M * BLOCK_WAVE_COUNT_N * NUM_THREADS_PER_WAVE;

constexpr int BLOCK_SIZE_M = WAVE_SIZE_M * BLOCK_WAVE_COUNT_M;
constexpr int BLOCK_SIZE_N = WAVE_SIZE_N * BLOCK_WAVE_COUNT_N;
constexpr int BLOCK_SIZE_K = WAVE_SIZE_K;

constexpr int REG_TILE_SIZE_M = 32;
constexpr int REG_TILE_SIZE_N = 32;
constexpr int REG_TILE_SIZE_K = 16;

static_assert(REG_TILE_SIZE_M == REG_TILE_SIZE_N, "REG_TILE_SIZE_M must be equal to REG_TILE_SIZE_N");

__device__ inline void load(T *global, int global_rows, int global_cols, T *reg, int tile_row_idx, int tile_col_idx) {
  static_assert(NUM_THREADS_PER_WAVE % REG_TILE_SIZE_M == 0, "NUM_THREADS_PER_WAVE must be divisible by REG_TILE_SIZE_M");
  constexpr int contiguous_elements_to_load = REG_TILE_SIZE_K / (NUM_THREADS_PER_WAVE / REG_TILE_SIZE_M);

  static_assert(contiguous_elements_to_load == 8, "TEST FAILED: contiguous_elements_to_load != 8");

  int stride_bw_rows = global_cols;
  using T2 = std::array<T, contiguous_elements_to_load>;

  int laneid = threadIdx.x % NUM_THREADS_PER_WAVE;

  int row_win_tile = laneid % REG_TILE_SIZE_M;
  int col_win_tile = (laneid / REG_TILE_SIZE_M) * contiguous_elements_to_load;

  auto global_ptr = &global[
      // Row
      (tile_row_idx * REG_TILE_SIZE_M + row_win_tile) * stride_bw_rows +
      // Col
      (tile_col_idx * REG_TILE_SIZE_K + col_win_tile)];

  // Assume both global and reg are row-major
  *reinterpret_cast<T2 *>(reg) = *reinterpret_cast<T2 *>(global_ptr);
}

__device__ inline void store(T *global, int global_rows, int global_cols, T *reg, int tile_row_idx, int tile_col_idx) {
  static_assert(NUM_THREADS_PER_WAVE % REG_TILE_SIZE_M == 0, "NUM_THREADS_PER_WAVE must be divisible by REG_TILE_SIZE_M");
  constexpr int contiguous_elements_to_store = REG_TILE_SIZE_N / (NUM_THREADS_PER_WAVE / REG_TILE_SIZE_M);
  constexpr bool should_swizzle_4_slices = true;

  int laneid = threadIdx.x % NUM_THREADS_PER_WAVE;

  int row_win_tile = laneid % REG_TILE_SIZE_M;
  int col_win_tile = (laneid / REG_TILE_SIZE_M) * contiguous_elements_to_store;

  int global_row_start = tile_row_idx * REG_TILE_SIZE_M;
  int global_col_start = tile_col_idx * REG_TILE_SIZE_N;

#pragma unroll
  for (int i = 0; i < contiguous_elements_to_store; i++) {
    // Assume RT is col-major and global is row-major -- here, we use the (col, row) coordinates from within the tile instead of (row, col)
    int global_row_offset = col_win_tile + i;
    int global_col_offset = row_win_tile;

    if (should_swizzle_4_slices) {
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

__device__ inline void mma_ABt_old(rt_fl_32x32 &c_reg, rt_bf_32x16 const &a_reg, rt_bf_32x16 const &b_reg) {
  // Decompose the MNK=32x32x16 matmuls into two 32x32x8 matmuls that accumulate into the same registers
  using ab_t = __attribute__((__vector_size__(4 * sizeof(short)))) short const;
  using cd_t = __attribute__((__vector_size__(16 * sizeof(float)))) float;
  auto &c = reinterpret_cast<cd_t &>(c_reg[0]);

  auto &a1 = reinterpret_cast<ab_t const &>(a_reg[0]);
  auto &b1 = reinterpret_cast<ab_t const &>(b_reg[0]);
  c = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a1, b1, c, 0, 0, 0);

  auto &a2 = reinterpret_cast<ab_t const &>(a_reg[4]);
  auto &b2 = reinterpret_cast<ab_t const &>(b_reg[4]);
  c = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a2, b2, c, 0, 0, 0);
}

__device__ inline void mma_ABt(rt_fl<32, 32> &c_reg, rt_bf<32, 16> const &a_reg, rt_bf<32, 16> const &b_reg) {
  // Decompose the MNK=32x32x16 matmuls into two 32x32x8 matmuls that accumulate into the same registers
  using ab_t = __attribute__((__vector_size__(4 * sizeof(short)))) short const;
  using cd_t = __attribute__((__vector_size__(16 * sizeof(float)))) float;
  auto &c = reinterpret_cast<cd_t &>(c_reg.tiles[0][0].data[0]);

  auto &a1 = reinterpret_cast<ab_t const &>(a_reg.tiles[0][0].data[0]);
  auto &b1 = reinterpret_cast<ab_t const &>(b_reg.tiles[0][0].data[0]);
  c = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a1, b1, c, 0, 0, 0);

  auto &a2 = reinterpret_cast<ab_t const &>(a_reg.tiles[0][0].data[a_reg.packed_per_thread / 2]);
  auto &b2 = reinterpret_cast<ab_t const &>(b_reg.tiles[0][0].data[b_reg.packed_per_thread / 2]);
  c = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a2, b2, c, 0, 0, 0);
}

__global__ __launch_bounds__(NUM_THREADS) void gpu_matmul_ABt_ker(globals g) {
  auto &[A, B, C] = g;
  auto M = C.rows();
  auto N = C.cols();
  auto K = A.cols();

  rt_fl_32x32 c_reg_old[WAVE_TILE_COUNT_M][WAVE_TILE_COUNT_N];
  rt_fl<32, 32> c_reg[WAVE_TILE_COUNT_M][WAVE_TILE_COUNT_N];

  for (int i = 0; i < WAVE_TILE_COUNT_M; i++)
    for (int j = 0; j < WAVE_TILE_COUNT_N; j++) {
      c_reg[i][j] = 0;
      c_reg_old[i][j] = {0};
    }

  int waveid = threadIdx.x / NUM_THREADS_PER_WAVE;
  int waveid_m = waveid / BLOCK_WAVE_COUNT_N;
  int waveid_n = waveid % BLOCK_WAVE_COUNT_N;
  int wave_start_m = blockIdx.x * (BLOCK_SIZE_M / REG_TILE_SIZE_M) + waveid_m * (WAVE_SIZE_M / REG_TILE_SIZE_M);
  int wave_start_n = blockIdx.y * (BLOCK_SIZE_N / REG_TILE_SIZE_N) + waveid_n * (WAVE_SIZE_N / REG_TILE_SIZE_N);

  rt_bf<32, 16> a_reg[WAVE_TILE_COUNT_M], b_reg[WAVE_TILE_COUNT_N];
  rt_bf<32, 32> c_reg_half[WAVE_TILE_COUNT_M][WAVE_TILE_COUNT_N];
  rt_bf_32x32 c_reg_half_old[WAVE_TILE_COUNT_M][WAVE_TILE_COUNT_N];

  // GEMM mainloop
  for (int k_block = 0; k_block < K; k_block += WAVE_SIZE_K) {
    rt_bf_32x16 a_reg_old[WAVE_TILE_COUNT_M], b_reg_old[WAVE_TILE_COUNT_N];

    for (int i = 0; i < WAVE_TILE_COUNT_M; i++)
      load(a_reg[i], A, {wave_start_m + i, k_block / WAVE_SIZE_K});
    for (int i = 0; i < WAVE_TILE_COUNT_N; i++)
      load(b_reg[i], B, {wave_start_n + i, k_block / WAVE_SIZE_K});

    for (int i = 0; i < WAVE_TILE_COUNT_M; i++)
      load(A.raw_ptr, M, K, &a_reg_old[i][0], wave_start_m + i, k_block / WAVE_SIZE_K);
    for (int i = 0; i < WAVE_TILE_COUNT_N; i++)
      load(B.raw_ptr, N, K, &b_reg_old[i][0], wave_start_n + i, k_block / WAVE_SIZE_K);

    for (int i = 0; i < a_reg_old[0].size(); i += 2) {
      a_reg[0].tiles[0][0].data[i / 2] = {a_reg_old[0][i], a_reg_old[0][i + 1]};
      b_reg[0].tiles[0][0].data[i / 2] = {b_reg_old[0][i], b_reg_old[0][i + 1]};
    }

    for (int i = 0; i < WAVE_TILE_COUNT_M; i++)
      for (int j = 0; j < WAVE_TILE_COUNT_N; j++)
        mma_ABt(c_reg[i][j], a_reg[i], b_reg[j]);
    for (int i = 0; i < WAVE_TILE_COUNT_M; i++)
      for (int j = 0; j < WAVE_TILE_COUNT_N; j++) {
        mma_ABt_old(c_reg_old[i][j], a_reg_old[i], b_reg_old[j]);
      }
  }
  for (int i = 0; i < WAVE_TILE_COUNT_M; i++)
    for (int j = 0; j < WAVE_TILE_COUNT_N; j++)
      for (int k = 0; k < c_reg[i][j].height; k++)
        for (int l = 0; l < c_reg[i][j].width; l++)
          for (int m = 0; m < c_reg[i][j].tiles[k][l].packed_per_thread; m++)
            c_reg_half[i][j].tiles[k][l].data[m] = base_types::convertor<bf16_2, float2>::convert(c_reg[i][j].tiles[k][l].data[m]);

  // for (int i = 0; i < WAVE_TILE_COUNT_M; i++)
  //   for (int j = 0; j < WAVE_TILE_COUNT_N; j++)
  //     for (int k = 0; k < c_reg[i][j].tiles[0][0].packed_per_thread; k++)
  //       c_reg_half[i][j].tiles[0][0].data[k] = base_types::convertor<bf16_2, float2>::convert(c_reg[i][j].tiles[0][0].data[k]);

  // for (int i = 0; i < WAVE_TILE_COUNT_M; i++)
  //   for (int j = 0; j < WAVE_TILE_COUNT_N; j++) {
  //     for (int k = 0; k < c_reg_half_old[i][j].size(); k++)
  //       c_reg_half_old[i][j][k] = base_types::convertor<bf16, float>::convert(c_reg_old[i][j][k]);
  // for (int k = 0; k < c_reg_half_old[i][j].size() / 2; k++)
  //   c_reg_half[i][j].tiles[0][0].data[k] = {c_reg_half_old[i][j][k * 2], c_reg_half_old[i][j][k * 2 + 1]};
  // }

  for (int i = 0; i < WAVE_TILE_COUNT_M; i++)
    for (int j = 0; j < WAVE_TILE_COUNT_N; j++) {
      // store(C.raw_ptr, M, N, reinterpret_cast<T *>(&c_reg_half_old[i][j][0]), wave_start_m + i, wave_start_n + j);
      store(C.raw_ptr, M, N, reinterpret_cast<T *>(&c_reg_half[i][j].tiles[0][0].data[0]), wave_start_m + i, wave_start_n + j);
    }
}

void gpu_matmul_ABt(T *A, T *B, T *C, int M, int N, int K, std::vector<T> &h_C) {
  dim3 block(NUM_THREADS_PER_WAVE * NUM_WAVES);
  dim3 grid((M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N);
  std::cout << "Problem Shape: (" << M << ", " << N << ", " << K << ")" << std::endl;
  std::cout << "Launching with grid (" << grid.x << ", " << grid.y << ", " << grid.z << ") with block (" << block.x << ", " << block.y << ", " << block.z << ")" << std::endl;
  float ms = 0;

  using gl_t = globals::abc_t;

  gl_t g_A(A, 1, 1, M, K);
  gl_t g_B(B, 1, 1, N, K);
  gl_t g_C(C, 1, 1, M, N);

  globals g{g_A, g_B, g_C};

  // warmup kernel
  gpu_matmul_ABt_ker<<<grid, block>>>(g);

  constexpr int num_iters = 0;
  for (int i = 0; i < num_iters; i++) {
    kernel_timer t(&ms, 1.0f / num_iters);
    gpu_matmul_ABt_ker<<<grid, block>>>(g);
  }

  int flops = 2 * M * N * K;
  float gflops = flops / (ms * 1e3);
  // std::cout << "GFLOPS: " << gflops << std::endl;

  hipCheck(hipMemcpy(h_C.data(), C, M * N * sizeof(T), hipMemcpyDeviceToHost));
}

int main() {
  int M = BLOCK_SIZE_M;
  int N = BLOCK_SIZE_N;
  int K = BLOCK_SIZE_K;

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