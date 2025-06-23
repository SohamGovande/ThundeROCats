#include <array>
#include <kittens.hpp>

using namespace kittens;

struct globals {
  struct layout {
    // base sizes - feel free to change these
    static constexpr coord_mnk mma_atom_size{32, 32, 16};
    // keep the k dimension on both of these equal to 1 - it doesn't make sense.
    static constexpr coord_mnk wave_tile_count{2, 2, 1};
    static constexpr coord_mnk block_wave_count{2, 2, 1};

    // derived sizes - do not change these
    static constexpr coord_mnk wave_size = mma_atom_size * wave_tile_count;
    static constexpr int num_waves = wave_tile_count.m * wave_tile_count.n * wave_tile_count.k;
    static constexpr int num_threads = num_waves * WAVE_THREADS;
    static constexpr coord_mnk block_size = wave_size * block_wave_count;
  };
  using abc_t = gl<bf16, -1, -1, -1, -1>;

  abc_t A, B, C;
};

using layout = globals::layout;

__global__ __launch_bounds__(layout::num_threads) void gpu_matmul_ABt_ker(globals g) {
  auto &[A, B, C] = g;
  auto M = C.rows();
  auto N = C.cols();
  auto K = A.cols();

  rt_bf<32, 16> a_reg[layout::wave_tile_count.m], b_reg[layout::wave_tile_count.n];
  rt_fl<32, 32, ducks::rt_layout::col> c_reg[layout::wave_tile_count.m][layout::wave_tile_count.n];
  rt_bf<32, 32, ducks::rt_layout::col> c_reg_half[layout::wave_tile_count.m][layout::wave_tile_count.n];

  for (int i = 0; i < layout::wave_tile_count.m; i++)
    for (int j = 0; j < layout::wave_tile_count.n; j++)
      c_reg[i][j] = 0;

  int waveid = threadIdx.x / WAVE_THREADS;
  int waveid_m = waveid / layout::block_wave_count.n;
  int waveid_n = waveid % layout::block_wave_count.n;
  int wave_start_m = blockIdx.x * (layout::block_size.m / layout::mma_atom_size.m) + waveid_m * (layout::wave_size.m / layout::mma_atom_size.m);
  int wave_start_n = blockIdx.y * (layout::block_size.n / layout::mma_atom_size.n) + waveid_n * (layout::wave_size.n / layout::mma_atom_size.n);

  // GEMM mainloop
  for (int k_block = 0; k_block < K; k_block += layout::wave_size.k) {
    for (int i = 0; i < layout::wave_tile_count.m; i++)
      load(a_reg[i], A, {wave_start_m + i, k_block / layout::wave_size.k});
    for (int i = 0; i < layout::wave_tile_count.n; i++)
      load(b_reg[i], B, {wave_start_n + i, k_block / layout::wave_size.k});

    for (int i = 0; i < layout::wave_tile_count.m; i++)
      for (int j = 0; j < layout::wave_tile_count.n; j++)
        mma_ABt(c_reg[i][j], a_reg[i], b_reg[j]);
  }
  for (int i = 0; i < layout::wave_tile_count.m; i++)
    for (int j = 0; j < layout::wave_tile_count.n; j++)
      for (int k = 0; k < c_reg[i][j].height; k++)
        for (int l = 0; l < c_reg[i][j].width; l++)
          for (int m = 0; m < c_reg[i][j].tiles[k][l].packed_per_thread; m++)
            c_reg_half[i][j].tiles[k][l].data[m] = base_types::convertor<bf16_2, float2>::convert(c_reg[i][j].tiles[k][l].data[m]);

  for (int i = 0; i < layout::wave_tile_count.m; i++)
    for (int j = 0; j < layout::wave_tile_count.n; j++)
      store(C, c_reg_half[i][j], {wave_start_m + i, wave_start_n + j});
}

void gpu_matmul_ABt(bf16 *A, bf16 *B, bf16 *C, int M, int N, int K, std::vector<bf16> &h_C) {
  dim3 block(WAVE_THREADS * layout::num_waves);
  dim3 grid((M + layout::block_size.m - 1) / layout::block_size.m, (N + layout::block_size.n - 1) / layout::block_size.n);
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

  hipCheck(hipMemcpy(h_C.data(), C, M * N * sizeof(bf16), hipMemcpyDeviceToHost));
}

int main() {
  int M = layout::block_size.m * 20;
  int N = layout::block_size.n * 10;
  int K = layout::block_size.k * 2;

  auto [h_A, d_A] = init<fill_random, bf16>(M * K);
  auto [h_B, d_B] = init<fill_random, bf16>(K * N);
  auto [h_C, d_C] = init<fill_zeros, bf16>(M * N);

  auto h_C_ref = h_C;
  cpu_matmul<bf16, /* A */ false, /* B.bf16 */ true>(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);
  gpu_matmul_ABt(d_A, d_B, d_C, M, N, K, h_C);

  assert_equal(h_C_ref, h_C);

  print_tensor_to_file<bf16>("matmul.csv", {{"A", h_A.data(), M, K}, {"B", h_B.data(), N, K}, {"C_ref", h_C_ref.data(), M, N}, {"C", h_C.data(), M, N}});

  hipCheck(hipFree(d_A));
  hipCheck(hipFree(d_B));
  hipCheck(hipFree(d_C));

  return 0;
}