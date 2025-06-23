#include <array>
#include <kittens.hpp>

using namespace kittens;

namespace mm_ABt_ker {
struct layout {
  // base sizes - feel free to change M/N dimensions on these
  static constexpr coord_mnk wave_tile_count{2, 1, 1};
  static constexpr coord_mnk block_wave_count{2, 2, 1};

  // derived  (or constant) sizes - do not change these
  static constexpr coord_mnk mma_atom_size{32, 32, 16};
  static constexpr coord_mnk wave_size = mma_atom_size * wave_tile_count;
  static constexpr int num_waves = block_wave_count.m * block_wave_count.n * block_wave_count.k;
  static constexpr int num_threads = num_waves * WAVE_THREADS;
  static constexpr coord_mnk block_size = wave_size * block_wave_count;
};
struct locals {
  using layout = mm_ABt_ker::layout;
  static constexpr int wave_tile_size_m = layout::wave_tile_count.m * layout::mma_atom_size.m;
  static constexpr int wave_tile_size_n = layout::wave_tile_count.n * layout::mma_atom_size.n;

  rt_bf<wave_tile_size_m, 16> a_reg;
  rt_bf<wave_tile_size_n, 16> b_reg;
  rt_fl<wave_tile_size_m, wave_tile_size_n, ducks::rt_layout::col> c_reg;
  rt_bf<wave_tile_size_m, wave_tile_size_n, ducks::rt_layout::col> c_reg_half;
};
struct globals {
  using abc_t = gl<bf16, -1, -1, -1, -1>;
  abc_t A, B, C;
};
}; // namespace mm_ABt_ker

using layout = mm_ABt_ker::layout;

template <ducks::rt::all RT_D, ducks::rt::all RT_S>
__device__ inline void copy(RT_D &dst, RT_S &src) {
  static_assert(RT_D::height == RT_S::height, "Heights must match");
  static_assert(RT_D::width == RT_S::width, "Widths must match");
  static_assert(RT_D::packed_per_thread == RT_S::packed_per_thread, "Packed per thread must match");

  for (int k = 0; k < dst.height; k++)
    for (int l = 0; l < dst.width; l++)
      for (int m = 0; m < dst.tiles[k][l].packed_per_thread; m++) {
        dst.tiles[k][l].data[m] = base_types::convertor<typename RT_D::dtype, typename RT_S::dtype>::convert(src.tiles[k][l].data[m]);
        if (base_types::convertor<float, typename RT_D::T>::convert(dst.tiles[k][l].data[m].x) != 16.0f) {
          printf("dst.tiles[%d][%d].data[%d] = %f\n", k, l, m, base_types::convertor<float, typename RT_D::T>::convert(dst.tiles[k][l].data[m].x));
        }
      }
}

__global__ __launch_bounds__(layout::num_threads) void gpu_matmul_ABt_ker(mm_ABt_ker::globals g) {
  int wave_start_m = blockIdx.x * (layout::block_size.m / layout::wave_size.m) + (waveid() / layout::block_wave_count.n);
  int wave_start_n = blockIdx.y * (layout::block_size.n / layout::wave_size.n) + (waveid() % layout::block_wave_count.n);
  mm_ABt_ker::locals l;
  zero(l.c_reg);
  for (int k_block = 0; k_block < g.A.cols(); k_block += layout::wave_size.k) {
    load(l.a_reg, g.A, {wave_start_m, k_block / layout::wave_size.k});
    load(l.b_reg, g.B, {wave_start_n, k_block / layout::wave_size.k});
    mma_ABt(l.c_reg, l.a_reg, l.b_reg);
  }
  copy(l.c_reg_half, l.c_reg);
  store(g.C, l.c_reg_half, {wave_start_m, wave_start_n});
}

void gpu_matmul_ABt(bf16 *A, bf16 *B, bf16 *C, int M, int N, int K, std::vector<bf16> &h_C) {
  dim3 block(WAVE_THREADS * layout::num_waves);
  dim3 grid((M + layout::block_size.m - 1) / layout::block_size.m, (N + layout::block_size.n - 1) / layout::block_size.n);
  std::cout << "Problem Shape: (" << M << ", " << N << ", " << K << ")" << std::endl;
  std::cout << "Launching with grid (" << grid.x << ", " << grid.y << ", " << grid.z << ") with block (" << block.x << ", " << block.y << ", " << block.z << ")" << std::endl;
  float ms = 0;

  using gl_t = mm_ABt_ker::globals::abc_t;

  gl_t g_A(A, 1, 1, M, K);
  gl_t g_B(B, 1, 1, N, K);
  gl_t g_C(C, 1, 1, M, N);

  mm_ABt_ker::globals g{g_A, g_B, g_C};

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
  int M = layout::block_size.m;
  int N = layout::block_size.n;
  int K = layout::block_size.k;

  auto [h_A, d_A] = init<fill_ones, bf16>(M * K);
  auto [h_B, d_B] = init<fill_ones, bf16>(K * N);
  auto [h_C, d_C] = init<fill_ones, bf16>(M * N);

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