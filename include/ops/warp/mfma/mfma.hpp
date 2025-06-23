#pragma once

#include "../../../common/common.hpp"
#include "../../../types/types.hpp"

namespace kittens {

template <int M, int N, int K>
__device__ inline void mma_ABt(rt_fl<M, N, ducks::rt_layout::col> &c_reg, rt_bf<M, K, ducks::rt_layout::row> const &a_reg, rt_bf<N, K, ducks::rt_layout::row> const &b_reg) {
  static_assert(M % 32 == 0, "M must be divisible by 32");
  static_assert(N % 32 == 0, "N must be divisible by 32");
  static_assert(K % 16 == 0, "K must be divisible by 16");

  constexpr int M_tiles = M / 32;
  constexpr int N_tiles = N / 32;
  constexpr int K_tiles = K / 16;

  static_assert(K_tiles == 1, "K must be 1 (FOR NOW)");

#pragma unroll
  for (int m = 0; m < M_tiles; m++) {
#pragma unroll
    for (int n = 0; n < N_tiles; n++) {
#pragma unroll
      for (int k = 0; k < K_tiles; k++) {
        // Decompose the MNK=32x32x16 matmuls into two 32x32x8 matmuls that accumulate into the same registers
        using ab_t = __attribute__((__vector_size__(4 * sizeof(short)))) short const;
        using cd_t = __attribute__((__vector_size__(16 * sizeof(float)))) float;
        auto &c = reinterpret_cast<cd_t &>(c_reg.tiles[m][n].data[0]);

        auto &a1 = reinterpret_cast<ab_t const &>(a_reg.tiles[m][k].data[0]);
        auto &b1 = reinterpret_cast<ab_t const &>(b_reg.tiles[n][k].data[0]);
        c = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a1, b1, c, 0, 0, 0);

        auto &a2 = reinterpret_cast<ab_t const &>(a_reg.tiles[m][k].data[a_reg.packed_per_thread / 2]);
        auto &b2 = reinterpret_cast<ab_t const &>(b_reg.tiles[n][k].data[b_reg.packed_per_thread / 2]);
        c = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a2, b2, c, 0, 0, 0);
      }
    }
  }
}

} // namespace kittens