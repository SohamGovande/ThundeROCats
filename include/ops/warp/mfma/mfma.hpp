#pragma once

#include "../../../common/common.hpp"
#include "../../../types/types.hpp"

namespace kittens {

__device__ inline void mma_ABt(rt_fl<32, 32, ducks::rt_layout::col> &c_reg, rt_bf<32, 16, ducks::rt_layout::row> const &a_reg, rt_bf<32, 16, ducks::rt_layout::row> const &b_reg) {
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

}