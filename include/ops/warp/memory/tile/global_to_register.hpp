/**
 * @file
 * @brief Functions for transferring data directly between global memory and registers and back.
 */

#pragma once

#include "../../../../common/common.hpp"

namespace kittens {

namespace detail {

template <typename T, typename U>
__device__ inline void load_rt_base(T *global, int global_cols, U *reg) {
  constexpr int REG_TILE_SIZE_M = 32;
  constexpr int REG_TILE_SIZE_K = 16;

  static_assert(WAVE_THREADS % REG_TILE_SIZE_M == 0, "WAVE_THREADS must be divisible by REG_TILE_SIZE_M");
  constexpr int contiguous_elements_to_load = REG_TILE_SIZE_K / (WAVE_THREADS / REG_TILE_SIZE_M);

  static_assert(contiguous_elements_to_load == 8, "TEST FAILED: contiguous_elements_to_load != 8");

  int stride_bw_rows = global_cols;
  using T2 = std::array<T, contiguous_elements_to_load>;

  int laneid = kittens::laneid();

  int row_win_tile = laneid % REG_TILE_SIZE_M;
  int col_win_tile = (laneid / REG_TILE_SIZE_M) * contiguous_elements_to_load;

  auto global_ptr = &global[
      // Row
      (row_win_tile)*stride_bw_rows +
      // Col
      (col_win_tile)];

  // Assume both global and reg are row-major
  *reinterpret_cast<T2 *>(reg) = *reinterpret_cast<T2 *>(global_ptr);
}
} // namespace detail

template <int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD = coord<RT>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
  using T2 = RT::dtype;
  using U = typename GL::dtype;

  const int row_stride = src.template stride<axis>();
  using U2 = typename base_types::packing<U>::packed_type;

#pragma unroll
  for (int row_tile = 0; row_tile < RT::height; row_tile++) {
#pragma unroll
    for (int col_tile = 0; col_tile < RT::width; col_tile++) {
      auto new_coord = idx.template unit_coord<axis, 3>();
      new_coord.r += row_tile * RT::tile_size_row;
      new_coord.c += col_tile * RT::tile_size_col;

      auto src_ptr = (U *)&src[(new_coord)];

      auto &base_tile = dst.tiles[row_tile][col_tile];
      detail::load_rt_base(src_ptr, row_stride, base_tile.data);
    }
  }
}

template <ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD = coord<RT>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
  load<2>(dst, src, idx);
}

} // namespace kittens