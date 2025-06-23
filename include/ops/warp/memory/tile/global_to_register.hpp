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

namespace detail {
template <typename T, typename U>
__device__ inline void store_rt_base(T *global, int global_cols, U *reg) {
  constexpr int REG_TILE_SIZE_M = 32;
  constexpr int REG_TILE_SIZE_N = 32;

  constexpr int contiguous_elements_to_store = REG_TILE_SIZE_N / (WAVE_THREADS / REG_TILE_SIZE_M);
  constexpr bool should_swizzle_4_slices = true;

  int laneid = threadIdx.x % WAVE_THREADS;

  int row_win_tile = laneid % REG_TILE_SIZE_M;
  int col_win_tile = (laneid / REG_TILE_SIZE_M) * contiguous_elements_to_store;

#pragma unroll
  for (int i = 0; i < contiguous_elements_to_store; i++) {
    // printf("laneid: %d, i: %d, row_win_tile: %d, col_win_tile: %d\n", laneid, i, row_win_tile, col_win_tile);
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
    global[global_row_offset * global_cols + global_col_offset] = reinterpret_cast<T const *>(reg)[i];
  }
}
} // namespace detail

template <int axis, ducks::gl::all GL, ducks::rt::col_layout RT, ducks::coord::tile COORD = coord<RT>>
__device__ inline static void store(GL &dst, const RT &src, const COORD &idx) {
  static_assert(RT::width % 2 == 0, "RT::width must be even");
  using T2 = RT::dtype;
  using U = typename GL::dtype;

  const int row_stride = dst.template stride<axis>();
  using U2 = typename base_types::packing<U>::packed_type;

#pragma unroll
  for (int row_tile = 0; row_tile < RT::height; row_tile++) {
#pragma unroll
    for (int col_tile = 0; col_tile < RT::width; col_tile += 2) {
      auto new_coord = idx.template unit_coord<axis, 3>();
      // switch row and col
      new_coord.c += row_tile * RT::tile_size_row;
      new_coord.r += col_tile * RT::tile_size_col;
      // if (threadIdx.x % 64 == 0) {
      // printf("new_coord: %d, %d, %d, %d\n", new_coord.b, new_coord.d, new_coord.r, new_coord.c);
      // }
      auto dst_ptr = (U *)&dst[(new_coord)];

      auto &base_tile = src.tiles[row_tile][col_tile];
      detail::store_rt_base(dst_ptr, row_stride, base_tile.data);
    }
  }
}

template <ducks::gl::all GL, ducks::rt::col_layout RT, ducks::coord::tile COORD = coord<RT>>
__device__ inline static void store(GL &dst, const RT &src, const COORD &idx) {
  store<2>(dst, src, idx);
}

} // namespace kittens