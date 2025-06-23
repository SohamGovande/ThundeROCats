#pragma once

#include "rt_base.hpp"
#include <concepts>
#include <type_traits>

namespace kittens {
namespace ducks {
namespace rt {
struct identifier {
};
} // namespace rt
} // namespace ducks

template <typename _T, int _rows, int _cols, ducks::rt_layout::all _layout = ducks::rt_layout::row>
struct rt {
  using identifier = ducks::rt::identifier;          ///< Type identifier for the rt structure.
  using layout = _layout;                            ///< Layout of the matrix tile.
  static_assert(kittens::ducks::base_types::T1<_T>); // confirm it's a supported type
  using T = kittens::base_types::packing<_T>::unpacked_type;
  using T2 = kittens::base_types::packing<_T>::packed_type;
  using dtype = T2; ///< Data type of the matrix elements

  using base_tile = rt_base<T, layout>;

  static constexpr int rows = _rows; ///< Total number of rows.
  static_assert(rows % base_tile::tile_size_row == 0, "Rows must be divisible by the tile size (32)");
  static constexpr int cols = _cols; ///< Total number of columns.
  static_assert(cols % base_tile::tile_size_col == 0, "Columns must be divisible by the tile size (16)");
  static constexpr int height = rows / base_tile::tile_size_row;                              ///< Height in subtiles.
  static constexpr int width = cols / base_tile::tile_size_col;                               ///< Width in subtiles.
  static constexpr int tile_size_row = base_tile::tile_size_row;                              ///< Size of the base tile.
  static constexpr int tile_size_col = base_tile::tile_size_col;                              ///< Size of the base tile.
  static constexpr int num_elements = base_tile::num_elements * width * height;               ///< Total number of elements.
  static constexpr int elements_per_thread = base_tile::elements_per_thread * width * height; ///< Elements handled per thread.
  static constexpr int packed_per_thread = base_tile::packed_per_thread * width * height;     ///< Packed elements per thread.
  static constexpr int packed_per_tile = base_tile::packed_per_thread;                        ///< Packed elements per tile.

  base_tile tiles[height][width]; ///< The actual storage for the matrix tile, organized in subtiles.

  __device__ inline void operator=(const T &value) {
    T2 value2 = base_types::packing<T>::pack(value);
#pragma unroll
    for (int i = 0; i < height; i++) {
#pragma unroll
      for (int j = 0; j < width; j++) {
#pragma unroll
        for (int k = 0; k < packed_per_tile; k++) {
          tiles[i][j].data[k] = value2;
        }
      }
    }
  }
  template <typename U>
  __device__ inline void operator=(const rt<U, rows, cols, layout> &other) {
    using U2 = base_types::packing<U>::packed_type;
#pragma unroll
    for (int i = 0; i < height; i++) {
#pragma unroll
      for (int j = 0; j < width; j++) {
#pragma unroll
        for (int k = 0; k < packed_per_tile; k++) {
          tiles[i][j].data[k] = base_types::convertor<T2, U2>::convert(other.tiles[i][j].data[k]);
        }
      }
    }
  }
}; // struct rt

/* ----------  CONCEPTS  ---------- */

namespace ducks {
namespace rt {
/**
 * @brief Concept for all register tiles.
 * @tparam T The type to check against the concept requirements.
 *
 * Requires:
 * - T has a nested type identifier that is the same as rt::identifier.
 */
template <typename T>
concept all = requires {
  typename T::identifier;                                // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::rt::identifier
/**
 * @brief Concept for register tiles with row layout.
 * @tparam T The type to check against the concept requirements.
 *
 * Requires:
 * - T is a register tile.
 * - T has an internal type layout that is ducks::rt_layout::row.
 */
template <typename T>
concept row_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::row>;
/**
 * @brief Concept for register tiles with col layout.
 * @tparam T The type to check against the concept requirements.
 *
 * Requires:
 * - T is a register tile.
 * - T has an internal type layout that is ducks::rt_layout::col.
 */
template <typename T>
concept col_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::col>;

} // namespace rt
} // namespace ducks

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// layout and type wrappers

template <int _r, int _c, ducks::rt_layout::all layout = ducks::rt_layout::row>
using rt_fl = rt<float, _r, _c, layout>;
template <int _r, int _c, ducks::rt_layout::all layout = ducks::rt_layout::row>
using rt_bf = rt<bf16, _r, _c, layout>;
template <int _r, int _c, ducks::rt_layout::all layout = ducks::rt_layout::row>
using rt_hf = rt<half, _r, _c, layout>;
} // namespace kittens