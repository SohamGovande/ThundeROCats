/**
 * @file
 * @brief The basic 16x16 register tile on which larger register tiles are built.
 */

#pragma once

#include <type_traits>

#include "../../common/common.hpp"
#include "rt_layout.hpp"
// #include "rv_layout.hpp"

namespace kittens {

template <typename T>
constexpr int TILE_ROW_DIM = 32;
template <typename T>
constexpr int TILE_COL_DIM = 16;

/* ----------  BASE 16x16 SUBTILE STRUCT  ---------- */

namespace ducks {
/**
 * @namespace rt_base
 *
 * @brief The namespace where concepts and abstract types for register base (16x16) tiles live.
 */
namespace rt_base {
/**
 * @brief A dummy type used to identify register base tiles.
 *
 * For a type to quack like an rt_base, it should define its identifier as ducks::rt_base::identifier.
 * If a type quacks like ducks::rt_base::identifier, it will be treated as an rt_base by compiler checks.
 */
struct identifier {};
} // namespace rt_base
} // namespace ducks

/**
 * @brief Basic tile structure for computation in registers.
 *
 * @tparam T2 The packed data type used for the matrix elements.
 * @tparam _layout The layout of the base tile, either row-major or column-major.
 *
 * This type is a primarily utility for building larger inline templates
 * out of PTX primitives and managing layouts.
 *
 * In general, you probably want a row-major tile, unless you specifically want to call mma
 */
template <typename _T, ducks::rt_layout::all _layout>
struct rt_base {
  using identifier = ducks::rt_base::identifier;     ///< Type identifier for the rt_base structure.
  using layout = _layout;                            ///< Layout of the matrix tile.
  static_assert(kittens::ducks::base_types::T1<_T>); // confirm it's a supported type
  using T = kittens::base_types::packing<_T>::unpacked_type;
  using T2 = kittens::base_types::packing<_T>::packed_type;
  using dtype = T2; ///< Data type of the matrix elements

  static_assert(
      std::is_same_v<dtype, bf16_2> || std::is_same_v<dtype, float2> || std::is_same_v<dtype, half_2>,
      "rt_base was provided an unsupported type.");

  static constexpr int tile_size_row = kittens::TILE_ROW_DIM<T>; // < Tile size is a constant 16 for everyone
  static constexpr int tile_size_col = kittens::TILE_COL_DIM<T>;
  static constexpr int rows = tile_size_row;                              ///< Number of rows.
  static constexpr int cols = tile_size_col;                              ///< Number of cols.
  static constexpr int num_elements = rows * cols;                        // 256 (64 for fp8e4m3)
  static constexpr int elements_per_thread = num_elements / WAVE_THREADS; // 8 (2 for fp8e4m3)

  static constexpr int packed_per_thread = (elements_per_thread / base_types::packing<dtype>::num()); // 4
  static constexpr int registers_per_thread = packed_per_thread * sizeof(dtype) / 4;                  // 4 or 8, registers are 32-bit words

  dtype data[packed_per_thread]; ///< The actual storage for the base tile
};

/* ----------  CONCEPTS  ---------- */

namespace ducks {
namespace rt_base {
/**
 * @brief Concept for all register base tiles.
 * @tparam T The type to check against the concept requirements.
 *
 * Requires:
 * - T has a nested type identifier that is the same as rt_base::identifier.
 */
template <typename T>
concept all = requires {
  typename T::identifier;                                // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::rt::identifier
} // namespace rt_base
} // namespace ducks

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

template <ducks::rt_layout::all L = ducks::rt_layout::row>
using rt_base_fl = rt_base<float, L>;
template <ducks::rt_layout::all L = ducks::rt_layout::row>
using rt_base_bf = rt_base<bf16, L>;
template <ducks::rt_layout::all L = ducks::rt_layout::row>
using rt_base_hf = rt_base<half, L>;
} // namespace kittens