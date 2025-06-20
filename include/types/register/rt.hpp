#pragma once

#include <concepts>
#include <type_traits>
#include "rt_base.hpp"

namespace kittens
{
    namespace ducks
    {
        namespace rt
        {
            struct identifier
            {
            };
        } // namespace rt
    } // namespace ducks

    template <typename _T, int _rows, int _cols>
    struct rt
    {
        using identifier = ducks::rt::identifier;
        using T = _T;
        using dtype = T;

        static constexpr int rows = _rows; ///< Total number of rows.
        static_assert(rows % rt_base<T>::tile_size_row == 0, "Rows must be divisible by the tile size");
        static constexpr int cols = _cols; ///< Total number of columns.
        static_assert(cols % rt_base<T>::tile_size_col == 0, "Columns must be divisible by the tile size");
        static constexpr int height = rows / rt_base<T>::tile_size_row;                              ///< Height in subtiles.
        static constexpr int width = cols / rt_base<T>::tile_size_col;                               ///< Width in subtiles.
        static constexpr int tile_size_row = rt_base<T>::tile_size_row;                              ///< Size of the base tile.
        static constexpr int tile_size_col = rt_base<T>::tile_size_col;                              ///< Size of the base tile.
        static constexpr int num_elements = rt_base<T>::num_elements * width * height;               ///< Total number of elements.
        static constexpr int elements_per_thread = rt_base<T>::elements_per_thread * width * height; ///< Elements handled per thread.
        static constexpr int packed_per_thread = rt_base<T>::packed_per_thread * width * height;     ///< Packed elements per thread.
        static constexpr int packed_per_tile = rt_base<T>::packed_per_thread;                        ///< Packed elements per tile.

        rt_base<T> tiles[height][width]; ///< The actual storage for the matrix tile, organized in subtiles.

        __device__ inline void
        operator=(const T &value)
        {
#pragma unroll
            for (int i = 0; i < height; i++)
            {
#pragma unroll
                for (int j = 0; j < width; j++)
                {
#pragma unroll
                    for (int k = 0; k < packed_per_tile; k++)
                    {
                        tiles[i][j].data[k] = value;
                    }
                }
            }
        }
    }; // struct rt

    template <int _r, int _c>
    using rt_fl = rt<float, _r, _c>;
    template <int _r, int _c>
    using rt_bf = rt<bf16, _r, _c>;
    template <int _r, int _c>
    using rt_hf = rt<fp16, _r, _c>;

    namespace ducks
    {
        namespace rt
        {
            /**
             * @brief Concept for all register tiles.
             * @tparam T The type to check against the concept requirements.
             *
             * Requires:
             * - T has a nested type identifier that is the same as rt::identifier.
             */
            template <typename T>
            concept all = requires {
                typename T::identifier;                              // Checks if T::identifier exists
            } && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::rt::identifier
        } // namespace rt
    } // namespace ducks
} // namespace kittens