#pragma once

#include <type_traits>

namespace kittens
{

    template <typename T>
    constexpr int TILE_ROW_DIM = 16;
    template <typename T>
    constexpr int TILE_COL_DIM = 16;

    /* ----------  BASE 16x16 SUBTILE STRUCT  ---------- */

    namespace ducks
    {
        /**
         * @namespace rt_base
         *
         * @brief The namespace where concepts and abstract types for register base (16x16) tiles live.
         */
        namespace rt_base
        {
            /**
             * @brief A dummy type used to identify register base tiles.
             *
             * For a type to quack like an rt_base, it should define its identifier as ducks::rt_base::identifier.
             * If a type quacks like ducks::rt_base::identifier, it will be treated as an rt_base by compiler checks.
             */
            struct identifier
            {
            };
        }
    } // namespace ducks

    /**
     * @brief Basic tile structure for computation in registers.
     *
     * @tparam T2 The packed data type used for the matrix elements.
     */
    template <typename _T>
    struct rt_base
    {
        using identifier = ducks::rt_base::identifier; ///< Type identifier for the rt_base structure.
        using T = _T;
        using dtype = T; ///< Data type of the matrix elements

        static constexpr int tile_size_row = kittens::TILE_ROW_DIM<T>; // < Tile size is a constant 16 for everyone
        static constexpr int tile_size_col = kittens::TILE_COL_DIM<T>;
        static constexpr int rows = tile_size_row;                    ///< Number of rows.
        static constexpr int cols = tile_size_col;                    ///< Number of cols.
        static constexpr int num_elements = rows * cols;              // 256 (64 for fp8e4m3)
        static constexpr int elements_per_thread = num_elements / 32; // 8 (2 for fp8e4m3)

        static constexpr int packed_per_thread = (elements_per_thread / 1);                // 4
        static constexpr int registers_per_thread = packed_per_thread * sizeof(dtype) / 4; // 4 or 8, registers are 32-bit words

        dtype data[packed_per_thread]; ///< The actual storage for the base tile
    };

    /* ----------  CONCEPTS  ---------- */

    namespace ducks
    {
        namespace rt_base
        {
            /**
             * @brief Concept for all register base tiles.
             * @tparam T The type to check against the concept requirements.
             *
             * Requires:
             * - T has a nested type identifier that is the same as rt_base::identifier.
             */
            template <typename T>
            concept all = requires {
                typename T::identifier;                              // Checks if T::identifier exists
            } && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::rt::identifier
        } // namespace rt
    } // namespace ducks

    /* ----------  WRAPPERS FOR PRETTINESS  ---------- */

    using rt_base_fl = rt_base<float>;
    using rt_base_bf = rt_base<bf16>;
    using rt_base_hf = rt_base<half>;
}