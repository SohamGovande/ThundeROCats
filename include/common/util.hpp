/**
 * @file
 * @brief General utilities for ThunderKittens.
 */

#pragma once

#include <concepts>
#include <memory>
#include <stdint.h>
#include <type_traits>

/**
 * @namespace kittens
 *
 * @brief The main namespace of ThunderKittens.
 */
namespace kittens {

/* ----------  GENERAL CONSTANTS FOR KITTENS  ---------- */

/**
 * @brief Constant representing number of threads in a warp.
 */
constexpr int WAVE_THREADS{64};
/**

 * @brief Get the warp ID of the current thread.
 * @return The warp ID.
 */
__device__ __forceinline__ int waveid() { return threadIdx.x / WAVE_THREADS; }
/**
 * @brief Get the lane ID of the current thread within its warp.
 * @return The lane ID.
 */
__device__ __forceinline__ int laneid() { return threadIdx.x % WAVE_THREADS; }

#ifdef KITTENS_MI355X
constexpr int MAX_SHARED_MEMORY = 163840; // 160KB for CDNA 4
#else
constexpr int MAX_SHARED_MEMORY = 65536; // 64KB for MI300 and below
#endif

struct transpose {
  static constexpr int N = 0; // not transposed
  static constexpr int T = 1; // transposed
};
struct axis {
  static constexpr int ROW = 0; // row axis of a tile
  static constexpr int COL = 1; // column axis of a tile
};

/* ----------  TYPE HELPERS  ---------- */

/**
 * @namespace ducks
 *
 * @brief ThunderKittens' namespace for template metaprogramming..
 *
 * This includes primarily dummy types and concept wrappers, along
 * with a few additional utilities.
 */
namespace ducks {

/**
 * @brief A type representing an empty default for a template.
 */
struct default_type {};

// This macro can't be done as a template, so it doesn't really have a location in kittens.
#define typeof(A) typename std::remove_const<typename std::remove_reference<decltype(A)>::type>::type

} // namespace ducks

/* ----------  SHUFFLE UTILS  ---------- */

/**
 * @brief Mask constant for all active threads in a warp.
 */
static constexpr uint32_t MASK_ALL = 0xFFFFFFFF;

// Joyously stolen from https://github.com/NVIDIA/cutlass/blob/5c447dd84f8ae0e1d48ff9a2eae26ce8c4958101/include/cute/container/alignment.hpp#L51
#define KITTENS_ALIGN_AS(n) alignas(n)

#define KITTENS_DEFAULT_ALIGN KITTENS_ALIGN_AS(16)

/**
 * @brief Dummy structure for alignment purposes. Needed for WGMMA and TMA calls.
 */
struct KITTENS_DEFAULT_ALIGN alignment_dummy {
  int dummy;
};
/**
 * @brief Very simple allocator for dynamic shared memory. Advances pointer and tracks alignments.
 * @tparam default_alignment The default alignment this allocator will enforce. If <=0 (default -1) it will not align.
 */
template <int default_alignment = 16>
struct shared_allocator {
  int *ptr;

private:
  // Recursive template to generate N-dimensional array type
  template <typename A, size_t... dims>
  struct variadic_array;
  template <typename A, size_t first_dim, size_t... rest_dims>
  struct variadic_array<A, first_dim, rest_dims...> {
    using type = typename variadic_array<A, rest_dims...>::type[first_dim];
  };
  template <typename A>
  struct variadic_array<A> {
    using type = A;
  };
  template <typename A, size_t... dims>
  using variadic_array_t = typename variadic_array<A, dims...>::type;

  template <int alignment>
  __device__ inline void align_ptr() {
    if constexpr (alignment > 0) {
      uint64_t p = reinterpret_cast<uint64_t>(ptr);
      if (p % alignment != 0) {
        ptr = (int *)(p + (alignment - (p % alignment)));
      }
    }
  }

public:
  /**
   * @brief Construct a new shared allocator using a pointer to extern shared memory.
   * @param[in] _ptr Pointer to the start of the extern shared memory.
   */
  __device__ shared_allocator(int *_ptr) : ptr(_ptr) {}
  /**
   * @brief Allocate shared memory for a single instance or N-dimensional array of type A.
   * @tparam A The type of the object to allocate.
   * @tparam dims... A list of dimensions for the N-dimensional array.
   * @return Reference to the allocated object.
   */
  template <typename A, size_t... dims>
  __device__ inline variadic_array_t<A, dims...> &allocate() {
    // static_assert(sizeof(A) % default_alignment == 0, "Type is not aligned properly for array allocation");
    align_ptr<default_alignment>();
    using at = variadic_array_t<A, dims...>;
    at *p = reinterpret_cast<at *>(ptr);
    ptr += sizeof(at) / sizeof(int);
    return *p;
  }
  /**
   * @brief Allocate shared memory for a single instance or N-dimensional array of type A.
   * @tparam alignment An alignment to enforce for this particular object.
   * @tparam A The type of the object to allocate.
   * @tparam dims... A list of dimensions for the N-dimensional array.
   * @return Reference to the allocated object.
   */
  template <int alignment, typename A, size_t... dims>
  __device__ inline variadic_array_t<A, dims...> &allocate() {
    // static_assert(sizeof(A) % alignment == 0, "Type is not aligned properly for array allocation");
    align_ptr<alignment>();
    using at = variadic_array_t<A, dims...>;
    at *p = reinterpret_cast<at *>(ptr);
    ptr += sizeof(at) / sizeof(int);
    return *p;
  }
};

} // namespace kittens