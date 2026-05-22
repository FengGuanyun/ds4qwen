// Q4 Metal common definitions shared across all kernels.
// Adapted from ds4's implicit common definitions.

#pragma once

#include <metal_stdlib>
using namespace metal;

// SIMD width (32 for Apple GPUs)
constexpr short N_SIMDWIDTH = 32;

// GGUF Q8_0 block format.
constexpr short QK8_0 = 32;
struct block_q8_0 {
    half    d;
    int8_t  qs[QK8_0];
};

// Number of output rows processed per threadgroup.
constexpr short N_R0_Q8_0 = 2;

// Helper: is_same
template<typename T, typename U> struct is_same { static constexpr bool value = false; };
template<typename T> struct is_same<T, T> { static constexpr bool value = true; };

// Helper: create filled simdgroup matrix
template<typename T, int N>
METAL_FUNC simdgroup_matrix<T, N, N> make_filled_simdgroup_matrix(T val) {
    simdgroup_matrix<T, N, N> m;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            m[i][j] = val;
    return m;
}

// Function constant base for dispatch specialization.
constexpr short FC_MUL_MV = 0;
constexpr short FC_MUL_MM = 2;
