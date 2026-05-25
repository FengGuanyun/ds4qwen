// Q4 Metal common definitions shared across all kernels.
// Adapted from ds4's implicit common definitions.

#include <metal_stdlib>
using namespace metal;

// SIMD width (32 for Apple GPUs)
constant short N_SIMDWIDTH = 32;

// GGUF Q8_0 block format.
constant short QK8_0 = 32;
struct block_q8_0 {
    half    d;
    int8_t  qs[QK8_0];
};

// Number of output rows processed per threadgroup.
constant short N_R0_Q8_0 = 2;

// Helper: create filled simdgroup matrix using Metal's built-in

// Function constant base for dispatch specialization.
constant short FC_MUL_MV = 0;
constant short FC_MUL_MM = 2;

// Shared matmul args struct
struct q4_metal_args_mul_mv {
    int ne00;
    int ne01;
    int ne02;
    ulong nb00;
    ulong nb01;
    ulong nb02;
    ulong nb03;
    int ne10;
    int ne11;
    int ne12;
    ulong nb10;
    ulong nb11;
    ulong nb12;
    ulong nb13;
    int ne0;
    int ne1;
    int nr0;
    short r2;
    short r3;
};
