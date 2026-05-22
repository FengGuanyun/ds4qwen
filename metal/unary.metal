// Q4 unary elementwise kernels.

struct q4_metal_args_unary {
    int32_t  ne00;
    uint64_t nb1;
};

// SiLU: out[i] = x[i] / (1 + exp(-x[i]))
[[host_name("kernel_unary_silu")]]
kernel void kernel_unary_silu(
        constant q4_metal_args_unary & args,
        device const float * src0,
        device       float * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3 tptg[[threads_per_threadgroup]]) {
    const uint row = tgpig.x;
    const uint offset = row * args.nb1 / sizeof(float);

    for (uint i = tpitg.x; i < args.ne00; i += tptg.x) {
        const float x = src0[offset + i];
        dst[offset + i] = x / (1.0f + exp(-x));
    }
}

// Sigmoid: out[i] = 1 / (1 + exp(-x[i]))
[[host_name("kernel_unary_sigmoid")]]
kernel void kernel_unary_sigmoid(
        constant q4_metal_args_unary & args,
        device const float * src0,
        device       float * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3 tptg[[threads_per_threadgroup]]) {
    const uint row = tgpig.x;
    const uint offset = row * args.nb1 / sizeof(float);

    for (uint i = tpitg.x; i < args.ne00; i += tptg.x) {
        const float x = src0[offset + i];
        dst[offset + i] = 1.0f / (1.0f + exp(-x));
    }
}

// Add scalar: out[i] = x[i] + s
[[host_name("kernel_unary_add_scalar")]]
kernel void kernel_unary_add_scalar(
        constant q4_metal_args_unary & args,
        device const float * src0,
        device       float * dst,
        constant     float & scale,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3 tptg[[threads_per_threadgroup]]) {
    const uint row = tgpig.x;
    const uint offset = row * args.nb1 / sizeof(float);

    for (uint i = tpitg.x; i < args.ne00; i += tptg.x) {
        dst[offset + i] = src0[offset + i] + scale;
    }
}
