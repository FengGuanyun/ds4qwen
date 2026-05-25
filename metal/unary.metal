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

// SiLU with clamping + multiply: out[i] = clamp_silu(gate[i]) * up[i]
// Clamps gate to [-clamp, clamp] before applying SiLU, then multiplies by up.
[[host_name("kernel_silu_clamped_mul")]]
kernel void kernel_silu_clamped_mul(
        device       float * out,
        device const float * gate,
        device const float * up,
        constant     float & clamp,
        constant uint64_t & n,
        uint tid[[thread_position_in_grid]]) {
    if (tid >= n) return;
    float g = gate[tid];
    if (g > clamp) g = clamp;
    if (g < -clamp) g = -clamp;
    out[tid] = (g / (1.0f + exp(-g))) * up[tid];
}

// Softplus: out[i] = log(1 + exp(-|x[i]|)) + max(x[i], 0)
// Numerically stable version: log1p(exp(-|x|)) + max(x, 0)
[[host_name("kernel_unary_softplus")]]
kernel void kernel_unary_softplus(
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
        const float abs_x = fabs(x);
        // log(1 + exp(-|x|)) - safe from overflow since exp(-|x|) <= 1
        dst[offset + i] = log(1.0f + exp(-abs_x)) + fmax(x, 0.0f);
    }
}

// Delta rule: recurrent state update for Gated DeltaNet decode.
// Per v_head: sk = state @ k; delta = v - sk; state = state*exp(g) + b*outer(delta,k); out = state @ k
// Then: out *= silu(z), RMS norm per head.
// Each threadgroup handles one v_head.
[[host_name("kernel_delta_rule_decode")]]
kernel void kernel_delta_rule_decode(
        device const float * state_in,    // [n_v_heads, head_v_dim, head_k_dim] input state
        device       float * state_out,   // [n_v_heads, head_v_dim, head_k_dim] output state
        device const float * k_exp,       // [n_v_heads, head_k_dim] expanded keys
        device const float * v_raw,       // [n_v_heads, head_v_dim] values
        device const float * gate,        // [n_v_heads] decay gate
        device const float * beta,        // [n_v_heads] input gate (sigmoid)
        device const float * z_raw,       // [n_v_heads, head_v_dim] SiLU gate
        device const float * ssm_norm_w,  // [head_v_dim] RMS norm weight
        device       float * output,      // [n_v_heads, head_v_dim] output
        constant uint32_t & n_v_heads,
        constant uint32_t & head_v_dim,
        constant uint32_t & head_k_dim,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3 tptg[[threads_per_threadgroup]]) {
    const uint vi = tgpig.x;  // v_head index
    if (vi >= n_v_heads) return;

    const uint hvd = head_v_dim;
    const uint hkd = head_k_dim;

    device const float *state_i = state_in + vi * hvd * hkd;
    device const float *k_i = k_exp + vi * hkd;
    device const float *v_i = v_raw + vi * hvd;
    device const float *z_i = z_raw + vi * hvd;
    device float *out_i = output + vi * hvd;
    device float *state_o_i = state_out + vi * hvd * hkd;

    const float gi = gate[vi];
    const float bi = beta[vi];
    const float eg = exp(gi);

    // Compute sk = state @ k, delta = v - sk
    // Each thread computes one element of delta
    for (uint d = tpitg.x; d < hvd; d += tptg.x) {
        float acc = 0.0f;
        for (uint j = 0; j < hkd; j++) {
            acc += state_i[d * hkd + j] * k_i[j];
        }
        // delta[d] stored in a local variable, we'll use it below
        // Actually we need it for state update too, so let's store to output temporarily
        out_i[d] = v_i[d] - acc;  // out_i holds delta temporarily
    }

    // Synchronize threadgroup before state update
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // State update: state_o = state*eg + b*outer(delta, k)
    // delta is in out_i, we need to read it back
    // Each thread updates one element of the state matrix
    // We need hvd * hkd threads, but we only have hvd threads per threadgroup
    // So each thread updates one row of the state matrix
    for (uint d = tpitg.x; d < hvd; d += tptg.x) {
        const float delta_d = out_i[d];
        for (uint j = 0; j < hkd; j++) {
            state_o_i[d * hkd + j] = state_i[d * hkd + j] * eg + bi * delta_d * k_i[j];
        }
    }

    // Compute output = state_o @ k
    for (uint d = tpitg.x; d < hvd; d += tptg.x) {
        float acc = 0.0f;
        for (uint j = 0; j < hkd; j++) {
            acc += state_o_i[d * hkd + j] * k_i[j];
        }
        out_i[d] = acc;  // output

        // SiLU gate: out *= z * sigmoid(z)
        const float z = z_i[d];
        out_i[d] *= z / (1.0f + exp(-z));

        // RMS norm per head: out /= rms * ssm_norm_w
        // We need to compute the sum of squares first
        // This requires a reduction across threads, which is complex
        // For now, skip per-head RMS norm and do it later
    }
}

// RMS norm per head: applies RMSNorm independently to each v_head's output.
[[host_name("kernel_rms_norm_per_head")]]
kernel void kernel_rms_norm_per_head(
        device       float * inout,      // [n_v_heads, head_v_dim] input/output
        device const float * weight,     // [head_v_dim] norm weights
        constant uint32_t & n_v_heads,
        constant uint32_t & head_v_dim,
        constant float & eps,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3 tptg[[threads_per_threadgroup]]) {
    const uint vi = tgpig.x;
    if (vi >= n_v_heads) return;

    const uint hvd = head_v_dim;
    device float *out_i = inout + vi * hvd;
    device const float *w_i = weight;

    // Compute sum of squares
    float ss = 0.0f;
    for (uint d = tpitg.x; d < hvd; d += tptg.x) {
        ss += out_i[d] * out_i[d];
    }

    // Reduce across threadgroup
    threadgroup float shmem[32];
    shmem[tpitg.x] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = 0.0f;
    for (uint d = 0; d < 32 && d < hvd; d++) ss += shmem[d];

    const float rms = sqrt(ss / hvd + eps);
    const float scale = 1.0f / rms;

    // Apply scale and weight
    for (uint d = tpitg.x; d < hvd; d += tptg.x) {
        out_i[d] = out_i[d] * scale * w_i[d];
    }
}
