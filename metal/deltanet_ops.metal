// Q4 DeltaNet-specific kernels.

struct q4_metal_args_deltanet_ops {
    uint32_t n;          // primary dimension
    uint32_t m;          // secondary dimension
    uint32_t n_v_heads;
    uint32_t head_dim;
    uint32_t n_k_groups;
    uint32_t conv_pos;
    float    eps;
};

// Vector-matrix Q4_K multiply: out[out_dim] = vec[in_dim] @ W[out_dim, in_dim]
// Each thread computes one output element, with input vector in threadgroup memory.
[[host_name("kernel_vec_matmul_q4k")]]
kernel void kernel_vec_matmul_q4k(
        device const uchar * weight,   // Q4_K: [out_dim, in_dim]
        device const float * vec,      // [in_dim]
        device       float * out,      // [out_dim]
        constant uint32_t & out_dim,
        constant uint32_t & in_dim,
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3 tptg[[threads_per_threadgroup]],
        threadgroup float * shmem) {
    const uint tid = tpitg.x + tptg.x * tptg.y;  // global thread ID within group
    const uint n_threads = tptg.x * tptg.y;

    /* QK_K = 256, already defined as macro from quant.metal */
    const int blocks_per_row = in_dim / QK_K;

    // Load input vector into threadgroup memory
    for (uint i = tid; i < in_dim; i += n_threads) {
        shmem[i] = vec[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread processes one output row
    uint row = tid;
    if (row >= out_dim) return;

    device const uchar *row_ptr = weight + (uint64_t)row * blocks_per_row * 144;

    float acc = 0.0f;
    for (int bi = 0; bi < blocks_per_row; bi++) {
        device const uchar *block_ptr = row_ptr + bi * 144;

        const float d  = q4k_f16_to_f32(((device const ushort *)block_ptr)[0]);
        const float dm = q4k_f16_to_f32(((device const ushort *)block_ptr)[1]);
        device const uchar *scales = block_ptr + 2;
        device const uchar *qs = block_ptr + 14;

        // Precompute scales and mn for all 8 groups
        float sc_arr[8], mn_arr[8];
        for (int g = 0; g < 8; g++) {
            if (g < 6) {
                sc_arr[g] = (float)(scales[g] & 0x3F);
                mn_arr[g] = (float)(scales[6 + g] & 0x3F);
            } else {
                const int gi = g - 6;
                sc_arr[g] = (float)((scales[4 + gi] >> 4) | ((scales[gi] & 0xC0) >> 2));
                mn_arr[g] = (float)((scales[10 + gi] >> 4) | ((scales[6 + gi] & 0xC0) >> 2));
            }
        }

        for (int grp = 0; grp < 8; grp++) {
            const float dmul = (grp >= 6) ? (d / 16.0f) : d;
            const float sc = sc_arr[grp];
            const float mn = mn_arr[grp];
            const int qs_off_grp = (grp / 2) * 32;
            const int base_elem = bi * QK_K + grp * 32;

            for (int sub = 0; sub < 2; sub++) {
                const int qs_off = qs_off_grp + sub * 16;
                const int shift = sub * 4;
                const int mask = sub == 0 ? 0x0F : 0xF0;
                const int elem_base = base_elem + sub * 16;
                for (int idx = 0; idx < 16; idx++) {
                    const uint8_t qbyte = qs[qs_off + idx];
                    const float q = (float)((qbyte & mask) >> shift);
                    const float w = dmul * sc * q - dm * mn;
                    acc += w * shmem[elem_base + idx];
                }
            }
        }
    }
    out[row] = acc;
}

// Fused conv1D + split + L2 norm + expand for DeltaNet.
// Phase 1: conv1D (parallelized over qkv_dim)
[[host_name("kernel_deltanet_conv_split")]]
kernel void kernel_deltanet_conv_split(
        device const float * qkv_raw,    // [qkv_dim]
        device const float * conv_buf,   // [conv_kernel, qkv_dim] ring buffer (read-only history)
        device       float * conv_buf_out, // [conv_kernel, qkv_dim] ring buffer (write new row)
        device const float * conv_w,     // [qkv_dim] conv1d weights
        device       float * q_exp,      // [n_v_heads, head_k_dim]
        device       float * k_exp,      // [n_v_heads, head_k_dim]
        device       float * v_out,      // [n_v_heads, head_v_dim]
        constant uint32_t & qkv_dim,
        constant uint32_t & n_k_groups,
        constant uint32_t & n_v_heads,
        constant uint32_t & head_k_dim,
        constant uint32_t & head_v_dim,
        constant uint32_t & repeat,
        constant uint32_t & conv_pos,
        uint tid[[thread_position_in_grid]]) {

    const uint qi = tid;

    if (qi >= qkv_dim) return;

    // Write qkv_raw into conv_buf at current position
    conv_buf_out[conv_pos * qkv_dim + qi] = qkv_raw[qi];

    // Conv1D: conv_qkv[qi] = sum_{k=0}^{3} conv_w[k*qkv_dim + qi] * conv_buf[(pos-1-k)%4][qi]
    float qkv_val = 0.0f;
    for (uint k = 0; k < 4; k++) {
        const int ki = (int)conv_pos - 1 - (int)k;
        const uint ri = ((ki % 4) + 4) % 4;
        qkv_val += conv_w[k * qkv_dim + qi] * conv_buf[ri * qkv_dim + qi];
    }

    const uint k_dim = n_k_groups * head_k_dim;  // 2048

    if (qi < k_dim) {
        // Q path: normalize per group and expand
        const uint g = qi / head_k_dim;
        const uint d = qi % head_k_dim;
        // Compute L2 norm for q group
        float nq = 0.0f;
        for (uint dd = 0; dd < head_k_dim; dd++) {
            float v = qkv_raw[g * head_k_dim + dd];
            nq += v * v;
        }
        nq = sqrt(nq);
        float inv_nq = (nq > 1.0e-6f) ? (1.0f / nq) : 1.0f;
        float q_val = qkv_raw[g * head_k_dim + d] * inv_nq;
        for (uint r = 0; r < repeat; r++) {
            const uint vi = g * repeat + r;
            if (vi < n_v_heads) {
                q_exp[vi * head_k_dim + d] = q_val;
            }
        }

        // K path: same for k group
        float nk = 0.0f;
        for (uint dd = 0; dd < head_k_dim; dd++) {
            float v = qkv_raw[k_dim + g * head_k_dim + dd];
            nk += v * v;
        }
        nk = sqrt(nk);
        float inv_nk = (nk > 1.0e-6f) ? (1.0f / nk) : 1.0f;
        float k_val = qkv_raw[k_dim + g * head_k_dim + d] * inv_nk;
        for (uint r = 0; r < repeat; r++) {
            const uint vi = g * repeat + r;
            if (vi < n_v_heads) {
                k_exp[vi * head_k_dim + d] = k_val;
            }
        }
    }

    // V path: copy conv result
    if (qi >= 2 * k_dim && qi < qkv_dim) {
        const uint vi = (qi - 2 * k_dim) / head_v_dim;
        const uint d  = (qi - 2 * k_dim) % head_v_dim;
        if (vi < n_v_heads && d < head_v_dim) {
            v_out[vi * head_v_dim + d] = qkv_val;
        }
    }
}

// Gate transforms: gate = softplus(alpha + bias) * a, beta = sigmoid(beta_raw)
[[host_name("kernel_deltanet_gate_transform")]]
kernel void kernel_deltanet_gate_transform(
        device const float * alpha_raw,
        device const float * beta_raw,
        device       float * gate_out,
        device       float * beta_out,
        device const float * dt_bias,
        device const float * ssm_a,
        constant uint32_t & n,
        uint tid[[thread_position_in_grid]]) {
    if (tid >= n) return;

    // Softplus: log(1 + exp(-|x|)) + max(x, 0)
    float biased = alpha_raw[tid] + dt_bias[tid];
    float abs_b = fabs(biased);
    float sp = log(1.0f + exp(-abs_b)) + fmax(biased, 0.0f);
    gate_out[tid] = sp * ssm_a[tid];

    // Sigmoid
    float b = beta_raw[tid];
    beta_out[tid] = 1.0f / (1.0f + exp(-b));
}

// Delta rule per v_head: sk = state @ k; delta = v - sk; state = state*exp(g) + b*outer(delta,k); out = state @ k
// One threadgroup per v_head.
[[host_name("kernel_delta_rule")]]
kernel void kernel_delta_rule(
        device       float * state,      // [n_v_heads, head_v_dim, head_k_dim] in/out
        device const float * k_exp,      // [n_v_heads, head_k_dim]
        device const float * v_raw,      // [n_v_heads, head_v_dim]
        device const float * gate,       // [n_v_heads]
        device const float * beta,       // [n_v_heads]
        device       float * output,     // [n_v_heads, head_v_dim]
        constant uint32_t & n_v_heads,
        constant uint32_t & head_v_dim,
        constant uint32_t & head_k_dim,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3 tptg[[threads_per_threadgroup]]) {

    const uint vi = tgpig.x;
    if (vi >= n_v_heads) return;

    const uint hvd = head_v_dim;
    const uint hkd = head_k_dim;

    device float *state_i = state + vi * hvd * hkd;
    device const float *k_i = k_exp + vi * hkd;
    device const float *v_i = v_raw + vi * hvd;

    const float gi = gate[vi];
    const float bi = beta[vi];
    const float eg = exp(gi);

    // Phase 1: sk = state @ k, delta = v - sk
    threadgroup float delta_arr[128];

    for (uint d = tpitg.x; d < hvd; d += tptg.x) {
        float acc = 0.0f;
        for (uint j = 0; j < hkd; j++) {
            acc += state_i[d * hkd + j] * k_i[j];
        }
        delta_arr[d] = v_i[d] - acc;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: state = state * eg + bi * outer(delta, k)
    // hvd * hkd = 128 * 128 = 16384 elements, 32 threads => 512 per thread
    for (uint idx = tpitg.x; idx < hvd * hkd; idx += tptg.x) {
        const uint d = idx / hkd;
        const uint j = idx % hkd;
        state_i[d * hkd + j] = state_i[d * hkd + j] * eg + bi * delta_arr[d] * k_i[j];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: output = state @ k
    for (uint d = tpitg.x; d < hvd; d += tptg.x) {
        float acc = 0.0f;
        for (uint j = 0; j < hkd; j++) {
            acc += state_i[d * hkd + j] * k_i[j];
        }
        output[vi * hvd + d] = acc;
    }
}

// SiLU gate + RMS norm per v_head (matches CPU path exactly)
[[host_name("kernel_deltanet_silu_rms")]]
kernel void kernel_deltanet_silu_rms(
        device       float * inout,      // [n_v_heads, head_v_dim] in: output, out: silu+norm result
        device const float * z_raw,      // [n_v_heads, head_v_dim] SiLU gate
        device const float * ssm_norm_w, // [head_v_dim] RMS norm weight
        constant uint32_t & n_v_heads,
        constant uint32_t & head_v_dim,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3 tptg[[threads_per_threadgroup]]) {

    const uint vi = tgpig.x;
    if (vi >= n_v_heads) return;

    const uint hvd = head_v_dim;
    device float *out_i = inout + vi * hvd;
    device const float *z_i = z_raw + vi * hvd;

    // SiLU gate: out *= z * sigmoid(z)
    for (uint d = tpitg.x; d < hvd; d += tptg.x) {
        const float z = z_i[d];
        out_i[d] *= z / (1.0f + exp(-z));
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // RMS norm per head
    float ss = 0.0f;
    for (uint d = tpitg.x; d < hvd; d += tptg.x) {
        ss += out_i[d] * out_i[d];
    }

    // Reduce
    threadgroup float shmem[32];
    shmem[tpitg.x] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ss = 0.0f;
    for (uint d = 0; d < 32; d++) ss += shmem[d];

    const float rms = sqrt(ss / hvd + 1e-5f);
    const float scale = 1.0f / rms;

    for (uint d = tpitg.x; d < hvd; d += tptg.x) {
        out_i[d] = out_i[d] * scale * ssm_norm_w[d];
    }
}

// Vector-matrix Q5_K multiply: out[out_dim] = vec[in_dim] @ W[out_dim, in_dim]
// Q5_K block: d(2) + dmin(2) + scales(12) + qh(32) + qs(128) = 176 bytes
[[host_name("kernel_vec_matmul_q5k")]]
kernel void kernel_vec_matmul_q5k(
        device const uchar * weight,   // Q5_K: [out_dim, in_dim]
        device const float * vec,      // [in_dim]
        device       float * out,      // [out_dim]
        constant uint32_t & out_dim,
        constant uint32_t & in_dim,
        uint tid[[thread_position_in_grid]]) {
    if (tid >= out_dim) return;

    /* QK_K = 256, already defined as macro from quant.metal */
    const int blocks_per_row = in_dim / QK_K;

    device const uchar *row_ptr = weight + (uint64_t)tid * blocks_per_row * 176;

    float acc = 0.0f;
    for (int bi = 0; bi < blocks_per_row; bi++) {
        device const uchar *block_ptr = row_ptr + bi * 176;
        device const ushort *d_ptr = (device const ushort *)block_ptr;
        device const uchar *scales = block_ptr + 4;
        device const uchar *qh = block_ptr + 16;
        device const uchar *qs = block_ptr + 48;

        const float d = q4k_f16_to_f32(d_ptr[0]);
        const float dm = q4k_f16_to_f32(d_ptr[1]);

        for (int grp = 0; grp < 8; grp++) {
            const bool hi = grp >= 6;
            const float dmul = hi ? (d / 16.0f) : d;

            float sc, mn;
            if (grp < 6) {
                sc = (float)(scales[grp] & 0x3F);
                mn = (float)(scales[6 + grp] & 0x3F);
            } else {
                const int gi = grp - 6;
                sc = (float)((scales[4 + gi] >> 4) | ((scales[gi] & 0xC0) >> 2));
                mn = (float)((scales[10 + gi] >> 4) | ((scales[6 + gi] & 0xC0) >> 2));
            }

            const int qs_off_grp = (grp / 2) * 32;
            for (int sub = 0; sub < 2; sub++) {
                const uint8_t mask = sub == 0 ? 0x0F : 0xF0;
                const int qs_off = qs_off_grp + sub * 16;
                for (int idx = 0; idx < 16; idx++) {
                    const uint8_t qbyte = qs[qs_off + idx];
                    const int e = bi * QK_K + grp * 32 + sub * 16 + idx;
                    uint8_t q = (qbyte & mask) >> (sub * 4);
                    // High bit from qh: 1 bit per element
                    if (e < in_dim) {
                        const uint8_t h = (qh[e / 8] >> (e % 8)) & 0x01;
                        q |= (h << 4);
                        const float w = dmul * (float)q - dm * mn;
                        acc += w * vec[e];
                    }
                }
            }
        }
    }
    out[tid] = acc;
}
