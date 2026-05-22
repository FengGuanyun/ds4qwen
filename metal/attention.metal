// Q4 Flash Attention kernel for GQA (Grouped Query Attention).
// Supports Qwen3.6-27B with 24Q / 4KV heads.

struct q4_metal_args_flash_attn {
    uint32_t n_q_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t n_tokens;     // number of query tokens (1 for decode)
    uint32_t kv_len;       // current KV cache length
    uint32_t q_per_kv;     // n_q_heads / n_kv_heads = 6 for Qwen3.6-27B
    float    scale;
    float    logit_softcap;
    uint64_t q_stride;     // bytes per query head
    uint64_t k_stride;     // bytes per KV head in cache
    uint64_t v_stride;     // bytes per KV head in cache
    uint64_t out_stride;   // bytes per output head
};

// GQA Flash Attention (decode path: n_tokens=1).
// Computes softmax(Q @ K.T / sqrt(d) + softcap) @ V for each Q head,
// broadcasting KV heads across Q heads.
[[host_name("kernel_flash_attn_gqa_decode")]]
kernel void kernel_flash_attn_gqa_decode(
        constant q4_metal_args_flash_attn & args,
        device const float * Q,            // [n_q_heads, head_dim]
        device const float * K_cache,      // [kv_len, n_kv_heads, head_dim]
        device const float * V_cache,      // [kv_len, n_kv_heads, head_dim]
        device       float * out,          // [n_q_heads, head_dim]
        threadgroup float * shmem [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3 tptg[[threads_per_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    const uint q_head_idx = tgpig.x; // which Q head this threadgroup handles

    if (q_head_idx >= args.n_q_heads) return;

    // Map Q head to KV head (GQA broadcasting)
    const uint kv_head_idx = q_head_idx / args.q_per_kv;

    device const float *q = Q + q_head_idx * args.head_dim;
    device const float *k = K_cache + kv_head_idx * args.k_stride;
    device const float *v = V_cache + kv_head_idx * args.v_stride;

    const float inv_sqrt_d = args.scale;

    // Compute attention scores: Q @ K.T for each KV position
    threadgroup float *scores = shmem;
    threadgroup float *weights = shmem + args.kv_len;

    float lmax = -INFINITY;
    for (uint pos = tpitg.x; pos < args.kv_len; pos += tptg.x) {
        float score = 0.0f;
        for (uint d = 0; d < args.head_dim; d++) {
            score += q[d] * k[pos * args.head_dim + d];
        }
        score *= inv_sqrt_d;
        if (args.logit_softcap > 1.0e-6f) {
            score = args.logit_softcap * tanh(score / args.logit_softcap);
        }
        scores[pos] = score;
        lmax = max(lmax, score);
    }

    // Reduce max across threads
    float max_val = simd_max(lmax);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tpitg.x == 0 && tiisg == 0) weights[0] = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_val = simd_sum(max_val > weights[0] ? max_val : weights[0]); // approximate

    // Compute exp and sum
    float lsum = 0.0f;
    for (uint pos = tpitg.x; pos < args.kv_len; pos += tptg.x) {
        const float e = exp(scores[pos] - max_val);
        weights[pos] = e;
        lsum += e;
    }

    float sum = simd_sum(lsum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tpitg.x == 0 && tiisg == 0) {
        for (uint i = 0; i < tptg.x; i++) sum += weights[i];
        weights[args.kv_len] = sum; // store sum at end
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_sum = 1.0f / weights[args.kv_len];

    // Compute output: weighted sum of V
    for (uint d = tpitg.x; d < args.head_dim; d += tptg.x) {
        float acc = 0.0f;
        for (uint pos = 0; pos < args.kv_len; pos++) {
            acc += weights[pos] * v[pos * args.head_dim + d];
        }
        out[q_head_idx * args.head_dim + d] = acc * inv_sum;
    }
}
