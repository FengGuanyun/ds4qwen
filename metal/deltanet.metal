// Q4 Gated DeltaNet kernel.
// Implements the linear recurrent state update for Qwen3.6-27B's hybrid architecture.
//
// For 48 of 64 layers, Qwen3.6 uses Gated DeltaNet instead of full attention.
// The recurrent state update is:
//   s' = (1 - dt) * s + dt * (b * v)
//   output = s'
//
// This is O(n) in context length but has sequential dependency.

struct q4_metal_args_deltanet {
    uint32_t n_embd;
    uint32_t head_dim;
    uint32_t n_kv_heads;
    uint32_t n_q_heads;
    uint32_t n_tokens;   // 1 for decode, >1 for prefill
    uint32_t q_per_kv;
};

// DeltaNet decode: single token state update.
// Each threadgroup handles one KV head.
[[host_name("kernel_deltanet_step")]]
kernel void kernel_deltanet_step(
        constant q4_metal_args_deltanet & args,
        device const float * x,          // [n_embd] input activation
        device const float * a_gate,     // [n_kv_heads, head_dim] decay gate weights
        device const float * b_proj,     // [n_kv_heads, head_dim] input projection
        device const float * dt_gate,    // [n_kv_heads, head_dim] timestep
        device       float * state,      // [n_kv_heads, head_dim] recurrent state (in/out)
        device       float * out,        // [n_q_heads, head_dim] output (broadcast from KV)
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3 tptg[[threads_per_threadgroup]]) {
    const uint kv_head = tgpig.x;

    if (kv_head >= args.n_kv_heads) return;

    const uint base = kv_head * args.head_dim;
    const float scale_factor = 1.0f / args.n_embd;

    // Compute x projected to head_dim (simplified: assume x is already projected)
    // In practice, W_a @ h_norm, W_b @ h_norm, W_dt @ h_norm are computed before this kernel

    // Update state: s' = (1 - dt) * s + dt * (b * v)
    // Here x serves as the combined input (already through projections)
    for (uint d = tpitg.x; d < args.head_dim; d += tptg.x) {
        const uint idx = base + d;
        const float a = a_gate[idx];
        const float b = b_proj[idx];
        const float dt = dt_gate[idx];
        const float v = x[idx];  // simplified: v comes from the same projection

        const float s = state[idx];
        const float s_new = (1.0f - dt) * s + dt * b * v;
        state[idx] = s_new;
    }

    // Broadcast KV head state to Q heads
    for (uint qi = 0; qi < args.q_per_kv; qi++) {
        const uint q_head = kv_head * args.q_per_kv + qi;
        if (q_head >= args.n_q_heads) break;

        for (uint d = tpitg.x; d < args.head_dim; d += tptg.x) {
            out[q_head * args.head_dim + d] = state[base + d];
        }
    }
}

// DeltaNet prefill: process n_tokens sequentially.
// Each threadgroup handles one KV head. Sequential token loop inside kernel.
[[host_name("kernel_deltanet_prefill")]]
kernel void kernel_deltanet_prefill(
        constant q4_metal_args_deltanet & args,
        device const float * q_in,       // [n_tokens, n_q_heads, head_dim]
        device const float * k_in,       // [n_tokens, n_kv_heads, head_dim]
        device const float * v_in,       // [n_tokens, n_kv_heads, head_dim]
        device const float * a_gate,     // [n_tokens, n_kv_heads, head_dim]
        device const float * b_proj,     // [n_tokens, n_kv_heads, head_dim]
        device const float * dt_gate,    // [n_tokens, n_kv_heads, head_dim]
        device       float * state,      // [n_kv_heads, head_dim] recurrent state (in/out)
        device       float * out,        // [n_tokens, n_q_heads, head_dim]
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3 tptg[[threads_per_threadgroup]]) {
    const uint kv_head = tgpig.x;

    if (kv_head >= args.n_kv_heads) return;

    const uint base = kv_head * args.head_dim;

    // Process tokens sequentially (can't parallelize across tokens for recurrence)
    for (uint t = 0; t < args.n_tokens; t++) {
        const uint tok_offset_q = t * args.n_q_heads * args.head_dim;
        const uint tok_offset_kv = t * args.n_kv_heads * args.head_dim;

        for (uint d = tpitg.x; d < args.head_dim; d += tptg.x) {
            const uint idx = base + d;
            const uint kv_idx = tok_offset_kv + idx;

            const float a = a_gate[kv_idx];
            const float b = b_proj[kv_idx];
            const float dt_val = dt_gate[kv_idx];
            const float v = v_in[kv_idx];

            const float s = state[idx];
            const float s_new = (1.0f - dt_val) * s + dt_val * b * v;
            state[idx] = s_new;

            // Broadcast to Q heads
            for (uint qi = 0; qi < args.q_per_kv; qi++) {
                const uint q_head = kv_head * args.q_per_kv + qi;
                if (q_head < args.n_q_heads) {
                    const uint q_idx = tok_offset_q + q_head * args.head_dim + d;
                    out[q_idx] = s_new;
                }
            }
        }

        // Synchronize before next token
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
