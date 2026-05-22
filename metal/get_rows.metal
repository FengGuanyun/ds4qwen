// Q4 token embedding lookup kernel.

struct q4_metal_args_get_rows {
    uint32_t n_vocab;
    uint32_t n_embd;
    uint64_t row_bytes;
};

// Lookup token embeddings: out[i] = token_embd[tokens[i]]
[[host_name("kernel_get_rows_f32_i32")]]
kernel void kernel_get_rows_f32_i32(
        constant q4_metal_args_get_rows & args,
        device const char * src0,
        device const int  * src1,
        device       char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3 tptg[[threads_per_threadgroup]]) {
    const uint token_idx = tgpig.x;
    const uint token = src1[token_idx];

    if (token >= args.n_vocab) return;

    device const float *x = (device const float *)(src0 + token * args.row_bytes);
    device float *y = (device float *)(dst + token_idx * args.n_embd * sizeof(float));

    const uint n_floats = args.n_embd;
    for (uint i = tpitg.x; i < n_floats; i += tptg.x) {
        y[i] = x[i];
    }
}
