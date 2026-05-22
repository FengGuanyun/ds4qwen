// Q4 Full RoPE kernel - applies rotary position embedding to all head_dim dimensions.
// Qwen3.6 uses full RoPE (not partial/tail-only like ds4).

struct q4_metal_args_rope {
    int32_t  ne0;
    int32_t  ne1;
    uint64_t nb1;
    float    freq_base;
    float    freq_scale;
    uint32_t pos0;
    uint32_t head_dim;
    uint32_t n_head;
};

// Full RoPE: rotates each pair of dimensions (2i, 2i+1) by freq * (pos0 + row/n_head).
// Only processes the first head_dim dimensions (the rotary part).
kernel void kernel_rope_full(
        constant q4_metal_args_rope & args,
        device const char * src0,
        device       char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const uint row = tgpig.x;  // head index within the batch
    const uint batch = tgpig.y; // batch index
    const uint n_rows = args.ne1; // total heads * batch

    if (row >= args.ne1) return;

    const uint pos = args.pos0 + batch;
    device const float *x = (device const float *)(src0 + row * args.nb1);
    device float *y = (device float *)(dst + row * args.nb1);

    const float scale = 1.0f / args.freq_scale;

    // Each thread processes pairs of dimensions
    for (uint i = tpitg.x; i < args.head_dim / 2; i += tptg.x) {
        const float freq = scale / pow(args.freq_base, 2.0f * i / args.head_dim);
        const float theta = freq * pos;
        const float cos_theta = cos(theta);
        const float sin_theta = sin(theta);

        const float x0 = x[2 * i];
        const float x1 = x[2 * i + 1];

        y[2 * i]     = x0 * cos_theta - x1 * sin_theta;
        y[2 * i + 1] = x0 * sin_theta + x1 * cos_theta;
    }

    // Copy non-rotary dimensions as-is
    for (uint i = tpitg.x + args.head_dim; i < args.ne0; i += tptg.x) {
        y[i] = x[i];
    }
}

// Vectorized RoPE with float4 loads for prefill.
kernel void kernel_rope_full_4(
        constant q4_metal_args_rope & args,
        device const char * src0,
        device       char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const uint row = tgpig.x;
    const uint batch = tgpig.y;

    if (row >= args.ne1) return;

    const uint pos = args.pos0 + batch;
    device const float4 *x4 = (device const float4 *)(src0 + row * args.nb1);
    device float4 *y4 = (device float4 *)(dst + row * args.nb1);

    const float scale = 1.0f / args.freq_scale;
    const uint n_pairs = args.head_dim / 2;

    // Process 4 dimensions (2 pairs) per iteration
    for (uint i = tpitg.x; i < n_pairs / 2; i += tptg.x) {
        const uint p0 = i * 2;
        const uint p1 = i * 2 + 1;

        const float f0 = scale / pow(args.freq_base, 2.0f * p0 / args.head_dim);
        const float f1 = scale / pow(args.freq_base, 2.0f * p1 / args.head_dim);

        const float4 v = x4[i];
        float4 out;

        float theta = f0 * pos;
        float c = cos(theta), s = sin(theta);
        out[0] = v[0] * c - v[1] * s;
        out[1] = v[0] * s + v[1] * c;

        theta = f1 * pos;
        c = cos(theta); s = sin(theta);
        out[2] = v[2] * c - v[3] * s;
        out[3] = v[2] * s + v[3] * c;

        y4[i] = out;
    }
}
