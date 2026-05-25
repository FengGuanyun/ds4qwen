// Q4 quantized matmul kernels - Q4_K and Q6_K weights with F32 activations.
// Follows ds4's dequantize-on-the-fly pattern for decode matvec.

// common.metal inlined above

#define QK_K 256

struct block_q4_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[QK_K / 2];
};

struct block_q6_K {
    uchar ql[QK_K / 2];
    uchar qh[QK_K / 4];
    char scales[QK_K / 16];
    half d;
};

/* Manual FP16 -> FP32 conversion matching the CPU's f16_to_f32().
 * Metal's (float)half type uses a different byte ordering than GGUF's FP16. */
static float q4k_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            exp = 1;
            while ((mant & 0x0400) == 0) { mant <<= 1; exp--; }
            mant &= 0x03ff;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    /* Use thread pointer cast to avoid as_type issues */
    thread uint32_t bits_copy = bits;
    return *(thread float *)&bits_copy;
}

/* =========================================================================
 * Q4_K matrix-vector multiply.
 * out[out_row] = x[in_dim] @ W[in_dim, out_row] where W is Q4_K.
 *
 * For decode (single input vector): each threadgroup computes NR0=2 output rows.
 * 32 threads each handle 8 elements per QK_K block (256 elements = 32*8).
 *
 * Argument layout (matching existing q4_metal kernels):
 *   buf[0]: src0 (weight), buf[1]: src1 (input), buf[2]: dst (output)
 *   bytes[3]: args struct
 * ========================================================================= */
[[host_name("kernel_mul_mv_q4_k_f32")]]
kernel void kernel_mul_mv_q4_k_f32(
        device const char *src0          [[buffer(0)]],
        device const char *src1          [[buffer(1)]],
        device       char *dst           [[buffer(2)]],
        constant q4_metal_args_mul_mv &args [[buffer(3)]],
        threadgroup  char *shmem         [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short NR0 = 2;
    constexpr short NW = 32;  /* simd width */

    const int nb = args.ne00 / QK_K;
    const int r0 = tgpig.x * NR0;  /* starting output row for this group */
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;
    device const float *y = (device const float *)(src1 + offset1);

    device const block_q4_K *ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row) * args.nb01 +
            (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;
        ax[row] = (device const block_q4_K *)(src0 + offset0);
    }

    float sumf[NR0] = {0.f};

    /* Each thread handles 8 elements per block: elements [tiisg*8 .. tiisg*8+7] */
    const short e0 = tiisg * 8;  /* 0, 8, 16, ..., 248 */

    for (int ib = 0; ib < nb; ib++) {
        for (short row = 0; row < NR0; ++row) {
            device const block_q4_K *xb = &ax[row][ib];

            /* Use manual FP16 conversion to match CPU's f16_to_f32 */
            device const uint16_t *raw16 = (device const uint16_t *)xb;
            const float d = q4k_f16_to_f32(raw16[0]);
            const float dm = q4k_f16_to_f32(raw16[1]);

            float acc = 0.f;
            for (short i = 0; i < 8; ++i) {
                const short e = e0 + i;
                const short grp = e / 32;       /* group 0..7 */
                const short pos = e % 32;       /* position in group 0..31 */
                const short sub = pos / 16;     /* sub-group within group: 0 or 1 */
                const short idx = pos % 16;     /* index in sub-group 0..15 */

                /* Scale and min extraction */
                const bool is_high_grp = grp >= 6;
                const float dmul = is_high_grp ? (d / 16.0f) : d;

                /* Get scale/min pair for this group */
                float sc_val, mn_val;
                if (grp < 6) {
                    sc_val = (float)(xb->scales[grp] & 0x3F);
                    mn_val = (float)(xb->scales[6 + grp] & 0x3F);
                } else {
                    /* Groups 6,7 from packed bytes */
                    const short gi = grp - 6;  /* 0 or 1 */
                    sc_val = (float)((xb->scales[4 + gi] >> 4) | ((xb->scales[gi] & 0xC0) >> 2));
                    mn_val = (float)((xb->scales[10 + gi] >> 4) | ((xb->scales[6 + gi] & 0xC0) >> 2));
                }

                /* qs offset: group 0-1 at base, group 2-3 at +32, group 4-5 at +64, group 6-7 at +96 */
                const short qs_off = (grp / 2) * 32 + sub * 16;
                const uchar qbyte = xb->qs[qs_off + idx];
                const uchar mask = sub == 0 ? 0x0F : 0xF0;
                const float q = (float)((qbyte & mask) >> (sub * 4));

                acc += y[e] * (dmul * sc_val * q - dm * mn_val);
            }
            sumf[row] += acc;
        }

        y += QK_K;
    }

    /* SIMD reduce and write */
    threadgroup float *shmem_f32[NR0];
    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *)shmem + NW * row;
        if (sgitg == 0) shmem_f32[row][tiisg] = 0.0f;
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) shmem_f32[row][sgitg] = sumf[row];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *dst_f32 = (device float *)dst +
        (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    for (short row = 0; row < NR0 && r0 + row < args.ne01; ++row) {
        if (tiisg == 0 && sgitg == 0) {
            dst_f32[r0 + row] = shmem_f32[row][0];
        }
    }
}

/* =========================================================================
 * Q6_K matrix-vector multiply.
 * out[out_row] = x[in_dim] @ W[in_dim, out_row] where W is Q6_K.
 * Block layout: 256 elements, 210 bytes.
 *   - ql[128]: low 4 bits, 2 elements per byte
 *   - qh[64]:  high 2 bits, 4 elements per byte
 *   - scales[16]: per-16-element scales (signed int8)
 *   - d: half scale factor
 * Block has 4 quarters of 64 elements. Each quarter has 4 scale values (16 el each).
 *
 * Argument layout (matching existing q4_metal kernels):
 *   buf[0]: src0 (weight), buf[1]: src1 (input), buf[2]: dst (output)
 *   bytes[3]: args struct
 * ========================================================================= */
[[host_name("kernel_mul_mv_q6_k_f32")]]
kernel void kernel_mul_mv_q6_k_f32(
        device const char *src0          [[buffer(0)]],
        device const char *src1          [[buffer(1)]],
        device       char *dst           [[buffer(2)]],
        constant q4_metal_args_mul_mv &args [[buffer(3)]],
        threadgroup  char *shmem         [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short NR0 = 2;
    constexpr short NW = 32;

    const int nb = args.ne00 / QK_K;
    const int r0 = tgpig.x * NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;
    device const float *y = (device const float *)(src1 + offset1);

    device const block_q6_K *ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row) * args.nb01 +
            (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;
        ax[row] = (device const block_q6_K *)((device char *)src0 + offset0);
    }

    float sumf[NR0] = {0.f};

    /* 32 threads cover one Q6_K block (256 elements).
     * Each thread handles 8 elements.
     * Thread tiisg maps to element indices: tiisg*8 + [0..7] within the block. */
    const short elem_base = tiisg * 8;  /* 0, 8, 16, ..., 248 */

    /* Precompute per-element dequantization metadata */
    uchar ql_idx[8];   /* byte index in ql */
    uchar ql_shift[8]; /* shift to extract low 4 bits */
    uchar qh_idx[8];   /* byte index in qh */
    uchar qh_shift[8]; /* shift to extract high 2 bits */
    uchar scale_idx[8];/* index in scales[16] */

    for (short i = 0; i < 8; ++i) {
        const short e = elem_base + i;    /* element index 0..255 */
        const short q = e / 64;            /* quarter 0..3 */
        const short p = e % 64;            /* position within quarter 0..63 */

        /* ql: 32 bytes per quarter, 2 elements per byte */
        ql_idx[i] = (uchar)(q * 32 + p / 2);
        ql_shift[i] = (uchar)((p % 2) * 4);

        /* qh: 16 bytes per quarter, 4 elements per byte */
        qh_idx[i] = (uchar)(q * 16 + p / 4);
        qh_shift[i] = (uchar)((p % 4) * 2);

        /* scales: 4 per quarter, 16 elements per scale */
        scale_idx[i] = (uchar)(q * 4 + p / 16);
    }

    for (int ib = 0; ib < nb; ib++) {
        /* Load 8 y values */
        /* Load 8 contiguous y values for this thread */
        float yl[8];
        for (short i = 0; i < 8; ++i) yl[i] = y[elem_base + i];

        for (short row = 0; row < NR0; row++) {
            device const block_q6_K *xb = &ax[row][ib];

            float acc = 0.f;
            for (short i = 0; i < 8; ++i) {
                uint8_t v = (xb->ql[ql_idx[i]] >> ql_shift[i]) & 0x0F;
                uint8_t h = (xb->qh[qh_idx[i]] >> qh_shift[i]) & 0x03;
                int v6 = (int)(v | (h << 4)) - 32;  /* signed: 0..63 -> -32..31 */
                acc += yl[i] * v6 * (float)xb->scales[scale_idx[i]];
            }
            sumf[row] += (float)xb->d * acc;
        }

        y += QK_K;  /* advance by 256 elements */
    }

    /* Handle remainder (when in_dim is not a multiple of 256) */
    const int processed = nb * QK_K;
    const int rem = args.ne00 - processed;
    if (rem > 0) {
        /* For simplicity, skip remainder - caller should pad dimensions */
    }

    /* Reduce across SIMD group and write */
    threadgroup float *shmem_f32[NR0];
    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *)shmem + NW * row;
        if (sgitg == 0) shmem_f32[row][tiisg] = 0.0f;
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) shmem_f32[row][sgitg] = sumf[row];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *dst_f32 = (device float *)dst +
        (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    for (short row = 0; row < NR0 && r0 + row < args.ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) dst_f32[r0 + row] = tot;
    }
}

/* =========================================================================
 * Fused Q4_K matrix-vector multiply: 4 separate matmuls from same input.
 *
 * Computes in a single dispatch:
 *   dst0[out0] = x[in_dim] @ W0[in_dim, out0]  (Q4_K)
 *   dst1[out1] = x[in_dim] @ W1[in_dim, out1]  (Q4_K)
 *   dst2[out2] = x[in_dim] @ W2[in_dim, out2]  (Q4_K)
 *   dst3[out3] = x[in_dim] @ W3[in_dim, out3]  (Q4_K)
 *
 * All weights share the same input vector x. This replaces 4 separate
 * kernel dispatches for DeltaNet (qkv_raw, alpha_raw, beta_raw, z_raw).
 * ========================================================================= */

struct q4_fused4_args {
    int in_dim;
    int out0, out1, out2, out3;
    ulong woff0, woff1, woff2, woff3;
    ulong ooff0, ooff1, ooff2, ooff3;
};

[[host_name("kernel_mul_mv_q4_k_fused4")]]
kernel void kernel_mul_mv_q4_k_fused4(
        device const char *w0             [[buffer(0)]],
        device const char *w1             [[buffer(1)]],
        device const char *w2             [[buffer(2)]],
        device const char *w3             [[buffer(3)]],
        device const float *src1          [[buffer(4)]],
        device       float *dst           [[buffer(5)]],
        constant q4_fused4_args &args     [[buffer(6)]],
        threadgroup  char *shmem          [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    constexpr short NR0 = 2;
    constexpr short NW = 32;

    /* Threadgroup z selects which of the 4 matmuls this group computes. */
    const int matmul_id = (int)tgpig.z;

    const int out_dim = (matmul_id == 0) ? args.out0 :
                        (matmul_id == 1) ? args.out1 :
                        (matmul_id == 2) ? args.out2 : args.out3;
    const ulong woff = (matmul_id == 0) ? args.woff0 :
                       (matmul_id == 1) ? args.woff1 :
                       (matmul_id == 2) ? args.woff2 : args.woff3;
    const ulong ooff = (matmul_id == 0) ? args.ooff0 :
                       (matmul_id == 1) ? args.ooff1 :
                       (matmul_id == 2) ? args.ooff2 : args.ooff3;

    /* Early exit if this matmul has no output rows */
    if (out_dim <= 0) return;

    const int nb = args.in_dim / QK_K;
    const int r0 = (int)tgpig.x * NR0;

    /* Skip if this threadgroup's rows are beyond the output dimension */
    if (r0 >= out_dim) return;

    device const block_q4_K *w_base = (device const block_q4_K *)(w0 + woff);
    if (matmul_id == 1) w_base = (device const block_q4_K *)(w1 + woff);
    else if (matmul_id == 2) w_base = (device const block_q4_K *)(w2 + woff);
    else if (matmul_id == 3) w_base = (device const block_q4_K *)(w3 + woff);

    device const float *y = src1;
    float sumf[NR0] = {0.f};
    const short e0 = tiisg * 8;

    for (int ib = 0; ib < nb; ib++) {
        for (short row = 0; row < NR0; ++row) {
            const uint row_stride = (uint)(args.in_dim / QK_K);
            device const block_q4_K *xb = &w_base[(r0 + row) * row_stride + ib];

            device const uint16_t *raw16 = (device const uint16_t *)xb;
            const float d = q4k_f16_to_f32(raw16[0]);
            const float dm = q4k_f16_to_f32(raw16[1]);

            float acc = 0.f;
            for (short i = 0; i < 8; ++i) {
                const short e = e0 + i;
                const short grp = e / 32;
                const short pos = e % 32;
                const short sub = pos / 16;
                const short idx = pos % 16;

                const bool is_high_grp = grp >= 6;
                const float dmul = is_high_grp ? (d / 16.0f) : d;

                float sc_val, mn_val;
                if (grp < 6) {
                    sc_val = (float)(xb->scales[grp] & 0x3F);
                    mn_val = (float)(xb->scales[6 + grp] & 0x3F);
                } else {
                    const short gi = grp - 6;
                    sc_val = (float)((xb->scales[4 + gi] >> 4) | ((xb->scales[gi] & 0xC0) >> 2));
                    mn_val = (float)((xb->scales[10 + gi] >> 4) | ((xb->scales[6 + gi] & 0xC0) >> 2));
                }

                const short qs_off = (grp / 2) * 32 + sub * 16;
                const uchar qbyte = xb->qs[qs_off + idx];
                const uchar mask = sub == 0 ? 0x0F : 0xF0;
                const float q = (float)((qbyte & mask) >> (sub * 4));

                acc += y[e] * (dmul * sc_val * q - dm * mn_val);
            }
            sumf[row] += acc;
        }
        y += QK_K;
    }

    /* SIMD reduce and write */
    threadgroup float *shmem_f32[NR0];
    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *)shmem + 32 * row;
        if (sgitg == 0) shmem_f32[row][tiisg] = 0.0f;
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) shmem_f32[row][sgitg] = sumf[row];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0 && r0 + row < out_dim; ++row) {
        if (tiisg == 0 && sgitg == 0) {
            dst[ooff + (uint64_t)r0 + row] = shmem_f32[row][0];
        }
    }
}
