// Q4 Metal matmul kernels - adapted from ds4 dense.metal.
// Supports Q8_0 and F16 weights with F32 activations for Qwen3.6-27B.

// common.metal inlined above

constant short FC_mul_mv_nsg   [[function_constant(0)]];
constant short FC_mul_mv_nxpsg [[function_constant(1)]];

// Q8_0 matrix-vector multiply for decode.
// out[out_row] = x[in_dim] @ W[in_dim, out_row] where W is Q8_0.
template<short NR0>
static inline void q4_mv_reduce_and_write(
        device float * dst_f32,
        float sumf[NR0],
        const int r0,
        const int ne01,
        ushort tiisg,
        ushort sgitg,
        threadgroup char * shmem) {
    constexpr short NW = N_SIMDWIDTH;
    threadgroup float * shmem_f32[NR0];

    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *) shmem + NW * row;
        if (sgitg == 0) shmem_f32[row][tiisg] = 0.0f;
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) shmem_f32[row][sgitg] = sumf[row];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) dst_f32[r0 + row] = tot;
    }
}

[[host_name("kernel_mul_mv_q8_0_f32")]]
kernel void kernel_mul_mv_q8_0_f32(
        constant q4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = 1;  /* Fixed for decode; FC_mul_mv_nsg default is 0 */
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NR0 = 2;

    const int nb = args.ne00 / QK8_0;
    const int r0 = tgpig.x * NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;
    device const float *y = (device const float *)(src1 + offset1);

    device const block_q8_0 *ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row) * args.nb01 +
            (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;
        ax[row] = (device const block_q8_0 *)((device char *)src0 + offset0);
    }

    float sumf[NR0] = {0.f};
    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;
    float yl[NQ];
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        for (short i = 0; i < NQ; ++i) yl[i] = yb[i];
        for (short row = 0; row < NR0; row++) {
            device const int8_t *qs = ax[row][ib].qs + il * NQ;
            float sumq = 0.f;
            for (short i = 0; i < NQ; ++i) sumq += qs[i] * yl[i];
            sumf[row] += sumq * ax[row][ib].d;
        }
        yb += NSG * NQ * QK8_0;
    }

    device float *dst_f32 = (device float *)dst +
        (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    q4_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

// F16 matrix-vector multiply for decode.
[[host_name("kernel_mul_mv_f16_f32")]]
kernel void kernel_mul_mv_f16_f32(
        constant q4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = 1;  /* Fixed for decode; FC_mul_mv_nsg default is 0 */
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB = 32;
    constexpr short NF = 8;
    constexpr short NR0 = 2;

    const int nb = args.ne00 / NB;
    const int r0 = tgpig.x * NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;
    device const float *y = (device const float *)(src1 + offset1);

    device const half *ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row) * args.nb01 +
            (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;
        ax[row] = (device const half *)((device char *)src0 + offset0);
    }

    float sumf[NR0] = {0.f};
    const short ix = tiisg / (NW / NF);
    const short il = tiisg % (NW / NF);
    const int ib0 = sgitg * NF + ix;
    float yl[NF];
    device const float *yb = y + ib0 * NB + il * NF;

    for (int ib = ib0; ib < nb; ib += NSG * NF) {
        for (short i = 0; i < NF; ++i) yl[i] = yb[i];
        for (short row = 0; row < NR0; row++) {
            device const half *xb = ax[row] + ib * NB + il * NF;
            float sumq = 0.f;
            for (short i = 0; i < NF; ++i) sumq += (float)xb[i] * yl[i];
            sumf[row] += sumq;
        }
        yb += NSG * NF * NW;
    }

    for (int i = nb * NB + sgitg * NW + tiisg; i < args.ne00; i += NW * NSG) {
        for (short row = 0; row < NR0; row++) sumf[row] += (float)ax[row][i] * y[i];
    }

    device float *dst_f32 = (device float *)dst +
        (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    q4_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

// Shared gate+up SwiGLU: processes two projections from same input,
// computes SiLU(gate) * up in-place.
[[host_name("kernel_shared_gate_up_swiglu_q8_0")]]
kernel void kernel_shared_gate_up_swiglu_q8_0(
        constant q4_metal_args_mul_mv & args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        constant     float &clamp_value,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = 1;  /* Fixed for decode; FC_mul_mv_nsg default is 0 */
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NR0 = 2;

    const int nb = args.ne00 / QK8_0;
    const int r0 = tgpig.x * NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;
    device const float *y = (device const float *)(src1 + offset1);

    device const block_q8_0 *ag[NR0];
    device const block_q8_0 *au[NR0];
    for (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row) * args.nb01 +
            (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;
        ag[row] = (device const block_q8_0 *)((device char *)src0_gate + offset0);
        au[row] = (device const block_q8_0 *)((device char *)src0_up + offset0);
    }

    float sumg[NR0] = {0.f};
    float sumu[NR0] = {0.f};
    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;
    float yl[NQ];
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        for (short i = 0; i < NQ; ++i) yl[i] = yb[i];
        for (short row = 0; row < NR0; ++row) {
            device const int8_t *qg = ag[row][ib].qs + il * NQ;
            device const int8_t *qu = au[row][ib].qs + il * NQ;
            float sg = 0.f, su = 0.f;
            for (short i = 0; i < NQ; ++i) {
                sg += qg[i] * yl[i];
                su += qu[i] * yl[i];
            }
            sumg[row] += sg * ag[row][ib].d;
            sumu[row] += su * au[row][ib].d;
        }
        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *sh_gate = (threadgroup float *)shmem;
    threadgroup float *sh_up = sh_gate + NW * NR0;
    for (short row = 0; row < NR0; ++row) {
        if (sgitg == 0) { sh_gate[row * NW + tiisg] = 0.0f; sh_up[row * NW + tiisg] = 0.0f; }
        sumg[row] = simd_sum(sumg[row]);
        sumu[row] = simd_sum(sumu[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) { sh_gate[row * NW + sgitg] = sumg[row]; sh_up[row * NW + sgitg] = sumu[row]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *gate_f32 = (device float *)dst_gate + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    device float *up_f32   = (device float *)dst_up + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    device float *mid_f32  = (device float *)dst_mid + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (short row = 0; row < NR0 && r0 + row < args.ne01; ++row) {
        float gate = simd_sum(sh_gate[row * NW + tiisg]);
        float up = simd_sum(sh_up[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            gate_f32[r0 + row] = gate;
            up_f32[r0 + row] = up;
            float g = clamp_value > 1.0e-6f ? min(gate, clamp_value) : gate;
            float u = clamp_value > 1.0e-6f ? clamp(up, -clamp_value, clamp_value) : up;
            mid_f32[r0 + row] = (g / (1.0f + exp(-g))) * u;
        }
    }
}

// Tiled matrix-matrix for prefill (batched tokens).
[[host_name("kernel_mul_mm_f16_f32")]]
kernel void kernel_mul_mm_f16_f32(
        constant q4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0 = 64, NR1 = 32, NK = 32;
    constexpr int NL0 = NK / 16, NL1 = NK / 8;

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);

    const short nr0 = min(args.ne01 - r0, NR0);
    const short nr1 = min(args.ne0 - r1, NR1);

    const short lr0 = min((short)tiitg / NL0, nr0 - 1);
    const short lr1 = min((short)tiitg / NL1, nr1 - 1);
    const short il0 = tiitg % NL0;

    const int i12 = im % args.ne12;
    const int i13 = im / args.ne12;
    const uint64_t offset0 = (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;

    device const half *x = (device const half *)(src0 + args.nb01 * (r0 + lr0) + offset0) + il0;

    const short iy = 8 * (tiitg % NL1);
    device const float *y = (device const float *)(
        src1 + args.nb13 * i13 + args.nb12 * i12 + args.nb11 * (r1 + lr1) + args.nb10 * iy);

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        half temp_a[16];
        for (short i = 0; i < 16; i++) temp_a[i] = x[i];

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            *(sa + 64 * (8 * sx + sy) + 8 * ly + lx) = loop_k + 16 * il0 + i < args.ne00 ? temp_a[i] : 0;
        }

        for (short i = 0; i < 8; ++i) {
            const short sx = tiitg % NL1;
            const short sy = (tiitg / NL1) / 8;
            const short lx = i;
            const short ly = (tiitg / NL1) % 8;
            *(sb + 64 * (4 * sx + sy) + 8 * ly + lx) = loop_k + iy + i < args.ne00 ? y[i] : 0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half *lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half *lsmb = sb + 2 * 64 * (sgitg / 2);

        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    device float *C = (device float *)dst + (r0 + 32 * (sgitg & 1)) +
        (r1 + 16 * (sgitg >> 1)) * args.ne01 + im * args.ne0 * args.ne01;
    for (short i = 0; i < 8; i++) simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * args.ne01 * (i / 4), args.ne01, 0, false);
}
