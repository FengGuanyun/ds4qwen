// Q4 RMSNorm kernel - standard layer normalization for Qwen models.

struct q4_metal_args_norm {
    int32_t  ne00;
    int32_t  ne00_t;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    eps;
};

// RMSNorm with weight: out[row] = x[row] / rms(x[row]) * weight
[[host_name("kernel_rms_norm_mul_f32")]]
kernel void kernel_rms_norm_mul_f32(
        constant q4_metal_args_norm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) shmem_f32[tiisg] = 0.0f;

    const int row = tgpig.x;
    device const float *x = (device const float *)(src0 + row * args.nb1);
    device const float *w = (device const float *)(src1);

    float sumf = 0.0f;
    for (int i = tpitg.x; i < args.ne00; i += ntg.x) sumf += x[i] * x[i];
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) shmem_f32[sgitg] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float scale = 1.0f / sqrt(sumf / args.ne00 + args.eps);
    device float *y = (device float *)(dst + row * args.nb1);

    for (int i = tpitg.x; i < args.ne00; i += ntg.x) {
        y[i] = x[i] * scale * w[i];
    }
}

// RMSNorm without weight: out[row] = x[row] / rms(x[row])
[[host_name("kernel_rms_norm_f32")]]
kernel void kernel_rms_norm_f32(
        constant q4_metal_args_norm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) shmem_f32[tiisg] = 0.0f;

    const int row = tgpig.x;
    device const float *x = (device const float *)(src0 + row * args.nb1);

    float sumf = 0.0f;
    for (int i = tpitg.x; i < args.ne00; i += ntg.x) sumf += x[i] * x[i];
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) shmem_f32[sgitg] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float scale = 1.0f / sqrt(sumf / args.ne00 + args.eps);
    device float *y = (device float *)(dst + row * args.nb1);

    for (int i = tpitg.x; i < args.ne00; i += ntg.x) {
        y[i] = x[i] * scale;
    }
}
