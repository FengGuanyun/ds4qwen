// Q4 softmax kernel - row softmax for attention scores.

struct q4_metal_args_softmax {
    int32_t  ne00;
    int32_t  ne01;
    uint64_t nb1;
    float    scale;
};

// Row softmax: out[row] = softmax(in[row] * scale)
template<typename T>
kernel void kernel_soft_max(
        constant q4_metal_args_softmax & args,
        device const  char * src0,
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3 tptg[[threads_per_threadgroup]]) {
    const int row = tgpig.x;
    device const float *psrc = (device const float *)(src0 + row * args.nb1);
    device float *pdst = (device float *)(dst + row * args.nb1);

    // Find max
    float lmax = -INFINITY;
    for (int i = tpitg.x; i < args.ne00; i += tptg.x) {
        lmax = max(lmax, psrc[i] * args.scale);
    }

    float max_val = simd_max(lmax);
    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) buf[tiisg] = -INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) buf[sgitg] = max_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // Compute exp and sum
    float lsum = 0.0f;
    for (int i = tpitg.x; i < args.ne00; i += tptg.x) {
        const float e = exp(psrc[i] * args.scale - max_val);
        pdst[i] = e;
        lsum += e;
    }

    float sum = simd_sum(lsum);
    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) buf[tiisg] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) buf[sgitg] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    const float inv_sum = 1.0f / sum;
    for (int i = tpitg.x; i < args.ne00; i += tptg.x) {
        pdst[i] *= inv_sum;
    }
}

typedef decltype(kernel_soft_max<float>) kernel_soft_max_t;

[[host_name("kernel_soft_max_f32")]]   kernel kernel_soft_max_t kernel_soft_max<float>;
