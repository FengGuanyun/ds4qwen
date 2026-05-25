// Q4 elementwise kernels - residual add and unary elementwise ops.

// Residual add: x[i] += residual[i]
[[host_name("kernel_residual_add")]]
kernel void kernel_residual_add(
        device float *x,
        device const float *residual,
        constant uint64_t &n,
        uint tid[[thread_position_in_grid]]) {
    if (tid >= n) return;
    x[tid] += residual[tid];
}

// Elementwise multiply: out[i] = a[i] * b[i]
[[host_name("kernel_elementwise_mul")]]
kernel void kernel_elementwise_mul(
        device float *out,
        device const float *a,
        device const float *b,
        constant uint64_t &n,
        uint tid[[thread_position_in_grid]]) {
    if (tid >= n) return;
    out[tid] = a[tid] * b[tid];
}

// In-encoder GPU copy: dst[dst_off..] = src[src_off..]
[[host_name("kernel_tensor_copy")]]
kernel void kernel_tensor_copy(
        device const char *src,
        device char *dst,
        constant uint64_t &src_offset,
        constant uint64_t &dst_offset,
        constant uint64_t &bytes,
        uint tid[[thread_position_in_grid]]) {
    if (tid >= bytes) return;
    dst[dst_offset + tid] = src[src_offset + tid];
}
