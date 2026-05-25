#ifndef Q4_GPU_H
#define Q4_GPU_H

#include <stdbool.h>
#include <stdint.h>

/* =========================================================================
 * GPU Tensor and Command Lifetime.
 * =========================================================================
 *
 * Opaque device tensor used by the Q4-specific GPU executor.
 *
 * The public GPU API is tensor-resident: activations, KV state, and scratch
 * buffers stay device-owned across the whole prefill/decode command sequence.
 */
typedef struct q4_gpu_tensor q4_gpu_tensor;

int q4_gpu_init(void);
void q4_gpu_cleanup(void);

q4_gpu_tensor *q4_gpu_tensor_alloc(uint64_t bytes);
q4_gpu_tensor *q4_gpu_tensor_view(const q4_gpu_tensor *base, uint64_t offset, uint64_t bytes);
void q4_gpu_tensor_free(q4_gpu_tensor *tensor);
uint64_t q4_gpu_tensor_bytes(const q4_gpu_tensor *tensor);
uint64_t q4_gpu_tensor_offset(const q4_gpu_tensor *tensor);
void *q4_gpu_tensor_contents(q4_gpu_tensor *tensor);
int q4_gpu_tensor_fill_f32(q4_gpu_tensor *tensor, float value, uint64_t count);
int q4_gpu_tensor_write(q4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
int q4_gpu_tensor_read(const q4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
int q4_gpu_tensor_copy(q4_gpu_tensor *dst, uint64_t dst_offset,
                          const q4_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes);

int q4_gpu_begin_commands(void);
int q4_gpu_flush_commands(void);
int q4_gpu_end_commands(void);
int q4_gpu_synchronize(void);

int q4_gpu_set_model_map(const void *model_map, uint64_t model_size);
int q4_gpu_set_model_fd(int fd);
int q4_gpu_set_model_map_range(const void *model_map, uint64_t model_size,
                                uint64_t map_offset, uint64_t map_size);
int q4_gpu_cache_model_range(const void *model_map, uint64_t model_size,
                              uint64_t offset, uint64_t bytes, const char *label);
int q4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes);
void q4_gpu_set_quality(bool quality);
void q4_gpu_print_memory_report(const char *label);

/* =========================================================================
 * Embedding Lookup.
 * ========================================================================= */

int q4_gpu_embed_token_tensor(
        q4_gpu_tensor *out,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       weight_offset,
        uint32_t       n_vocab,
        uint32_t       token,
        uint32_t       n_embd);

int q4_gpu_embed_tokens_tensor(
        q4_gpu_tensor       *out,
        const q4_gpu_tensor *tokens,
        const void          *model_map,
        uint64_t             model_size,
        uint64_t             weight_offset,
        uint32_t             n_vocab,
        uint32_t             n_tokens,
        uint32_t             n_embd);

/* =========================================================================
 * Core Kernels: Matmul, Norm, RoPE, Attention, DeltaNet.
 * ========================================================================= */

/* Q8_0 matrix-vector multiply: out[n_out] = x[n_in] @ W[n_in, n_out]
 * W is Q8_0 quantized at model_map + weight_offset.
 * x is F32 on device. out is F32 on device. */
int q4_gpu_matmul_q8_0_tensor(
        q4_gpu_tensor *out,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       weight_offset,
        uint64_t       in_dim,
        uint64_t       out_dim,
        const q4_gpu_tensor *x,
        uint32_t       n_tok);

int q4_gpu_matmul_q4_k_tensor(
        q4_gpu_tensor *out,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       weight_offset,
        uint64_t       in_dim,
        uint64_t       out_dim,
        const q4_gpu_tensor *x,
        uint32_t       n_tok);

/* Fused: 4 Q4_K matmuls from same input in one dispatch. */
int q4_gpu_matmul_q4_k_fused4_tensor(
        q4_gpu_tensor *out,
        const void    *model_map,
        uint64_t       model_size,
        const uint64_t weight_offsets[4],
        const uint32_t out_dims[4],
        uint64_t       in_dim,
        const q4_gpu_tensor *x);

/* Q6_K matrix-vector multiply. Same signature as q8_0 variant. */
int q4_gpu_matmul_q6_k_tensor(
        q4_gpu_tensor *out,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       weight_offset,
        uint64_t       in_dim,
        uint64_t       out_dim,
        const q4_gpu_tensor *x,
        uint32_t       n_tok);

/* Dispatch to the correct quantization kernel based on tensor type. */
int q4_gpu_matmul_any_tensor(
        q4_gpu_tensor *out,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       weight_offset,
        uint64_t       in_dim,
        uint64_t       out_dim,
        uint32_t       weight_type,
        const q4_gpu_tensor *x,
        uint32_t       n_tok);

/* Shared gate+up SwiGLU: gate = x @ W_gate (Q8_0), up = x @ W_up (Q8_0),
 * mid = SiLU(gate) * up. Writes gate, up, mid to separate tensors. */
int q4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        q4_gpu_tensor *gate,
        q4_gpu_tensor *up,
        q4_gpu_tensor *mid,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       gate_offset,
        uint64_t       up_offset,
        uint64_t       in_dim,
        uint64_t       out_dim,
        const q4_gpu_tensor *x,
        float          clamp);

/* RMSNorm: out[rows, n] = x[rows, n] / rms(x) * weight[n] */
int q4_gpu_rms_norm_weight_rows_tensor(
        q4_gpu_tensor *out,
        const q4_gpu_tensor *x,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       weight_offset,
        uint32_t       n,
        uint32_t       rows,
        float          eps);

/* Full RoPE applied to all head_dim dimensions (not just tail). */
int q4_gpu_rope_full_tensor(
        q4_gpu_tensor *x,
        uint32_t       n_tok,
        uint32_t       n_head,
        uint32_t       head_dim,
        uint32_t       pos0,
        float          freq_base,
        float          freq_scale);

/* GQA Flash Attention with KV cache. */
int q4_gpu_flash_attn_tensor(
        q4_gpu_tensor *out,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       q_offset,
        uint64_t       k_offset,
        uint64_t       v_offset,
        const q4_gpu_tensor *kv_cache_k,
        const q4_gpu_tensor *kv_cache_v,
        uint32_t       n_q_heads,
        uint32_t       n_kv_heads,
        uint32_t       head_dim,
        uint32_t       n_tokens,
        uint32_t       pos0,
        uint32_t       kv_len,
        float          logit_softcap);

/* Write K/V to KV cache at given position. */
int q4_gpu_kv_cache_store_tensor(
        const q4_gpu_tensor *k,
        const q4_gpu_tensor *v,
        q4_gpu_tensor       *cache_k,
        q4_gpu_tensor       *cache_v,
        uint32_t             pos,
        uint32_t             n_kv_heads,
        uint32_t             head_dim);

/* Gated DeltaNet state update for decode (single token). */
int q4_gpu_deltanet_step_tensor(
        q4_gpu_tensor *out,
        q4_gpu_tensor *state,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       a_offset,
        uint64_t       b_offset,
        uint64_t       dt_offset,
        const q4_gpu_tensor *x,
        uint32_t       n_embd,
        uint32_t       head_dim,
        uint32_t       n_kv_heads);

/* Gated DeltaNet prefill: process n_tokens sequentially. */
int q4_gpu_deltanet_prefill_tensor(
        q4_gpu_tensor *out,
        q4_gpu_tensor *state,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       a_offset,
        uint64_t       b_offset,
        uint64_t       dt_offset,
        const q4_gpu_tensor *x,
        uint32_t       n_tokens,
        uint32_t       n_embd,
        uint32_t       head_dim,
        uint32_t       n_kv_heads,
        uint32_t       n_q_heads);

/* Elementwise: y[i] = x[i] + scale */
int q4_gpu_add_tensor(
        q4_gpu_tensor *y,
        const q4_gpu_tensor *x,
        uint64_t       n,
        float          scale);

/* Elementwise: out = SiLU(x) */
int q4_gpu_silu_tensor(
        q4_gpu_tensor *out,
        const q4_gpu_tensor *x,
        uint64_t       n);

/* Fused: out = SiLU(clamp(gate)) * up, all on GPU */
int q4_gpu_silu_clamped_mul_tensor(
        q4_gpu_tensor *out,
        const q4_gpu_tensor *gate,
        const q4_gpu_tensor *up,
        uint64_t       n,
        float          clamp);

/* Elementwise: out = a * b (broadcast b across rows) */
int q4_gpu_mul_rows_tensor(
        q4_gpu_tensor *out,
        const q4_gpu_tensor *a,
        const q4_gpu_tensor *b,
        uint32_t       rows,
        uint32_t       cols);

/* Residual add: x = x + residual */
int q4_gpu_residual_add_tensor(
        q4_gpu_tensor *x,
        const q4_gpu_tensor *residual,
        uint64_t       n);

/* =========================================================================
 * DeltaNet GPU Operations.
 * ========================================================================= */

/* Vector-matrix Q4_K multiply: out[out_dim] = vec[in_dim] @ W[out_dim, in_dim] */
int q4_gpu_vec_matmul_q4k_tensor(
        q4_gpu_tensor *out,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       weight_offset,
        uint64_t       in_dim,
        uint64_t       out_dim,
        const q4_gpu_tensor *x);

/* Fused conv1D + split + L2 norm + expand for DeltaNet. */
int q4_gpu_deltanet_conv_split_tensor(
        const q4_gpu_tensor *qkv_raw,
        const q4_gpu_tensor *conv_buf,
        const q4_gpu_tensor *conv_buf_out,
        const void          *model_map,
        uint64_t             model_size,
        uint64_t             conv_w_offset,
        q4_gpu_tensor       *q_exp,
        q4_gpu_tensor       *k_exp,
        q4_gpu_tensor       *v_out,
        uint32_t             qkv_dim,
        uint32_t             n_k_groups,
        uint32_t             n_v_heads,
        uint32_t             head_k_dim,
        uint32_t             head_v_dim,
        uint32_t             repeat,
        uint32_t             conv_pos);

/* Gate transforms: gate = softplus(alpha + bias) * a, beta = sigmoid(beta) */
int q4_gpu_deltanet_gate_transform_tensor(
        const q4_gpu_tensor *alpha_raw,
        const q4_gpu_tensor *beta_raw,
        q4_gpu_tensor       *gate_out,
        q4_gpu_tensor       *beta_out,
        const void          *model_map,
        uint64_t             model_size,
        uint64_t             dt_bias_offset,
        uint64_t             ssm_a_offset,
        uint32_t             n);

/* Delta rule per v_head: sk = state @ k; delta = v - sk; state update; out = state @ k */
int q4_gpu_delta_rule_tensor(
        q4_gpu_tensor *state,
        const q4_gpu_tensor *k_exp,
        const q4_gpu_tensor *v_raw,
        const q4_gpu_tensor *gate,
        const q4_gpu_tensor *beta,
        q4_gpu_tensor *output,
        uint32_t n_v_heads,
        uint32_t head_v_dim,
        uint32_t head_k_dim);

/* SiLU gate + RMS norm per v_head */
int q4_gpu_deltanet_silu_rms_tensor(
        q4_gpu_tensor *inout,
        const q4_gpu_tensor *z_raw,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       ssm_norm_w_offset,
        uint32_t       n_v_heads,
        uint32_t       head_v_dim);

/* Vector-matrix Q5_K multiply: out[out_dim] = vec[in_dim] @ W[out_dim, in_dim] */
int q4_gpu_vec_matmul_q5k_tensor(
        q4_gpu_tensor *out,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       weight_offset,
        uint64_t       in_dim,
        uint64_t       out_dim,
        const q4_gpu_tensor *x);

/* Softmax over last dimension for each row. */
int q4_gpu_softmax_tensor(
        q4_gpu_tensor *x,
        uint32_t       rows,
        uint32_t       cols);

#endif /* Q4_GPU_H */
