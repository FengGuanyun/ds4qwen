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

/* Softmax over last dimension for each row. */
int q4_gpu_softmax_tensor(
        q4_gpu_tensor *x,
        uint32_t       rows,
        uint32_t       cols);

#endif /* Q4_GPU_H */
