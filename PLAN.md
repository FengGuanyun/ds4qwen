# Q4 (QwenStar 4) Research & Development Plan

## Project Overview

Q4 is a native inference engine for **Qwen3.6-27B**, following the same philosophy
as [ds4](https://github.com/antirez/ds4): one model at a time, validated against
official logits, tested at long context, with dedicated quantization and
disk-based KV cache.

## Model: Qwen3.6-27B vs DeepSeek V4 Flash

| Feature | DeepSeek V4 Flash | Qwen3.6-27B |
|---------|-------------------|-------------|
| Type | MoE (284B total, ~13B active) | Dense (27B) |
| Attention | Traditional GQA | Hybrid: 48x Gated DeltaNet + 16x Gated Attention |
| KV heads | N/A | 4 KV heads (GQA 24:4, naturally low KV cache) |
| Context | 1M tokens | 262K (extendable to 1M) |
| FFN | MoE experts | SwiGLU dense, 17408 intermediate dim |
| Layers | Unknown | 64 layers (16 blocks x 3 DeltaNet + 1 Attention) |

### Why Qwen3.6-27B is Interesting

1. **DeltaNet** provides linear (O(n)) scaling for long context in 48 of 64 layers
2. **GQA 24:4** means only 4 KV heads -> much smaller KV cache than standard
3. Dense model -> simpler to quantize and deploy than MoE
4. Flagship-level coding performance, surpassing larger MoE models
5. Apache 2.0 license

## Quantization Strategy

### Current State

Q4 supports the following quantization formats:

| Format | GGUF Table | CPU Kernel | GPU Kernel | In Use |
|--------|------------|------------|------------|--------|
| Q8_0 | Yes | Yes | Yes | Partial FFN tensors |
| Q4_K | Yes | Yes | Yes | Primary quant format |
| Q5_K | Yes | Yes | Yes | ssm_out |
| Q6_K | Yes | Yes | Yes | attn_qkv, attn_v |
| Q2_K | Yes (recognized only) | No | No | Not implemented |
| Q3_K | Yes (recognized only) | No | No | Not implemented |

The GGUF format table at `q4.c:498` registers Q2_K block size (84 bytes/256 weights,
~2.625 bits/weight) but no dequantization kernels exist.

### Phase 2: 2-bit Quantization Plan

To support 2-bit quantization, the following work is needed:

#### 2.1 CPU Kernels (q4.c)
- Add Q2_K dequantize + F32 matvec kernel (reference existing Q4_K/Q5_K/Q6_K pattern)
- Add `Q4_TENSOR_Q2_K` to the tensor type enum
- Add `tensor_expect_q2_k_layout()` validation helper

#### 2.2 GPU Interface (q4_gpu.h)
- Add `q4_gpu_matmul_q2_k_tensor()` declaration
- Add `q4_gpu_vec_matmul_q2k_tensor()` for vector-matrix multiply

#### 2.3 Metal Kernels (metal/quant.metal)
- Implement Q2_K block dequantization in Metal
- Add Q2_K matmul kernel matching the GPU interface

#### 2.4 Mixed Quantization Strategy
For dense models, unlike MoE where only routed experts are quantized, we need
a per-tensor-type strategy:

```
Tensor                     | Quant    | Reason
-------------------------- | -------- | ------
token_embd                 | Q4_K     | First projection quality
attn_qkv (DeltaNet)        | Q6_K     | q/k/v combined, sensitive
attn_q / attn_k / attn_v   | Q6_K     | Attention quality
attn_output                | Q4_K     | Output projection
ssm_conv1d / ssm_alpha/... | F32      | Small tensors, keep precision
ssm_out                    | Q5_K     | DeltaNet output projection
ffn_gate / ffn_up / ffn_down| Q2_K    | FFN is most space, least sensitive
output                     | Q6_K     | Vocabulary projection critical
```

Expected model size with Q2_K FFN: ~10-11 GB (vs ~17 GB for Q4_K_M)

#### 2.5 GGUF Generation
Use llama.cpp's quantize tool with imatrix to generate Q2_K GGUF:

```bash
# Generate importance matrix
llama-imatrix -m qwen3.6-27b-f16.gguf -f calibration.txt -o qwen3.6-27b.imatrix

# Quantize with mixed strategy via --tensor-type overrides
llama-quantize --imatrix qwen3.6-27b.imatrix \
  qwen3.6-27b-f16.gguf qwen3.6-27b-q2k.gguf Q2_K \
  --tensor-type "output.weight=Q6_K" \
  --tensor-type "token_embd.weight=Q4_K"
```

## Disk KV Cache (Phase 3)

### Current State
`q4_kvstore.c` and `q4_server.c` are Phase 3 stubs.

### Design (following DS4 architecture)

Reference: `ds4/ds4_kvstore.c` and `ds4/ds4_kvstore.h` (~1000 lines, relatively independent)

Key concepts:
- **SHA1-based checkpoint files**: Each KV state saved to disk with SHA1 of prompt prefix as key
- **Automatic eviction**: LRU + hit count + 6-hour half-life scoring
- **Continued checkpoints**: Save intermediate states every ~10000 tokens
- **Trailer hooks**: Protocol-specific metadata (tool mappings, response state)

```
./q4-server --ctx 262144 --kv-disk-dir /tmp/q4-kv --kv-disk-space-mb 32768
```

### Why This Matters for 24GB MacBook

```
Component          | Q4_K_M (~17GB) | Q2_K Mixed (~11GB)
------------------ | -------------- | -------------------
Model weights      | ~17 GB         | ~11 GB
macOS system       | ~4 GB          | ~4 GB
Available for KV   | ~3 GB          | ~9 GB
Max context (MLA)  | ~10-15K tokens | ~32-64K tokens
With disk KV       | unlimited*     | unlimited*
```

*Disk KV solves capacity; bandwidth is the bottleneck. DeltaNet's linear
attention state is smaller than traditional KV cache, making disk-based
operation more feasible than for standard attention models.

## Implementation Phases

| Phase | Feature | Status | Priority |
|-------|---------|--------|----------|
| Phase 1 | Core engine + Metal backend | Done | - |
| Phase 1 | CLI (q4) + Benchmark (q4-bench) | Done | - |
| Phase 2 | 2-bit quantization (Q2_K kernels) | Not started | High |
| Phase 3 | Disk KV cache (q4_kvstore) | Stub only | High |
| Phase 3 | HTTP server (q4-server) | Stub only | Medium |
| Phase 4 | CUDA backend | Not started | Low |
| Phase 4 | Native coding agent | Not started | Low |

## References

- DS4 original project: https://github.com/antirez/ds4
- DS4 KV store design: ds4/ds4_kvstore.c, ds4/ds4_kvstore.h
- llama.cpp quantization: https://github.com/ggml-org/llama.cpp
- Qwen3.6-27B paper: https://qwen.ai/blog?id=qwen3.6-27B
