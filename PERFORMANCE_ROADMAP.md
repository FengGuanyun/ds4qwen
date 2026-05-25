# GPU 推理性能优化路线图

## 现状

- **基准速度**: ~196ms/token (5-step decode) / ~215ms/token (30-step avg)
- **目标速度**: 50ms/token (20 tok/s)
- **差距**: ~4x
- **GPU 内存**: ~490MB (非瓶颈)
- **每 token GPU 函数调用数**: ~36+ (每层 8-11 次 dispatch × 64 层)
- **模型**: Qwen3.6-27B Hybrid (64 layers, 32 DeltaNet + 32 Attention)

## 已完成优化

| # | 优化项 | 收益 | 状态 |
|---|--------|------|------|
| 1 | 修复 tensor_read/write offset bug | 正确性 | DONE |
| 2 | FFN gate clamping (clamp=10.0) | 正确性 | DONE |
| 3 | SSM state 保持 GPU (移除 per-layer readback) | ~100ms | DONE |
| 4 | DeltaNet 全量 GPU 化 | ~50ms | DONE |
| 5 | FFN 元素操作 GPU 化 (SiLU×up) | ~20ms | DONE |
| 6 | 移除 per-layer debug readback | ~10ms | DONE |
| 7 | 优化 Q5_K ssm_out matmul | ~5ms | DONE |
| 8 | FP16 激活转换 | 收益为负(已回退) | REJECTED |
| 9 | 融合 DeltaNet 4×Q4_K matmul | ~12ms | DONE |
| 10 | 融合 FFN gate+up matmul | ~2ms | DONE |

## 待优化项（按预期收益排序）

### Tier 1: 最大收益

#### 1. GPU Graph 编码（预期 ~50-80ms 收益）
**问题**: 每个 token 都要重新 encode 整个 command buffer（begin → encode 36+ kernels → flush → end）。
**方案**: 将整个前向传播编码为可重放的 Metal graph（MTLIndirectCommandBuffer），每个 token 只需 replay。
**参考**: ds4 使用 `metal_graph_eval_token_raw_swa` 实现 graph replay。
**优先级**: 最高 — 达到 50ms/token 的必经之路，预计带来 2-3x 提升。

#### 2. 融合 Attention QKV 投影（预期 ~10-15ms 收益）
**问题**: Attention 层 3 次独立 matmul（Q=12288, K=1024, V=1024），32 层 = 96 次 dispatch。
**方案**: 用 fused4 kernel 合并为 1 次 dispatch/层，省 64 次 dispatch。
**难点**: Q 和 KV 输出维度差异大（12288 vs 1024），需优化 fused kernel 的 threadgroup 分配，避免浪费计算。

#### 3. 融合 FFN down + residual add（预期 ~5ms 收益）
**问题**: FFN 路径 `mid → down matmul → residual add`，两次 dispatch。
**方案**: fused kernel 在 matmul 写出的同时加 residual。

### Tier 2: 中等收益

#### 4. 融合 Norm + Matmul（预期 ~5ms 收益）
**问题**: 每层都有 `RMSNorm → matmul` 模式，norm 输出直接作为 matmul 输入。
**方案**: 在 matmul kernel 内部 inline norm 计算，消除 norm 的独立 dispatch。
**影响范围**: DeltaNet pre-norm, Attention pre-norm, FFN pre-norm, post-attention norm — 每层 2-3 次 norm。

#### 5. 优化 Flash Attention decode 路径（预期 ~5-10ms 收益）
**问题**: attention kernel 对每个 Q head 遍历整个 KV cache，O(kv_len × head_dim × n_heads)。
**方案**:
- tiled attention 减少 shared memory 压力
- 对 small batch 使用更高效的线程布局
- KV cache 增长时性能下降明显（196ms → 215ms over 30 steps）

#### 6. 减少 tensor_copy 的 per-call 开销（预期 ~5ms 收益）
**问题**: `q4_gpu_tensor_copy` 每次创建独立 MTLCommandBuffer（commit → wait），硬同步点。
**方案**: 在同一个 encoder 内完成数据搬运（blit 或 compute copy），避免同步。
**注意**: 此前尝试 batching 导致 encoder 冲突崩溃，需要更谨慎的 encoder 管理。

### Tier 3: 小收益

#### 7. Embedding lookup 移到 GPU（预期 ~1ms 收益）
**问题**: embedding 在 CPU 做，然后 upload 5120 float 到 GPU。
**方案**: 使用 `kernel_get_rows_f32_i32` 或实现 Q4_K dequant get_rows。
**优先级**: 低 — 收益小但实现简单。

#### 8. 减少 KV cache 内存（预期 ~0ms 速度收益）
**问题**: KV cache 是 F32，占用大量 GPU 内存。
**方案**: 使用 FP8 压缩（参考 ds4）。
**影响**: 主要影响最大 context 大小，对 decode 速度影响有限。

#### 9. FP16 激活（prefill 模式）
**状态**: decode 模式已验证收益为负（conversion overhead > bandwidth savings）。
**可能性**: prefill 模式（序列长、中间 tensor 大）可能仍有收益。

## 性能分析数据

### 每层 dispatch 计数
| 层类型 | dispatch/层 | 层数 | 总 dispatch |
|--------|------------|------|-------------|
| DeltaNet | ~8 | 32 | 256 |
| Attention | ~11 | 32 | 352 |
| FFN | ~5 | 64 | 320 |
| **总计** | | **64** | **~928** |

### 每 dispatch 平均耗时
- 196ms / 928 dispatch ≈ 0.21ms/dispatch（含内核计算 + Metal API overhead）
- Metal dispatch overhead 约 10-50μs/call

### KV cache 增长影响
- 5-step: 196ms/token
- 10-step: 199ms/token
- 30-step: 215ms/token（attention 计算量随 kv_len 线性增长）

## 文件索引

| 文件 | 职责 |
|------|------|
| `q4.c` | GPU forward path |
| `q4_metal.m` | Metal GPU backend |
| `q4_gpu.h` | GPU API 声明 |
| `metal/quant.metal` | Q4_K/Q6_K matmul kernels + fused4 |
| `metal/attention.metal` | Flash Attention kernel |
| `metal/deltanet_ops.metal` | DeltaNet kernels |
| `metal/elementwise.metal` | Elementwise kernels |
| `metal/unary.metal` | Unary kernels |
