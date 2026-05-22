# QwenStar 4

QwenStar 4 is a native inference engine specific for **Qwen3.6-27B**. It is
intentionally narrow: not a generic GGUF runner, not a wrapper around another
runtime — it is completely self-contained.

We support the following backends:
* **Metal** is our primary target. Starting from MacBooks with 32GB of RAM.
* **NVIDIA CUDA** with special care for DGX Spark and consumer GPUs.

## Motivations

Qwen3.6-27B is a dense 27B parameter model with a hybrid attention architecture:

1. It uses **Gated DeltaNet** (linear recurrent attention) for 48 of 64 layers,
   giving sub-quadratic scaling for long context.
2. **Gated Attention** (full softmax) appears every 4th layer to maintain global
   context and strong retrieval capabilities.
3. **GQA with 24Q/4KV heads** provides efficient attention with low KV cache.
4. The model features a context window of **262K tokens** (extendable to 1M).
5. Being a dense model, it is simpler to quantize and deploy than MoE variants.
6. The **SwiGLU FFN** with ~18944 intermediate dimensions provides strong capacity.
7. It performs at flagship level on coding benchmarks, outperforming much larger
   MoE models like Qwen3.5-397B-A17B.

This project follows the same philosophy as [ds4](https://github.com/antirez/ds4):
one model at a time, validated against official logits, tested at long context,
and integrated with a ready-to-use HTTP API compatible with OpenAI clients.

## Architecture

Qwen3.6-27B parameters:
- **64 layers** in 16 repeating blocks of (3× DeltaNet + 1× Attention)
- **5120** hidden dimension
- **248,320** vocabulary size (padded)
- **24 Q heads / 4 KV heads** (Grouped Query Attention)
- **128** head dimension
- **~18,944** FFN intermediate dimension (SwiGLU)
- Standard residual connections, full RoPE

## Building

```bash
# macOS (Metal, default)
make

# CPU-only (diagnostics, no production use)
make cpu

# Linux with CUDA
make cuda CUDA_ARCH=sm_120   # adjust for your GPU
```

Requires a C99 compiler. Metal backend requires macOS 13+. CUDA backend requires
nvcc and a working CUDA toolkit installation.

## Usage

### CLI

```bash
# Interactive REPL
./q4 -m qwen3.6-27b.gguf

# One-shot generation
./q4 -m qwen3.6-27b.gguf -p "Explain quantum computing" -n 200

# With specific context size
./q4 -m qwen3.6-27b.gguf --ctx-size 32768
```

### Server

```bash
./q4-server -m qwen3.6-27b.gguf --port 8080

# Then use with any OpenAI-compatible client:
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

### Benchmarking

```bash
./q4-bench -m qwen3.6-27b.gguf --ctx-start 2048 --ctx-max 65536 --step-incr 2048
```

## GGUF Download

Use the provided script to download the model from HuggingFace:

```bash
./download_model.sh
```

This downloads the Qwen3.6-27B GGUF (Q4_K_M quantized, ~17GB) from the Unsloth
or bartowski GGUF repositories.

## Model Card

See [MODEL_CARD.md](MODEL_CARD.md) for details about Qwen3.6-27B architecture,
sampling parameters, and known limitations.

## Acknowledgements

This project would not exist without **llama.cpp and GGML**, which opened the
path for local inference engines. We are thankful to the GGML authors and
contributors. Some quantization format definitions and CPU dot logic are
adapted from the GGML codebase.

## Status

This is **beta quality** code. The inference engine, Metal backend, and CLI
interface are functional. The HTTP server and KV cache persistence are
work in progress.

## License

Apache 2.0
