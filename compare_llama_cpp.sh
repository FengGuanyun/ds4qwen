#!/usr/bin/env bash
# Compare ds4qwen vs llama.cpp inference speed and memory.
#
# Usage:
#   ./compare_llama_cpp.sh model.gguf [llama-bench-path]
#
# Requirements:
#   - q4-bench binary in current directory (or ../build/)
#   - llama-bench binary (from llama.cpp build)
#
# Output: TSV comparison of prefill throughput, decode throughput, and memory.
set -euo pipefail

MODEL="${1:?Usage: $0 model.gguf [llama-bench-path]}"
LLAMA_BENCH="${2:-}"
Q4_BENCH="${Q4_BENCH_BIN:-./q4-bench}"

# Find q4-bench if not in current directory
if [ ! -x "$Q4_BENCH" ]; then
    if [ -x "../build/q4-bench" ]; then
        Q4_BENCH="../build/q4-bench"
    else
        echo "error: q4-bench not found at $Q4_BENCH or ../build/q4-bench"
        exit 1
    fi
fi

# Find llama-bench
if [ -z "$LLAMA_BENCH" ]; then
    for candidate in \
        ./llama-bench \
        ../llama.cpp/build/bin/llama-bench \
        /usr/local/bin/llama-bench \
        $(which llama-bench 2>/dev/null); do
        if [ -x "$candidate" ]; then
            LLAMA_BENCH="$candidate"
            break
        fi
    done
    if [ -z "$LLAMA_BENCH" ]; then
        echo "warning: llama-bench not found, skipping llama.cpp comparison"
        echo "  Build llama.cpp: cd llama.cpp && cmake -B build && cmake --build build --target llama-bench"
    fi
fi

CTX_SIZE=4096
PROMPT_LEN=512
DECODE_STEPS=64
N_THREADS=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 8)

echo "========================================"
echo "ds4qwen vs llama.cpp benchmark"
echo "========================================"
echo "Model:     $MODEL"
echo "Context:   $CTX_SIZE"
echo "Prefill:   $PROMPT_LEN tokens"
echo "Decode:    $DECODE_STEPS steps"
echo "Threads:   $N_THREADS"
echo ""

# ---------- ds4qwen benchmark ----------
echo "--- ds4qwen (CPU) ---"
$Q4_BENCH -m "$MODEL" --cpu --ctx $CTX_SIZE -t $N_THREADS \
    --prefill $PROMPT_LEN --decode $DECODE_STEPS 2>&1 | tee /tmp/q4_bench_output.txt

echo ""

# ---------- llama.cpp benchmark ----------
if [ -n "$LLAMA_BENCH" ]; then
    echo "--- llama.cpp ---"

    # llama-bench format: pp (prompt processing), tg (text generation)
    $LLAMA_BENCH \
        -m "$MODEL" \
        --model-type qwen3 \
        -ngl 0 \
        -n $DECODE_STEPS \
        -pp $PROMPT_LEN \
        -tg $DECODE_STEPS \
        -c $CTX_SIZE \
        -t $N_THREADS \
        -npl 1 \
        -fa 0 \
        2>&1 | tee /tmp/llama_bench_output.txt

    echo ""
    echo "========================================"
    echo "Comparison summary"
    echo "========================================"

    # Extract ds4qwen metrics
    Q4_PREFILL=$(grep 'prefill_tok_per_sec' /tmp/q4_bench_output.txt 2>/dev/null | awk '{print $2}')
    Q4_DECODE=$(grep 'decode_tok_per_sec' /tmp/q4_bench_output.txt 2>/dev/null | awk '{print $2}')
    Q4_LOAD=$(grep 'load_time_ms' /tmp/q4_bench_output.txt 2>/dev/null | awk '{print $2}')
    Q4_MODEL_MB=$(grep 'model_total_mb' /tmp/q4_bench_output.txt 2>/dev/null | awk '{print $2}')

    # Extract llama.cpp metrics (depends on output format)
    LLAMA_PP=$(grep 'pp' /tmp/llama_bench_output.txt 2>/dev/null | tail -1 | awk '{for(i=1;i<=NF;i++) if($i ~ /pp_ms/) print $(i+1)}')
    LLAMA_TG=$(grep 'tg' /tmp/llama_bench_output.txt 2>/dev/null | tail -1 | awk '{for(i=1;i<=NF;i++) if($i ~ /tg_ms/) print $(i+1)}')

    echo ""
    if [ -n "$Q4_PREFILL" ] && [ -n "$Q4_DECODE" ]; then
        printf "%-20s %15s %15s\n" "Metric" "ds4qwen" "llama.cpp"
        printf "%-20s %15s %15s\n" "------" "-------" "---------"
        printf "%-20s %15s %15s\n" "Prefill (tok/s)" "$Q4_PREFILL" "${LLAMA_PP:+computed}"
        printf "%-20s %15s %15s\n" "Decode (tok/s)" "$Q4_DECODE" "${LLAMA_TG:+computed}"
        printf "%-20s %15s %15s\n" "Load time (ms)" "${Q4_LOAD:-N/A}" "N/A"
        printf "%-20s %15s %15s\n" "Model memory (MB)" "${Q4_MODEL_MB:-N/A}" "N/A"
    else
        echo "ds4qwen metrics not available (CPU path may not be fully implemented yet)"
    fi
else
    echo "llama.cpp benchmark skipped (llama-bench not found)"
    echo ""
    echo "To compare with llama.cpp:"
    echo "  1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp"
    echo "  2. Build: cd llama.cpp && cmake -B build && cmake --build build --target llama-bench"
    echo "  3. Run: $0 $MODEL ./llama.cpp/build/bin/llama-bench"
fi
