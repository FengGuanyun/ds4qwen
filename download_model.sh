#!/usr/bin/env bash
# Download Qwen3.6-27B GGUF from HuggingFace.
# Defaults to the Unsloth Q4_K_M quant (~17GB).
set -euo pipefail

REPO="${Q4_MODEL_REPO:-unsloth/Qwen3.6-27B-GGUF}"
FILE="${Q4_MODEL_FILE:-Qwen3.6-27B.Q4_K_M.gguf}"
OUTPUT="${Q4_MODEL_PATH:-${FILE}}"
HF_TOKEN="${HF_TOKEN:-}"

URL="https://huggingface.co/${REPO}/resolve/main/${FILE}?download=true"

echo "Downloading ${FILE} from ${REPO}..."

if command -v hf &>/dev/null; then
    hf download "${REPO}" --include "${FILE}" --local-dir .
    if [ "${FILE}" != "${OUTPUT}" ]; then
        ln -sf "${FILE}" "${OUTPUT}"
    fi
elif command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "${REPO}" "${FILE}" --local-dir .
    if [ "${FILE}" != "${OUTPUT}" ]; then
        ln -sf "${FILE}" "${OUTPUT}"
    fi
else
    # Fallback: curl with resume support.
    curl -L -C - -o "${OUTPUT}" \
        ${HF_TOKEN:+-H "Authorization: Bearer ${HF_TOKEN}"} \
        "${URL}"
fi

echo "Model saved to ${OUTPUT}"
ls -lh "${OUTPUT}"
