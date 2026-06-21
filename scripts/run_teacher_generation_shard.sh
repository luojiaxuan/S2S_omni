#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
INPUT="${INPUT:-work/gigaspeech_pilot/compression_teacher_requests.jsonl}"
OUT_DIR="${OUT_DIR:-work/gigaspeech_pilot}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
LOG_EVERY="${LOG_EVERY:-25}"

source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false

python scripts/generate_teacher_labels.py \
  --input "${INPUT}" \
  --output "${OUT_DIR}/compression_teacher_labels_shard${SHARD_INDEX}.jsonl" \
  --sft-output "${OUT_DIR}/compression_sft_shard${SHARD_INDEX}.jsonl" \
  --backend transformers \
  --model "${MODEL}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --log-every "${LOG_EVERY}" \
  --resume
