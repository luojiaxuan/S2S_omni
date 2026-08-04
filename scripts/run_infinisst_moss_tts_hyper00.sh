#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/data/datasets/infinisst-moss-tts-en-zh-segments-v1}"
S2S_OMNI_ROOT="${S2S_OMNI_ROOT:-/data/S2S_omni}"
MOSS_TTS_ROOT="${MOSS_TTS_ROOT:-/data/MOSS-TTS}"
RUN_ROOT="${RUN_ROOT:-/data/S2S_omni_runs/moss_tts_infinisst_$(date +%Y%m%d_%H%M%S)}"

SPLIT="${SPLIT:-train}"
MOSS_BASE_URL="${MOSS_BASE_URL:-http://127.0.0.1:8000}"
MODEL_PATH="${MODEL_PATH:-OpenMOSS-Team/MOSS-TTS-Realtime}"
CODEC_PATH="${CODEC_PATH:-OpenMOSS-Team/MOSS-Audio-Tokenizer}"

PREP_NUM_PROCESSES="${PREP_NUM_PROCESSES:-4}"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-${S2S_OMNI_ROOT}/configs/accelerate_ddp_4gpu.yaml}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SKIP_GENERATE="${SKIP_GENERATE:-0}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

mkdir -p "${RUN_ROOT}/raw" "${RUN_ROOT}/prepared" "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/logs"

REQUEST_JSONL="${DATASET_ROOT}/manifest/${SPLIT}_moss_requests.jsonl"
RAW_JSONL="${RUN_ROOT}/raw/${SPLIT}_moss_raw.jsonl"
REJECTED_JSONL="${RUN_ROOT}/raw/${SPLIT}_moss_rejected.jsonl"
PREPARED_JSONL="${RUN_ROOT}/prepared/${SPLIT}_with_codes.jsonl"
TRAIN_JSONL="${PREPARED_JSONL%.jsonl}.rank*.jsonl"
OUTPUT_DIR="${RUN_ROOT}/checkpoints/moss_tts_realtime_infinisst_${SPLIT}"

echo "{\"run_root\":\"${RUN_ROOT}\",\"dataset_root\":\"${DATASET_ROOT}\",\"split\":\"${SPLIT}\",\"moss_base_url\":\"${MOSS_BASE_URL}\"}" \
  | tee "${RUN_ROOT}/run_config.json"

if [[ "${SKIP_GENERATE}" != "1" ]]; then
  python "${S2S_OMNI_ROOT}/scripts/generate_moss_realtime_targets.py" \
    --input-jsonl "${REQUEST_JSONL}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-jsonl "${RAW_JSONL}" \
    --rejected-jsonl "${REJECTED_JSONL}" \
    --base-url "${MOSS_BASE_URL}" \
    --model "${MODEL_PATH}" \
    --workers 1 \
    --path-mode absolute \
    2>&1 | tee "${RUN_ROOT}/logs/01_generate_targets.log"
fi

cd "${MOSS_TTS_ROOT}"

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  accelerate launch --num_processes "${PREP_NUM_PROCESSES}" \
    moss_tts_realtime/finetuning/prepare_data.py \
    --codec-path "${CODEC_PATH}" \
    --device auto \
    --input-jsonl "${RAW_JSONL}" \
    --output-jsonl "${PREPARED_JSONL}" \
    2>&1 | tee "${RUN_ROOT}/logs/02_prepare_codes.log"
fi

accelerate launch --config_file "${ACCELERATE_CONFIG_FILE}" \
  moss_tts_realtime/finetuning/sft.py \
  --model-path "${MODEL_PATH}" \
  --codec-path "${CODEC_PATH}" \
  --train-jsonl "${TRAIN_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning-rate "${LEARNING_RATE}" \
  --num-epochs "${NUM_EPOCHS}" \
  --num-workers "${NUM_WORKERS}" \
  --mixed-precision bf16 \
  2>&1 | tee "${RUN_ROOT}/logs/03_train.log"
