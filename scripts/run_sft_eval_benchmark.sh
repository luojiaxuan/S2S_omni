#!/usr/bin/env bash
set -euo pipefail

ADAPTER_DIR="${1:?adapter dir is required}"
OUTPUT_DIR="${2:?output dir is required}"

MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"
AUDIO_NUM_SAMPLES="${AUDIO_NUM_SAMPLES:-8}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"

DEV_GOLD="${DEV_GOLD:-work/gigaspeech_heldout_eval_20260622/splits/dev/policy_sample_manifest.jsonl}"
TEST_GOLD="${TEST_GOLD:-work/gigaspeech_heldout_eval_20260622/splits/test/policy_sample_manifest.jsonl}"

mkdir -p "${OUTPUT_DIR}"

run_split() {
  local split="$1"
  local gold="$2"
  local split_dir="${OUTPUT_DIR}/${split}"
  local pred="${split_dir}/predictions.jsonl"

  mkdir -p "${split_dir}"
  python3 scripts/eval_lora_adapter_generation.py \
    --input "${gold}" \
    --adapter "${ADAPTER_DIR}" \
    --output "${pred}" \
    --summary "${split_dir}/generation_summary.json" \
    --model "${MODEL}" \
    --num-samples "${NUM_SAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --device-map "${DEVICE_MAP}"

  python3 scripts/evaluate_predictions.py \
    --gold "${gold}" \
    --pred "${pred}" \
    --prediction-field base_prediction \
    --output "${split_dir}/base_metrics.jsonl" \
    --summary "${split_dir}/base_metrics_summary.json"

  python3 scripts/evaluate_predictions.py \
    --gold "${gold}" \
    --pred "${pred}" \
    --prediction-field evo_prediction \
    --output "${split_dir}/evo_metrics.jsonl" \
    --summary "${split_dir}/evo_metrics_summary.json"
}

run_split dev "${DEV_GOLD}"
run_split test "${TEST_GOLD}"

python3 scripts/generate_talker_audio_samples.py \
  --input "${DEV_GOLD}" \
  --adapter "${ADAPTER_DIR}" \
  --output-dir "${OUTPUT_DIR}/audio_samples_dev" \
  --model "${MODEL}" \
  --num-samples "${AUDIO_NUM_SAMPLES}" \
  --device-map "${DEVICE_MAP}"
