#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-${PWD}}"
S2S_RUN_ROOT="${S2S_RUN_ROOT:-/mnt/data/jiaxuanluo/S2S_omni_runs}"
OUT_ROOT="${OUT_ROOT:-${S2S_RUN_ROOT}/rasst_softwav_full_$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
TRAIN_JSONL="${TRAIN_JSONL:-/mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline.jsonl}"
DEV_JSONL="${DEV_JSONL:-/mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline_dev.jsonl}"
GPU="${GPU:-0,1,2,3,4,5,6,7}"
PYTHON="${PYTHON:-python3}"
MFA_JOBS="${MFA_JOBS:-8}"
TARGET_SHARDS="${TARGET_SHARDS:-1}"
TARGET_SHARD_INDEX="${TARGET_SHARD_INDEX:-0}"
TRAIN_MAX_RECORDS="${TRAIN_MAX_RECORDS:-12500}"
EVAL_MAX_RECORDS="${EVAL_MAX_RECORDS:-50}"

mkdir -p "${OUT_ROOT}"/logs
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

echo "[1/8] Generate train full-target Qwen3-Omni wav/codes"
"${PYTHON}" "${ROOT}/scripts/generate_rasst_omni_tts_targets.py" \
  --input "${TRAIN_JSONL}" \
  --output-dir "${OUT_ROOT}/train_targets" \
  --model "${MODEL}" \
  --device-map auto \
  --max-records "${TRAIN_MAX_RECORDS}" \
  --num-shards "${TARGET_SHARDS}" \
  --shard-index "${TARGET_SHARD_INDEX}" \
  --log-every 10 \
  2>&1 | tee "${OUT_ROOT}/logs/01_generate_train_targets.shard${TARGET_SHARD_INDEX}.log"

if [[ "${TARGET_SHARDS}" != "1" ]]; then
  echo "[INFO] TARGET_SHARDS=${TARGET_SHARDS}; run all shards, then rerun with SKIP_TARGETS=1 for MFA/manifest/train."
  exit 0
fi

echo "[2/8] MFA align train targets"
MFA_JOBS="${MFA_JOBS}" bash "${ROOT}/scripts/run_mfa_align_rasst_targets.sh" \
  "${OUT_ROOT}/train_targets" \
  "${OUT_ROOT}/train_targets/mfa_aligned" \
  2>&1 | tee "${OUT_ROOT}/logs/02_mfa_train.log"

echo "[3/8] Build train turn manifest"
"${PYTHON}" "${ROOT}/scripts/build_rasst_softwav_manifest.py" \
  --rasst-jsonl "${TRAIN_JSONL}" \
  --target-manifest "${OUT_ROOT}/train_targets/target_manifest.jsonl" \
  --textgrid-dir "${OUT_ROOT}/train_targets/mfa_aligned" \
  --output-dir "${OUT_ROOT}/train_manifest" \
  --max-records "${TRAIN_MAX_RECORDS}" \
  --speed-assignment cycle \
  --speed-factors 1.0,1.35,1.7,2.0 \
  --log-every 100 \
  2>&1 | tee "${OUT_ROOT}/logs/03_build_train_manifest.log"

echo "[4/8] Generate dev full-target Qwen3-Omni wav/codes"
"${PYTHON}" "${ROOT}/scripts/generate_rasst_omni_tts_targets.py" \
  --input "${DEV_JSONL}" \
  --output-dir "${OUT_ROOT}/dev_targets" \
  --model "${MODEL}" \
  --device-map auto \
  --log-every 10 \
  2>&1 | tee "${OUT_ROOT}/logs/04_generate_dev_targets.log"

echo "[5/8] MFA align dev targets"
MFA_JOBS="${MFA_JOBS}" bash "${ROOT}/scripts/run_mfa_align_rasst_targets.sh" \
  "${OUT_ROOT}/dev_targets" \
  "${OUT_ROOT}/dev_targets/mfa_aligned" \
  2>&1 | tee "${OUT_ROOT}/logs/05_mfa_dev.log"

echo "[6/8] Build dev turn manifest with all speed variants"
"${PYTHON}" "${ROOT}/scripts/build_rasst_softwav_manifest.py" \
  --rasst-jsonl "${DEV_JSONL}" \
  --target-manifest "${OUT_ROOT}/dev_targets/target_manifest.jsonl" \
  --textgrid-dir "${OUT_ROOT}/dev_targets/mfa_aligned" \
  --output-dir "${OUT_ROOT}/dev_manifest" \
  --speed-assignment all \
  --speed-factors 1.0,1.35,1.7,2.0 \
  --log-every 50 \
  2>&1 | tee "${OUT_ROOT}/logs/06_build_dev_manifest.log"

echo "[7/8] Tokenization/processor smoke on train manifest"
"${PYTHON}" "${ROOT}/scripts/train_qwen3_omni_softwav_lora.py" \
  --train-manifest "${OUT_ROOT}/train_manifest/turn_manifest.jsonl" \
  --output-dir "${OUT_ROOT}/train_tokenize_smoke" \
  --model "${MODEL}" \
  --max-records 1 \
  --tokenize-smoke \
  2>&1 | tee "${OUT_ROOT}/logs/07_tokenize_smoke.log"

echo "[8/10] Full LoRA SFT"
"${PYTHON}" "${ROOT}/scripts/train_qwen3_omni_softwav_lora.py" \
  --train-manifest "${OUT_ROOT}/train_manifest/turn_manifest.jsonl" \
  --dev-manifest "${OUT_ROOT}/dev_manifest/turn_manifest.jsonl" \
  --output-dir "${OUT_ROOT}/checkpoints/qwen3_omni_rasst_softwav_lora" \
  --model "${MODEL}" \
  --epochs 1 \
  --gradient-accumulation-steps 16 \
  --max-codec-frames 96 \
  --logging-steps 5 \
  --save-steps 100 \
  2>&1 | tee "${OUT_ROOT}/logs/08_train_full.log"

echo "[9/10] Generate dev predictions"
"${PYTHON}" "${ROOT}/scripts/generate_rasst_softwav_outputs.py" \
  --manifest "${OUT_ROOT}/dev_manifest/turn_manifest.jsonl" \
  --output-dir "${OUT_ROOT}/dev_predictions" \
  --model "${MODEL}" \
  --adapter "${OUT_ROOT}/checkpoints/qwen3_omni_rasst_softwav_lora/final" \
  --max-records "${EVAL_MAX_RECORDS}" \
  2>&1 | tee "${OUT_ROOT}/logs/09_generate_dev_predictions.log"

echo "[10/10] Evaluate dev predictions"
"${PYTHON}" "${ROOT}/scripts/evaluate_rasst_softwav_outputs.py" \
  --manifest "${OUT_ROOT}/dev_manifest/turn_manifest.jsonl" \
  --predictions "${OUT_ROOT}/dev_predictions/predictions.jsonl" \
  --output "${OUT_ROOT}/dev_eval/metrics.jsonl" \
  --summary "${OUT_ROOT}/dev_eval/summary.json" \
  --html "${OUT_ROOT}/dev_eval/index.html" \
  2>&1 | tee "${OUT_ROOT}/logs/10_eval_dev.log"

echo "[DONE] ${OUT_ROOT}"
