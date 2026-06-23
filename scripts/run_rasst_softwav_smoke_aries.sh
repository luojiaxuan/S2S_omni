#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-${PWD}}"
S2S_RUN_ROOT="${S2S_RUN_ROOT:-/mnt/data/jiaxuanluo/S2S_omni_runs}"
OUT_ROOT="${OUT_ROOT:-${S2S_RUN_ROOT}/rasst_softwav_smoke_$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
TRAIN_JSONL="${TRAIN_JSONL:-/mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline.jsonl}"
GPU="${GPU:-0,1,2,3,4,5,6,7}"
PYTHON="${PYTHON:-python3}"
MAX_RECORDS="${MAX_RECORDS:-20}"
MFA_JOBS="${MFA_JOBS:-8}"

mkdir -p "${OUT_ROOT}"/logs
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

echo "[1/5] Generate full-target Qwen3-Omni wav/codes"
"${PYTHON}" "${ROOT}/scripts/generate_rasst_omni_tts_targets.py" \
  --input "${TRAIN_JSONL}" \
  --output-dir "${OUT_ROOT}/targets" \
  --model "${MODEL}" \
  --max-records "${MAX_RECORDS}" \
  --device-map auto \
  --log-every 1 \
  2>&1 | tee "${OUT_ROOT}/logs/01_generate_targets.log"

echo "[2/5] MFA align generated target wavs"
MFA_JOBS="${MFA_JOBS}" bash "${ROOT}/scripts/run_mfa_align_rasst_targets.sh" \
  "${OUT_ROOT}/targets" \
  "${OUT_ROOT}/targets/mfa_aligned" \
  2>&1 | tee "${OUT_ROOT}/logs/02_mfa_align.log"

echo "[3/5] Build turn-level soft-wav manifest"
"${PYTHON}" "${ROOT}/scripts/build_rasst_softwav_manifest.py" \
  --rasst-jsonl "${TRAIN_JSONL}" \
  --target-manifest "${OUT_ROOT}/targets/target_manifest.jsonl" \
  --textgrid-dir "${OUT_ROOT}/targets/mfa_aligned" \
  --output-dir "${OUT_ROOT}/manifest" \
  --max-records "${MAX_RECORDS}" \
  --speed-assignment cycle \
  --speed-factors 1.0,1.35,1.7,2.0 \
  --log-every 1 \
  2>&1 | tee "${OUT_ROOT}/logs/03_build_manifest.log"

echo "[4/5] Tokenization/processor smoke"
"${PYTHON}" "${ROOT}/scripts/train_qwen3_omni_softwav_lora.py" \
  --train-manifest "${OUT_ROOT}/manifest/turn_manifest.jsonl" \
  --output-dir "${OUT_ROOT}/train_smoke" \
  --model "${MODEL}" \
  --max-records 1 \
  --tokenize-smoke \
  2>&1 | tee "${OUT_ROOT}/logs/04_tokenize_smoke.log"

echo "[5/5] One optimizer-step smoke"
"${PYTHON}" "${ROOT}/scripts/train_qwen3_omni_softwav_lora.py" \
  --train-manifest "${OUT_ROOT}/manifest/turn_manifest.jsonl" \
  --output-dir "${OUT_ROOT}/train_1step" \
  --model "${MODEL}" \
  --max-records 4 \
  --max-steps 1 \
  --gradient-accumulation-steps 1 \
  --max-codec-frames 16 \
  --save-steps 1 \
  --logging-steps 1 \
  2>&1 | tee "${OUT_ROOT}/logs/05_train_1step.log"

echo "[DONE] ${OUT_ROOT}"
