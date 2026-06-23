#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-sglang-omni-jiaxuan-aries-e2e-0623-smoke}"
ROOT_HOST="${ROOT_HOST:-/mnt/data/jiaxuanluo/S2S_omni}"
RUN_ROOT="${RUN_ROOT:-/mnt/data3/jiaxuanluo/s2s_omni_aries/runs/s2st_s2sonly_12k_$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="$(basename "${RUN_ROOT}")"
C_RUN_ROOT="${C_RUN_ROOT:-/data/runs/${RUN_NAME}}"
TRAIN_JSONL="${TRAIN_JSONL:-/mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline.jsonl}"
TTS_URL="${TTS_URL:-http://127.0.0.1:18112/v1/audio/speech}"
TTS_MODELS_URL="${TTS_MODELS_URL:-http://127.0.0.1:18112/v1/models}"
MAX_RECORDS="${MAX_RECORDS:-12500}"
TTS_WORKERS="${TTS_WORKERS:-8}"
MFA_JOBS="${MFA_JOBS:-8}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
TEXT_CE_WEIGHT="${TEXT_CE_WEIGHT:-0.05}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
MAX_CODEC_FRAMES="${MAX_CODEC_FRAMES:-32}"

mkdir -p "${RUN_ROOT}/logs"

echo "{\"run_root\":\"${RUN_ROOT}\",\"container_run_root\":\"${C_RUN_ROOT}\",\"max_records\":${MAX_RECORDS},\"tts_workers\":${TTS_WORKERS},\"text_ce_weight\":${TEXT_CE_WEIGHT},\"learning_rate\":\"${LEARNING_RATE}\",\"max_codec_frames\":${MAX_CODEC_FRAMES}}"

docker exec "${CONTAINER_NAME}" bash -lc "curl -fsS '${TTS_MODELS_URL}' >/dev/null"

echo "[1/4] Generate Qwen3-TTS HTTP targets"
docker exec \
  -e C_RUN_ROOT="${C_RUN_ROOT}" \
  -e TRAIN_JSONL="${TRAIN_JSONL}" \
  -e TTS_URL="${TTS_URL}" \
  -e MAX_RECORDS="${MAX_RECORDS}" \
  -e TTS_WORKERS="${TTS_WORKERS}" \
  "${CONTAINER_NAME}" \
  bash -lc '
    source /data/.venv/bin/activate
    cd /data/repo/S2S_omni
    export PYTHONPATH=/data/repo/S2S_omni:${PYTHONPATH:-}
    python scripts/generate_rasst_http_tts_targets.py \
      --input "${TRAIN_JSONL}" \
      --output-dir "${C_RUN_ROOT}/targets" \
      --url "${TTS_URL}" \
      --max-records "${MAX_RECORDS}" \
      --workers "${TTS_WORKERS}" \
      --timeout-s 900 \
      --log-every 25
  ' 2>&1 | tee -a "${RUN_ROOT}/logs/01_generate_http_targets.log"

echo "[2/4] MFA align target wavs"
MFA_JOBS="${MFA_JOBS}" bash "${ROOT_HOST}/scripts/run_mfa_align_rasst_targets.sh" \
  "${RUN_ROOT}/targets" \
  "${RUN_ROOT}/targets/mfa_aligned" \
  2>&1 | tee -a "${RUN_ROOT}/logs/02_mfa_align.log"

echo "[3/4] Build turn-level S2ST manifest"
docker exec \
  -e C_RUN_ROOT="${C_RUN_ROOT}" \
  -e TRAIN_JSONL="${TRAIN_JSONL}" \
  -e MAX_RECORDS="${MAX_RECORDS}" \
  "${CONTAINER_NAME}" \
  bash -lc '
    source /data/.venv/bin/activate
    cd /data/repo/S2S_omni
    export PYTHONPATH=/data/repo/S2S_omni:${PYTHONPATH:-}
    python scripts/build_rasst_softwav_manifest.py \
      --rasst-jsonl "${TRAIN_JSONL}" \
      --target-manifest "${C_RUN_ROOT}/targets/target_manifest.jsonl" \
      --textgrid-dir "${C_RUN_ROOT}/targets/mfa_aligned" \
      --output-dir "${C_RUN_ROOT}/manifest" \
      --max-records "${MAX_RECORDS}" \
      --speed-assignment cycle \
      --speed-factors 1.0,1.35,1.7,2.0 \
      --log-every 100
  ' 2>&1 | tee -a "${RUN_ROOT}/logs/03_build_manifest.log"

echo "[4/4] Train S2S-only all-LoRA soft-wav adapter"
docker exec \
  -e C_RUN_ROOT="${C_RUN_ROOT}" \
  -e TRAIN_GPUS="${TRAIN_GPUS}" \
  -e MODEL="${MODEL}" \
  -e TEXT_CE_WEIGHT="${TEXT_CE_WEIGHT}" \
  -e LEARNING_RATE="${LEARNING_RATE}" \
  -e MAX_CODEC_FRAMES="${MAX_CODEC_FRAMES}" \
  "${CONTAINER_NAME}" \
  bash -lc '
    source /data/.venv/bin/activate
    cd /data/repo/S2S_omni
    export PYTHONPATH=/data/repo/S2S_omni:${PYTHONPATH:-}
    export HF_HOME=/root/.cache/huggingface
    export HF_HUB_CACHE=/root/.cache/huggingface/hub
    export XDG_CACHE_HOME=/data/.cache
    export TMPDIR=/data/tmp
    export TOKENIZERS_PARALLELISM=false
    export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
    python scripts/train_qwen3_omni_softwav_lora.py \
      --train-manifest "${C_RUN_ROOT}/manifest/turn_manifest.jsonl" \
      --output-dir "${C_RUN_ROOT}/checkpoints/s2sonly_textce_${TEXT_CE_WEIGHT}_all_lora_softwav" \
      --model "${MODEL}" \
      --device-map auto \
      --epochs 1 \
      --gradient-accumulation-steps 16 \
      --max-codec-frames "${MAX_CODEC_FRAMES}" \
      --save-steps 100 \
      --logging-steps 10 \
      --learning-rate "${LEARNING_RATE}" \
      --warmup-steps 0 \
      --train-scope all \
      --text-ce-weight "${TEXT_CE_WEIGHT}" \
      --codec-ce-weight 0 \
      --wav-l1-weight 0.2 \
      --stft-weight 0.5 \
      --no-detach-talker-condition-thinker \
      --gradient-checkpointing \
      --code2wav-grad-mode soft \
      --soft-temperature-start 1.5 \
      --soft-temperature-end 1.5
  ' 2>&1 | tee -a "${RUN_ROOT}/logs/04_train_s2sonly.log"

echo "[DONE] ${RUN_ROOT}"
