#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-sglang-omni-jiaxuan-aries-hibiki-zero}"
RUN_ROOT="${RUN_ROOT:-/mnt/data3/jiaxuanluo/s2s_omni_hibiki_zero/runs/smoke_$(date +%Y%m%d_%H%M%S)}"
C_RUN_ROOT="${C_RUN_ROOT:-/data/runs/$(basename "${RUN_ROOT}")}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:?Set SOURCE_MANIFEST to a fr/es/pt/de -> en source JSONL}"
TTS_URLS="${TTS_URLS:-}"
TTS_URL="${TTS_URL:-http://127.0.0.1:18112/v1/audio/speech}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:30000/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
TEACHER_BACKEND="${TEACHER_BACKEND:-openai}"
MAX_PER_LANG="${MAX_PER_LANG:-5}"
MAX_RECORDS="${MAX_RECORDS:-20}"
TTS_WORKERS="${TTS_WORKERS:-4}"
MFA_BIN="${MFA_BIN:-mfa}"
MFA_MODEL="${MFA_MODEL:-english_us_arpa}"
MFA_DICT="${MFA_DICT:-english_us_arpa}"

mkdir -p "${RUN_ROOT}/logs"
echo "{\"run_root\":\"${RUN_ROOT}\",\"container_run_root\":\"${C_RUN_ROOT}\",\"source_manifest\":\"${SOURCE_MANIFEST}\"}"

docker exec \
  -e C_RUN_ROOT="${C_RUN_ROOT}" \
  -e SOURCE_MANIFEST="${SOURCE_MANIFEST}" \
  -e MAX_PER_LANG="${MAX_PER_LANG}" \
  "${CONTAINER_NAME}" \
  bash -lc '
    source /data/.venvs/hibiki-zero/bin/activate
    cd /data/repo/S2S_omni
    export PYTHONPATH=/data/repo/S2S_omni:${PYTHONPATH:-}
    python scripts/hibiki_zero_prepare_sources.py \
      --input "${SOURCE_MANIFEST}" \
      --output-dir "${C_RUN_ROOT}/data" \
      --max-per-lang "${MAX_PER_LANG}" \
      --split
  ' 2>&1 | tee "${RUN_ROOT}/logs/01_prepare_sources.log"

docker exec \
  -e C_RUN_ROOT="${C_RUN_ROOT}" \
  -e TEACHER_BACKEND="${TEACHER_BACKEND}" \
  -e TEACHER_MODEL="${TEACHER_MODEL}" \
  -e OPENAI_BASE_URL="${OPENAI_BASE_URL}" \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  "${CONTAINER_NAME}" \
  bash -lc '
    source /data/.venvs/hibiki-zero/bin/activate
    cd /data/repo/S2S_omni
    export PYTHONPATH=/data/repo/S2S_omni:${PYTHONPATH:-}
    python scripts/hibiki_zero_generate_teacher_text.py \
      --input "${C_RUN_ROOT}/data/source_manifest.jsonl" \
      --output "${C_RUN_ROOT}/teacher/teacher_manifest.jsonl" \
      --backend "${TEACHER_BACKEND}" \
      --model "${TEACHER_MODEL}" \
      --base-url "${OPENAI_BASE_URL}" \
      --api-key "${OPENAI_API_KEY}" \
      --max-records '"${MAX_RECORDS}"'
  ' 2>&1 | tee "${RUN_ROOT}/logs/02_teacher_text.log"

docker exec \
  -e C_RUN_ROOT="${C_RUN_ROOT}" \
  -e TTS_URL="${TTS_URL}" \
  -e TTS_URLS="${TTS_URLS}" \
  -e TTS_WORKERS="${TTS_WORKERS}" \
  "${CONTAINER_NAME}" \
  bash -lc '
    source /data/.venvs/hibiki-zero/bin/activate
    cd /data/repo/S2S_omni
    export PYTHONPATH=/data/repo/S2S_omni:${PYTHONPATH:-}
    python scripts/hibiki_zero_generate_tts_targets.py \
      --input "${C_RUN_ROOT}/teacher/teacher_manifest.jsonl" \
      --output-dir "${C_RUN_ROOT}/tts" \
      --url "${TTS_URL}" \
      --urls "${TTS_URLS}" \
      --backend moss_tts_http \
      --workers "${TTS_WORKERS}"
  ' 2>&1 | tee "${RUN_ROOT}/logs/03_tts.log"

"${MFA_BIN}" align \
  "${RUN_ROOT}/tts/mfa_corpus" \
  "${MFA_DICT}" \
  "${MFA_MODEL}" \
  "${RUN_ROOT}/mfa_aligned" \
  --clean \
  --overwrite \
  2>&1 | tee "${RUN_ROOT}/logs/04_mfa.log"

docker exec \
  -e C_RUN_ROOT="${C_RUN_ROOT}" \
  "${CONTAINER_NAME}" \
  bash -lc '
    source /data/.venvs/hibiki-zero/bin/activate
    cd /data/repo/S2S_omni
    export PYTHONPATH=/data/repo/S2S_omni:${PYTHONPATH:-}
    python scripts/hibiki_zero_slice_mfa_chunks.py \
      --tts-manifest "${C_RUN_ROOT}/tts/tts_manifest.jsonl" \
      --textgrid-dir "${C_RUN_ROOT}/mfa_aligned" \
      --output-dir "${C_RUN_ROOT}/sft" \
      --speed-assignment cycle \
      --speed-factors 1.0,1.35,1.7,2.0
    python scripts/hibiki_zero_validate_manifest.py \
      --manifest "${C_RUN_ROOT}/sft/sft_sample_manifest.jsonl" \
      --output "${C_RUN_ROOT}/sft/validation.json"
    python scripts/hibiki_zero_run_baseline.py \
      --manifest "${C_RUN_ROOT}/sft/sft_sample_manifest.jsonl" \
      --output-dir "${C_RUN_ROOT}/baseline" \
      --max-records 20
    python scripts/hibiki_zero_render_html.py \
      --manifest "${C_RUN_ROOT}/sft/sft_sample_manifest.jsonl" \
      --predictions "${C_RUN_ROOT}/baseline/baseline_predictions.jsonl" \
      --output "${C_RUN_ROOT}/index.html" \
      --max-samples 20
  ' 2>&1 | tee "${RUN_ROOT}/logs/05_slice_baseline_html.log"

echo "[DONE] ${RUN_ROOT}"
