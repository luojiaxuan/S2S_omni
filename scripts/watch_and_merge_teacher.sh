#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-work/gigaspeech_pilot}"
LOG_DIR="${LOG_DIR:-logs}"
NUM_SHARDS="${NUM_SHARDS:-2}"
REQUESTS="${REQUESTS:-${OUT_DIR}/compression_teacher_requests.jsonl}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
MERGED_LABELS="${MERGED_LABELS:-${OUT_DIR}/compression_teacher_labels.jsonl}"
MERGED_SFT="${MERGED_SFT:-${OUT_DIR}/compression_sft.jsonl}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${OUT_DIR}/compression_teacher_summary.json}"

source .venv/bin/activate

pid_for_shard() {
  local shard="$1"
  local pid_file="${LOG_DIR}/teacher_generation_shard${shard}.pid"
  if [[ -s "${pid_file}" ]]; then
    awk 'NF {print $1; exit}' "${pid_file}"
    return 0
  fi
  ps -u "$(whoami)" -o pid=,args= \
    | grep "[p]ython scripts/generate_teacher_labels.py" \
    | grep -- "--shard-index ${shard}" \
    | awk '{print $1; exit}'
}

label_files=()
for shard in $(seq 0 "$((NUM_SHARDS - 1))"); do
  label_files+=("${OUT_DIR}/compression_teacher_labels_shard${shard}.jsonl")
done

while true; do
  any_running=0
  for shard in $(seq 0 "$((NUM_SHARDS - 1))"); do
    pid="$(pid_for_shard "${shard}" || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      any_running=1
    fi
  done

  date -Is
  for shard in $(seq 0 "$((NUM_SHARDS - 1))"); do
    labels="${OUT_DIR}/compression_teacher_labels_shard${shard}.jsonl"
    sft="${OUT_DIR}/compression_sft_shard${shard}.jsonl"
    printf 'shard=%s labels=' "${shard}"
    wc -l "${labels}" 2>/dev/null | awk '{print $1}' || true
    printf 'shard=%s sft=' "${shard}"
    wc -l "${sft}" 2>/dev/null | awk '{print $1}' || true
  done

  if [[ "${any_running}" -eq 0 ]]; then
    break
  fi
  sleep "${SLEEP_SECONDS}"
done

python scripts/merge_teacher_labels.py \
  --requests "${REQUESTS}" \
  --labels "${label_files[@]}" \
  --output "${MERGED_LABELS}" \
  --sft-output "${MERGED_SFT}"

python scripts/summarize_teacher_labels.py --pretty "${MERGED_LABELS}" > "${SUMMARY_OUTPUT}"
cat "${SUMMARY_OUTPUT}"
