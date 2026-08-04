#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
DATASET_ROOT="${DATASET_ROOT:-/data/datasets/infinisst-moss-tts-en-zh-segments-v1}"
S2S_OMNI_ROOT="${S2S_OMNI_ROOT:-/data/S2S_omni}"
MOSS_TTS_ROOT="${MOSS_TTS_ROOT:-/data/MOSS-TTS}"
MODEL_PATH="${MODEL_PATH:-OpenMOSS-Team/MOSS-TTS-Realtime}"
CODEC_PATH="${CODEC_PATH:-OpenMOSS-Team/MOSS-Audio-Tokenizer}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
PREP_NUM_PROCESSES="${PREP_NUM_PROCESSES:-4}"
PREP_BATCH_SIZE="${PREP_BATCH_SIZE:-16}"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-${S2S_OMNI_ROOT}/configs/accelerate_ddp_4gpu.yaml}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-60}"

mkdir -p "${RUN_ROOT}/raw" "${RUN_ROOT}/prepared" "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/logs"

wait_for_generation() {
  while true; do
    local alive=0
    for pidfile in "${RUN_ROOT}"/pids/generate_targets_full_shard*.pid; do
      [[ -e "${pidfile}" ]] || continue
      if kill -0 "$(cat "${pidfile}")" 2>/dev/null; then
        alive=1
      fi
    done
    if [[ "${alive}" == "0" ]]; then
      break
    fi
    date
    for split in train dev; do
      local total=0
      for shard in 0 1 2 3; do
        local path="${RUN_ROOT}/raw/${split}_moss_raw_shard${shard}.jsonl"
        local count=0
        [[ -f "${path}" ]] && count="$(wc -l < "${path}")"
        total=$((total + count))
      done
      echo "[wait] ${split}_accepted=${total}"
    done
    sleep "${POLL_INTERVAL_S}"
  done
}

combine_split() {
  local split="$1"
  local raw="${RUN_ROOT}/raw/${split}_moss_raw.jsonl"
  : > "${raw}"
  for shard in 0 1 2 3; do
    local path="${RUN_ROOT}/raw/${split}_moss_raw_shard${shard}.jsonl"
    [[ -f "${path}" ]] && cat "${path}" >> "${raw}"
  done
  echo "[combine] ${split} accepted $(wc -l < "${raw}") -> ${raw}"

  local rejected="${RUN_ROOT}/raw/${split}_moss_rejected.jsonl"
  : > "${rejected}"
  for shard in 0 1 2 3; do
    local path="${RUN_ROOT}/raw/${split}_moss_rejected_shard${shard}.jsonl"
    [[ -f "${path}" ]] && cat "${path}" >> "${rejected}"
  done
  echo "[combine] ${split} rejected $(wc -l < "${rejected}") -> ${rejected}"
}

stop_moss_serving() {
  python3 - "${RUN_ROOT}" <<'PY'
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

run = Path(sys.argv[1])
roots = []
for pidfile in sorted(run.glob("moss_realtime_serve*.pid")):
    try:
        pid = int(pidfile.read_text().strip())
        os.kill(pid, 0)
    except Exception:
        continue
    roots.append(pid)

out = subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True)
children = {}
for line in out.splitlines():
    try:
        pid, ppid = map(int, line.split())
    except Exception:
        continue
    children.setdefault(ppid, []).append(pid)

targets = []
stack = roots[:]
while stack:
    pid = stack.pop()
    targets.append(pid)
    stack.extend(children.get(pid, []))

for sig in (signal.SIGTERM, signal.SIGKILL):
    any_live = False
    for pid in sorted(set(targets), reverse=True):
        try:
            os.kill(pid, sig)
            any_live = True
        except ProcessLookupError:
            pass
    time.sleep(2)
    if not any_live:
        break
print({"stopped_moss_serving_pids": sorted(set(targets))})
PY
}

prepare_split() {
  local split="$1"
  local raw="${RUN_ROOT}/raw/${split}_moss_raw.jsonl"
  local prepared="${RUN_ROOT}/prepared/${split}_with_codes.jsonl"
  cd "${MOSS_TTS_ROOT}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" PYTHONPATH="${MOSS_TTS_ROOT}" \
  accelerate launch --num_processes "${PREP_NUM_PROCESSES}" \
    moss_tts_realtime/finetuning/prepare_data.py \
    --codec-path "${CODEC_PATH}" \
    --device auto \
    --input-jsonl "${raw}" \
    --output-jsonl "${prepared}" \
    --batch-size "${PREP_BATCH_SIZE}" \
    2>&1 | tee "${RUN_ROOT}/logs/02_prepare_${split}.log"
}

train_model() {
  local train_jsonl="${RUN_ROOT}/prepared/train_with_codes.rank*.jsonl"
  local output_dir="${RUN_ROOT}/checkpoints/moss_tts_realtime_infinisst_train"
  cd "${MOSS_TTS_ROOT}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" PYTHONPATH="${MOSS_TTS_ROOT}" \
  accelerate launch --config_file "${ACCELERATE_CONFIG_FILE}" \
    moss_tts_realtime/finetuning/sft.py \
    --model-path "${MODEL_PATH}" \
    --codec-path "${CODEC_PATH}" \
    --train-jsonl "${train_jsonl}" \
    --output-dir "${output_dir}" \
    --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning-rate "${LEARNING_RATE}" \
    --num-epochs "${NUM_EPOCHS}" \
    --num-workers "${NUM_WORKERS}" \
    --mixed-precision bf16 \
    2>&1 | tee "${RUN_ROOT}/logs/03_train.log"
}

wait_for_generation
combine_split train
combine_split dev
stop_moss_serving
prepare_split train
prepare_split dev
train_model
