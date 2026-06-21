#!/usr/bin/env bash
set -euo pipefail

PID_FILE="${PID_FILE:-logs/sft_train.pid}"
TRAIN_LOG="${TRAIN_LOG:-logs/sft_train.log}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/qwen3_omni_gigaspeech_mixed_compression_sft}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"

if [[ ! -s "${PID_FILE}" ]]; then
  echo "missing pid file: ${PID_FILE}" >&2
  exit 1
fi

pid="$(awk 'NF {print $1; exit}' "${PID_FILE}")"
while true; do
  echo "===== $(date -Is) ====="
  if kill -0 "${pid}" 2>/dev/null; then
    echo "status=running pid=${pid}"
  else
    echo "status=exited pid=${pid}"
  fi

  echo "-- gpu --"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true

  echo "-- latest trainer logs --"
  TRAIN_LOG_PATH="${TRAIN_LOG}" python3 - <<'PY' || true
from pathlib import Path
import os

path = Path(os.environ["TRAIN_LOG_PATH"])
if path.exists():
    text = path.read_text(errors="replace").replace("\r", "\n")
    needles = (
        "loss",
        "train_loss",
        "global_step",
        "epoch",
        "train_runtime",
        "train_samples_per_second",
        "/249",
        "checkpoint",
    )
    lines = [line for line in text.splitlines() if any(needle in line for needle in needles)]
    for line in lines[-30:]:
        print(line[-260:])
PY

  echo "-- output dir --"
  ls -lh "${OUTPUT_DIR}" 2>/dev/null || true
  find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -print 2>/dev/null | sort | tail -n 5 || true

  if ! kill -0 "${pid}" 2>/dev/null; then
    exit 0
  fi
  sleep "${INTERVAL_SECONDS}"
done
