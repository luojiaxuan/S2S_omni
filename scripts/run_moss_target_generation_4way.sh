#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/data/datasets/infinisst-moss-tts-en-zh-segments-v1}"
S2S_OMNI_ROOT="${S2S_OMNI_ROOT:-/data/S2S_omni}"
RUN_ROOT="${RUN_ROOT:-/data/S2S_omni_runs/moss_tts_infinisst_$(date +%Y%m%d_%H%M%S)}"
PORTS="${PORTS:-48731,49157,52391,54863}"
SPLITS="${SPLITS:-train,dev}"
TIMEOUT_S="${TIMEOUT_S:-240}"
LOG_EVERY="${LOG_EVERY:-250}"
PATH_MODE="${PATH_MODE:-absolute}"

mkdir -p "${RUN_ROOT}/raw" "${RUN_ROOT}/logs" "${RUN_ROOT}/pids"

IFS=',' read -r -a PORT_ARRAY <<< "${PORTS}"
NUM_SHARDS="${#PORT_ARRAY[@]}"

python3 - "${RUN_ROOT}" "${DATASET_ROOT}" "${SPLITS}" "${NUM_SHARDS}" <<'PY'
import sys
from pathlib import Path

run = Path(sys.argv[1])
dataset = Path(sys.argv[2])
splits = [s for s in sys.argv[3].split(",") if s]
num_shards = int(sys.argv[4])

for split in splits:
    outs = [
        (run / "raw" / f"{split}_full_requests_shard{i}.jsonl").open("w")
        for i in range(num_shards)
    ]
    try:
        with (dataset / "manifest" / f"{split}_moss_requests.jsonl").open() as f:
            for idx, line in enumerate(f):
                outs[idx % num_shards].write(line)
    finally:
        for out in outs:
            out.close()
PY

for shard in "${!PORT_ARRAY[@]}"; do
  port="${PORT_ARRAY[$shard]}"
  (
    set -euo pipefail
    IFS=',' read -r -a SPLIT_ARRAY <<< "${SPLITS}"
    for split in "${SPLIT_ARRAY[@]}"; do
      python3 "${S2S_OMNI_ROOT}/scripts/generate_moss_realtime_targets.py" \
        --input-jsonl "${RUN_ROOT}/raw/${split}_full_requests_shard${shard}.jsonl" \
        --dataset-root "${DATASET_ROOT}" \
        --output-jsonl "${RUN_ROOT}/raw/${split}_moss_raw_shard${shard}.jsonl" \
        --rejected-jsonl "${RUN_ROOT}/raw/${split}_moss_rejected_shard${shard}.jsonl" \
        --base-url "http://127.0.0.1:${port}" \
        --workers 1 \
        --timeout-s "${TIMEOUT_S}" \
        --path-mode "${PATH_MODE}" \
        --resume \
        --allow-rejections \
        --log-every "${LOG_EVERY}"
    done
  ) > "${RUN_ROOT}/logs/01_generate_targets_full_shard${shard}.log" 2>&1 &
  echo "$!" > "${RUN_ROOT}/pids/generate_targets_full_shard${shard}.pid"
done

wait
