#!/usr/bin/env bash
set -euo pipefail

# Shard v2 row requests 4 ways and generate long row targets against the
# 4 local MOSS serving processes.

RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
S2S_OMNI_ROOT="${S2S_OMNI_ROOT:-/data/S2S_omni}"
PORTS="${PORTS:-48731,49157,52391,54863}"
SPLITS="${SPLITS:-train,dev}"
FIXED_REF="${FIXED_REF:?FIXED_REF is required (server-visible wav path)}"

mkdir -p "${RUN_ROOT}/raw" "${RUN_ROOT}/logs" "${RUN_ROOT}/pids" "${RUN_ROOT}/wavs"

IFS=',' read -r -a PORT_ARRAY <<< "${PORTS}"
NUM_SHARDS="${#PORT_ARRAY[@]}"

python3 - "${RUN_ROOT}" "${SPLITS}" "${NUM_SHARDS}" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
splits = [s for s in sys.argv[2].split(",") if s]
num_shards = int(sys.argv[3])

for split in splits:
    rows = (run / "raw" / f"{split}_v2_rows.jsonl").read_text(encoding="utf-8").splitlines()
    outs = [
        (run / "raw" / f"{split}_v2_rows_shard{i}.jsonl").open("w", encoding="utf-8")
        for i in range(num_shards)
    ]
    try:
        for idx, line in enumerate(r for r in rows if r.strip()):
            outs[idx % num_shards].write(line + "\n")
    finally:
        for out in outs:
            out.close()
    print(json.dumps({"split": split, "rows": len(rows), "shards": num_shards}))
PY

for shard in "${!PORT_ARRAY[@]}"; do
  port="${PORT_ARRAY[$shard]}"
  (
    set -euo pipefail
    IFS=',' read -r -a SPLIT_ARRAY <<< "${SPLITS}"
    for split in "${SPLIT_ARRAY[@]}"; do
      mkdir -p "${RUN_ROOT}/wavs/${split}"
      python3 "${S2S_OMNI_ROOT}/scripts/generate_moss_realtime_long_targets.py" \
        --input-jsonl "${RUN_ROOT}/raw/${split}_v2_rows_shard${shard}.jsonl" \
        --output-jsonl "${RUN_ROOT}/raw/${split}_v2_raw_shard${shard}.jsonl" \
        --rejected-jsonl "${RUN_ROOT}/raw/${split}_v2_rejected_shard${shard}.jsonl" \
        --wav-dir "${RUN_ROOT}/wavs/${split}" \
        --base-url "http://127.0.0.1:${port}" \
        --fixed-ref "${FIXED_REF}" \
        --log-every 25
    done
  ) > "${RUN_ROOT}/logs/01_generate_v2_shard${shard}.log" 2>&1 &
  echo "$!" > "${RUN_ROOT}/pids/generate_v2_shard${shard}.pid"
done

wait
