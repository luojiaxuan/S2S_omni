#!/usr/bin/env bash
# Same checkpoint, fed at the granularity it was finetuned for: one turn per 1.92 s chunk.
set -uo pipefail
R=/data/tmp/runs/20260830-200824-401864000
W=/data/delta_tts
SW=/d4/S2S_omni_runs/moss_tts_infinisst_v2_20260804/acl_bench/tts_rows
export PYTHONPATH="$R/env/site"
export CUDA_VISIBLE_DEVICES=1

python - "$SW" "$W/rows_chunk192.jsonl" <<'PY'
import json, sys
sw, out = sys.argv[1], sys.argv[2]
with open(out, "w", encoding="utf-8") as fh:
    for tid in ("268", "110", "117"):
        row = json.loads(open(f"{sw}/talk{tid}.chunk192.swrow.jsonl", encoding="utf-8").readline())
        row["num_segments"] = len(row["segments"])
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(row["row_id"], row["num_segments"], "segments")
PY

run () {
  local shard="$1"
  mkdir -p "$W/gran"
  python "$R/eval/code/w0.py" \
    --model-path /data/delta_tts/ckpt \
    --codec-path "$R/eval/resources/c1" \
    --moss-tts-root "$R/eval/resources/m0" \
    --fixed-ref "$R/eval/input/ref.wav" \
    --rows-jsonl "$W/rows_chunk192.jsonl" \
    --out-dir "$W/gran" \
    --summary-jsonl "$W/gran.s$shard.jsonl" \
    --num-shards 3 --shard-id "$shard" \
    --device cuda \
    --min-runaway-floor-s 15 --max-seconds-per-char 0.6 \
    --sliding-window 11 --soft-reset-keep 3 --continuous-codec-context \
    --seed 42 --log-every 1 > "$W/gran.s$shard.log" 2>&1
  echo "EXIT_gran.s$shard=$?" >> "$W/gran.s$shard.log"
}
for shard in 0 1 2; do run "$shard" & done
wait
echo GRAN_DONE > "$W/gran.status"
