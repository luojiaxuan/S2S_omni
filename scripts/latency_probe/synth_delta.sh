#!/usr/bin/env bash
# Swap only the TTS weights: the colleague's delta-finetuned checkpoint against our own v8,
# same InfiniSST phrase rows, same fixed speaker prompt, same sliding window, same guard.
set -uo pipefail
R=/data/tmp/runs/20260830-200824-401864000
W=/data/delta_tts
mkdir -p "$W" || exit 1
export PYTHONPATH="$R/env/site"
export CUDA_VISIBLE_DEVICES=1          # container GPU 1 = host idx 6

python - "$R/eval/input/c2.jsonl" "$W/rows.jsonl" <<'PY'
import json, sys
keep = ("talk110_phrv2ep1fix_full", "talk117_phrv2ep1fix_full", "talk268_phrv2ep1fix_full")
with open(sys.argv[2], "w", encoding="utf-8") as out:
    for line in open(sys.argv[1], encoding="utf-8"):
        row = json.loads(line)
        if row["row_id"] in keep:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
print("rows:", sum(1 for _ in open(sys.argv[2], encoding="utf-8")))
PY

run () {                       # name, model, shards, shard_id
  local name="$1" model="$2" shards="$3" shard="$4"
  mkdir -p "$W/$name"
  python "$R/eval/code/w0.py" \
    --model-path "$model" \
    --codec-path "$R/eval/resources/c1" \
    --moss-tts-root "$R/eval/resources/m0" \
    --fixed-ref "$R/eval/input/ref.wav" \
    --rows-jsonl "$W/rows.jsonl" \
    --out-dir "$W/$name" \
    --summary-jsonl "$W/$name.s$shard.jsonl" \
    --num-shards "$shards" --shard-id "$shard" \
    --device cuda \
    --min-runaway-floor-s 15 --max-seconds-per-char 0.6 \
    --sliding-window 11 --soft-reset-keep 3 --continuous-codec-context \
    --seed 42 --log-every 1 > "$W/$name.s$shard.log" 2>&1
  echo "EXIT_$name.s$shard=$?" >> "$W/$name.s$shard.log"
}

for shard in 0 1 2; do
  run delta /data/delta_tts/ckpt 3 "$shard" &
done
# our own v8 on talk 268 only (shard 2 of 3 selects it), to complete the same-image baseline
run oursv8 "$R/outputs/model/checkpoint-epoch-0" 3 2 &
wait
echo SYNTH_DONE > "$W/status"
