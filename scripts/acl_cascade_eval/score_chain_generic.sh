#!/usr/bin/env bash
# note (luojiaxuan): 完整打分链，参数化 (tag, mode)。与 base baseline 和 v3 的
# qwen3asr run 同一口径，四个 run 之间以及与 v3 历史数字都可直接对照。
# usage: score_chain_generic.sh <tag> <mode>
set -u
tag="$1"; mode="$2"
BENCH=/data/S2S_omni_runs/moss_tts_infinisst_v2_20260804/acl_bench
RUN=acl6060_live_enzh_cascade_moss${tag}_${mode}_chunk192_speed1
RD=$BENCH/rundirs/$RUN
SEG_PY=/data/venvs/segale_eval2/bin/python

echo "[1/5] ASR"
python3 "$BENCH/score_generic.py" "$tag" "$mode" || exit 2
echo "[2/5] SEGALE inputs"
"$SEG_PY" /data/S2S_omni/scripts/build_acl6060_segale_inputs.py --run-dir "$RD" || exit 3
echo "[3/5] SEGALE alignment"
"$SEG_PY" /data/S2S_omni/scripts/run_acl6060_segale_alignment.py \
  --run-dir "$RD" --speech-latency-repo /data/speech-to-speech-latency \
  --target-lang zh --device cuda || exit 4
echo "[4/5] BLEU"
"$SEG_PY" /data/S2S_omni/scripts/build_acl6060_xcomet_input.py \
  --run-dir "$RD" --output-jsonl "$RD/xcomet_input.jsonl" \
  --summary-json "$RD/bleu_summary.json" --bleu-tokenizer zh || exit 5
echo "[5/5] XCOMET-XL"
"$SEG_PY" /data/S2S_omni/scripts/run_acl6060_xcomet_xl.py \
  --input-jsonl "$RD/xcomet_input.jsonl" --output-jsonl "$RD/xcomet_segments.jsonl" \
  --summary-json "$RD/xcomet_summary.json" --reference-free --batch-size 4 || exit 6

python3 -c "
import json
b=json.load(open('$RD/bleu_summary.json')); x=json.load(open('$RD/xcomet_summary.json'))
print('RESULT $tag $mode  BLEU %.2f  XCOMET %.4f  null %d/%d (%.1f%%)' % (
  b['bleu'], x['xcomet_xl'], b['null_alignments'], b['segments'], 100*b['null_alignment_ratio']))
"
echo "CHAIN_DONE $tag $mode"
