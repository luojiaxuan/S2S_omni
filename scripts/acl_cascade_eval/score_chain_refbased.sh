#!/usr/bin/env bash
# note (luojiaxuan): 新 canonical 打分链（用户裁定 2026-08-11，台账 4.-16）：
#   - ASR：自托管 Qwen3-ASR-1.7B（127.0.0.1:47500，plain sglang）；
#   - BLEU：SEGALE 句级，null（under/over-translation）保留为空假设，不剔除；
#   - XCOMET-XL：reference-based（对齐 Open-LiveTranslate 的模式），但 null
#     主动置零（fixed_xcomet_xl_score=0.0，input 构建器既有行为）。
# 与旧 _gptasr 目录并存：qwen 路径的 rundir 无后缀，不覆盖旧 canonical。
# usage: score_chain_refbased.sh <gpu> <tag> <modearg>
set -u
g="$1"; tag="$2"; mode="$3"
BENCH=/data/S2S_omni_runs/moss_tts_infinisst_v2_20260804/acl_bench
RUN=acl6060_live_enzh_cascade_moss${tag}_${mode}_chunk192_speed1
RD=$BENCH/rundirs/$RUN
# note (luojiaxuan): hyper01 上的 SEGALE venv 叫 acl6060-segale，路径不同，
# 用 SEG_PY 环境变量覆盖；缺省仍指 hyper00 的 segale_eval2。
SEG_PY="${SEG_PY:-/data/venvs/segale_eval2/bin/python}"
cd /data/S2S_omni

echo "[1/5] Qwen3-ASR"
python3 "$BENCH/score_generic.py" "$tag" "$mode" qwen3 || exit 2
echo "[2/5] SEGALE inputs"
"$SEG_PY" scripts/build_acl6060_segale_inputs.py --run-dir "$RD" || exit 3
echo "[3/5] SEGALE alignment (GPU $g)"
CUDA_VISIBLE_DEVICES="$g" "$SEG_PY" scripts/run_acl6060_segale_alignment.py \
  --run-dir "$RD" --speech-latency-repo /data/speech-to-speech-latency \
  --target-lang zh --device cuda || exit 4
echo "[4/5] BLEU（句级，null 保留）"
"$SEG_PY" scripts/build_acl6060_xcomet_input.py \
  --run-dir "$RD" --output-jsonl "$RD/xcomet_input.jsonl" \
  --summary-json "$RD/bleu_summary.json" --bleu-tokenizer zh || exit 5
echo "[5/5] XCOMET-XL reference-based（null 置零）"
CUDA_VISIBLE_DEVICES="$g" "$SEG_PY" scripts/run_acl6060_xcomet_xl.py \
  --input-jsonl "$RD/xcomet_input.jsonl" --output-jsonl "$RD/xcomet_segments.jsonl" \
  --summary-json "$RD/xcomet_summary.json" --batch-size 4 || exit 6

# note (luojiaxuan): 漏译率报固定参考句口径（台账 4.-21）；块级口径保留在
# summary 里但不再作为 headline。字段缺失就 KeyError 失败，不兜底。
python3 -c "
import json
b=json.load(open('$RD/bleu_summary.json')); x=json.load(open('$RD/xcomet_summary.json'))
print('NEWOP_RESULT $tag $mode  BLEU %.2f  XCOMET-ref %.4f  under %d/%d (%.1f%%)  over_blocks %d' % (
  b['bleu'], x['xcomet_xl'], b['under_translation_source_segments'], b['source_segments'],
  100*b['under_translation_source_ratio'], b['over_translation_alignments']))
"
echo "NEWOP_CHAIN_DONE $tag $mode"
