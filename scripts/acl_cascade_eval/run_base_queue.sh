#!/usr/bin/env bash
# note (luojiaxuan): 原生 MOSS-TTS-Realtime（未做任何 SFT）作为级联 baseline。
# 除 --model-path 外，所有配置与 v3+session-reset 操作点逐字对齐：同一份
# InfiniSST chunk=1.92s 文本、同一固定中文音色、同样 11-turn 会话重置、
# 同样 runaway floor，因此唯一变量是 checkpoint。
set -u
RUN=/data/S2S_omni_runs/moss_tts_infinisst_v2_20260804
g="$1"; shift
cd /data/S2S_omni
for talk in "$@"; do
  CUDA_VISIBLE_DEVICES="$g" TORCHDYNAMO_DISABLE=1 python3 scripts/moss_multiturn_infer.py \
    --model-path "OpenMOSS-Team/MOSS-TTS-Realtime" \
    --fixed-ref "$RUN/fixed_ref/fixed_zh_ref.wav" \
    --rows-jsonl "$RUN/acl_bench/tts_rows/talk$talk.chunk192.rows.jsonl" \
    --out-dir "$RUN/acl_bench/tts_wavs_base/talk$talk.chunk192" \
    --summary-jsonl "$RUN/acl_bench/tts_wavs_base/talk$talk.chunk192.summary.jsonl" \
    --device cuda --min-runaway-floor-s 15 --log-every 20 \
    >> "$RUN/acl_bench/logs/tts_base_gpu$g.log" 2>&1
done
echo "BASE_QUEUE_DONE gpu=$g talks=$*"
