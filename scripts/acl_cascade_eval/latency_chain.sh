#!/usr/bin/env bash
# note (luojiaxuan): 单个配置的延迟全链（容器内跑）：timing 仿真+ASR ->
# SEGALE inputs -> alignment(GPU) -> LongYAAL。usage: latency_chain.sh <gpu> <modearg>
set -u
g="$1"; modearg="$2"
B=/data/S2S_omni_runs/moss_tts_infinisst_v2_20260804/acl_bench
RD=$B/rundirs/acl6060_live_enzh_cascade_mossv6_${modearg}_chunk192_speed1_gptasr_latency
SEG_PY=/data/venvs/segale_eval2/bin/python
cd /data/S2S_omni
export PYTHONPATH=/data/S2S_omni:/data/S2S_omni/scripts

echo "[1/4] timing 仿真 + 带时间戳 ASR ($modearg)"
python3 scripts/build_cascade_speech_timing.py --bench "$B" --tag v6 \
  --modearg "$modearg" --api-key-file /data/openai_key.txt || exit 2
echo "[2/4] SEGALE inputs"
"$SEG_PY" scripts/build_acl6060_segale_inputs.py --run-dir "$RD" || exit 3
echo "[3/4] SEGALE alignment (GPU $g)"
CUDA_VISIBLE_DEVICES="$g" "$SEG_PY" scripts/run_acl6060_segale_alignment.py \
  --run-dir "$RD" --speech-latency-repo /data/speech-to-speech-latency \
  --target-lang zh --device cuda || exit 4
echo "[3.5/4] quality_summary（longyaal 依赖，写在 segale_alignment 缺省位置）"
"$SEG_PY" scripts/build_acl6060_xcomet_input.py --run-dir "$RD" \
  --output-jsonl "$RD/xcomet_input.jsonl" --bleu-tokenizer zh || exit 6
echo "[4/4] LongYAAL"
"$SEG_PY" scripts/run_acl6060_segale_longyaal.py \
  --run-dir "$RD" --speech-latency-repo /data/speech-to-speech-latency || exit 5
python3 -c "
import json
s = json.load(open('$RD/segale_longyaal/summary.json'))
print('LATENCY_RESULT $modearg  LongYAAL_cu %.0f  LongYAAL_ca %.0f  end_cu %.0f  end_ca %.0f  bleu %.2f' % (
  s['longyaal_cu'], s['longyaal_ca'], s['ending_offset_cu_ms_mean'], s['ending_offset_ca_ms_mean'], s['bleu']))
"
echo "LATENCY_CHAIN_DONE $modearg"
