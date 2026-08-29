#!/usr/bin/env bash
# note (luojiaxuan): 单次受控 SimulEval 推理。参数逐字沿用 baseline A/B，
# 只有 --lora-path 变化，保证与 4.-23 的对照可比。
# note (luojiaxuan): --model-type w2v2_qwen25 必须显式给——agents/infinisst.py
# 默认是 w2v2_llama31，用它加载 Qwen 权重会静默输出乱码（台账 4.-23）。
set -eu
tag="$1"; lora="$2"; src="${3:-dev.source}"
source /home/jiaxuanluo/miniconda3/bin/activate infinisst
cd /home/jiaxuanluo/InfiniSST
export PYTHONPATH=$PWD:/mnt/gemini/data2/jiaxuanluo/hydra_fs:/mnt/gemini/data2/jiaxuanluo/fa_abitrue:/mnt/gemini/data2/jiaxuanluo/tf47:/mnt/aries/data6/jiaxuanluo/fairseq-0.12.2
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TOKENIZERS_PARALLELISM=false

ROOT=/mnt/gemini/data1/jiaxuanluo/acl6060_eval
OUT=/mnt/gemini/data2/jiaxuanluo/runs/infer_phrase/${tag}
[ -e "$OUT" ] && { echo "REFUSE: $OUT exists — 先删干净再跑，避免新旧产物混淆"; exit 1; }
mkdir -p "$OUT"

# note (luojiaxuan): LoRA 键名对不上时 agent 会静默跑无适配器的模型，先校验再跑。
python "$(dirname "$0")/check_lora_keys.py" "$lora" \
  /mnt/gemini/data2/jiaxuanluo/stage2_M=12_norm0_qwen2.5-7b-instruct_rope.bin || exit 1

simuleval \
    --agent agents/infinisst.py \
    --model-type w2v2_qwen25 \
    --source-segment-size 1920 \
    --latency-multiplier 2 \
    --max-latency-multiplier 12 \
    --source-lang English --target-lang Chinese \
    --min-start-sec 0 \
    --source ${ROOT}/${src} --target ${ROOT}/dev.target \
    --output "$OUT" \
    --w2v2-path /mnt/aries/data6/xixu/demo/wav2_vec_vox_960h_pl.pt \
    --w2v2-type w2v2 --ctc-finetuned True --audio-normalize 0 \
    --length-shrink-cfg "[(1024,2,2)] * 2" \
    --block-size 48 --max-cache-size 576 \
    --max-llm-cache-size 1000 --always-cache-system-prompt \
    --max-new-tokens 20 --beam 4 \
    --no-repeat-ngram-lookback 100 --no-repeat-ngram-size 5 \
    --repetition-penalty 1.2 \
    --model-name /mnt/aries/data6/jiaxuanluo/Qwen2.5-7B-Instruct \
    --state-dict-path /mnt/gemini/data2/jiaxuanluo/stage1_M=12_norm0_qwen2.5-7b-instruct_rope.bin \
    --lora-path "$lora" --lora-rank 32 \
    --quality-metrics BLEU --eval-latency-unit char --sacrebleu-tokenizer zh
echo "INFER_EXIT=$? tag=$tag"
