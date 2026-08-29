#!/usr/bin/env bash
# note (luojiaxuan): v2 训练收尾——转 checkpoint、对齐键名、双档推理、出结构判据。
# note (luojiaxuan): 等 checkpoint 稳定再转：NFS 上 16GB 的 model_states 要写约
# 7 分钟，中途读会拿到撕裂档（zip central directory 报错）。
set -eu
source /home/jiaxuanluo/miniconda3/bin/activate infinisst
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
PP=/mnt/gemini/data2/jiaxuanluo/phrase_pipeline
G=/mnt/gemini/data2/jiaxuanluo
CK=$G/runs/infinisst_phrase_v2/last.ckpt
REF=$G/stage2_M=12_norm0_qwen2.5-7b-instruct_rope.bin

prev=""; stable=0
while [ $stable -lt 3 ]; do
  cur=$(stat -c '%s-%Y' "$CK/checkpoint/mp_rank_00_model_states.pt")
  [ "$cur" = "$prev" ] && stable=$((stable + 1)) || stable=0
  prev="$cur"; sleep 40
done
echo "ckpt stable: $prev"

cd "$CK"
python zero_to_fp32.py . "$G/v2final_raw"
python "$PP/strip_lightning_prefix.py" "$G/v2final_raw/pytorch_model.bin" "$G/stage2_phrase_v2_fixed.bin" "$REF"
rm -rf "$G/v2final_raw"

python -c "
import torch
ref = torch.load('$REF', map_location='cpu', weights_only=True)
sd = torch.load('$G/stage2_phrase_v2_fixed.bin', map_location='cpu', weights_only=True)
rel = sum((ref[k].float() - sd[k].float()).norm().item() / ref[k].float().norm().item() for k in ref)
print(f'权重相对现役 stage2 平均变化 {rel / len(ref) * 100:.3f}%')
"

# note (luojiaxuan): 1× 与 1.5× 是两次独立推理（1.5× 跑变速音频），并行到两张卡。
CUDA_VISIBLE_DEVICES=0 bash "$PP/run_infer.sh" phrv2 "$G/stage2_phrase_v2_fixed.bin" \
  > "$G/runs/infer_phrase/phrv2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 bash "$PP/run_infer.sh" phrv2s150 "$G/stage2_phrase_v2_fixed.bin" dev.source.speed150 \
  > "$G/runs/infer_phrase/phrv2s150.log" 2>&1 &
wait

T=/mnt/gemini/data1/jiaxuanluo/acl6060_eval
python "$PP/instances_to_turns.py" "$G/runs/infer_phrase/phrv2/instances.log" "$T/turns_phrv2" phrv2
python "$PP/instances_to_turns.py" "$G/runs/infer_phrase/phrv2s150/instances.log" "$T/turns_phrv2s150" phrv2s150
# note (luojiaxuan): 只比对本脚本自己产出的两档，其余档位各自产出后单独并表，
# 避免某个目录尚未生成时在长脚本末尾崩掉。
python "$PP/cmp_turns.py" "baseline 现役=$T/turns_baseline" \
  "v2 final 1x=$T/turns_phrv2" "v2 final 1.5x=$T/turns_phrv2s150"
cat "$G/runs/infer_phrase/phrv2/scores.tsv"
echo FINALIZE_DONE
