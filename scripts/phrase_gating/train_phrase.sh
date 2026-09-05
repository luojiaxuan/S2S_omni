#!/usr/bin/env bash
# Phrase-gated thinker SFT. Every hyperparameter mirrors the origin checkpoint's args.json
# (gigaspeech-zh-s_origin-bsz4); the only deliberate change is the dataset, whose releases
# were moved to phrase boundaries by phrase_gate_traj.py. 4 GPUs x micro 1 x accum 1 is the
# original global batch of 4.
set -x
W=/mnt/gemini/home/jiaxuanluo/phrase_sft_20260904
source "$W/env.sh"
BASE=/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct
DATA=/mnt/gemini/data/jiaxuanluo/phrase_gating_20260904/train_s_zh_phrase_ours.jsonl
TAG="${1:-smoke}"
EXTRA=()
[ "$TAG" = "smoke" ] && EXTRA=(--max_steps 15 --save_steps 100000)

# note (luojiaxuan): PCIe P2P between these A6000s deadlocks NCCL — a bare four-rank
# all-reduce hangs indefinitely and the trainer sits at 100%% GPU with no memory growth.
# Disabling P2P makes it complete; measured on aries 2026-09-04.
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=2,3,4,5
export NPROC_PER_NODE=4
export MASTER_PORT=29517

"$W/env/bin/swift" sft \
  --model "$BASE" --model_type qwen3_omni \
  --dataset "$DATA" \
  --train_type lora --lora_rank 32 --lora_alpha 32 --lora_dropout 0.05 \
  --target_modules all-linear \
  --freeze_vit true --freeze_aligner true \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 --lr_scheduler_type cosine --warmup_ratio 0.05 \
  --weight_decay 0.01 --max_grad_norm 1.0 \
  --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-8 \
  --max_length 2048 \
  --gradient_checkpointing true \
  --deepspeed zero3 \
  --seed 42 \
  --logging_steps 5 --save_steps 200 --save_total_limit 3 \
  --dataloader_num_workers 4 \
  --output_dir "$W/out_$TAG" \
  "${EXTRA[@]}"
echo "TRAIN_${TAG}_EXIT=$?"
