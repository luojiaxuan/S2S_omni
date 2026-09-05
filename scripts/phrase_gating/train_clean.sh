#!/usr/bin/env bash
# Phrase-gated thinker SFT. Every hyperparameter mirrors the origin checkpoint's args.json
# (gigaspeech-zh-s_origin-bsz4); the only deliberate change is the dataset, whose releases
# were moved to phrase boundaries by phrase_gate_traj.py.
#
# 4 GPUs x micro 1 x accum 1 reproduces the original global batch of 4.
#
# No deepspeed: an H200 holds all 38B parameters (~118 GB observed of 143 GB), so sharding
# them only adds communication. Measured on this host: ZeRO-3 41.8 s/step, plain DDP
# 23.9 s/step, with GPU utilisation and dataloader workers unchanged between the two.
#
# Every flag stays on the continuation lines below with NO comments among them: a '#' after
# a backslash continuation ends the command, and everything after it is silently dropped —
# that is how an earlier run lost --deepspeed, --seed, --save_steps and --output_dir while
# still appearing to start correctly.
set -x
W=/data/phrase_sft
export TMPDIR="$W/tmp" XDG_CACHE_HOME="$W/cache" HF_HOME=/root/.cache/huggingface \
       PYTHONPATH="$W/pyshim"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
BASE=$W/Qwen3-Omni-30B-A3B-Instruct
DATA=$W/train_s_zh_phrase_local.jsonl
TAG="${1:-smoke}"
EXTRA=()
[ "$TAG" = "argcheck" ] && EXTRA=(--max_steps 2)

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export MASTER_PORT=$(( 29000 + ($$ % 900) ))

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
  --seed 42 \
  --logging_steps 5 --save_steps 200 --save_total_limit 3 \
  --dataloader_num_workers 16 \
  --output_dir "$W/out_$TAG" \
  "${EXTRA[@]}"
echo "TRAIN_${TAG}_EXIT=$?"
