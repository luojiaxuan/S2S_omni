#!/usr/bin/env bash
# note (luojiaxuan): the OLT Megatron SFT stages on hyper01 (4xH200), with the turn-end loss plugin that
# note (luojiaxuan): finetune/scripts/swift_plugins/ added on 2026-09-09. The 2026-09-05 phrase run predates it and
# note (luojiaxuan): trained under ms-swift's default loss, where an intermediate EMPTY assistant turn carries no
# note (luojiaxuan): supervised token at all. That is the decision phrase gating exists to teach, and the phrase
# note (luojiaxuan): corpus is 29.3% empty turns against the word-aligned corpus's 10.9%, so the phrase arm lost
# note (luojiaxuan): 2.7x more supervision than the arm it was being compared against.
#
# note (luojiaxuan):   megatron_hyper01.sh convert   HF base -> Megatron (mcore), once per base
# note (luojiaxuan):   megatron_hyper01.sh train     megatron sft with --loss_scale $LOSS
# note (luojiaxuan):   megatron_hyper01.sh export    merged bf16 HF checkpoint the cascade serves
#
# note (luojiaxuan): ARM=phrase|word selects the manifest; LOSS defaults to empty_turn_end_w0.5, the best 1.92 s
# note (luojiaxuan): setting in the recipe owner's own grid (44.03 BLEU against 43.59 at weight 1 and 36.87 under
# note (luojiaxuan): the default loss). Flags stay on the continuation lines with no comments among them: a '#'
# note (luojiaxuan): after a backslash swallows the rest of the command.
set -euo pipefail
STAGE="${1:?convert|train|export|probe}"
IMG=modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.1
ARM="${ARM:-phrase}"
LOSS="${LOSS:-empty_turn_end_w0.5}"
W=/data/phrase_sft2
HOSTW=/data04/jaxan/phrase_sft2
BASE=/root/.cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695
MCORE=$W/mcore_base
case "$ARM" in
  phrase) DATA=$W/train_s_zh_phrase_ours_local.jsonl ;;
  word)   DATA=$W/train_s_zh_origin_local.jsonl ;;
  *) echo "ARM must be phrase or word, got '$ARM'" >&2; exit 2 ;;
esac
CKPT=$W/run_${ARM}_${LOSS}
PLUGIN=$W/plugins/weighted_turn_end.py
[ "$LOSS" = assistant_turn_end ] && PLUGIN=$W/plugins/assistant_turn_end.py
GPUS="${GPUS:-4}"
DEVICES="${DEVICES:-4,5,6,7}"
PORT=$(( 20000 + ($$ % 20000) ))
NAME="${NAME:-sglang-omni-jaxan-3}"

if [ -n "$(docker ps -q --filter "name=^$NAME\$")" ]; then
  echo "container $NAME is running (see docker ps and \$HOME/jiaxuanluo-map.txt); refusing to replace it" >&2
  exit 5
fi
docker rm "$NAME" >/dev/null 2>&1 || true
docker run --rm --init --name "$NAME" --gpus "\"device=$DEVICES\"" --shm-size=64g \
  -v /data04/jaxan:/data -v /data04/cache/huggingface:/root/.cache/huggingface \
  -e MEGATRON_LM_PATH=/data/serving_ab/olt/third_party/Megatron-LM -e MODELSCOPE_CACHE=$W/cache/modelscope \
  -e PYTHONPATH= -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1 -e NCCL_DEBUG=WARN \
  -e PYTHONUNBUFFERED=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e ENABLE_AUDIO_OUTPUT=False -e MASTER_PORT="$PORT" -e NPROC_PER_NODE="$GPUS" \
  -e HF_HOME=/root/.cache/huggingface -e TMPDIR=$W/tmp \
  "$IMG" bash -c "$(cat <<EOF
set -x
mkdir -p "$W/cache/modelscope" "$W/tmp" "$CKPT"
case "$STAGE" in
  convert)
    swift export --model "$BASE" --to_mcore true --torch_dtype bfloat16 --output_dir "$MCORE"
    ;;
  train)
    megatron sft \
      --load "$MCORE" \
      --dataset "$DATA" \
      --external_plugins "$PLUGIN" --loss_scale "$LOSS" \
      --split_dataset_ratio 0.01 --data_seed 42 --seed 42 \
      --load_from_cache_file true \
      --train_type lora --lora_rank 32 --lora_alpha 32 --target_modules all-linear \
      --freeze_llm false --freeze_vit true --freeze_aligner true \
      --vit_gradient_checkpointing false \
      --packing true --max_length 2048 \
      --expert_model_parallel_size "$GPUS" \
      --moe_permute_fusion true --moe_grouped_gemm true --moe_shared_expert_overlap true \
      --moe_aux_loss_coeff 1e-3 \
      --micro_batch_size 1 --global_batch_size 4 \
      --recompute_granularity full --recompute_method uniform --recompute_num_layers 1 \
      --finetune true --cross_entropy_loss_fusion true \
      --lr 1e-4 --lr_warmup_fraction 0.05 --min_lr 1e-5 --weight_decay 0.01 --clip_grad 1.0 \
      --max_epochs 1 \
      --save "$CKPT/mcore" --add_version false \
      --log_interval 10 --eval_interval 200 --save_interval 200 \
      --num_workers 8 --dataset_num_proc 8 \
      --no_save_optim true --no_save_rng true \
      --attention_backend flash
    ;;
  export)
    swift export --mcore_adapters "$CKPT/mcore" --to_hf true --torch_dtype bfloat16 --output_dir "$CKPT/hf"
    ls "$CKPT/hf" | head
    ;;
  probe)
    python -c "import torch, swift; import swift.megatron; print('swift', swift.__version__, '| torch', torch.__version__, '| gpus', torch.cuda.device_count())"
    python -c "import sys; sys.path.insert(0, '$W/plugins'); import weighted_turn_end as w; print('loss names:', sorted(n for n in dir(w) if 'w0' in n) or 'registered at import')"
    ls "$MEGATRON_LM_PATH/megatron/core/__init__.py" "$BASE/config.json" "$DATA" "$PLUGIN"
    ;;
esac
echo "STAGE_${STAGE}_EXIT=\$?"
EOF
)"
