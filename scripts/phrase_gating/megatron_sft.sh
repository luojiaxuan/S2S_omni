#!/usr/bin/env bash
# The three stages of OLT's finetune/recipes/omni_sft_recipe.sh, run under Docker on
# hyper01 with the recipe's own image and arguments. The recipe itself refuses to run
# outside Apptainer (it verifies the .sif's labels), so the commands are reproduced here
# verbatim; every training flag below is the recipe's.
#
#   megatron_sft.sh convert   HF base -> Megatron (mcore) checkpoint, once
#   megatron_sft.sh train     megatron sft, LoRA r32 on all-linear, EP=4, global batch 4, 1 epoch
#   megatron_sft.sh export    merged bf16 HF checkpoint the OLT cascade serves
#
# Flags stay on the continuation lines with no comments among them.
set -euo pipefail
STAGE="${1:?convert|train|export}"
IMG=modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.1
H=/data04/jaxan
BASE=/data/phrase_sft/Qwen3-Omni-30B-A3B-Instruct
MCORE=/data/phrase_sft/Qwen3-Omni-30B-A3B-Instruct-mcore
DATA=/data/phrase_sft/train_s_zh_phrase_local.jsonl
CKPT=/data/phrase_sft/megatron_run
MEGATRON=/data/serving_ab/olt/third_party/Megatron-LM
GPUS="${GPUS:-4}"
DEVICES="${DEVICES:-0,1,2,3}"
PORT=$(( 20000 + ($$ % 20000) ))
NAME="sglang-omni-jaxan-mg-$STAGE"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run --rm --init --name "$NAME" --gpus "\"device=$DEVICES\"" --ipc=host --shm-size=64g \
  -v "$H":/data \
  -e MEGATRON_LM_PATH="$MEGATRON" -e MODELSCOPE_CACHE=/data/phrase_sft/cache/modelscope \
  -e PYTHONPATH= -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1 -e NCCL_DEBUG=WARN \
  -e PYTHONUNBUFFERED=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e ENABLE_AUDIO_OUTPUT=False -e MASTER_PORT="$PORT" -e NPROC_PER_NODE="$GPUS" \
  -e HF_HOME=/data/phrase_sft/cache/huggingface \
  "$IMG" bash -c "$(cat <<EOF
set -x
mkdir -p /data/phrase_sft/cache/modelscope
case "$STAGE" in
  convert)
    swift export --model "$BASE" --to_mcore true --torch_dtype bfloat16 --output_dir "$MCORE"
    ;;
  train)
    mkdir -p "$CKPT/mcore"
    megatron sft \
      --load "$MCORE" \
      --dataset "$DATA" \
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
esac
echo "STAGE_${STAGE}_EXIT=\$?"
EOF
)"
