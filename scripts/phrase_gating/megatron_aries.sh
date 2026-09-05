#!/usr/bin/env bash
# The three stages of OLT's finetune/recipes/omni_sft_recipe.sh, run under Docker on aries
# with the recipe's own image and arguments. The recipe refuses to run outside Apptainer
# (it verifies the .sif's labels), so its commands are reproduced verbatim here.
#
#   megatron_aries.sh convert   HF base -> Megatron (mcore) checkpoint, once
#   megatron_aries.sh train     megatron sft, LoRA r32 all-linear, EP=4, global batch 4, 1 epoch
#   megatron_aries.sh export    merged bf16 HF checkpoint the OLT cascade serves
#
# Everything lives on the gemini NFS (aries' local disks are full). Flags stay on the
# continuation lines with no comments among them.
set -euo pipefail
STAGE="${1:?convert|train|export}"
IMG=modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.1
W=/mnt/gemini/home/jiaxuanluo/phrase_sft_20260904
BASE=/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct
MCORE=$W/Qwen3-Omni-30B-A3B-Instruct-mcore
DATA=/mnt/gemini/data/jiaxuanluo/phrase_gating_20260904/train_s_zh_phrase_ours.jsonl
CKPT=$W/megatron_run
MEGATRON=$W/third_party/Megatron-LM
GPUS="${GPUS:-4}"
DEVICES="${DEVICES:-3,4,5,6}"
PORT=$(( 20000 + ($$ % 20000) ))
NAME="sglang-omni-jaxan-1"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run --rm --init --name "$NAME" --gpus "\"device=$DEVICES\"" --ipc=host --shm-size=64g \
  -v /mnt/gemini/home:/mnt/gemini/home -v /mnt/gemini/data:/mnt/gemini/data -v /mnt/gemini/data2:/mnt/gemini/data2 \
  -e MEGATRON_LM_PATH="$MEGATRON" -e MODELSCOPE_CACHE="$W/cache/modelscope" \
  -e PYTHONPATH= -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1 -e NCCL_DEBUG=WARN \
  -e PYTHONUNBUFFERED=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e ENABLE_AUDIO_OUTPUT=False -e MASTER_PORT="$PORT" -e NPROC_PER_NODE="$GPUS" \
  -e HF_HOME="$W/cache/huggingface" -e TMPDIR="$W/tmp" \
  "$IMG" bash -c "$(cat <<EOF
set -x
mkdir -p "$W/cache/modelscope" "$W/tmp"
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
  probe)
    python -c "import torch, swift; import swift.megatron; print('swift', swift.__version__, '| torch', torch.__version__, '| gpus', torch.cuda.device_count())"
    ls "$MEGATRON/megatron/core/__init__.py" "$BASE/config.json" "$DATA"
    ;;
esac
echo "STAGE_${STAGE}_EXIT=\$?"
EOF
)"
