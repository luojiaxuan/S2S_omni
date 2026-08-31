#!/usr/bin/env bash
#SBATCH --job-name=sglang-omni-tts
#SBATCH --account=guests
#SBATCH --qos=guest-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /home/guests/zhen/sglang-omni-tts/logs/train_%j.out
#SBATCH -e /home/guests/zhen/sglang-omni-tts/logs/train_%j.err

# note (luojiaxuan): tilde 是 spot 型容量（guest 账号低优先级，可能被抢占/requeue）。
# 恢复策略：每 200 步存 checkpoint；sft.py 未显式传 resume 时会自动扫
# outputs/model/checkpoint-step-*（带 trainer_state.json 的最新一个）续训。所有状态都在 $HOME（jail 持久盘），计算节点本地盘不落任何东西。
set -u
ROOT=/home/guests/zhen/sglang-omni-tts
cd "$ROOT"
export HF_HOME=$ROOT/cache/hf
export TMPDIR=$ROOT/tmp
mkdir -p logs cache/hf tmp outputs/model
export PYTHONPATH=$ROOT/code/src
export NCCL_NVLS_ENABLE=0

env/venv/bin/python -m torch.distributed.run --nproc_per_node 3 \
  code/src/moss_tts_realtime/finetuning/sft.py \
  --model-path assets/base \
  --codec-path assets/codec \
  --train-jsonl data/train_v9.jsonl \
  --output-dir outputs/model \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 5 \
  --learning-rate 1e-5 \
  --num-epochs 1 \
  --num-workers 4 \
  --mixed-precision bf16 \
  --attn-implementation sdpa \
  --seed 42 \
  --checkpointing-steps 200
echo "TRAIN_EXIT=$?"
