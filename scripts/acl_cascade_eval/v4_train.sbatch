#!/usr/bin/env bash
#SBATCH --job-name=v4_train
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=700G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/guests/zhen/s2s_omni_v4/logs/train_%j.out
#SBATCH --error=/home/guests/zhen/s2s_omni_v4/logs/train_%j.err

# note (luojiaxuan): 同一节点上同时训练处理组与对照组，两边 global batch 都是
# 15（= v3 原始配置 1 x 3 x 5），其余超参逐项照抄 ckpt_v3/finetune_args.json：
#   GPU 0-2  v4 数据集（自生成历史）  1 x 3 x 5 = 15
#   GPU 3-5  v3 数据集（手写污染，对照）1 x 3 x 5 = 15
# 两边必须用**相同的进程数**：accelerate 每个 optimizer step 会把 lr
# scheduler 推进 num_processes 次，而 sft.py 算 num_training_steps 时没有
# 乘回来，于是调度会提前 num_processes 倍跑完（实测 5 进程时 lr 在
# step 306/1333 就归零，3 进程时约 463/1390）。进程数不同 = 有效学习步数
# 不同，会直接污染 A/B，所以两组都固定 3 进程（与 v3 原始 run 一致）。
# 记录已带 audio_codes/ref_audio_codes（align_slice 阶段就编码好了），所以
# 和 v3 原始 run 一样直接喂 sft.py，不需要 prepare_data 这一步。
# 之所以连 v3 也重训一遍：原 v3 是在 hyper00 的 H200 上跑的，重训对照组可以
# 排除硬件/软件栈/数据顺序带来的差异，让 v3 vs v4 只剩「污染来源」一个变量。
set -u
ROOT=/home/guests/zhen/s2s_omni_v4
PY=/home/guests/zhen/miniconda3/envs/s2s_v4/bin/python
ACC=/home/guests/zhen/miniconda3/envs/s2s_v4/bin/accelerate
MOSS=$ROOT/code/MOSS-TTS/moss_tts_realtime   # 包与脚本都在这一层
export HF_HOME=/home/guests/zhen/.cache/huggingface
export TORCHDYNAMO_DISABLE=1
export TOKENIZERS_PARALLELISM=false

echo "[$(date -Is)] job=$SLURM_JOB_ID node=$(hostname) restart=${SLURM_RESTART_COUNT:-0}"

train_one() {
  local tag=$1 devices=$2 nproc=$3 accum=$4 prepared=$5
  local out=$ROOT/runs/ckpt_$tag
  mkdir -p "$out"
  cd "$MOSS"
  CUDA_VISIBLE_DEVICES="$devices" PYTHONPATH="$MOSS" \
  "$ACC" launch --num_processes "$nproc" --mixed_precision bf16 \
    --main_process_port $((29500 + RANDOM % 500)) \
    finetuning/sft.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Realtime \
    --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer \
    --train-jsonl "$prepared" \
    --output-dir "$out" \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps "$accum" \
    --learning-rate 1e-5 \
    --num-epochs 1 \
    --num-workers 2 \
    --mixed-precision bf16 \
    --attn-implementation sdpa \
    > "$ROOT/logs/train_$tag.log" 2>&1
  echo "TRAIN_EXIT tag=$tag rc=$?"
}

train_one v4 "0,1,2" 3 5 "$ROOT/data/rows/train_v4.jsonl" &
P4=$!
train_one v3ctl "3,4,5" 3 5 "$ROOT/data/multiturn/prepared_v3/train_v3.jsonl" &
P3=$!
wait $P4 $P3

echo "[$(date -Is)] TRAIN_BOTH_EXITED"
for tag in v4 v3ctl; do
  if ls "$ROOT/runs/ckpt_$tag"/*/model.safetensors >/dev/null 2>&1 || ls "$ROOT/runs/ckpt_$tag"/model.safetensors >/dev/null 2>&1; then
    echo "TRAIN_OK_$tag"
  else
    echo "TRAIN_MISSING_$tag"
  fi
done
echo "V4_TRAIN_DONE"
