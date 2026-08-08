#!/usr/bin/env bash
#SBATCH --job-name=v4_selfhist
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=700G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/guests/zhen/s2s_omni_v4/logs/selfhist_%j.out
#SBATCH --error=/home/guests/zhen/s2s_omni_v4/logs/selfhist_%j.err

# note (luojiaxuan): Tilde 是 guest QoS + PreemptMode=REQUEUE，作业随时可能被
# 抢占重排队，所以每个 shard 都必须能断点续跑：输出按 shard 独立追加，脚本启动
# 时读回已完成的 row id 跳过（gen_moss_self_history.py 内置）。所有状态都写在
# /home/guests/zhen 这块持久盘上，不依赖计算节点本地盘或 /tmp。
set -u
ROOT=/home/guests/zhen/s2s_omni_v4
PY=$ROOT/../miniconda3/envs/s2s_v4/bin/python
OUT=${OUT:-$ROOT/runs/selfhist}
RECORDS=${RECORDS:-$ROOT/data/rows/gen_targets.jsonl}
BATCH=${BATCH:-64}
WINDOW=${WINDOW:-8}
mkdir -p "$OUT" "$ROOT/logs"

echo "[$(date -Is)] job=$SLURM_JOB_ID node=$(hostname) restart_count=${SLURM_RESTART_COUNT:-0}"
echo "[records] $(wc -l < "$RECORDS") rows | batch=$BATCH window=$WINDOW"

for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i TORCHDYNAMO_DISABLE=1 \
  "$PY" "$ROOT/code/S2S_omni/scripts/gen_moss_self_history.py" \
    --model-path "$ROOT/data/ckpt_v3" \
    --moss-tts-root "$ROOT/code/MOSS-TTS/moss_tts_realtime" \
    --fixed-ref "$ROOT/data/multiturn/fixed_ref/fixed_zh_ref.wav" \
    --records-jsonl "$RECORDS" \
    --out-jsonl "$OUT/self_hist.shard$i.jsonl" \
    --heartbeat "$OUT/hb.shard$i.json" \
    --num-shards 8 --shard-id $i \
    --batch-size "$BATCH" --sliding-window "$WINDOW" \
    --log-every 20 \
    > "$ROOT/logs/selfhist_shard$i.log" 2>&1 &
done
wait

total=$(cat "$OUT"/self_hist.shard*.jsonl 2>/dev/null | wc -l)
echo "[$(date -Is)] ALL_SHARDS_EXITED rows_total=$total"
if [ "$total" -ge "$(wc -l < "$RECORDS")" ]; then
  echo "SELFHIST_COMPLETE"
else
  echo "SELFHIST_PARTIAL rows=$total — resubmit to resume (skip-existing by row id)"
fi
