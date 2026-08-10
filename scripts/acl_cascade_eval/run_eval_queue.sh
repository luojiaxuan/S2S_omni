#!/usr/bin/env bash
# note (luojiaxuan): 通用评测队列——按 (checkpoint, 模式) 参数化，配置逐字
# 沿用 v3 的两个 run，保证 v4 / v3ctl / 已有 v3 三者可比：
#   sliding: --sliding-window 11 --min-runaway-floor-s 15，输入 swrow（整场一行）
#   reset  : 无 sliding 标志，输入 rows（每 11 turn 一个会话）
# usage: run_eval_queue.sh <gpu> <tag> <ckpt_path> <mode> <talk...>
set -u
RUN=/data/S2S_omni_runs/moss_tts_infinisst_v2_20260804
g="$1"; tag="$2"; ckpt="$3"; mode="$4"; shift 4
# note (luojiaxuan): PREFIX 选输入档位——chunk192（1x）或 speed125/speed150。
# 输出目录带上 PREFIX，避免不同速度档互相覆盖。
PREFIX="${PREFIX:-chunk192}"
cd /data/S2S_omni

# note (luojiaxuan): mode 支持 sliding / slidingN / reset。slidingN 用来扫
# 窗口大小——回答"reset 不硬切、保留一半历史"是否可行：sliding6 与 reset11
# 的平均历史长度都约 5 轮，唯一差别是 sliding 永不清零、reset 周期性清零。
case "$mode" in
  sliding)  rows_suffix="swrow"; extra="--sliding-window 11" ;;
  # note (luojiaxuan): slidingpin = 滑窗 w=11，但首位固定为整场第一个 turn。
  # 窗口长度不变，唯一变量是首位那个 turn 自带"从静音起音"。用来在不重训的
  # 前提下检验静音锚点假设（台账 4.-9 / 4.-10）。必须排在 sliding* 之前，
  # 否则会被当成窗口大小解析成 --sliding-window pin。
  slidingpin) rows_suffix="swrow"; extra="--sliding-window 11 --pin-first-turn" ;;
  sliding*) rows_suffix="swrow"; extra="--sliding-window ${mode#sliding}" ;;
  # anchorNN = reset + 韵律锚点，NN 为十分之一秒（anchor10 = 1.0s）
  anchor*)  rows_suffix="rows";  extra="--reset-carry-seconds $(awk "BEGIN{print ${mode#anchor}/10}")" ;;
  # note (luojiaxuan): guard = 滑窗 w=11 + 短 turn 吞并护栏（阈值 1.5 帧/字）
  guard)    rows_suffix="swrow"; extra="--sliding-window 11 --min-frames-per-char 1.5" ;;
  *)        rows_suffix="rows";  extra="" ;;
esac

for talk in "$@"; do
  out="$RUN/acl_bench/tts_wavs_${tag}_${mode}_${PREFIX}/talk$talk.$PREFIX"
  # note (luojiaxuan): 已完成的 talk 跳过，作业被打断后可直接重跑
  if [ -s "$out.summary.jsonl" ] && [ -s "$RUN/acl_bench/tts_wavs_${tag}_${mode}_${PREFIX}/talk$talk.$PREFIX.done" ]; then
    echo "skip talk$talk (done)"; continue
  fi
  CUDA_VISIBLE_DEVICES="$g" TORCHDYNAMO_DISABLE=1 python3 scripts/moss_multiturn_infer.py \
    --model-path "$ckpt" \
    --fixed-ref "$RUN/fixed_ref/fixed_zh_ref.wav" \
    --rows-jsonl "$RUN/acl_bench/tts_rows/talk$talk.$PREFIX.$rows_suffix.jsonl" \
    --out-dir "$out" \
    --summary-jsonl "$out.summary.jsonl" \
    --device cuda --min-runaway-floor-s 15 --log-every 50 $extra \
    >> "$RUN/acl_bench/logs/eval_${tag}_${mode}_${PREFIX}_gpu$g.log" 2>&1 \
    && touch "$RUN/acl_bench/tts_wavs_${tag}_${mode}_${PREFIX}/talk$talk.$PREFIX.done"
done
echo "EVAL_QUEUE_DONE tag=$tag mode=$mode gpu=$g talks=$*"
