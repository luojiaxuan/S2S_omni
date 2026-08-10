#!/usr/bin/env bash
# note (luojiaxuan): 逐个 talk 跑，**每个 talk 启动前才挑卡**，OOM 后换卡重试。
#
# 为什么需要这个：hyper00 是共享机，显存占用几分钟就变一次。此前的做法是
# 先扫一遍空闲卡再把 5 个 talk 分派下去，结果在"扫描"与"启动"之间别的项目
# 占掉了其中一张（实测 GPU2 三分钟内从 527 MiB 涨到 133 GB），那两个 talk
# 直接 OOM 退出，而外层脚本只看到队列正常结束。
#
# usage: run_talks_resilient.sh <tag> <ckpt> <mode> <talk...>
set -u
RUN=/data/S2S_omni_runs/moss_tts_infinisst_v2_20260804
B=$RUN/acl_bench
PREFIX="${PREFIX:-chunk192}"
tag="$1"; ckpt="$2"; mode="$3"; shift 3
NEED_MIB="${NEED_MIB:-45000}"
MAX_TRY="${MAX_TRY:-6}"

pick_gpu() {
  # note (luojiaxuan): 按剩余显存降序挑，排除本轮已经失败过的卡。
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
    | awk -F', ' -v need="$NEED_MIB" -v bad="$1" '
        { free = $2 - $3; if (free < need) next;
          split(bad, b, ","); for (i in b) if (b[i] == $1) next;
          print free, $1 }' \
    | sort -rn | head -1 | awk '{print $2}'
}

for talk in "$@"; do
  done_marker="$B/tts_wavs_${tag}_${mode}_${PREFIX}/talk$talk.$PREFIX.done"
  [ -s "$done_marker" ] && { echo "skip talk$talk (done)"; continue; }
  bad=""
  for try in $(seq 1 "$MAX_TRY"); do
    g=$(pick_gpu "$bad")
    if [ -z "$g" ]; then
      echo "talk$talk: 暂无满足 ${NEED_MIB}MiB 的卡，等 300s 后重试 (try $try)"
      sleep 300; bad=""; continue
    fi
    echo "[$(date -Is)] talk$talk -> GPU $g (try $try)"
    bash "$B/run_eval_queue.sh" "$g" "$tag" "$ckpt" "$mode" "$talk"
    [ -s "$done_marker" ] && { echo "talk$talk OK on GPU $g"; break; }
    echo "talk$talk 在 GPU $g 上失败，换卡"
    bad="$bad,$g"
  done
  [ -s "$done_marker" ] || echo "TALK_FAILED $talk"
done
echo "RESILIENT_DONE tag=$tag mode=$mode"
