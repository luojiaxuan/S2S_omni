#!/usr/bin/env bash
# note (luojiaxuan): Mac-side orchestrator for protocol A on aries: the three thinkers (theirs / our word-aligned /
# note (luojiaxuan): our phrase-gated) compared two ways on the 5-talk ACL 60-60 dev split, all in one container.
# note (luojiaxuan):   deterministic A/B  thinker temperature 0 + TTS greedy (TTS_SAMPLE=0): pipeline is
# note (luojiaxuan):                      deterministic, so one run isolates the thinker difference from sampling noise.
# note (luojiaxuan):   sampled band       published settings (thinker 0.6, TTS sampled), three runs per thinker for a
# note (luojiaxuan):                      mean and spread at the deployment operating point.
# note (luojiaxuan): 12 runs, each writes its own metrics.json and is skipped on resume. Stages:
# note (luojiaxuan):   wait   the hyper01->aries local transfer finished (image + support + 3 thinkers on NVMe)
# note (luojiaxuan):   up     pick free GPUs, create the container, register the map
# note (luojiaxuan):   smoke  one deterministic MAX_DOCS=1 run on the phrase thinker to prove the ported env works
# note (luojiaxuan):   sweep  the 12 runs
# note (luojiaxuan):   down   remove the container and its map line
# note (luojiaxuan): CA (computation-aware) latency here is A6000-timed and only comparable within aries; CU and the
# note (luojiaxuan): text metrics (BLEU/XCOMET) are hardware-independent. FROM=<stage> resumes; each stage logs to $STATUS.
set -uo pipefail
STATUS="${STATUS:?}"
FROM="${FROM:-wait}"
HERE=$(cd "$(dirname "$0")" && pwd)
A="aries"
O="-o RemoteCommand=none -o ConnectTimeout=30 -o ServerAliveInterval=60 -o ServerAliveCountMax=20"
SSH="ssh $O"
D3=/mnt/data3/jiaxuanluo/serving_ab
D4=/mnt/data4/jiaxuanluo/serving_ab_thinkers
CNAME_FILE=$HERE/.aries_cname
TRANSFER_LOG="${TRANSFER_LOG:-$HERE/to_aries_local.log}"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$STATUS"; }
ok() { say "STAGE $1 OK $2"; }
fail() { say "STAGE $1 FAIL $2"; exit 1; }
active=0; want() { [ "$1" = "$FROM" ] && active=1; [ $active = 1 ]; }

# note (luojiaxuan): container GPU ordinals 0,1 = thinker (TP=2), 2 = TTS -- fixed by the --gpus device order at `up`.
declare -a ARMS=(
  "theirs_det   92001 92501 /data/serving_ab/thinkers/theirs owaski-infinisst-thinker-phrase-zh          0 0"
  "word_det     92011 92511 /data/thinkers_extra/ours_word   gavinlaw-infinisst-no-tmsft-origin-bsz4-zh  0 0"
  "phrase_det   92021 92521 /data/serving_ab/thinkers/phrase gavinlaw-infinisst-thinker-phrase-gated-zh  0 0"
  "theirs_s1    92002 92502 /data/serving_ab/thinkers/theirs owaski-infinisst-thinker-phrase-zh          0.6 1"
  "word_s1      92012 92512 /data/thinkers_extra/ours_word   gavinlaw-infinisst-no-tmsft-origin-bsz4-zh  0.6 1"
  "phrase_s1    92022 92522 /data/serving_ab/thinkers/phrase gavinlaw-infinisst-thinker-phrase-gated-zh  0.6 1"
  "theirs_s2    92003 92503 /data/serving_ab/thinkers/theirs owaski-infinisst-thinker-phrase-zh          0.6 1"
  "word_s2      92013 92513 /data/thinkers_extra/ours_word   gavinlaw-infinisst-no-tmsft-origin-bsz4-zh  0.6 1"
  "phrase_s2    92023 92523 /data/serving_ab/thinkers/phrase gavinlaw-infinisst-thinker-phrase-gated-zh  0.6 1"
  "theirs_s3    92004 92504 /data/serving_ab/thinkers/theirs owaski-infinisst-thinker-phrase-zh          0.6 1"
  "word_s3      92014 92514 /data/thinkers_extra/ours_word   gavinlaw-infinisst-no-tmsft-origin-bsz4-zh  0.6 1"
  "phrase_s3    92024 92524 /data/serving_ab/thinkers/phrase gavinlaw-infinisst-thinker-phrase-gated-zh  0.6 1"
)
THEIRS_SHA8=7d29be87; WORD_SHA8=fd0a5c8f; PHRASE_SHA8=83a95f5b
sha_of() { case "$1" in theirs*) echo $THEIRS_SHA8;; word*) echo $WORD_SHA8;; phrase*) echo $PHRASE_SHA8;; esac; }

if want wait; then
  while :; do
    grep -q TO_ARIES_LOCAL_DONE "$TRANSFER_LOG" 2>/dev/null && break
    grep -q "FAIL" "$TRANSFER_LOG" 2>/dev/null && fail wait "transfer reported FAIL: $(grep FAIL "$TRANSFER_LOG" | tail -1)"
    sleep 120
  done
  sizes=$($SSH $A "du -sh $D3/thinkers/theirs $D3/thinkers/phrase $D4/ours_word 2>/dev/null | cut -f1 | tr '\n' ' '")
  for p in "$D3/olt" "$D3/venvs/olt-thinker/bin/vllm" "$D3/venvs/olt-moss/bin/python" "$D3/tts/model.safetensors" "$D3/checkpoints/XCOMET-XL/checkpoints/model.ckpt" "$D3/thinkers/theirs/config.json" "$D3/thinkers/phrase/config.json" "$D4/ours_word/config.json"; do
    $SSH $A "test -e $p" || fail wait "missing after transfer: $p"
  done
  $SSH $A "docker images --format '{{.Repository}}:{{.Tag}}' | grep -qx jaxanluo/sglang-omni:dev" || fail wait "image jaxanluo/sglang-omni:dev not on aries"
  ok wait "thinkers $sizes; image + support + xcomet present"
fi

if want up; then
  # note (luojiaxuan): pick three GPUs with <2 GB resident. thinker wants two, TTS one; prefer an NVLink pair for the
  # note (luojiaxuan): thinker but SYS is acceptable (NCCL P2P is off anyway). Refuse if fewer than three are free.
  free=$($SSH $A "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '\$2<20000{print \$1, \$2}' | sort -k2 -n | head -5 | awk '{print \$1}' | tr '\n' ' '")
  say "5 emptiest usable GPUs on aries (thinker TP=4=first four, TTS=fifth): [$free]"
  read -r g0 g1 g2 g3 g4 rest <<<"$free"
  [ -n "${g4:-}" ] || fail up "need 5 usable GPUs (<20GB used) for TP=4 thinker + TTS, have [$free]"
  DEV="$g0,$g1,$g2,$g3,$g4"
  taken=$( { $SSH $A "docker ps -a --filter name=sglang-omni-jaxan --format '{{.Names}}'; grep -aoE '^sglang-omni-jaxan-[0-9]+' \$HOME/jiaxuanluo-map.txt" 2>/dev/null; } | grep -oE '[0-9]+$' | sort -un)
  n=1; while echo "$taken" | grep -qx "$n"; do n=$((n+1)); done
  CNAME=sglang-omni-jaxan-$n; echo "$CNAME" > "$CNAME_FILE"; echo "$DEV" > "$HERE/.aries_dev"
  $SSH $A "docker run -d --init --name $CNAME --gpus '\"device=$DEV\"' --shm-size=32g \
    -v $D3:/data/serving_ab -v $D4:/data/thinkers_extra -v \$HOME/.keys:/root/.keys:ro \
    -e PYTHONPATH=/data/serving_ab/pyshim -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1 \
    jaxanluo/sglang-omni:dev bash -c 'sleep infinity' >/dev/null" || fail up "docker run"
  $SSH $A "printf '%s\tgpus=idx%s\thost=aries\thost_data=%s(:/data/serving_ab)+%s\tdesc=protocol-A cascade sweep: 3 thinkers x (1 deterministic + 3 sampled) on ACL dev 5 talks; jobs 920xx\tcreated=%s\t收尾:12 run 出齐即删\n' '$CNAME' '$DEV' '$D3' '$D4' '$(date -u +%FT%TZ)' >> \$HOME/jiaxuanluo-map.txt"
  scp -q "$HERE/aries_run.sh" $A:$D3/phrase_gated_data/aries_run.sh || fail up "scp aries_run.sh"
  ok up "$CNAME on GPUs $DEV"
fi

CNAME=$(cat "$CNAME_FILE" 2>/dev/null || echo "")
[ -n "$CNAME" ] || fail sweep "no container name recorded (run the up stage)"
runarm() { # tag job sjob ckpt rev temp tts_sample max_docs
  local tag=$1 job=$2 sjob=$3 ckpt=$4 rev=$5 temp=$6 ts=$7 md=$8 sha; sha=$(sha_of "$tag")
  $SSH $A "docker exec $CNAME env TAG=$tag JOB=$job SJOB=$sjob CKPT_DIR=$ckpt REV_NAME=$rev SHA8=$sha \
    THINKER_TEMPERATURE=$temp TTS_SAMPLE=$ts MAX_DOCS=$md THINKER_TP=4 THINKER_GPUS=0,1,2,3 TTS_GPU=4 THINKER_GPU_UTIL=0.70 \
    bash /data/serving_ab/phrase_gated_data/aries_run.sh" 2>&1 | tail -6
}

if want smoke; then
  out=$(runarm phrase_smoke 92099 92599 /data/serving_ab/thinkers/phrase gavinlaw-infinisst-thinker-phrase-gated-zh 0 0 1)
  say "smoke: $(echo "$out" | tr '\n' ' ' | cut -c1-200)"
  echo "$out" | grep -q "RUN_phrase_smoke_DONE" || fail smoke "smoke run did not finish; see aries_phrase_smoke_*.log"
  ok smoke "deterministic 1-doc run scored"
fi

if want sweep; then
  for arm in "${ARMS[@]}"; do
    read -r tag job sjob ckpt rev temp ts <<<"$arm"
    out=$(runarm "$tag" "$job" "$sjob" "$ckpt" "$rev" "$temp" "$ts" 5)
    line=$(echo "$out" | grep -aE "RUN_.*_DONE|SKIP|FAIL|CU:|CA:" | tr '\n' ' ' | cut -c1-220)
    # note (luojiaxuan): the real success criterion is metrics.json on disk, not the runarm stdout -- a multi-hour
    # note (luojiaxuan): ssh docker exec can lose its trailing lines (RUN_DONE) even when the work finished.
    if $SSH $A "test -f /mnt/data3/jiaxuanluo/serving_ab/results/s2st_moss-delta_dev_1920ms_$job/metrics.json"; then
      ok "sweep/$tag" "$line"
    else
      fail "sweep/$tag" "no metrics.json; $line"
    fi
  done
fi

if want collect; then
  scp -q "$HERE/aries_collect.py" $A:$D3/phrase_gated_data/aries_collect.py || fail collect "scp collect"
  tbl=$($SSH $A "docker exec $CNAME python3 /data/serving_ab/phrase_gated_data/aries_collect.py" 2>&1)
  say "RESULTS:"; while IFS= read -r l; do say "$l"; done <<<"$tbl"
  ok collect "table written"
fi

if want down; then
  $SSH $A "docker rm -f $CNAME >/dev/null 2>&1 || { sleep 3; docker rm -f $CNAME >/dev/null 2>&1; }; sed -i \"/^$CNAME\t/d\" \$HOME/jiaxuanluo-map.txt; echo containers=\$(docker ps -a --filter name=sglang-omni-jaxan -q | wc -l) map=\$(grep -acE '^sglang-omni-jaxan-[0-9]+\s' \$HOME/jiaxuanluo-map.txt)"
  ok down "removed $CNAME"
fi
say "ARIES DRIVER DONE"
