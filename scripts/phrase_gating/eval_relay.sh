#!/usr/bin/env bash
# note (luojiaxuan): unattended second half of the 2x2: wait for retrain_relay.sh to export both arms, then score
# note (luojiaxuan): P_fixed and W_fixed under the aries-matched conditions. Runs detached; the eval container is
# note (luojiaxuan): long-lived (docker exec per cell) unlike the --rm training containers, so it is registered in
# note (luojiaxuan): the map at creation and removed when both cells are scored.
# note (luojiaxuan): It waits for RELAY DONE rather than starting early: export_word needs the same four GPUs.
set -uo pipefail
W=/data04/jaxan/phrase_sft2
LOSS=empty_turn_end_w0.5
ST=$W/eval_relay.status
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ST"; }

while :; do
  grep -aq "RELAY DONE" $W/relay.status 2>/dev/null && break
  grep -aq "FAIL" $W/relay.status 2>/dev/null && { say "ABORT: retrain relay failed"; exit 1; }
  pgrep -fc "retrain_rela[y]" >/dev/null 2>&1 || { grep -aq "RELAY DONE" $W/relay.status 2>/dev/null || { say "ABORT: relay process gone before RELAY DONE"; exit 1; }; }
  sleep 60
done
for arm in phrase word; do
  [ -f "$W/run_${arm}_${LOSS}/hf/config.json" ] || { say "ABORT: $arm export missing"; exit 1; }
done
say "OK both arms exported; starting evaluation"

taken=$( { docker ps -a --filter name=sglang-omni-jaxan --format '{{.Names}}'; grep -aoE '^sglang-omni-jaxan-[0-9]+' $HOME/jiaxuanluo-map.txt 2>/dev/null; } | grep -oE '[0-9]+$' | sort -un)
n=1; while echo "$taken" | grep -qx "$n"; do n=$((n+1)); done
N=sglang-omni-jaxan-$n
docker run -d --init --name "$N" --gpus '"device=4,5,6,7"' --shm-size=32g \
  -v /data04/jaxan:/data -v /data04/cache/huggingface:/root/.cache/huggingface -v $HOME/.keys:/root/.keys:ro \
  -e PYTHONPATH=/data/serving_ab/pyshim -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1 \
  jaxanluo/sglang-omni:dev bash -c 'sleep infinity' >/dev/null || { say "ABORT: docker run"; exit 1; }
printf '%s\tgpus=idx4,5,6,7\thost=hyper01\thost_data=/data04/jaxan(:/data)\tdesc=2x2 交互检验的 fixed-loss 两格打分 (P_fixed 93001, W_fixed 93011), aries 口径\tcreated=%s\t收尾:两格出分即删\n' \
  "$N" "$(date -u +%FT%TZ)" >> $HOME/jiaxuanluo-map.txt
say "container $N up on GPUs 4,5,6,7"

rc=0
for arm in phrase word; do
  NAME=$N ARM=$arm LOSS=$LOSS bash $W/eval_2x2_hyper01.sh >> $W/eval_${arm}_driver.log 2>&1
  if grep -aq "EVAL_${arm}_OK" $W/eval_${arm}_driver.log; then
    say "OK eval $arm: $(grep -a "EVAL_${arm}_OK" $W/eval_${arm}_driver.log | tail -1)"
  else
    say "FAIL eval $arm (see $W/eval_${arm}_driver.log)"; rc=1
  fi
done

docker rm -f "$N" >/dev/null 2>&1 || { sleep 3; docker rm -f "$N" >/dev/null 2>&1; }
sed -i "/^$N\t/d" $HOME/jiaxuanluo-map.txt
say "EVAL RELAY DONE rc=$rc; containers=$(docker ps -a --filter name=sglang-omni-jaxan -q | wc -l)"
exit $rc
