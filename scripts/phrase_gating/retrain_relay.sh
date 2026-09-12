#!/usr/bin/env bash
# note (luojiaxuan): unattended relay for the corrected-loss retrain on hyper01, run detached so an ssh timeout
# note (luojiaxuan): cannot interrupt a stage (that already killed one docker run today). Stages, in order:
# note (luojiaxuan):   wait_phrase  poll until the phrase train stage has a non-empty mcore and exit 0
# note (luojiaxuan):   export_phrase / train_word / export_word
# note (luojiaxuan): Each stage's success criterion is its ARTIFACT, never the exit code: a container whose
# note (luojiaxuan): heredoc failed to expand still exits 0 (2026-09-12).
set -uo pipefail
W=/data04/jaxan/phrase_sft2
LOSS=empty_turn_end_w0.5
ST=$W/relay.status
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ST"; }
ck() { echo "$W/run_$1_$LOSS"; }
mcore_ok() { [ -d "$(ck $1)/mcore" ] && [ -n "$(ls -A "$(ck $1)/mcore" 2>/dev/null)" ]; }
hf_ok() { [ -f "$(ck $1)/hf/config.json" ] && [ -f "$(ck $1)/hf/model.safetensors.index.json" ]; }

while :; do
  grep -aq "STAGE_train_EXIT=0" $W/train_phrase.log 2>/dev/null && mcore_ok phrase && break
  grep -aqE "STAGE_train_EXIT=[1-9]" $W/train_phrase.log 2>/dev/null && { say "FAIL phrase train nonzero exit"; exit 1; }
  sleep 60
done
say "OK phrase train: mcore $(du -sh $(ck phrase)/mcore | cut -f1)"

for step in export_phrase train_word export_word; do
  case $step in
    export_phrase) ARM=phrase ST_NAME=export ;;
    train_word)    ARM=word   ST_NAME=train ;;
    export_word)   ARM=word   ST_NAME=export ;;
  esac
  NAME=sglang-omni-jaxan-3 ARM=$ARM LOSS=$LOSS DEVICES=4,5,6,7 GPUS=4 \
    bash $W/megatron_hyper01.sh $ST_NAME > $W/${step}.log 2>&1
  if [ "$ST_NAME" = train ]; then
    mcore_ok $ARM || { say "FAIL $step: no mcore"; exit 1; }
    say "OK $step: mcore $(du -sh $(ck $ARM)/mcore | cut -f1)"
  else
    hf_ok $ARM || { say "FAIL $step: no hf/config.json or index"; exit 1; }
    say "OK $step: hf $(du -sh $(ck $ARM)/hf | cut -f1)"
  fi
done
say "RELAY DONE both arms exported"
