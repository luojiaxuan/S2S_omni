#!/usr/bin/env bash
# note (luojiaxuan): hyper01-side steps for evaluating the phrase-gated thinker with the OLT cascade recipe.
# note (luojiaxuan): Runs on the hyper01 host; the eval itself runs inside a task-scoped container (named with the
# note (luojiaxuan): smallest sglang-omni-jaxan-<n> free in both docker ps -a and the map at `up` time) that sees
# note (luojiaxuan): GPUs 2,3,4 (thinker TP=2 on 0,1 and the TTS on 2 in container numbering). The env below is
# note (luojiaxuan): the one the earlier "ours" arm (job 90002) was generated and scored with, so the two runs
# note (luojiaxuan): differ only in the thinker checkpoint and its revision.
#
# note (luojiaxuan): hyper01_phrase_eval.sh up <sha8>   create the container and register it in the map
# note (luojiaxuan): hyper01_phrase_eval.sh cascade     generate 3 dev talks (SKIP_SCORING=1)
# note (luojiaxuan): hyper01_phrase_eval.sh compare     diff the new generation identity against job 90002
# note (luojiaxuan): hyper01_phrase_eval.sh score       ElevenLabs ASR + SEGALE/LongYAAL/BLEU/XCOMET, CU then CA
# note (luojiaxuan): hyper01_phrase_eval.sh down        remove the container and its map line
set -uo pipefail
CMD="${1:?up|cascade|compare|score|down}"
SAB=/data04/jaxan/serving_ab
CNAME_FILE=$SAB/thinker_phrase_gated.cname
if [ "$CMD" = up ]; then
  taken=$( { docker ps -a --filter name=sglang-omni-jaxan --format '{{.Names}}'; grep -aoE '^sglang-omni-jaxan-[0-9]+' "$HOME/jiaxuanluo-map.txt"; } | grep -oE '[0-9]+$' | sort -un)
  n=1; while echo "$taken" | grep -qx "$n"; do n=$((n+1)); done
  CNAME=sglang-omni-jaxan-$n
else
  CNAME=$(cat "$CNAME_FILE")
fi
GPUS=2,3,4
JOB=90003
SJOB=95003
REF_JOB=90002
RUN=/data/serving_ab/results/s2st_moss-delta_dev_1920ms_$JOB
COMMON="OLT_RESULTS_ROOT=/data/serving_ab/results OLT_VENV_ROOT=/data/serving_ab/venvs ACL_ROOT=/data/serving_ab/acl_root HF_HOME=/root/.cache/huggingface"

case "$CMD" in
up)
  SHA8="${2:?need the HF commit (8 hex) of the exported thinker}"
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPUS | awk '$1>2000' | wc -l)
  [ "$busy" = 0 ] || { echo "GPUs $GPUS are not free:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i $GPUS; exit 3; }
  [ "$(docker ps -a -q --filter "name=^$CNAME\$" | wc -l)" = 0 ] || { echo "$CNAME already exists"; exit 4; }
  docker run -d --init --name "$CNAME" --gpus "\"device=$GPUS\"" --ipc=host --shm-size=64g \
    -v /data04/jaxan:/data -v /data04/cache/huggingface:/root/.cache/huggingface -v /data04/jaxan/.keys:/root/.keys:ro \
    -e PYTHONPATH=/data/serving_ab/pyshim -e HF_HOME=/root/.cache/huggingface \
    vllm-omni:dev bash -c 'sleep infinity' >/dev/null
  printf '%s\tgpus=idx%s\thost=hyper01\thost_data=/data04/jaxan(:/data)\tdesc=phrase-gated thinker OLT cascade eval (3 dev docs, job %s) + scoring; thinker rev %s\tcreated=%s\t收尾:打分完成即删\n' \
    "$CNAME" "$GPUS" "$JOB" "$SHA8" "$(date -u +%FT%TZ)" >> "$HOME/jiaxuanluo-map.txt"
  echo "$SHA8" > "$SAB/thinker_phrase_gated.sha8"
  echo "$CNAME" > "$CNAME_FILE"
  docker ps --filter "name=^$CNAME\$" --format '{{.Names}} {{.Status}}'
  ;;
cascade)
  SHA8=$(cat "$SAB/thinker_phrase_gated.sha8")
  docker exec "$CNAME" bash -c "cd /data/serving_ab/olt && export HF_TOKEN=\$(cat /root/.keys/hf_token_gavinlaw) && \
    SLURM_JOB_ID=$JOB CKPT=/data/serving_ab/thinker_phrase_gated \
    THINKER_MODEL_REVISION=local:gavinlaw-infinisst-thinker-phrase-gated-zh:$SHA8 \
    THINKER_BACKEND=uv THINKER_GPUS=0,1 TTS_GPU=2 MAX_DOCS=3 SKIP_SCORING=1 $COMMON \
    MOSS_MODEL=/data/serving_ab/tts MOSS_MODEL_REVISION=local:owaski-moss-tts-realtime-delta-zh-125k:fc2d094d \
    bash eval/recipes/run_s2st_eval.sbatch 1.92 1.0 dev moss-delta" > "$SAB/cascade_phrase.log" 2>&1
  echo "CASCADE_phrase_EXIT=$?" | tee -a "$SAB/cascade_phrase.log"
  ls "$SAB/results/s2st_moss-delta_dev_1920ms_$JOB/generation_config.json" 2>&1
  ls "$SAB/results/s2st_moss-delta_dev_1920ms_$JOB/wavs_cu" 2>/dev/null | wc -l
  ;;
compare)
  python3 - "$SAB/results/s2st_moss-delta_dev_1920ms_$REF_JOB/generation_config.json" \
             "$SAB/results/s2st_moss-delta_dev_1920ms_$JOB/generation_config.json" <<'PY'
import json, sys
ref, new = (json.load(open(p)) for p in sys.argv[1:3])
expected = {"thinker_checkpoint", "thinker_checkpoint_revision", "generation_config_fingerprint"}
diff = sorted(k for k in set(ref) | set(new) if ref.get(k) != new.get(k))
for k in diff:
    tag = "expected" if k in expected else "UNEXPECTED"
    print(f"{tag:10} {k}: {str(ref.get(k))[:60]} -> {str(new.get(k))[:60]}")
print("unexpected_diffs=%d" % sum(k not in expected for k in diff))
PY
  ;;
score)
  docker exec "$CNAME" bash -c "cd /data/serving_ab/olt && export HF_TOKEN=\$(cat /root/.keys/hf_token_gavinlaw) && \
    ls /data/serving_ab/checkpoints/XCOMET-XL/checkpoints/model.ckpt >/dev/null && \
    SLURM_JOB_ID=$SJOB CUDA_VISIBLE_DEVICES=2 S2S_SCORE_REGIMES=sequential $COMMON \
    XCOMET_CKPT=/data/serving_ab/checkpoints/XCOMET-XL/checkpoints/model.ckpt \
    ELEVENLABS_KEY_FILE=/root/.keys/elevenlabs_sst_data S2S_ASR_RESPONSES_ROOT=/data/serving_ab/asr_responses \
    bash eval/recipes/run_s2s_score.sbatch $RUN moss-delta dev 1.0 1.92" > "$SAB/score_phrase.log" 2>&1
  echo "SCORE_phrase_EXIT=$?" | tee -a "$SAB/score_phrase.log"
  grep -aE "^  (CU|CA): " "$SAB/score_phrase.log" | cut -c1-220
  ls "$SAB/results/s2st_moss-delta_dev_1920ms_$JOB/metrics.json" 2>&1
  ;;
down)
  docker rm -f "$CNAME" >/dev/null 2>&1 || { sleep 3; docker rm -f "$CNAME" >/dev/null 2>&1 || true; }
  sed -i "/^$CNAME\t/d" "$HOME/jiaxuanluo-map.txt"
  echo "containers=$(docker ps -a --filter name=sglang-omni-jaxan -q | wc -l) map_lines=$(grep -acE '^sglang-omni-jaxan-[0-9]+\s' "$HOME/jiaxuanluo-map.txt")"
  ;;
esac
