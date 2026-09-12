#!/usr/bin/env bash
# note (luojiaxuan): score one corrected-loss cell (W_fixed or P_fixed) on hyper01 under EXACTLY the conditions
# note (luojiaxuan): W_old/P_old were scored with on aries, so the four cells of the 2x2 interaction test are
# note (luojiaxuan): comparable: thinker TP=4, --enforce-eager, greedy thinker (temperature 0), greedy TTS
# note (luojiaxuan): (TTS_SAMPLE=0), 1.92 s chunks, source speed 1.0, the 5-talk ACL dev split, ElevenLabs Scribe v2.
# note (luojiaxuan): Only empty_turn_end_w differs between the old and fixed cells -- that is the whole point.
#
# note (luojiaxuan): GPU layout differs from the aries run for one reason: hyper01 allows 4 GPUs per host while
# note (luojiaxuan): aries allowed the 5 that run used (TP=4 thinker + a dedicated TTS card). The TTS shares the
# note (luojiaxuan): thinker's last card here; on a 143 GB H200 the thinker holds ~15 GB/card and MOSS ~5 GB, so
# note (luojiaxuan): co-residency is free. TP stays 4, which is what actually affects the numerics.
set -uo pipefail
ARM="${ARM:?phrase|word}"
LOSS="${LOSS:-empty_turn_end_w0.5}"
IMG="${IMG:-vllm-omni:dev}"
W=/data/phrase_sft2
SAB=/data/serving_ab
case "$ARM" in
  phrase) JOB=${JOB:-93001}; SJOB=${SJOB:-93501}; REV=gavinlaw-infinisst-phrase-gated-zh-$LOSS ;;
  word)   JOB=${JOB:-93011}; SJOB=${SJOB:-93511}; REV=gavinlaw-infinisst-word-aligned-zh-$LOSS ;;
  *) echo "ARM must be phrase or word" >&2; exit 2 ;;
esac
CKPT=$W/run_${ARM}_${LOSS}/hf
RUN=$SAB/results/s2st_moss-delta_dev_1920ms_$JOB
NAME="${NAME:?container name}"
DEVICES="${DEVICES:-4,5,6,7}"
LOG=$W/eval_${ARM}

COMMON="OLT_RESULTS_ROOT=$SAB/results OLT_VENV_ROOT=$SAB/venvs ACL_ROOT=$SAB/acl6060_full HF_HOME=/root/.cache/huggingface \
  MOSS_MODEL=$SAB/tts MOSS_MODEL_REVISION=local:owaski-moss-tts-realtime-delta-zh-125k:fc2d094d \
  XCOMET_CKPT=$SAB/checkpoints/XCOMET-XL/checkpoints/model.ckpt \
  ELEVENLABS_KEY_FILE=/root/.keys/elevenlabs_sst_data S2S_ASR_RESPONSES_ROOT=$SAB/asr_responses"

docker exec "$NAME" bash -c "cd $SAB/olt && export HF_TOKEN=\$(cat /root/.keys/hf_token_gavinlaw) && \
  git config --global --add safe.directory '*' && \
  [ -f $RUN/generation_config.json ] || { rm -rf $RUN; env $COMMON \
    SLURM_JOB_ID=$JOB CKPT=$CKPT THINKER_MODEL_REVISION=local:$REV:fixedloss \
    THINKER_BACKEND=uv THINKER_TP=4 THINKER_GPUS=0,1,2,3 TTS_GPU=3 THINKER_GPU_UTIL=0.80 \
    THINKER_ENFORCE_EAGER=1 THINKER_TEMPERATURE=0 TTS_SAMPLE=0 MAX_DOCS=5 SKIP_SCORING=1 \
    bash eval/recipes/run_s2st_eval.sbatch 1.92 1.0 dev moss-delta; }" > ${LOG}_cascade.log 2>&1
echo "CASCADE_${ARM}_EXIT=$?" | tee -a ${LOG}_cascade.log
docker exec "$NAME" test -f $RUN/generation_config.json || { echo "FAIL cascade $ARM: no generation_config.json"; exit 1; }

docker exec "$NAME" bash -c "cd $SAB/olt && export HF_TOKEN=\$(cat /root/.keys/hf_token_gavinlaw) && \
  env $COMMON SLURM_JOB_ID=$SJOB CUDA_VISIBLE_DEVICES=3 S2S_SCORE_REGIMES=sequential \
  bash eval/recipes/run_s2s_score.sbatch $RUN moss-delta dev 1.0 1.92" > ${LOG}_score.log 2>&1
echo "SCORE_${ARM}_EXIT=$?" | tee -a ${LOG}_score.log
grep -aE "^  (CU|CA): " ${LOG}_score.log | cut -c1-200
docker exec "$NAME" python3 -c "
import json; m=json.load(open('$RUN/metrics.json'))['metrics']['CU']
assert 'BLEU' in m, 'scoring skipped: ' + json.dumps(m)
print('EVAL_${ARM}_OK BLEU', round(m['BLEU'],2), 'XCOMET', round(m['XCOMET_XL'],4))
"
