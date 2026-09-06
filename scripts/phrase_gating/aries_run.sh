#!/usr/bin/env bash
# note (luojiaxuan): one cascade generation + scoring of a single arm inside the aries eval container. Every path
# note (luojiaxuan): is the container path (/data/serving_ab is /mnt/data3/jiaxuanluo/serving_ab, the local NVMe copy
# note (luojiaxuan): so a 60 GB thinker reloads from disk in seconds instead of ~80 min over the gemini NFS). All the
# note (luojiaxuan): arm parameters arrive as environment variables so the Mac driver can sweep them:
# note (luojiaxuan):   TAG JOB SJOB CKPT_DIR REV_NAME SHA8 THINKER_TEMPERATURE TTS_SAMPLE MAX_DOCS
# note (luojiaxuan):   THINKER_GPUS TTS_GPU THINKER_GPU_UTIL   (the last three are container GPU ordinals)
# note (luojiaxuan): Skips its own work when metrics.json already exists, so the driver resumes without recomputing.
set -uo pipefail
: "${TAG:?} ${JOB:?} ${SJOB:?} ${CKPT_DIR:?} ${REV_NAME:?} ${SHA8:?}"
THINKER_TEMPERATURE="${THINKER_TEMPERATURE:-0.6}"
TTS_SAMPLE="${TTS_SAMPLE:-1}"
MAX_DOCS="${MAX_DOCS:-5}"
THINKER_TP="${THINKER_TP:-2}"
THINKER_GPUS="${THINKER_GPUS:-0,1}"
TTS_GPU="${TTS_GPU:-2}"
THINKER_GPU_UTIL="${THINKER_GPU_UTIL:-0.85}"
SAB=/data/serving_ab
RUN=$SAB/results/s2st_moss-delta_dev_1920ms_$JOB
LOG=$SAB/phrase_gated_data/aries_$TAG
export HF_TOKEN=$(cat /root/.keys/hf_token_gavinlaw)
git config --global --add safe.directory '*'
cd $SAB/olt

if [ -f "$RUN/metrics.json" ]; then echo "SKIP $TAG: metrics.json exists"; exit 0; fi

COMMON=(
  OLT_RESULTS_ROOT=$SAB/results OLT_VENV_ROOT=$SAB/venvs
  ACL_ROOT=$SAB/acl6060_full HF_HOME=$SAB/hf_home
  MOSS_MODEL=$SAB/tts MOSS_MODEL_REVISION=local:owaski-moss-tts-realtime-delta-zh-125k:fc2d094d
  XCOMET_CKPT=$SAB/checkpoints/XCOMET-XL/checkpoints/model.ckpt
  ELEVENLABS_KEY_FILE=/root/.keys/elevenlabs_sst_data
  S2S_ASR_RESPONSES_ROOT=$SAB/asr_responses
)

if [ ! -f "$RUN/generation_config.json" ]; then
  rm -rf "$RUN"
  env "${COMMON[@]}" \
    SLURM_JOB_ID=$JOB CKPT=$CKPT_DIR THINKER_MODEL_REVISION=local:$REV_NAME:$SHA8 \
    THINKER_BACKEND=uv THINKER_TP=$THINKER_TP THINKER_GPUS=$THINKER_GPUS TTS_GPU=$TTS_GPU THINKER_GPU_UTIL=$THINKER_GPU_UTIL \
    THINKER_TEMPERATURE=$THINKER_TEMPERATURE TTS_SAMPLE=$TTS_SAMPLE MAX_DOCS=$MAX_DOCS SKIP_SCORING=1 \
    bash eval/recipes/run_s2st_eval.sbatch 1.92 1.0 dev moss-delta > "${LOG}_cascade.log" 2>&1
  ec=$?
  echo "CASCADE_${TAG}_EXIT=$ec" | tee -a "${LOG}_cascade.log"
  [ $ec -eq 0 ] && [ -f "$RUN/generation_config.json" ] || { echo "FAIL cascade $TAG"; exit 1; }
fi

env "${COMMON[@]}" \
  SLURM_JOB_ID=$SJOB CUDA_VISIBLE_DEVICES=$TTS_GPU S2S_SCORE_REGIMES=sequential \
  bash eval/recipes/run_s2s_score.sbatch $RUN moss-delta dev 1.0 1.92 > "${LOG}_score.log" 2>&1
ec=$?
echo "SCORE_${TAG}_EXIT=$ec" | tee -a "${LOG}_score.log"
grep -aE "^  (CU|CA): " "${LOG}_score.log" | cut -c1-200
[ $ec -eq 0 ] && [ -f "$RUN/metrics.json" ] || { echo "FAIL score $TAG"; exit 1; }
echo "RUN_${TAG}_DONE"
