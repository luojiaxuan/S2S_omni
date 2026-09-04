#!/usr/bin/env bash
# Score the checkpoint-swap arm through the unchanged OLT stack, same recipe as PR #40.
set -uo pipefail
cd /d4/olt_build/olt2 || exit 1
export OLT_RESULTS_ROOT=/d4/olt_build/results3 \
       OLT_VENV_ROOT=/d4/olt_build/venvs2 \
       ACL_ROOT=/d4/olt_build/acl_root \
       HF_HOME=/root/.cache/huggingface \
       ELEVENLABS_KEY_FILE=/d4/.keys/elevenlabs_sst_data \
       S2S_ASR_RESPONSES_ROOT=/d4/olt_build/results3/asr_responses \
       XCOMET_CKPT=/d4/olt_build/checkpoints2/XCOMET-XL/checkpoints/model.ckpt \
       GPU_CU=0 GPU_CA=0 S2S_SCORE_REGIMES=sequential \
       SLURM_JOB_ID=delta-final CUDA_VISIBLE_DEVICES=0
mkdir -p "$OLT_RESULTS_ROOT"
bash eval/recipes/score_timeline.sh /data/delta_tts/timeline infinisst-moss-cascade \
  /data/delta_tts/identity_delta.json dev 1.0 1.92 > /d4/olt_build/score_delta.log 2>&1
echo "SCORE_DELTA_EXIT=$?" >> /d4/olt_build/score_delta.log
echo DELTA_SCORED > /d4/olt_build/score_delta.status
