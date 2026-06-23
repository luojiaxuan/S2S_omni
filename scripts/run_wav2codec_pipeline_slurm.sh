#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/data2/jiaxuanluo/S2S_omni}"
PY="${PY:-/home/jiaxuanluo/miniconda3/envs/infinisst/bin/python}"
PARTITION="${PARTITION:-aries}"
GPU_GRES="${GPU_GRES:-gpu:a6000:1}"
GENERATE_GPU_GRES="${GENERATE_GPU_GRES:-gpu:a6000:4}"
GENERATE_SCRIPT="${GENERATE_SCRIPT:-scripts/generate_omni_codec_pairs_sglang.py}"
GENERATE_EXTRA_ARGS="${GENERATE_EXTRA_ARGS:---thinker-tp-size 2 --gpu-thinker-tp 0,1 --gpu-talker 2 --gpu-code2wav 3 --gpu-image-encoder 3 --gpu-audio-encoder 3}"
SGLANG_OMNI_ROOT="${SGLANG_OMNI_ROOT:-/home/jiaxuanluo/sglang-omni}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
TSV="${TSV:-/mnt/taurus/data/siqiouyang/datasets/gigaspeech/train_xl_case_ft-qwen2.5-32b-instruct_marked_mfa_punc_asr.tsv}"
PAIR_ROOT="${PAIR_ROOT:-${ROOT}/work/omni_s2s_codec_pairs_25k_${DATE_TAG}}"
CKPT_ROOT="${CKPT_ROOT:-${ROOT}/checkpoints/wav2omni_codec_25k_${DATE_TAG}}"
MODE="${MODE:-help}"

cd "${ROOT}"

run_gpu() {
  srun --partition="${PARTITION}" --gres="${GPU_GRES}" "$@"
}

run_generate_gpu() {
  srun --partition="${PARTITION}" --gres="${GENERATE_GPU_GRES}" "$@"
}

case "${MODE}" in
  prepare)
    "${PY}" scripts/build_omni_codec_pair_manifest.py \
      --tsv "${TSV}" \
      --output-dir "${PAIR_ROOT}/manifests" \
      --target-counts "${TARGET_COUNTS:-train=25000,dev=500,test=500}"
    ;;
  generate)
    SPLIT="${SPLIT:-train}"
    SHARD_INDEX="${SHARD_INDEX:-0}"
    NUM_SHARDS="${NUM_SHARDS:-1}"
    run_generate_gpu "${PY}" "${GENERATE_SCRIPT}" \
      --input "${PAIR_ROOT}/manifests/${SPLIT}_manifest.jsonl" \
      --output-dir "${PAIR_ROOT}/${SPLIT}" \
      --sglang-omni-root "${SGLANG_OMNI_ROOT}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${SHARD_INDEX}" \
      ${GENERATE_EXTRA_ARGS}
    ;;
  generate_shard)
    SPLIT="${SPLIT:-train}"
    SHARD_INDEX="${SHARD_INDEX:-0}"
    NUM_SHARDS="${NUM_SHARDS:-1}"
    run_generate_gpu "${PY}" "${GENERATE_SCRIPT}" \
      --input "${PAIR_ROOT}/manifests/${SPLIT}_manifest.jsonl" \
      --output-dir "${PAIR_ROOT}/${SPLIT}_shards/shard_$(printf '%04d' "${SHARD_INDEX}")" \
      --sglang-omni-root "${SGLANG_OMNI_ROOT}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${SHARD_INDEX}" \
      ${GENERATE_EXTRA_ARGS}
    ;;
  merge)
    SPLIT="${SPLIT:-train}"
    NUM_SHARDS="${NUM_SHARDS:-1}"
    "${PY}" scripts/merge_omni_codec_pair_shards.py \
      --shard-root "${PAIR_ROOT}/${SPLIT}_shards" \
      --output-dir "${PAIR_ROOT}/${SPLIT}" \
      --num-shards "${NUM_SHARDS}"
    ;;
  audit)
    SPLIT="${SPLIT:-train}"
    run_gpu "${PY}" scripts/audit_omni_codec_pairs.py \
      --input "${PAIR_ROOT}/${SPLIT}/pairs.jsonl" \
      --output-dir "${PAIR_ROOT}/${SPLIT}_audit" \
      --num-samples "${NUM_SAMPLES:-8}" \
      --decode
    ;;
  train_overfit)
    run_gpu "${PY}" -m accelerate.commands.launch scripts/train_wav2codec.py \
      --train-manifest "${PAIR_ROOT}/train/pairs.jsonl" \
      --output-dir "${CKPT_ROOT}_overfit32" \
      --max-train-records 32 \
      --max-dev-records 32 \
      --overfit \
      --max-steps "${MAX_STEPS:-1000}" \
      --eval-every "${EVAL_EVERY:-100}" \
      --save-every "${SAVE_EVERY:-500}" \
      --per-device-batch-size "${BATCH_SIZE:-2}" \
      --gradient-accumulation-steps "${GRAD_ACCUM:-4}"
    ;;
  train_smoke)
    run_gpu "${PY}" -m accelerate.commands.launch scripts/train_wav2codec.py \
      --train-manifest "${PAIR_ROOT}/train/pairs.jsonl" \
      --dev-manifest "${PAIR_ROOT}/dev/pairs.jsonl" \
      --output-dir "${CKPT_ROOT}_smoke5k" \
      --max-train-records 5000 \
      --max-dev-records 500 \
      --epochs "${EPOCHS:-3}" \
      --per-device-batch-size "${BATCH_SIZE:-2}" \
      --gradient-accumulation-steps "${GRAD_ACCUM:-8}"
    ;;
  train_full)
    run_gpu "${PY}" -m accelerate.commands.launch scripts/train_wav2codec.py \
      --train-manifest "${PAIR_ROOT}/train/pairs.jsonl" \
      --dev-manifest "${PAIR_ROOT}/dev/pairs.jsonl" \
      --output-dir "${CKPT_ROOT}" \
      --epochs "${EPOCHS:-3}" \
      --per-device-batch-size "${BATCH_SIZE:-2}" \
      --gradient-accumulation-steps "${GRAD_ACCUM:-8}"
    ;;
  eval)
    SPLIT="${SPLIT:-test}"
    run_gpu "${PY}" scripts/eval_wav2codec.py \
      --manifest "${PAIR_ROOT}/${SPLIT}/pairs.jsonl" \
      --checkpoint "${CKPT_ROOT}" \
      --output-dir "${PAIR_ROOT}/${SPLIT}_eval" \
      --audio-samples "${NUM_SAMPLES:-8}"
    ;;
  *)
    cat <<EOF
Usage:
  MODE=prepare bash scripts/run_wav2codec_pipeline_slurm.sh
  MODE=generate SPLIT=train NUM_SHARDS=8 SHARD_INDEX=0 bash scripts/run_wav2codec_pipeline_slurm.sh
  MODE=generate_shard SPLIT=train NUM_SHARDS=8 SHARD_INDEX=0 bash scripts/run_wav2codec_pipeline_slurm.sh
  MODE=merge SPLIT=train NUM_SHARDS=8 bash scripts/run_wav2codec_pipeline_slurm.sh
  MODE=audit SPLIT=train bash scripts/run_wav2codec_pipeline_slurm.sh
  MODE=train_overfit bash scripts/run_wav2codec_pipeline_slurm.sh
  MODE=train_smoke bash scripts/run_wav2codec_pipeline_slurm.sh
  MODE=train_full bash scripts/run_wav2codec_pipeline_slurm.sh
  MODE=eval SPLIT=test bash scripts/run_wav2codec_pipeline_slurm.sh

Defaults:
  ROOT=${ROOT}
  PY=${PY}
  PARTITION=${PARTITION}
  PAIR_ROOT=${PAIR_ROOT}
  CKPT_ROOT=${CKPT_ROOT}
EOF
    ;;
esac
