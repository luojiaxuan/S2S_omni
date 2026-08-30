#!/usr/bin/env bash
set -euo pipefail

workspace=/workspace
site_packages="${workspace}/env/site"
ready_marker="${workspace}/env/moss-tts-ready"
base_model="${workspace}/cache/huggingface/hub/models--OpenMOSS-Team--MOSS-TTS-Realtime/snapshots/75682787d8e2fcc73faca37ba2931453ca9c4022"

export HF_HOME="${workspace}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export XDG_CACHE_HOME="${workspace}/cache/xdg"
export PIP_CACHE_DIR="${workspace}/cache/pip"
export TMPDIR="${workspace}/tmp"
export PYTHONPATH="${site_packages}:${workspace}/code/MOSS-TTS"

if [[ ! -f "${ready_marker}" ]]; then
  python -m pip install \
    --target "${site_packages}" \
    --cache-dir "${PIP_CACHE_DIR}" \
    "${workspace}/code/MOSS-TTS[finetune]" \
    "transformers==5.0.0"
  touch "${ready_marker}"
fi

python -m accelerate.commands.launch \
  --num_processes 5 \
  --mixed_precision bf16 \
  "${workspace}/code/MOSS-TTS/moss_tts_realtime/finetuning/sft.py" \
  --model-path "${base_model}" \
  --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer@3cd226ba2947efa357ef453bcad111b6eafba782 \
  --train-jsonl "${workspace}/data/train_matched.jsonl" \
  --output-dir "${workspace}/checkpoints/model" \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 3 \
  --learning-rate 1e-5 \
  --num-epochs 1 \
  --num-workers 2 \
  --mixed-precision bf16 \
  --attn-implementation sdpa \
  --seed 42 \
  --checkpointing-steps 500 \
  --resume-from-checkpoint latest \
  --checkpoint-request-file "${workspace}/checkpoints/request"
