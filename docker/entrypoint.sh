#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/.cache/pip}"
export WANDB_DIR="${WANDB_DIR:-/data/wandb}"
export TMPDIR="${TMPDIR:-/data/tmp}"
export S2S_RUN_ROOT="${S2S_RUN_ROOT:-/data/S2S_omni_runs}"
export MFA_HOME="${MFA_HOME:-/data}"
export MFA_BIN="${MFA_BIN:-/opt/micromamba/envs/mfa/bin/mfa}"
export PYTHONPATH="/opt/S2S_omni:${PYTHONPATH:-}"
export PATH="/opt/micromamba/envs/mfa/bin:/opt/micromamba/bin:${PATH}"

mkdir -p \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${XDG_CACHE_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${WANDB_DIR}" \
  "${TMPDIR}" \
  "${S2S_RUN_ROOT}"

exec "$@"

