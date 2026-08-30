#!/usr/bin/env bash
#SBATCH --job-name=jx-75129
#SBATCH --partition=main
#SBATCH --qos=guest-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:5
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180

set -euo pipefail

: "${TASK_ROOT:?}"

container_import_ref="pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime"
container_squashfs="${TASK_ROOT}/images/pytorch-2.9.1-cu128.sqsh"
checkpoint_request="${TASK_ROOT}/checkpoints/request"

mkdir -p \
  "${TASK_ROOT}/cache/enroot" \
  "${TASK_ROOT}/enroot/data" \
  "${TASK_ROOT}/enroot/runtime" \
  "${TASK_ROOT}/enroot/tmp" \
  "${TASK_ROOT}/images" \
  "${TASK_ROOT}/logs" \
  "${TASK_ROOT}/receipts"
rm -f "${checkpoint_request}"

export ENROOT_CACHE_PATH="${TASK_ROOT}/cache/enroot"
export ENROOT_DATA_PATH="${TASK_ROOT}/enroot/data"
export ENROOT_RUNTIME_PATH="${TASK_ROOT}/enroot/runtime"
export ENROOT_TEMP_PATH="${TASK_ROOT}/enroot/tmp"
export TASK_ROOT

restart_count="${SLURM_RESTART_COUNT:-0}"
allocation_path="${TASK_ROOT}/receipts/allocation-${SLURM_JOB_ID}-${restart_count}.txt"
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  hostname
  printf 'job_id=%s\n' "${SLURM_JOB_ID}"
  printf 'restart_count=%s\n' "${restart_count}"
  printf 'job_gpus=%s\n' "${SLURM_JOB_GPUS:-}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
  printf 'task_root=%s\n' "${TASK_ROOT}"
  printf 'container_import_ref=%s\n' "${container_import_ref}"
  df -Pk "${TASK_ROOT}"
  nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv,noheader
  sha256sum "${TASK_ROOT}/code/train.sh"
  sha256sum "${TASK_ROOT}/code/MOSS-TTS/moss_tts_realtime/finetuning/sft.py"
  sha256sum "${TASK_ROOT}/data/train_matched.jsonl"
} >"${allocation_path}"

if [[ ! -f "${container_squashfs}" ]]; then
  import_path="${container_squashfs}.partial-${SLURM_JOB_ID}-${restart_count}"
  enroot import --arch x86_64 --output "${import_path}" \
    "docker://${container_import_ref}"
  mv "${import_path}" "${container_squashfs}"
fi
sha256sum "${container_squashfs}" >>"${allocation_path}"

trap 'touch "${checkpoint_request}"' USR1
set +e
srun \
  --container-image="${container_squashfs}" \
  --container-mounts="${TASK_ROOT}:/workspace:rw" \
  --container-workdir=/workspace \
  --container-remap-root \
  --container-writable \
  --no-container-entrypoint \
  --container-env=TASK_ROOT,SLURM_JOB_ID,SLURM_RESTART_COUNT \
  /workspace/code/train.sh &
step_pid=$!
wait "${step_pid}"
step_status=$?
set -e

if [[ -f "${checkpoint_request}" ]]; then
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
fi
exit "${step_status}"
