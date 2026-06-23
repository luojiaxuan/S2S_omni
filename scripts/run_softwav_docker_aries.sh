#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-s2s-omni-softwav:dev}"
CONTAINER_NAME="${CONTAINER_NAME:-s2s-omni-jiaxuanluo-aries-softwav-$(date +%m%d-%H%M)}"

: "${HOST_DATA_DIR:?Set HOST_DATA_DIR to a local /mnt/data* personal directory first}"
: "${HOST_HF_CACHE_DIR:?Set HOST_HF_CACHE_DIR to an explicit local HF cache directory first}"

hostname
nvidia-smi
df -hT / /mnt/data /mnt/data2 /mnt/data3 /mnt/data4 /mnt/data5 /mnt/data6 /mnt/data7
docker ps -a --format '{{.Names}}' | sort | tail -100

mkdir -p \
  "${HOST_DATA_DIR}/tmp" \
  "${HOST_DATA_DIR}/.cache/pip" \
  "${HOST_DATA_DIR}/wandb" \
  "${HOST_DATA_DIR}/S2S_omni_runs" \
  "${HOST_HF_CACHE_DIR}"

docker run -itd \
  --shm-size 32g \
  --gpus all \
  -v "${HOST_HF_CACHE_DIR}:/root/.cache/huggingface" \
  -v "${HOST_DATA_DIR}:/data" \
  -v /mnt/gemini/data1:/mnt/gemini/data1:ro \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_CACHE=/root/.cache/huggingface/hub \
  -e XDG_CACHE_HOME=/data/.cache \
  -e PIP_CACHE_DIR=/data/.cache/pip \
  -e WANDB_DIR=/data/wandb \
  -e TMPDIR=/data/tmp \
  -e S2S_RUN_ROOT=/data/S2S_omni_runs \
  -e MFA_HOME=/data \
  --ipc=host \
  --ulimit nofile=65536:65536 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --privileged \
  --name "${CONTAINER_NAME}" \
  "${IMAGE}" \
  /bin/bash

echo "Started ${CONTAINER_NAME}"
echo "Example: docker exec -it ${CONTAINER_NAME} bash -lc 'cd /opt/S2S_omni && bash scripts/run_rasst_softwav_smoke_aries.sh'"

