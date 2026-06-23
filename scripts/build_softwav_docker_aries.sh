#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-s2s-omni-softwav:dev}"
BASE_IMAGE="${BASE_IMAGE:-hongccc/sglang-omni:dev}"

hostname
nvidia-smi || true
df -hT / /mnt/data /mnt/data2 /mnt/data3 /mnt/data4 /mnt/data5 /mnt/data6 /mnt/data7 || true
docker ps -a --format '{{.Names}}' | sort | tail -100 || true

docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -f "${ROOT}/docker/Dockerfile.softwav" \
  -t "${IMAGE}" \
  "${ROOT}"

echo "Built ${IMAGE}"

