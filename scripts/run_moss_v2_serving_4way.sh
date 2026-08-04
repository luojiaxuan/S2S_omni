#!/usr/bin/env bash
set -euo pipefail

# Launch 4 MOSS-TTS-Realtime serving processes (one per GPU) for the v2
# row-level target generation. Ports match the v1 layout.

RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
SGLANG_OMNI_ROOT="${SGLANG_OMNI_ROOT:-/data/sglang-omni-pr1192}"
MODEL_PATH="${MODEL_PATH:-OpenMOSS-Team/MOSS-TTS-Realtime}"
SERVE_CONFIG="${SERVE_CONFIG:-examples/configs/moss_tts_realtime.yaml}"
PORTS="${PORTS:-48731,49157,52391,54863}"
GPUS="${GPUS:-0,1,2,3}"
ALLOWED_MEDIA="${ALLOWED_MEDIA:-${RUN_ROOT}}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/pids"

IFS=',' read -r -a PORT_ARRAY <<< "${PORTS}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"

cd "${SGLANG_OMNI_ROOT}"
for i in "${!PORT_ARRAY[@]}"; do
  port="${PORT_ARRAY[$i]}"
  gpu="${GPU_ARRAY[$i]}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup sgl-omni serve \
    --model-path "${MODEL_PATH}" \
    --config "${SERVE_CONFIG}" \
    --allowed-local-media-path "${ALLOWED_MEDIA}" \
    --port "${port}" \
    > "${RUN_ROOT}/logs/00_moss_serve_gpu${gpu}_port${port}.log" 2>&1 &
  echo "$!" > "${RUN_ROOT}/pids/moss_serve_gpu${gpu}.pid"
  echo "[serve] gpu=${gpu} port=${port} pid=$(cat "${RUN_ROOT}/pids/moss_serve_gpu${gpu}.pid")"
done

echo "[serve] waiting for /v1/models on all ports"
for i in "${!PORT_ARRAY[@]}"; do
  port="${PORT_ARRAY[$i]}"
  for _ in $(seq 1 120); do
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:${port}/v1/models" || true)"
    [[ "${code}" == "200" ]] && break
    sleep 5
  done
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:${port}/v1/models" || true)"
  echo "[serve] port ${port} status=${code}"
  [[ "${code}" == "200" ]] || exit 1
done
echo "[serve] all ready"
