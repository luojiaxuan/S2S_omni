#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-hongccc/sglang-omni:dev}"
CONTAINER_NAME="${CONTAINER_NAME:-sglang-omni-jiaxuan-aries-hibiki-zero-$(date +%m%d-%H%M)}"
MIN_FREE_GB="${MIN_FREE_GB:-300}"
PRIMARY_ROOT="${PRIMARY_ROOT:-/mnt/data3/jiaxuanluo/s2s_omni_hibiki_zero}"
FALLBACK_ROOT="${FALLBACK_ROOT:-/mnt/data6/jiaxuanluo/s2s_omni_hibiki_zero}"
HOST_REPO="${HOST_REPO:-/mnt/data3/jiaxuanluo/S2S_omni}"
HOST_HF_CACHE_DIR="${HOST_HF_CACHE_DIR:-/mnt/data2/jiaxuanluo/.cache/huggingface}"
RECREATE_CONTAINER="${RECREATE_CONTAINER:-0}"

hostname
nvidia-smi || true
df -hT / /mnt/data /mnt/data2 /mnt/data3 /mnt/data4 /mnt/data5 /mnt/data6 /mnt/data7 || true
docker ps -a --format '{{.Names}}' | sort | tail -100 || true

free_gb() {
  df -BG --output=avail "$1" | tail -1 | tr -dc '0-9'
}

choose_root() {
  local primary_parent fallback_parent primary_free fallback_free
  primary_parent="$(dirname "${PRIMARY_ROOT}")"
  fallback_parent="$(dirname "${FALLBACK_ROOT}")"
  primary_free="$(free_gb "${primary_parent}")"
  fallback_free="$(free_gb "${fallback_parent}")"
  if [[ "${primary_free}" -ge "${MIN_FREE_GB}" ]]; then
    echo "${PRIMARY_ROOT}"
  elif [[ "${fallback_free}" -ge "${MIN_FREE_GB}" ]]; then
    echo "${FALLBACK_ROOT}"
  else
    echo "Neither ${primary_parent} (${primary_free}G) nor ${fallback_parent} (${fallback_free}G) has ${MIN_FREE_GB}G free." >&2
    exit 1
  fi
}

HOST_DATA_DIR="${HOST_DATA_DIR:-$(choose_root)}"
mkdir -p \
  "${HOST_DATA_DIR}/tmp" \
  "${HOST_DATA_DIR}/.cache/pip" \
  "${HOST_DATA_DIR}/.cache/uv" \
  "${HOST_DATA_DIR}/wandb" \
  "${HOST_HF_CACHE_DIR}" \
  "$(dirname "${HOST_REPO}")"

if [[ -f "${HOST_REPO}/pyproject.toml" && ! -d "${HOST_REPO}/.git" ]]; then
  echo "Using existing non-git repo snapshot at ${HOST_REPO}"
elif [[ ! -d "${HOST_REPO}/.git" ]]; then
  git clone https://github.com/luojiaxuan/S2S_omni.git "${HOST_REPO}"
else
  git -C "${HOST_REPO}" pull --ff-only
fi

docker pull "${IMAGE}"
DATA_MOUNTS=()
for path in /mnt/data /mnt/data1 /mnt/data2 /mnt/data3 /mnt/data4 /mnt/data6 /mnt/taurus/data /mnt/taurus/data1 /mnt/taurus/data2; do
  if [[ -d "${path}" ]]; then
    DATA_MOUNTS+=(-v "${path}:${path}:ro")
  fi
done
if [[ "${RECREATE_CONTAINER}" == "1" ]] && docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  docker start "${CONTAINER_NAME}" >/dev/null
else
  docker run -itd \
    --shm-size 32g \
    --gpus all \
    -v "${HOST_HF_CACHE_DIR}:/root/.cache/huggingface" \
    -v "${HOST_DATA_DIR}:/data" \
    -v "${HOST_REPO}:/data/repo/S2S_omni" \
    "${DATA_MOUNTS[@]}" \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_HUB_CACHE=/root/.cache/huggingface/hub \
    -e XDG_CACHE_HOME=/data/.cache \
    -e PIP_CACHE_DIR=/data/.cache/pip \
    -e WANDB_DIR=/data/wandb \
    -e TMPDIR=/data/tmp \
    --ipc=host \
    --ulimit nofile=65536:65536 \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --privileged \
    --name "${CONTAINER_NAME}" \
    "${IMAGE}" \
    /bin/zsh
fi

docker exec "${CONTAINER_NAME}" bash -lc '
  set -euo pipefail
  python3 -m pip install --upgrade pip uv
  if [[ -d /data/.venvs/hibiki-zero && ! -x /data/.venvs/hibiki-zero/bin/python ]]; then
    rm -rf /data/.venvs/hibiki-zero
  fi
  if [[ ! -x /data/.venvs/hibiki-zero/bin/python ]]; then
    uv venv --python 3.13 /data/.venvs/hibiki-zero
  fi
  source /data/.venvs/hibiki-zero/bin/activate
  uv pip install hibiki-zero soundfile numpy scipy sacrebleu requests datasets
  hibiki-zero --help >/data/log_hibiki_zero_help.txt 2>&1 || true
  python - <<'"'"'PY'"'"'
import importlib.util
missing = [m for m in ["soundfile", "numpy", "requests"] if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"missing modules: {missing}")
print("hibiki-zero env ok")
PY
'

cat <<EOF
container=${CONTAINER_NAME}
host_data_dir=${HOST_DATA_DIR}
host_repo=${HOST_REPO}

Run inside container:
  docker exec -it ${CONTAINER_NAME} /bin/zsh
  source /data/.venvs/hibiki-zero/bin/activate
  cd /data/repo/S2S_omni
EOF
