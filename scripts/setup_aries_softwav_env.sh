#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-${PWD}}"
VENV="${VENV:-/mnt/data/jiaxuanluo/venvs/s2s_omni_softwav}"
PYTHON="${PYTHON:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
USE_USER_SITE="${USE_USER_SITE:-0}"
USER_BASE="${USER_BASE:-/mnt/data/jiaxuanluo/python_userbase/s2s_omni_softwav}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/mnt/data/jiaxuanluo/.cache/pip}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/data/jiaxuanluo/.cache}"
TMPDIR="${TMPDIR:-/mnt/data/jiaxuanluo/tmp}"

mkdir -p "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TMPDIR}"
export PIP_CACHE_DIR
export XDG_CACHE_HOME
export TMPDIR

if [[ "${USE_USER_SITE}" != "1" ]]; then
  mkdir -p "$(dirname "${VENV}")"
  if [[ ! -d "${VENV}" ]]; then
    if ! "${PYTHON}" -m venv "${VENV}"; then
      echo "venv creation failed. Rerun with USE_USER_SITE=1 to install into the user site-packages." >&2
      exit 1
    fi
  fi
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  PIP=(python -m pip)
  INSTALL_PREFIX=()
else
  mkdir -p "${USER_BASE}"
  export PYTHONUSERBASE="${USER_BASE}"
  export PATH="${USER_BASE}/bin:${PATH}"
  PIP=("${PYTHON}" -m pip)
  INSTALL_PREFIX=(--user)
fi

"${PIP[@]}" install "${INSTALL_PREFIX[@]}" --upgrade pip setuptools wheel

"${PIP[@]}" install "${INSTALL_PREFIX[@]}" --index-url "${TORCH_INDEX_URL}" torch torchvision torchaudio
"${PIP[@]}" install "${INSTALL_PREFIX[@]}" -r "${ROOT}/requirements-train.txt"

"${PYTHON}" - <<'PY'
import importlib.util
mods = ["torch", "transformers", "peft", "qwen_omni_utils", "soundfile"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"missing modules after install: {missing}")
from PIL import Image
if not hasattr(Image, "Resampling"):
    raise SystemExit("Pillow is too old: PIL.Image.Resampling is missing")
print("env ok")
PY

if [[ "${USE_USER_SITE}" != "1" ]]; then
  echo "Activate with: source ${VENV}/bin/activate"
else
  echo "Installed into PYTHONUSERBASE=${USER_BASE}"
  echo "Use with: export PYTHONUSERBASE=${USER_BASE}; export PATH=${USER_BASE}/bin:\$PATH"
fi
