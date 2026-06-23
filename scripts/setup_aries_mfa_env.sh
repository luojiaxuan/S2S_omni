#!/usr/bin/env bash
set -euo pipefail

MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/mnt/data/jiaxuanluo/micromamba}"
MFA_ENV_NAME="${MFA_ENV_NAME:-mfa}"
MFA_PYTHON="${MFA_PYTHON:-3.10}"
TMPDIR="${TMPDIR:-/mnt/data/jiaxuanluo/tmp}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/data/jiaxuanluo/.cache}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${MAMBA_ROOT_PREFIX}/pkgs}"
MFA_HOME="${MFA_HOME:-/mnt/data/jiaxuanluo}"

mkdir -p "${MAMBA_ROOT_PREFIX}/bin" "${TMPDIR}" "${XDG_CACHE_HOME}" "${CONDA_PKGS_DIRS}" "${MFA_HOME}"
export MAMBA_ROOT_PREFIX
export CONDA_PKGS_DIRS
export TMPDIR
export XDG_CACHE_HOME
export HOME="${MFA_HOME}"
export PYTHONNOUSERSITE=1

MICROMAMBA="${MAMBA_ROOT_PREFIX}/bin/micromamba"
if [[ ! -x "${MICROMAMBA}" ]]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C "${MAMBA_ROOT_PREFIX}/bin" --strip-components=1 bin/micromamba
fi

if "${MICROMAMBA}" run -n "${MFA_ENV_NAME}" python -c "pass" >/dev/null 2>&1; then
  "${MICROMAMBA}" install -y -n "${MFA_ENV_NAME}" -c conda-forge \
    "python=${MFA_PYTHON}" \
    "kalpy<0.10" \
    montreal-forced-aligner
else
  "${MICROMAMBA}" create -y -n "${MFA_ENV_NAME}" -c conda-forge \
    "python=${MFA_PYTHON}" \
    "kalpy<0.10" \
    montreal-forced-aligner
fi

MFA_BIN="$("${MICROMAMBA}" run -n "${MFA_ENV_NAME}" python - <<'PY'
import shutil
print(shutil.which("mfa") or "")
PY
)"
if [[ -z "${MFA_BIN}" ]]; then
  echo "mfa not found in micromamba env: ${MFA_ENV_NAME}" >&2
  exit 1
fi
MFA_PIP="$(dirname "${MFA_BIN}")/pip"
"${MFA_PIP}" install --cache-dir "${XDG_CACHE_HOME}/pip" spacy-pkuseg dragonmapper hanziconv
"${MFA_BIN}" version

echo "MFA_BIN=${MFA_BIN}"
