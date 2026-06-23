#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:?usage: $0 TARGET_OUTPUT_DIR [ALIGN_OUTPUT_DIR]}"
ALIGN_DIR="${2:-${TARGET_DIR}/mfa_aligned}"
CORPUS_DIR="${TARGET_DIR}/mfa_corpus"
DICTIONARY="${MFA_DICTIONARY:-mandarin_mfa}"
ACOUSTIC_MODEL="${MFA_ACOUSTIC_MODEL:-mandarin_mfa}"
JOBS="${MFA_JOBS:-8}"
BEAM="${MFA_BEAM:-100}"
RETRY_BEAM="${MFA_RETRY_BEAM:-400}"
MFA_HOME="${MFA_HOME:-/mnt/data/jiaxuanluo}"
MFA_TMPDIR="${MFA_TMPDIR:-${MFA_HOME}/tmp}"
MFA_XDG_CACHE_HOME="${MFA_XDG_CACHE_HOME:-${MFA_HOME}/.cache}"
MFA_MICROMAMBA="${MFA_MICROMAMBA:-/mnt/data/jiaxuanluo/micromamba/bin/micromamba}"
MFA_ENV_NAME="${MFA_ENV_NAME:-mfa}"

mkdir -p "${MFA_TMPDIR}" "${MFA_XDG_CACHE_HOME}"
export HOME="${MFA_HOME}"
export TMPDIR="${MFA_TMPDIR}"
export XDG_CACHE_HOME="${MFA_XDG_CACHE_HOME}"
export PYTHONNOUSERSITE=1

if [[ -z "${MFA_BIN:-}" && -x "${MFA_MICROMAMBA}" ]]; then
  MFA_BIN="$("${MFA_MICROMAMBA}" run -n "${MFA_ENV_NAME}" python - <<'PY'
import shutil
print(shutil.which("mfa") or "")
PY
)"
fi
MFA_BIN="${MFA_BIN:-mfa}"

if ! command -v "${MFA_BIN}" >/dev/null 2>&1; then
  echo "mfa command not found: ${MFA_BIN}. Set MFA_BIN to a working Montreal Forced Aligner binary." >&2
  exit 127
fi
export PATH="$(dirname "${MFA_BIN}"):${PATH}"

if [[ ! -d "${CORPUS_DIR}" ]]; then
  echo "Missing MFA corpus dir: ${CORPUS_DIR}" >&2
  exit 2
fi

mkdir -p "${ALIGN_DIR}"

"${MFA_BIN}" model download dictionary "${DICTIONARY}" || true
"${MFA_BIN}" model download acoustic "${ACOUSTIC_MODEL}" || true

ALIGN_EXTRA_ARGS=()
if "${MFA_BIN}" align --help | grep -q -- "--beam"; then
  ALIGN_EXTRA_ARGS+=(--beam "${BEAM}" --retry_beam "${RETRY_BEAM}")
fi

"${MFA_BIN}" align \
  --clean \
  --overwrite \
  --num_jobs "${JOBS}" \
  "${ALIGN_EXTRA_ARGS[@]}" \
  "${CORPUS_DIR}" \
  "${DICTIONARY}" \
  "${ACOUSTIC_MODEL}" \
  "${ALIGN_DIR}"

echo "MFA alignment written to ${ALIGN_DIR}"
