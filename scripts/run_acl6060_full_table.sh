#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/luojiaxuan/Documents/Codex/2026-06-20/s"
REPO="${ROOT}/work/S2S_omni"
PYTHON_BIN="${ROOT}/outputs/floras_test_preview/.venv/bin/python"
OMNISTEVAL_BIN="${ROOT}/outputs/floras_test_preview/.venv/bin/omnisteval"
DATASET_ROOT="/tmp/rasst_main_result_data"
ARTIFACT_BASE="${REPO}/projects/acl6060_s2s_metrics_seed/artifacts"
OPENAI_KEY_FILE="/tmp/acl6060_keys/openai.key"
GEMINI_KEY_FILE="/tmp/acl6060_keys/gemini.key"
KIT_COOKIE_HEADER_FILE="${ROOT}/outputs/floras_live_pilot_refs/kit_shorten_60s_chunk1920/.lt2srv_cookie_header"
OUTPUT_BASE="/tmp/acl6060_live_sweep"
KIT_OUTPUT_BASE="/tmp/acl6060_kit_live_sweep"
CHUNK_MS="960"
RUN_GPT_GEMINI="1"
RUN_KIT="1"
RUN_METRICS="1"
RUN_XCOMET="0"

usage() {
  cat <<'EOF'
Usage: scripts/run_acl6060_full_table.sh [options]

Options:
  --dataset-root PATH
  --artifact-base PATH
  --openai-key-file PATH
  --gemini-key-file PATH
  --kit-cookie-header-file PATH
  --output-base PATH
  --kit-output-base PATH
  --chunk-ms N
  --skip-gpt-gemini
  --skip-kit
  --skip-metrics
  --run-xcomet
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --artifact-base) ARTIFACT_BASE="$2"; shift 2 ;;
    --openai-key-file) OPENAI_KEY_FILE="$2"; shift 2 ;;
    --gemini-key-file) GEMINI_KEY_FILE="$2"; shift 2 ;;
    --kit-cookie-header-file) KIT_COOKIE_HEADER_FILE="$2"; shift 2 ;;
    --output-base) OUTPUT_BASE="$2"; shift 2 ;;
    --kit-output-base) KIT_OUTPUT_BASE="$2"; shift 2 ;;
    --chunk-ms) CHUNK_MS="$2"; shift 2 ;;
    --skip-gpt-gemini) RUN_GPT_GEMINI="0"; shift ;;
    --skip-kit) RUN_KIT="0"; shift ;;
    --skip-metrics) RUN_METRICS="0"; shift ;;
    --run-xcomet) RUN_XCOMET="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

speed_tag() {
  local speed="$1"
  speed="${speed//./p}"
  echo "speed${speed}"
}

need_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 3
  fi
}

run_step() {
  local name="$1"
  shift
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${name}"
  "$@"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE ${name}"
}

is_complete_artifact() {
  local dir="$1"
  [[ -f "${dir}/instances.log" && -f "${dir}/run_config.json" ]] &&
    [[ "$(wc -l < "${dir}/instances.log" | tr -d ' ')" == "5" ]]
}

run_kit_row() {
  local lang="$1"
  local speed="$2"
  local tag="en${lang}_kit_chunk${CHUNK_MS}_$(speed_tag "${speed}")"
  local local_dir="${KIT_OUTPUT_BASE}/${tag}"
  local artifact_dir="${ARTIFACT_BASE}/acl6060_live_${tag}"
  if is_complete_artifact "${artifact_dir}"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP existing KIT ${tag}"
    return
  fi
  run_step "kit ${tag}" \
    "${PYTHON_BIN}" "${REPO}/scripts/run_acl6060_kit_live_eval.py" \
      --dataset-root "${DATASET_ROOT}" \
      --output-dir "${local_dir}" \
      --cookie-header-file "${KIT_COOKIE_HEADER_FILE}" \
      --api-key-file "${OPENAI_KEY_FILE}" \
      --target-lang "${lang}" \
      --chunk-ms "${CHUNK_MS}" \
      --speed-factor "${speed}" \
      --download-hf \
      --resume
  mkdir -p "${artifact_dir}"
  cp "${local_dir}/instances.log" "${local_dir}/responses.jsonl" "${local_dir}/run_config.json" "${artifact_dir}/"
}

mkdir -p "${OUTPUT_BASE}" "${KIT_OUTPUT_BASE}" "${ARTIFACT_BASE}"

if [[ "${RUN_GPT_GEMINI}" == "1" || "${RUN_KIT}" == "1" ]]; then
  need_file "${OPENAI_KEY_FILE}"
fi
if [[ "${RUN_GPT_GEMINI}" == "1" ]]; then
  need_file "${GEMINI_KEY_FILE}"
fi
if [[ "${RUN_KIT}" == "1" ]]; then
  need_file "${KIT_COOKIE_HEADER_FILE}"
fi

if [[ "${RUN_GPT_GEMINI}" == "1" ]]; then
  run_step "gpt/gemini en-zh missing 1.25x" \
    "${REPO}/scripts/run_acl6060_live_compare.sh" \
      --providers openai,gemini \
      --target-langs zh \
      --chunks "${CHUNK_MS}" \
      --speeds 1.25 \
      --dataset-root "${DATASET_ROOT}" \
      --output-base "${OUTPUT_BASE}" \
      --artifact-base "${ARTIFACT_BASE}" \
      --openai-key-file "${OPENAI_KEY_FILE}" \
      --gemini-key-file "${GEMINI_KEY_FILE}" \
      --no-score \
      --download-hf \
      --resume
  run_step "gpt/gemini en-de/en-ja all speeds" \
    "${REPO}/scripts/run_acl6060_live_compare.sh" \
      --providers openai,gemini \
      --target-langs de,ja \
      --chunks "${CHUNK_MS}" \
      --speeds 1,1.25,1.5 \
      --dataset-root "${DATASET_ROOT}" \
      --output-base "${OUTPUT_BASE}" \
      --artifact-base "${ARTIFACT_BASE}" \
      --openai-key-file "${OPENAI_KEY_FILE}" \
      --gemini-key-file "${GEMINI_KEY_FILE}" \
      --no-score \
      --download-hf \
      --resume
fi

if [[ "${RUN_KIT}" == "1" ]]; then
  for lang in zh de ja; do
    for speed in 1 1.25 1.5; do
      run_kit_row "${lang}" "${speed}"
    done
  done
fi

if [[ "${RUN_METRICS}" == "1" ]]; then
  metric_cmd=(
    "${PYTHON_BIN}" "${REPO}/scripts/run_acl6060_metric_pipeline.py"
    --artifact-base "${ARTIFACT_BASE}"
    --chunk-ms "${CHUNK_MS}"
    --python-bin "${PYTHON_BIN}"
    --omnisteval-bin "${OMNISTEVAL_BIN}"
  )
  if [[ "${RUN_XCOMET}" == "1" ]]; then
    metric_cmd+=(--run-xcomet)
  else
    metric_cmd+=(--no-run-xcomet)
  fi
  run_step "metric pipeline" "${metric_cmd[@]}"
fi
