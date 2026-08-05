#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OMNISTEVAL_BIN="${OMNISTEVAL_BIN:-omnisteval}"
SPEECH_LATENCY_REPO="${SPEECH_LATENCY_REPO:-}"
DATASET_ROOT="${DATASET_ROOT:-/tmp/rasst_main_result_data}"
ARTIFACT_BASE="${ARTIFACT_BASE:-${REPO}/projects/acl6060_s2s_metrics_seed/artifacts}"
OPENAI_KEY_FILE="${OPENAI_KEY_FILE:-/tmp/acl6060_keys/openai.key}"
GEMINI_KEY_FILE="${GEMINI_KEY_FILE:-/tmp/acl6060_keys/gemini.key}"
KIT_COOKIE_HEADER_FILE="${KIT_COOKIE_HEADER_FILE:-}"
OUTPUT_BASE="${OUTPUT_BASE:-/tmp/acl6060_live_sweep}"
KIT_OUTPUT_BASE="${KIT_OUTPUT_BASE:-/tmp/acl6060_kit_live_sweep}"
CHUNK_MS="960"
RUN_GPT_GEMINI="1"
RUN_KIT="1"
RUN_METRICS="1"
RUN_XCOMET="0"

usage() {
  cat <<'EOF'
Usage: scripts/run_acl6060_full_table.sh [options]

Options:
  --python-bin PATH
  --omnisteval-bin PATH
  --speech-latency-repo PATH
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
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --omnisteval-bin) OMNISTEVAL_BIN="$2"; shift 2 ;;
    --speech-latency-repo) SPEECH_LATENCY_REPO="$2"; shift 2 ;;
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

need_executable() {
  local executable="$1"
  if [[ ! -x "${executable}" ]] && ! command -v "${executable}" >/dev/null 2>&1; then
    echo "Missing required executable: ${executable}" >&2
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
  local summary="${dir}/target_speech_timing_summary.json"
  [[ -f "${dir}/instances.log" && -f "${dir}/run_config.json" && -f "${summary}" ]] &&
    [[ "$(wc -l < "${dir}/instances.log" | tr -d ' ')" == "5" ]] &&
    "${PYTHON_BIN}" -c \
      'import json,sys; data=json.load(open(sys.argv[1])); ok=data.get("samples")==5 and data.get("timing_method")=="target_speech_word_timestamp_to_pcm_packet_playout_v2"; raise SystemExit(0 if ok else 1)' \
      "${summary}"
}

has_kit_source_timeline() {
  local dir="$1"
  "${PYTHON_BIN}" -c \
    'import json,pathlib,sys; runs=sorted(pathlib.Path(sys.argv[1]).glob("[0-9][0-9][0-9]_*")); ok=len(runs)==5 and all(any(row.get("sent_at_s") is not None for row in json.load(open(run/"run.json")).get("postStats", [])) for run in runs); raise SystemExit(0 if ok else 1)' \
    "${dir}"
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
  local resume_flag="--no-resume"
  if has_kit_source_timeline "${local_dir}"; then
    resume_flag="--resume"
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
      "${resume_flag}"
  run_step "target speech timing ${tag}" \
    "${PYTHON_BIN}" "${REPO}/scripts/build_acl6060_target_speech_instances.py" \
      --run-dir "${local_dir}" \
      --api-key-file "${OPENAI_KEY_FILE}" \
      --resume
  mkdir -p "${artifact_dir}"
  cp \
    "${local_dir}/instances.log" \
    "${local_dir}/instances.provider_transcript.log" \
    "${local_dir}/responses.jsonl" \
    "${local_dir}/run_config.json" \
    "${local_dir}/target_speech_timing.jsonl" \
    "${local_dir}/target_speech_timing_summary.json" \
    "${artifact_dir}/"
}

mkdir -p "${OUTPUT_BASE}" "${KIT_OUTPUT_BASE}" "${ARTIFACT_BASE}"
need_executable "${PYTHON_BIN}"

if [[ "${RUN_GPT_GEMINI}" == "1" || "${RUN_KIT}" == "1" ]]; then
  need_file "${OPENAI_KEY_FILE}"
fi
if [[ "${RUN_GPT_GEMINI}" == "1" ]]; then
  need_file "${GEMINI_KEY_FILE}"
fi
if [[ "${RUN_KIT}" == "1" ]]; then
  need_file "${KIT_COOKIE_HEADER_FILE}"
fi
if [[ "${RUN_METRICS}" == "1" && ! -d "${SPEECH_LATENCY_REPO}" ]]; then
  echo "Missing SEGALE repository; pass --speech-latency-repo PATH" >&2
  exit 3
fi

if [[ "${RUN_GPT_GEMINI}" == "1" ]]; then
  run_step "gpt/gemini en-zh missing 1.25x" \
    "${REPO}/scripts/run_acl6060_live_compare.sh" \
      --python-bin "${PYTHON_BIN}" \
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
      --python-bin "${PYTHON_BIN}" \
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
    --speech-latency-repo "${SPEECH_LATENCY_REPO}"
  )
  if [[ "${RUN_XCOMET}" == "1" ]]; then
    metric_cmd+=(--run-xcomet --reference-free-xcomet)
  else
    metric_cmd+=(--no-run-xcomet)
  fi
  run_step "metric pipeline" "${metric_cmd[@]}"
fi
