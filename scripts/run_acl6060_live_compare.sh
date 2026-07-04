#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/luojiaxuan/Documents/Codex/2026-06-20/s"
REPO="${ROOT}/work/S2S_omni"
PYTHON_BIN="${ROOT}/outputs/floras_test_preview/.venv/bin/python"
DATASET_ROOT="/tmp/rasst_main_result_data"
OUTPUT_BASE="/tmp/acl6060_live_sweep"
ARTIFACT_BASE="${REPO}/projects/acl6060_s2s_metrics_seed/artifacts"
OPENAI_KEY_FILE="/tmp/acl6060_keys/openai.key"
GEMINI_KEY_FILE="/tmp/acl6060_keys/gemini.key"
REMOTE_BASE="/mnt/data2/jiaxuanluo/tmp/s2s_omni_acl6060_live_sweep_20260704"
REMOTE_RASST="/mnt/data2/jiaxuanluo/RASST"
REMOTE_PYTHON="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python"
REMOTE_FBK="/mnt/taurus/home/jiaxuanluo/FBK-fairseq"
REMOTE_MWER="/mnt/taurus/home/jiaxuanluo/mwerSegmenter"
PROVIDERS="openai,gemini"
CHUNKS="960,1920"
SPEEDS="1,1.5"
LIMIT="0"
START_INDEX="0"
MAX_AUDIO_SECONDS="0"
PACE_FLAG="--pace"
RESUME_FLAG="--resume"
NO_SCORE="0"
PROGRESS_INTERVAL_S="120"
OPENAI_RECEIVE_TIMEOUT_S="600"
GEMINI_RECEIVE_TIMEOUT_S="240"
GEMINI_POST_SEND_IDLE_S="8"
GEMINI_MAX_SESSION_INPUT_S="480"

usage() {
  cat <<'EOF'
Usage: scripts/run_acl6060_live_compare.sh [options]

Options:
  --providers openai,gemini
  --chunks 960,1920
  --speeds 1,1.5
  --dataset-root PATH
  --output-base PATH
  --artifact-base PATH
  --openai-key-file PATH
  --gemini-key-file PATH
  --limit N
  --start-index N
  --max-audio-seconds SECONDS
  --progress-interval-s SECONDS
  --openai-receive-timeout-s SECONDS
  --gemini-receive-timeout-s SECONDS
  --gemini-post-send-idle-s SECONDS
  --gemini-max-session-input-s SECONDS
  --no-score
  --pace | --no-pace
  --resume | --no-resume
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --providers) PROVIDERS="$2"; shift 2 ;;
    --chunks) CHUNKS="$2"; shift 2 ;;
    --speeds) SPEEDS="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-base) OUTPUT_BASE="$2"; shift 2 ;;
    --artifact-base) ARTIFACT_BASE="$2"; shift 2 ;;
    --openai-key-file) OPENAI_KEY_FILE="$2"; shift 2 ;;
    --gemini-key-file) GEMINI_KEY_FILE="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --start-index) START_INDEX="$2"; shift 2 ;;
    --max-audio-seconds) MAX_AUDIO_SECONDS="$2"; shift 2 ;;
    --progress-interval-s) PROGRESS_INTERVAL_S="$2"; shift 2 ;;
    --openai-receive-timeout-s) OPENAI_RECEIVE_TIMEOUT_S="$2"; shift 2 ;;
    --gemini-receive-timeout-s) GEMINI_RECEIVE_TIMEOUT_S="$2"; shift 2 ;;
    --gemini-post-send-idle-s) GEMINI_POST_SEND_IDLE_S="$2"; shift 2 ;;
    --gemini-max-session-input-s) GEMINI_MAX_SESSION_INPUT_S="$2"; shift 2 ;;
    --no-score) NO_SCORE="1"; shift ;;
    --pace) PACE_FLAG="--pace"; shift ;;
    --no-pace) PACE_FLAG="--no-pace"; shift ;;
    --resume) RESUME_FLAG="--resume"; shift ;;
    --no-resume) RESUME_FLAG="--no-resume"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

csv_to_array() {
  local raw="$1"
  raw="${raw//,/ }"
  printf '%s\n' ${raw}
}

speed_tag() {
  local speed="$1"
  speed="${speed//./p}"
  echo "speed${speed}"
}

run_step() {
  local name="$1"
  shift
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${name}"
  "$@"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE ${name}"
}

score_run() {
  local local_dir="$1"
  local tag="$2"
  local artifact_dir="$3"
  local remote_dir="${REMOTE_BASE}/${tag}"

  mkdir -p "${artifact_dir}"
  ssh -T -o RemoteCommand=none -o RequestTTY=no taurus "mkdir -p '${remote_dir}'"
  scp -q -O \
    "${local_dir}/instances.log" \
    "${local_dir}/responses.jsonl" \
    "${local_dir}/run_config.json" \
    "taurus:${remote_dir}/"
  ssh -T -o RemoteCommand=none -o RequestTTY=no taurus \
    "cd '${REMOTE_RASST}' && PATH='${REMOTE_MWER}':\$PATH '${REMOTE_PYTHON}' code/rasst/eval/offline_sst_eval/offline_streamlaal_eval.py --mode acl6060 --instances-log '${remote_dir}/instances.log' --lang-code zh --ref-file data/main_result/inputs/acl_zh/ref.txt --source-file data/main_result/inputs/acl_zh/source_text.txt --audio-yaml data/main_result/inputs/acl_zh/audio.yaml --glossary-acl6060 data/glossaries/acl6060_tagged_gt_raw_min_norm2.json --fbk-fairseq-root '${REMOTE_FBK}' --python-bin '${REMOTE_PYTHON}' --output-tsv '${remote_dir}/eval_results.tsv' --output-log '${remote_dir}/eval_results.log'"
  cp "${local_dir}/instances.log" "${local_dir}/responses.jsonl" "${local_dir}/run_config.json" "${artifact_dir}/"
  scp -q -O \
    "taurus:${remote_dir}/eval_results.tsv" \
    "taurus:${remote_dir}/eval_results.log" \
    "${artifact_dir}/"
}

mkdir -p "${OUTPUT_BASE}" "${ARTIFACT_BASE}"

for provider in $(csv_to_array "${PROVIDERS}"); do
  case "${provider}" in
    openai)
      if [[ ! -f "${OPENAI_KEY_FILE}" ]]; then
        echo "Missing OpenAI key file: ${OPENAI_KEY_FILE}" >&2
        exit 3
      fi
      ;;
    gemini)
      if [[ ! -f "${GEMINI_KEY_FILE}" ]]; then
        echo "Missing Gemini key file: ${GEMINI_KEY_FILE}" >&2
        exit 3
      fi
      ;;
    *)
      echo "Unsupported provider: ${provider}" >&2
      exit 2
      ;;
  esac

  for chunk in $(csv_to_array "${CHUNKS}"); do
    for speed in $(csv_to_array "${SPEEDS}"); do
      tag="${provider}_chunk${chunk}_$(speed_tag "${speed}")"
      local_dir="${OUTPUT_BASE}/${tag}"
      artifact_dir="${ARTIFACT_BASE}/acl6060_live_${tag}"
      key_file="${OPENAI_KEY_FILE}"
      receive_timeout="${OPENAI_RECEIVE_TIMEOUT_S}"
      runner_cmd=(
        "${PYTHON_BIN}" "${REPO}/projects/acl6060_s2s_metrics_seed/run_acl6060_live_stream_eval.py"
        --dataset-root "${DATASET_ROOT}"
        --output-dir "${local_dir}"
        --provider "${provider}"
        --api-key-file "${key_file}"
        --chunk-ms "${chunk}"
        --speed-factor "${speed}"
        --start-index "${START_INDEX}"
        --limit "${LIMIT}"
        --max-audio-seconds "${MAX_AUDIO_SECONDS}"
        --receive-timeout-s "${receive_timeout}"
        --progress-interval-s "${PROGRESS_INTERVAL_S}"
        "${PACE_FLAG}"
        "${RESUME_FLAG}"
      )
      if [[ "${provider}" == "gemini" ]]; then
        key_file="${GEMINI_KEY_FILE}"
        receive_timeout="${GEMINI_RECEIVE_TIMEOUT_S}"
        runner_cmd=(
          "${PYTHON_BIN}" "${REPO}/projects/acl6060_s2s_metrics_seed/run_acl6060_live_stream_eval.py"
          --dataset-root "${DATASET_ROOT}"
          --output-dir "${local_dir}"
          --provider "${provider}"
          --api-key-file "${key_file}"
          --chunk-ms "${chunk}"
          --speed-factor "${speed}"
          --start-index "${START_INDEX}"
          --limit "${LIMIT}"
          --max-audio-seconds "${MAX_AUDIO_SECONDS}"
          --receive-timeout-s "${receive_timeout}"
          --progress-interval-s "${PROGRESS_INTERVAL_S}"
          "${PACE_FLAG}"
          "${RESUME_FLAG}"
          --post-send-idle-s "${GEMINI_POST_SEND_IDLE_S}"
          --max-session-input-s "${GEMINI_MAX_SESSION_INPUT_S}"
        )
      fi

      run_step "acl6060 ${tag}" "${runner_cmd[@]}"

      if [[ "${NO_SCORE}" == "1" || "${LIMIT}" != "0" ||
        ( "${MAX_AUDIO_SECONDS}" != "0" && "${MAX_AUDIO_SECONDS}" != "0.0" ) ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP score ${tag}"
      else
        run_step "score ${tag}" score_run "${local_dir}" "${tag}" "${artifact_dir}"
      fi
    done
  done
done
