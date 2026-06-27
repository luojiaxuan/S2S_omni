#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Users/luojiaxuan/Documents/Codex/2026-06-20/s}"
REPO="${REPO:-${ROOT}/work/S2S_omni}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/outputs/floras_test_preview/.venv/bin/python}"
OUT_BASE="${OUT_BASE:-${ROOT}/outputs/floras_live_pilot_refs}"
MANIFEST="${MANIFEST:-${OUT_BASE}/live_runs.jsonl}"
LOG_DIR="${LOG_DIR:-${OUT_BASE}/logs}"
mkdir -p "${LOG_DIR}"

if [[ -z "${OPENAI_API_KEY:-}" ]] && command -v launchctl >/dev/null 2>&1; then
  export OPENAI_API_KEY="$(launchctl getenv OPENAI_API_KEY)"
fi
if [[ -z "${GEMINI_API_KEY:-}" ]] && command -v launchctl >/dev/null 2>&1; then
  export GEMINI_API_KEY="$(launchctl getenv GEMINI_API_KEY)"
fi
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"
: "${GEMINI_API_KEY:?GEMINI_API_KEY must be set}"

RUN_IDS=(
  "en-zh_mono_asr_test__0__speed_1"
  "en-zh_mono_asr_test__0__speed_1.5"
)
CHUNKS=(960 1920)

run_step() {
  local name="$1"
  shift
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${name}"
  "$@"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE ${name}"
}

for chunk in "${CHUNKS[@]}"; do
  openai_dir="${OUT_BASE}/openai_live_full_enzh_chunk${chunk}"
  for run_id in "${RUN_IDS[@]}"; do
    run_step "openai chunk=${chunk} run=${run_id}" \
      "${PYTHON_BIN}" "${REPO}/scripts/run_floras_openai_realtime.py" \
        --manifest "${MANIFEST}" \
        --output-dir "${openai_dir}" \
        --run-id "${run_id}" \
        --chunk-ms "${chunk}" \
        --pace \
        --receive-timeout-s 1200
  done
done

for chunk in "${CHUNKS[@]}"; do
  gemini_dir="${OUT_BASE}/gemini_live_full_enzh_chunk${chunk}_trim"
  for run_id in "${RUN_IDS[@]}"; do
    run_step "gemini chunk=${chunk} run=${run_id}" \
      "${PYTHON_BIN}" "${REPO}/scripts/run_floras_gemini_live.py" \
        --manifest "${MANIFEST}" \
        --output-dir "${gemini_dir}" \
        --run-id "${run_id}" \
        --chunk-ms "${chunk}" \
        --pace \
        --receive-timeout-s 240 \
        --post-send-idle-s 8
  done
done

for chunk in "${CHUNKS[@]}"; do
  run_step "openai full ASR chunk=${chunk}" \
    "${PYTHON_BIN}" "${REPO}/scripts/openai_transcribe_live_outputs.py" \
      --run-output-dir "${OUT_BASE}/openai_live_full_enzh_chunk${chunk}" \
      --output "${OUT_BASE}/openai_asr_full_enzh_chunk${chunk}.jsonl"
  run_step "gemini full ASR chunk=${chunk}" \
    "${PYTHON_BIN}" "${REPO}/scripts/openai_transcribe_live_outputs.py" \
      --run-output-dir "${OUT_BASE}/gemini_live_full_enzh_chunk${chunk}_trim" \
      --output "${OUT_BASE}/gemini_asr_full_enzh_chunk${chunk}_trim.jsonl"
done

for chunk in "${CHUNKS[@]}"; do
  run_step "openai eval chunk=${chunk}" \
    "${PYTHON_BIN}" "${REPO}/scripts/evaluate_floras_live_s2s.py" \
      --manifest "${MANIFEST}" \
      --run-output-dir "${OUT_BASE}/openai_live_full_enzh_chunk${chunk}" \
      --output-dir "${OUT_BASE}/openai_eval_full_enzh_chunk${chunk}_asr" \
      --asr-jsonl "${OUT_BASE}/openai_asr_full_enzh_chunk${chunk}.jsonl" \
      --target-context-s 20 \
      --coverage-judge none
  run_step "gemini eval chunk=${chunk}" \
    "${PYTHON_BIN}" "${REPO}/scripts/evaluate_floras_live_s2s.py" \
      --manifest "${MANIFEST}" \
      --run-output-dir "${OUT_BASE}/gemini_live_full_enzh_chunk${chunk}_trim" \
      --output-dir "${OUT_BASE}/gemini_eval_full_enzh_chunk${chunk}_trim_asr" \
      --asr-jsonl "${OUT_BASE}/gemini_asr_full_enzh_chunk${chunk}_trim.jsonl" \
      --target-context-s 20 \
      --coverage-judge none
done

run_step "combined dashboard" \
  "${PYTHON_BIN}" "${REPO}/scripts/render_floras_compare_dashboard.py" \
    --output-dir "${OUT_BASE}/compare_openai_gemini_enzh_full_chunks" \
    --title "FLORAS en-zh full Live Compare: OpenAI vs Gemini, 0.96s vs 1.92s chunks" \
    --run-id-prefix en-zh_ \
    --eval "openai_960=${OUT_BASE}/openai_eval_full_enzh_chunk960_asr" \
    --eval "gemini_960=${OUT_BASE}/gemini_eval_full_enzh_chunk960_trim_asr" \
    --eval "openai_1920=${OUT_BASE}/openai_eval_full_enzh_chunk1920_asr" \
    --eval "gemini_1920=${OUT_BASE}/gemini_eval_full_enzh_chunk1920_trim_asr"
