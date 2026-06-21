#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p work

python3 scripts/make_stress_manifest.py \
  --input data/examples/seed_segments.jsonl \
  --output work/stress_manifest.jsonl \
  --speed-factors 1.0,1.35,1.7

python3 scripts/make_sft_jsonl.py \
  --input work/stress_manifest.jsonl \
  --output work/sft.jsonl

python3 scripts/evaluate_predictions.py \
  --gold work/stress_manifest.jsonl \
  --pred data/examples/seed_segments.jsonl \
  --output work/eval.jsonl \
  --summary work/eval_summary.json

python3 scripts/train_text_lora_sft.py \
  --config configs/qwen3_omni_lora_sft.yaml \
  --dry-run

echo "smoke ok"
