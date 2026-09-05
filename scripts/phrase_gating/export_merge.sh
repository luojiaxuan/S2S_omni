#!/usr/bin/env bash
# Merge the phrase-gated LoRA into the base and export a full checkpoint the OLT cascade
# can serve through vLLM, the same way owaski/infinisst-thinker-phrase-zh is served.
# Usage: export_merge.sh [checkpoint-dir]   (defaults to the last checkpoint of out_full)
set -x
W=/data/phrase_sft
source /dev/null
export TMPDIR="$W/tmp" XDG_CACHE_HOME="$W/cache" HF_HOME=/root/.cache/huggingface
CKPT="${1:-$(ls -dt "$W"/out_full/*/checkpoint-* 2>/dev/null | head -1)}"
[ -n "$CKPT" ] || { echo "no checkpoint found under $W/out_full"; exit 1; }
OUT=$W/thinker_phrase_merged
rm -rf "$OUT"
"$W/env/bin/swift" export \
  --adapters "$CKPT" \
  --merge_lora true \
  --output_dir "$OUT"
echo "EXPORT_EXIT=$?"
ls -la "$OUT" | head -5
du -sh "$OUT"
