#!/usr/bin/env bash
# note (luojiaxuan): copy one 2x2 cell's artifacts off hyper01 into artifacts/ab_2x2/<CELL>/.
# note (luojiaxuan): metrics.json under serving_ab/results is written 0600 root, so the host user cannot read it and
# note (luojiaxuan): scp fails; a throwaway root container mounting /data04/jaxan reads it whether or not the
# note (luojiaxuan): evaluation container still exists. eval_relay.sh removes that container as soon as both cells
# note (luojiaxuan): score, so nothing here may depend on it being up.
# note (luojiaxuan):   pull_cell.sh W_fixed 93011
# note (luojiaxuan): --entrypoint /bin/cat is required: the image's entrypoint prints a CUDA banner to stdout, which
# note (luojiaxuan): a plain `docker run ... cat` prepends to the captured bytes (10796 instead of 10049).
# note (luojiaxuan): Each file lands in a temp copy that must parse as JSON before it replaces the archived one:
# note (luojiaxuan): a failed docker run still returns 0 bytes through the redirect, which would silently blank a
# note (luojiaxuan): good archive copy.
set -euo pipefail
CELL="${1:?cell name, e.g. W_fixed}"
JOB="${2:?slurm job id, e.g. 93011}"
HOST="${HOST:-hyper01}"
IMG="${IMG:-jaxanluo/sglang-omni:dev}"
RUN=/data/serving_ab/results/s2st_moss-delta_dev_1920ms_$JOB
DEST=$(cd "$(dirname "$0")/../.." && pwd)/artifacts/ab_2x2/$CELL
mkdir -p "$DEST"

for f in metrics.json render_report.json generation_config.json; do
  tmp=$(mktemp)
  ssh -o ConnectTimeout=30 "$HOST" \
    "docker run --rm --entrypoint /bin/cat -v /data04/jaxan:/data $IMG $RUN/$f" > "$tmp" 2>/dev/null || true
  if [ -s "$tmp" ] && python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$tmp" 2>/dev/null; then
    mv "$tmp" "$DEST/$f"
    echo "OK   $CELL/$f ($(wc -c < "$DEST/$f") bytes)"
  else
    rm -f "$tmp"
    echo "MISS $CELL/$f (absent or unreadable; archive copy left untouched)"
  fi
done
