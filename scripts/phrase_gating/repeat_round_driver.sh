#!/usr/bin/env bash
# note (luojiaxuan): Mac-side driver for a repeat round of the three-arm cascade comparison on hyper01 — the same
# note (luojiaxuan): three dev talks, the same env, one more sampled run per thinker — so the run-to-run spread
# note (luojiaxuan): behind the single-run margins (research_log (八)) is measured instead of assumed. Arms run one
# note (luojiaxuan): after another in one container (hyper01_phrase_eval.sh up/cascade/score with env overrides),
# note (luojiaxuan): the container is removed at the end. Each stage appends "STAGE <arm>/<step> OK|FAIL" to $STATUS.
set -uo pipefail
STATUS="${STATUS:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
SAB=/data04/jaxan/serving_ab
PGD=$SAB/phrase_gated_data
O="-o RemoteCommand=none -o ConnectTimeout=30 -o ServerAliveInterval=60 -o ServerAliveCountMax=10"
SSH="ssh $O"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$STATUS"; }
ok() { say "STAGE $1 OK $2"; }
fail() { say "STAGE $1 FAIL $2"; exit 1; }
E="bash $PGD/hyper01_phrase_eval.sh"
scp -q "$HERE/hyper01_phrase_eval.sh" hyper01:$PGD/ || fail setup "scp"
THEIRS_SHA8=$($SSH hyper01 "grep -a 'owaski/infinisst-thinker-phrase-zh' $PGD/dl_thinkers.log | awk '{print \$2}'")
[ -n "$THEIRS_SHA8" ] || fail setup "no resolved revision for the owaski thinker in dl_thinkers.log"
# note (luojiaxuan): arm = TAG JOB SJOB CKPT_DIR REV_NAME SHA8 ; jobs 9001x keep the round apart from the first
# note (luojiaxuan): runs (90001-90003) in the results tree and in the ASR response cache.
ARMS=(
  "theirs2 90011 95011 /data/serving_ab/thinkers/theirs owaski-infinisst-thinker-phrase-zh $THEIRS_SHA8"
  "word2 90012 95012 /data/serving_ab/thinkers/ours_word gavinlaw-infinisst-no-tmsft-origin-bsz4-zh fd0a5c8f"
  "phrase2 90013 95013 /data/serving_ab/thinker_phrase_gated gavinlaw-infinisst-thinker-phrase-gated-zh 83a95f5b"
)
first=1
for arm in "${ARMS[@]}"; do
  read -r TAG JOB SJOB CKPT REVN SHA8 <<<"$arm"
  ENV="TAG=$TAG JOB=$JOB SJOB=$SJOB CKPT_DIR=$CKPT REV_NAME=$REVN SHA8=$SHA8"
  if [ $first = 1 ]; then
    up=$($SSH hyper01 "MAP_DESC='OLT cascade repeat round (3 dev docs; theirs/word/phrase thinkers, jobs 90011-90013) + scoring' $ENV $E up $SHA8" 2>&1) || fail "$TAG/up" "$up"
    say "container: $(echo $up)"; first=0
  fi
  c=$($SSH hyper01 "$ENV $E cascade" 2>&1 | tr '\n' ' ')
  case "$c" in *"CASCADE_${TAG}_EXIT=0"*) ok "$TAG/cascade" "$c" ;; *) fail "$TAG/cascade" "$c (hyper01 $PGD/cascade_$TAG.log)" ;; esac
  cmp=$($SSH hyper01 "$ENV $E compare" 2>&1); say "$TAG identity vs 90002: $(echo "$cmp" | tail -1)"
  sc=$($SSH hyper01 "$ENV $E score" 2>&1); say "$sc"
  echo "$sc" | grep -q "SCORE_${TAG}_EXIT=0" || fail "$TAG/score" "see hyper01 $PGD/score_$TAG.log"
  ok "$TAG/score" "metrics.json written"
done
d=$($SSH hyper01 "$E down" 2>&1); ok down "$d"
say "ROUND DONE"
