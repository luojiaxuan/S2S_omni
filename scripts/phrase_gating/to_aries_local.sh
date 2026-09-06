# note (luojiaxuan): push the eval stack from hyper01 to aries LOCAL NVMe (gemini NFS reads at ~12 MB/s, far too
# note (luojiaxuan): slow to reload a 60 GB thinker per run). Support dirs + two thinkers go to /mnt/data3, the
# note (luojiaxuan): third thinker to /mnt/data4. The container mounts these so venv absolute paths (/data/serving_ab/*)
# note (luojiaxuan): resolve. Idempotent: rsync skips what already matches, so this is safe to re-run.
set -uo pipefail
A=jiaxuanluo@aries.cs.ucsb.edu
D3=/mnt/data3/jiaxuanluo/serving_ab
D4=/mnt/data4/jiaxuanluo/serving_ab_thinkers
S=/data04/jaxan/serving_ab
SSHO="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=60 -o ServerAliveCountMax=10"
R() { rsync -a --partial --inplace -e "ssh $SSHO" "$@"; }
say() { echo "$(date -u +%FT%TZ) $*"; }
ssh -n $SSHO $A "mkdir -p $D3/hf_home/hub $D3/thinkers $D4" || { say "FAIL mkdir"; exit 1; }
say "image: docker save|load"
if ssh -n $SSHO $A 'docker images --format "{{.Repository}}:{{.Tag}}" | grep -qx vllm-omni:dev'; then
  say "image OK: already on aries"
else
  docker save vllm-omni:dev | ssh $SSHO $A "docker load" && say "image OK" || say "FAIL image"
fi
for d in olt venvs checkpoints tts pyshim asr_responses phrase_gated_data acl6060_full; do
  src=$S/$d; [ "$d" = acl6060_full ] && src=/mnt/gemini/home/jiaxuanluo/serving_ab/acl6060_full
  ssh -n $SSHO $A "test -d $src" 2>/dev/null || true
  R "$src/" $A:$D3/$d/ && say "support OK: $d" || say "FAIL support: $d"
done
for h in models--OpenMOSS-Team--MOSS-Audio-Tokenizer models--sentence-transformers--LaBSE; do
  R /data04/cache/huggingface/hub/$h/ $A:$D3/hf_home/hub/$h/ && say "hf OK: $h" || say "FAIL hf: $h"
done
R $S/thinkers/theirs/ $A:$D3/thinkers/theirs/ && say "thinker OK: theirs $(ssh -n $SSHO $A du -sh $D3/thinkers/theirs | cut -f1)" || say "FAIL thinker theirs"
R $S/thinker_phrase_gated/ $A:$D3/thinkers/phrase/ && say "thinker OK: phrase $(ssh -n $SSHO $A du -sh $D3/thinkers/phrase | cut -f1)" || say "FAIL thinker phrase"
R $S/thinkers/ours_word/ $A:$D4/ours_word/ && say "thinker OK: ours_word $(ssh -n $SSHO $A du -sh $D4/ours_word | cut -f1)" || say "FAIL thinker ours_word"
say "TO_ARIES_LOCAL_DONE"
