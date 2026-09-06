# note (luojiaxuan): move the eval stack hyper01 -> aries local NVMe. The stack (olt/venvs/tts/checkpoints/codec)
# note (luojiaxuan): was written inside containers as root, so the host user cannot rsync it; a root container tars
# note (luojiaxuan): it with world-read first. Thinkers are host-owned and rsync directly. No image is moved -- aries
# note (luojiaxuan): already has jaxanluo/sglang-omni:dev (py3.12.3), and the venvs' cu12 wheels run there (tested).
set -uo pipefail
A=jiaxuanluo@aries.cs.ucsb.edu
D3=/mnt/data3/jiaxuanluo/serving_ab
D4=/mnt/data4/jiaxuanluo/serving_ab_thinkers
S=/data04/jaxan/serving_ab
SSHO="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=60 -o ServerAliveCountMax=20"
say() { echo "$(date -u +%FT%TZ) $*"; }
R() { rsync -a --partial --inplace -e "ssh $SSHO" "$@"; }
rm -f /data04/jaxan/stack.tar /data04/jaxan/hf.tar
docker run --rm -v /data04/jaxan:/data -v /data04/cache:/cache alpine sh -c '
  cd /data/serving_ab && tar cf /data/stack.tar olt venvs tts checkpoints pyshim asr_responses phrase_gated_data && chmod 644 /data/stack.tar
  cd /cache/huggingface/hub && tar cf /data/hf.tar models--OpenMOSS-Team--MOSS-Audio-Tokenizer models--sentence-transformers--LaBSE && chmod 644 /data/hf.tar
' && say "tarballs: $(ls -la /data04/jaxan/stack.tar /data04/jaxan/hf.tar | awk '{print $5}' | tr '\n' ' ')" || { say FAIL tar; exit 1; }
ssh -n $SSHO $A "mkdir -p $D3/hf_home/hub $D4" || { say FAIL mkdir; exit 1; }
R /data04/jaxan/stack.tar /data04/jaxan/hf.tar $A:$D3/ && say "tars rsynced" || { say FAIL rsync-tar; exit 1; }
ssh -n $SSHO $A "cd $D3 && tar xf stack.tar && tar xf hf.tar -C hf_home/hub && rm -f stack.tar hf.tar" && say "extracted on aries" || { say FAIL extract; exit 1; }
rm -f /data04/jaxan/stack.tar /data04/jaxan/hf.tar
R $S/thinkers/theirs/ $A:$D3/thinkers/theirs/ && say "thinker theirs OK $(ssh -n $SSHO $A du -sh $D3/thinkers/theirs | cut -f1)" || { say FAIL theirs; exit 1; }
R $S/thinker_phrase_gated/ $A:$D3/thinkers/phrase/ && say "thinker phrase OK $(ssh -n $SSHO $A du -sh $D3/thinkers/phrase | cut -f1)" || { say FAIL phrase; exit 1; }
R $S/thinkers/ours_word/ $A:$D4/ours_word/ && say "thinker ours_word OK $(ssh -n $SSHO $A du -sh $D4/ours_word | cut -f1)" || { say FAIL ours_word; exit 1; }
ssh -n $SSHO $A "rsync -a /mnt/gemini/home/jiaxuanluo/serving_ab/acl6060_full/ $D3/acl6060_full/ && find $D3/acl6060_full \( -name '._*' -o -name '.DS_Store' \) -delete" && say "acl6060_full OK" || { say FAIL acl; exit 1; }
say "TO_ARIES_LOCAL_DONE"
