# note (luojiaxuan): download the three thinkers from HF onto aries local NVMe (hyper01->aries rsync stalled;
# note (luojiaxuan): aries pulls HF at ~20 MB/s, faster and independent). Runs detached in the jaxanluo image
# note (luojiaxuan): (has the olt-thinker venv with huggingface_hub). Writes TO_ARIES_LOCAL_DONE into the shared
# note (luojiaxuan): transfer log the driver waits on, so the pipeline resumes automatically.
set -e
D3=/mnt/data3/jiaxuanluo/serving_ab
D4=/mnt/data4/jiaxuanluo/serving_ab_thinkers
export HF_TOKEN=$(cat /root/.keys/hf_token_gavinlaw)
PY=$D3/venvs/olt-thinker/bin/python
$PY - <<'PYEOF'
import os, time
from huggingface_hub import snapshot_download
jobs = [
    ("owaski/infinisst-thinker-phrase-zh", "7d29be87", "/mnt/data3/jiaxuanluo/serving_ab/thinkers/theirs"),
    ("gavinlaw/infinisst-thinker-phrase-gated-zh", "83a95f5b", "/mnt/data3/jiaxuanluo/serving_ab/thinkers/phrase"),
    ("gavinlaw/infinisst-no-tmsft-origin-bsz4-zh", "fd0a5c8f", "/mnt/data4/jiaxuanluo/serving_ab_thinkers/ours_word"),
]
for repo, rev, dst in jobs:
    t = time.time()
    snapshot_download(repo, revision=rev, local_dir=dst, max_workers=8)
    print(f"THINKER_OK {repo}@{rev} -> {dst} {time.time()-t:.0f}s", flush=True)
print("ALL_THINKERS_OK", flush=True)
PYEOF
