"""Push the v4 / v3-control checkpoints from Tilde to Hugging Face.

# note (luojiaxuan): Tilde 登录节点和 hyper 集群互相不可达，按全局规则artifact
# 经 HF Hub 中转，不走本地 Mac。上传后在 hyper 上 snapshot_download 即可评测。
"""
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path("/home/guests/zhen/s2s_omni_v4")
TARGETS = [
    ("v4", "gavinlaw/moss-tts-realtime-infinisst-en-zh-v4-selfhist"),
    ("v3ctl", "gavinlaw/moss-tts-realtime-infinisst-en-zh-v3-control-tilde"),
]

api = HfApi()
out = {}
for tag, repo in TARGETS:
    base = ROOT / f"runs/ckpt_{tag}"
    cands = sorted(base.glob("checkpoint-epoch-*")) or ([base] if (base / "model.safetensors").exists() else [])
    if not cands:
        print(f"SKIP {tag}: no checkpoint under {base}", flush=True)
        continue
    src = cands[-1]
    api.create_repo(repo, private=True, exist_ok=True)
    info = api.upload_folder(folder_path=str(src), repo_id=repo, commit_message=f"{tag} checkpoint from Tilde")
    rev = getattr(info, "oid", None) or str(info)
    print(f"UPLOADED {tag} {repo} {rev}", flush=True)
    out[tag] = {"repo": repo, "revision": rev, "src": str(src)}

(ROOT / "runs/upload_manifest.json").write_text(json.dumps(out, indent=1))
print("UPLOAD_DONE", json.dumps(out))
