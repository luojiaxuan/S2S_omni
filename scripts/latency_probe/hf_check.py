import os
os.environ["HF_TOKEN"] = open("/data04/jaxan/.keys/hf_token_gavinlaw").read().strip()
from huggingface_hub import HfApi
api = HfApi()
print("whoami:", api.whoami()["name"])
for rev in ("521e09fa", "main"):
    try:
        info = api.repo_info("gavinlaw/moss-tts-realtime-infinisst-en-zh-v8-phrase",
                             revision=rev, files_metadata=True)
    except Exception as exc:
        print(f"{rev}: {type(exc).__name__}: {str(exc)[:200]}")
        continue
    print(f"--- revision {rev} -> {info.sha[:12]}  last_modified={info.lastModified}")
    for s in info.siblings:
        if s.size:
            print(f"    {s.rfilename:<28} {s.size:>13,}  lfs_sha256={(s.lfs.get('sha256')[:16] if s.lfs else '-')}")
