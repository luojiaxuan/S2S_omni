"""Upload the exported phrase-gated thinker (merged bf16 HF weights + the Megatron LoRA
adapter) to a private HF model repo, the training jsonl to a private dataset repo, and
reconcile the file lists so a silently skipped LFS upload cannot pass as done."""
import argparse
import json
import pathlib

from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--hf", required=True, help="merged bf16 export directory")
ap.add_argument("--mcore", required=True, help="Megatron save dir (LoRA adapter iterations + args.json)")
ap.add_argument("--card", required=True, help="README.md for the model repo")
ap.add_argument("--data-repo", required=True)
ap.add_argument("--data", required=True, help="the phrase-gated training jsonl")
ap.add_argument("--data-card", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
api = HfApi()


def local_files(root, prefix=""):
    root = pathlib.Path(root)
    return sorted(prefix + str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


api.create_repo(a.repo, repo_type="model", private=True, exist_ok=True)
api.upload_folder(folder_path=a.hf, repo_id=a.repo, repo_type="model",
                  commit_message="merged bf16 export of the phrase-gated LoRA (OLT Megatron recipe)")
api.upload_folder(folder_path=a.mcore, repo_id=a.repo, repo_type="model", path_in_repo="mcore_lora",
                  commit_message="Megatron LoRA adapter iterations, args.json, train logging")
api.upload_file(path_or_fileobj=a.card, path_in_repo="README.md", repo_id=a.repo, repo_type="model",
                commit_message="model card")
want = local_files(a.hf) + local_files(a.mcore, "mcore_lora/")
have = set(api.list_repo_files(a.repo, repo_type="model"))
missing = sorted(set(want) - have)
sha = api.list_repo_commits(a.repo, repo_type="model")[0].commit_id

api.create_repo(a.data_repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_file(path_or_fileobj=a.data, path_in_repo=pathlib.Path(a.data).name, repo_id=a.data_repo,
                repo_type="dataset", commit_message="phrase-gated en->zh SFT trajectories")
api.upload_file(path_or_fileobj=a.data_card, path_in_repo="README.md", repo_id=a.data_repo,
                repo_type="dataset", commit_message="dataset card")
data_have = set(api.list_repo_files(a.data_repo, repo_type="dataset"))
data_sha = api.list_repo_commits(a.data_repo, repo_type="dataset")[0].commit_id

result = {"repo": a.repo, "sha": sha, "n_local": len(want), "n_remote": len(have), "missing": missing,
          "data_repo": a.data_repo, "data_sha": data_sha,
          "data_ok": pathlib.Path(a.data).name in data_have}
json.dump(result, open(a.out, "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: (len(v) if k == "missing" else v) for k, v in result.items()}, ensure_ascii=False))
