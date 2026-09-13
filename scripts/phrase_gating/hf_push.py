"""Push the two corrected-loss thinker exports to Hugging Face and verify them by content.

Each export becomes branch `empty-turn-end-w0.5` of the model repo that already holds the same
arm trained under the default loss scale, so the two losses are revisions of one model rather
than duplicate repos. Arms go one after the other because the account's upload quota is shared.
Runs on the hyper01 host; progress lines go to hf_push.status (START / OK / FAIL, then DONE).
"""
import hashlib
import os
import sys
import time

from huggingface_hub import HfApi

BRANCH = "empty-turn-end-w0.5"
ROOT = "/data04/jaxan/phrase_sft2"
STATUS = f"{ROOT}/hf_push.status"
SOURCE = "luojiaxuan/S2S_omni@148c0da"
ARMS = {
    "phrase": {
        "repo": "gavinlaw/infinisst-thinker-phrase-gated-zh",
        "corpus": "phrase-gated trajectories (`train_s_zh_phrase_ours`, dataset "
                  "`gavinlaw/infinisst-sft-phrase-gated-zh`), 29.3% empty turns",
        "bleu": 42.53,
        "xcomet": 0.7626,
    },
    "word": {
        "repo": "gavinlaw/infinisst-no-tmsft-origin-bsz4-zh",
        "corpus": "word-aligned trajectories (`train_s_zh_origin`), 10.9% empty turns",
        "bleu": 40.01,
        "xcomet": 0.7189,
    },
}

CARD = """# {repo} @ {branch}

Merged bf16 export of `Qwen/Qwen3-Omni-30B-A3B-Instruct` with a LoRA trained on {corpus},
under the turn-end loss scale `empty_turn_end_w0.5`: weight 0.5 on the end token of every
empty assistant turn, which the default loss scale leaves with no supervised token. `main`
holds the same arm trained under the default loss scale.

- Recipe: LoRA r32 / alpha 32 on all linear layers, packing, max_length 2048, global batch 4,
  lr 1e-4, one epoch, Megatron backend (ms-swift 3.9.1), 4x H200.
- Training and export: `scripts/phrase_gating/megatron_hyper01.sh` in {source}
  (ARM={arm}, LOSS=empty_turn_end_w0.5).
- Evaluation, ACL 60-60 dev (5 talks), CU, greedy thinker and greedy TTS: BLEU {bleu},
  XCOMET-XL {xcomet}. The 2x2 against the default-loss checkpoints is `artifacts/ab_2x2/`
  in the same repository.
"""


def say(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def local_digest(path, lfs):
    # note (luojiaxuan): HF records sha256 for LFS files and the git blob sha1 for the rest.
    h = hashlib.sha256() if lfs else hashlib.sha1(b"blob %d\0" % os.path.getsize(path))
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


api = HfApi()
for arm, spec in ARMS.items():
    repo = spec["repo"]
    local = f"{ROOT}/run_{arm}_empty_turn_end_w0.5/hf"
    api.create_branch(repo, branch=BRANCH, exist_ok=True)
    say(f"START {arm} -> {repo}@{BRANCH}")
    # note (luojiaxuan): the branch starts as a copy of main, whose file set differs from this export
    # note (luojiaxuan): (the word repo's main holds 30 files, 17 of them absent here). Without deletions
    # note (luojiaxuan): the branch would carry both weight sets and a snapshot download would pull both;
    # note (luojiaxuan): delete_patterns="*" makes the commit mirror the export.
    api.upload_folder(repo_id=repo, folder_path=local, revision=BRANCH, delete_patterns="*",
                      commit_message=f"Retrain under empty_turn_end_w0.5 ({SOURCE})")

    # note (luojiaxuan): content, not size. main holds the same architecture exported the same way
    # note (luojiaxuan): (on the phrase repo 25 of 26 files match it by name and byte count), so a size
    # note (luojiaxuan): check passes while the branch still carries the default-loss weights.
    # note (luojiaxuan): The sha reported below is the verified branch head, which exists whether or not
    # note (luojiaxuan): this run's upload had anything new to commit.
    info = api.model_info(repo, revision=BRANCH, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}
    local_names = os.listdir(local)
    bad = [f"{name}: not in the export"
           for name in sorted(set(remote) - set(local_names) - {".gitattributes", "README.md"})]
    for name in sorted(local_names):
        s = remote.get(name)
        if s is None:
            bad.append(f"{name}: absent")
        elif local_digest(os.path.join(local, name), s.lfs is not None) != (s.lfs.sha256 if s.lfs else s.blob_id):
            bad.append(f"{name}: content differs")
    if bad:
        say(f"FAIL {arm}: {len(bad)} problems, first {bad[:4]}")
        sys.exit(1)

    card = CARD.format(branch=BRANCH, arm=arm, source=SOURCE, **spec)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo,
                    revision=BRANCH, commit_message="Model card for the empty_turn_end_w0.5 revision")
    say(f"OK {arm}: {len(local_names)} files content-verified, nothing extra, at {repo}@{BRANCH} ({info.sha[:12]})")
say("HF PUSH DONE")
