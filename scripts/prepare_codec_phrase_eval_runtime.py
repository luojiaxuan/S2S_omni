#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path


REQUIREMENTS = [
    "Cython>=3.0",
    "sacrebleu[ja]>=2.5",
    "sentence-transformers>=3.0",
    "soundfile>=0.12",
    "spacy>=3.7",
]


def run(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def replace_symlink(path: Path, target: Path) -> None:
    if path.is_symlink() and path.resolve() == target.resolve():
        return
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.symlink_to(target, target_is_directory=True)


def main() -> None:
    task_root = Path(__file__).resolve().parents[2]
    eval_root = task_root / "eval"
    config = json.loads((task_root / "code" / "config.json").read_text())
    resources = config["evaluation"]["resources"]
    site_packages = task_root / "env" / "site"
    fingerprint = hashlib.sha256("\n".join(REQUIREMENTS).encode()).hexdigest()
    marker = task_root / "env" / "evaluation_requirements.sha256"
    if not marker.is_file() or marker.read_text().strip() != fingerprint:
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--no-deps",
                "--target",
                str(site_packages),
                *REQUIREMENTS,
            ]
        )
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--no-deps",
                "--target",
                str(site_packages),
                str(eval_root / "resources" / "r0" / "SEGALE"),
            ]
        )
        marker.write_text(fingerprint + "\n")

    sys.path.insert(0, str(site_packages))
    from huggingface_hub import snapshot_download

    token_path = Path("/data/.secrets/hf_token")
    token = token_path.read_text().strip() if token_path.is_file() else None
    dataset = Path(
        snapshot_download(
            repo_id=resources["dataset_repo"],
            repo_type="dataset",
            revision=resources["dataset_revision"],
            token=token,
        )
    )
    cache_root = Path("/data/hf-cache/hub")
    codec = next(
        (cache_root / "models--OpenMOSS-Team--MOSS-Audio-Tokenizer" / "snapshots").glob(
            resources["codec_revision"]
        )
    )
    asr = next(
        (cache_root / "models--Qwen--Qwen3-ASR-1.7B" / "snapshots").glob(
            resources["asr_revision"]
        )
    )
    replace_symlink(eval_root / "resources" / "d0", dataset)
    replace_symlink(eval_root / "resources" / "c0", codec)
    replace_symlink(eval_root / "resources" / "a0", asr)

    distributions = [
        "Cython",
        "sacrebleu",
        "sentence-transformers",
        "soundfile",
        "spacy",
        "vecalign",
    ]
    versions = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    (eval_root / "runtime_versions.json").write_text(
        json.dumps(versions, indent=2) + "\n", encoding="utf-8"
    )
    if any(value is None for value in versions.values()):
        raise RuntimeError(f"missing evaluation packages: {versions}")
    print(json.dumps(versions, indent=2), flush=True)


if __name__ == "__main__":
    main()
