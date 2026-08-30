#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


RESOLVED_REQUIREMENTS = [
    "numpy==1.26.4",
    "sacrebleu==2.6.0",
    "soundfile==0.14.0",
    "spacy==3.8.16",
    "spacy-pkuseg==0.0.33",
    "https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.8.0/zh_core_web_sm-3.8.0-py3-none-any.whl",
]
NO_DEPS_REQUIREMENTS = [
    "Cython==3.3.0",
    "sentence-transformers==6.0.0",
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
    site_packages = task_root / "env" / "eval-site"
    segale_source = eval_root / "resources" / "r0" / "SEGALE"
    fingerprint_inputs = [
        *RESOLVED_REQUIREMENTS,
        *NO_DEPS_REQUIREMENTS,
        resources["speech_latency_revision"],
    ]
    fingerprint = hashlib.sha256("\n".join(fingerprint_inputs).encode()).hexdigest()
    marker = task_root / "env" / "evaluation_requirements.sha256"
    if (
        not site_packages.is_dir()
        or not marker.is_file()
        or marker.read_text().strip() != fingerprint
    ):
        if site_packages.exists():
            shutil.rmtree(site_packages)
        site_packages.mkdir(parents=True)
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--target",
                str(site_packages),
                *RESOLVED_REQUIREMENTS,
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
                *NO_DEPS_REQUIREMENTS,
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
                str(segale_source),
            ]
        )
        dp_cores = list((segale_source / "vecalign").glob("dp_core*.so"))
        if len(dp_cores) != 1:
            raise RuntimeError(f"expected one compiled vecalign dp_core, got {dp_cores}")
        shutil.copy2(dp_cores[0], site_packages / "vecalign" / dp_cores[0].name)

    sys.path.insert(0, str(site_packages))
    sys.path.insert(0, str(segale_source))
    os.environ["PATH"] = os.pathsep.join(
        [str(site_packages / "bin"), os.environ.get("PATH", "")]
    )
    import numpy
    import portalocker
    import spacy
    import spacy_pkuseg
    import zh_core_web_sm
    from vecalign.dp_core import dense_dp

    del dense_dp, portalocker, spacy_pkuseg
    zh_core_web_sm.load()
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
    codec = Path(
        snapshot_download(
            repo_id=resources["codec_repo"],
            revision=resources["codec_revision"],
            token=token,
        )
    )
    asr = Path(
        snapshot_download(
            repo_id=resources["asr_repo"],
            revision=resources["asr_revision"],
            token=token,
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
        "spacy-pkuseg",
        "zh-core-web-sm",
    ]
    versions = {"numpy": numpy.__version__}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    versions["vecalign"] = (
        "source" if importlib.util.find_spec("vecalign") is not None else None
    )
    (eval_root / "runtime_versions.json").write_text(
        json.dumps(versions, indent=2) + "\n", encoding="utf-8"
    )
    if any(value is None for value in versions.values()):
        raise RuntimeError(f"missing evaluation packages: {versions}")
    marker.write_text(fingerprint + "\n")
    print(json.dumps(versions, indent=2), flush=True)


if __name__ == "__main__":
    main()
