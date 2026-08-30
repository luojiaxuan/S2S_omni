#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REQUIREMENTS = [
    "accelerate>=1.10.1",
    "einops==0.8.1",
    "huggingface_hub[hf_xet]",
    "librosa==0.11.0",
    "numpy==2.1.0",
    "orjson==3.11.4",
    "packaging",
    "psutil",
    "PyYAML==6.0.3",
    "safetensors==0.6.2",
    "scipy==1.16.2",
    "tiktoken==0.12.0",
    "transformers==5.0.0",
    "zstandard",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def install_dependencies(task_root: Path) -> Path:
    site_packages = task_root / "env" / "site"
    site_packages.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256("\n".join(REQUIREMENTS).encode()).hexdigest()
    marker = task_root / "env" / "requirements.sha256"
    if not marker.is_file() or marker.read_text().strip() != fingerprint:
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--target",
                str(site_packages),
                *REQUIREMENTS,
            ]
        )
        marker.write_text(fingerprint + "\n")
    return site_packages


def prepare_data(task_root: Path, config: dict, env: dict[str, str]) -> Path:
    from huggingface_hub import snapshot_download
    import zstandard

    data_dir = task_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "input.jsonl"
    if output_path.is_file():
        return output_path

    source = Path(
        snapshot_download(
            repo_id=config["data"]["repo"],
            repo_type="dataset",
            revision=config["data"]["revision"],
            allow_patterns=config["data"]["allow_patterns"],
        )
    )
    clean_path = data_dir / "clean.jsonl"
    stats_path = data_dir / "clean_stats.json"
    run(
        [
            sys.executable,
            str(task_root / "code" / "prepare.py"),
            "--trajectory-tsv",
            str(source / config["data"]["trajectory_tsv"]),
            "--prepared-jsonl",
            str(source / config["data"]["prepared_jsonl"]),
            "--output-jsonl",
            str(clean_path),
            "--stats-json",
            str(stats_path),
        ],
        env=env,
    )

    base_path = data_dir / "base.jsonl"
    with (source / config["data"]["base_jsonl_zst"]).open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            with base_path.open("wb") as sink:
                shutil.copyfileobj(reader, sink)
    run(
        [
            sys.executable,
            str(task_root / "code" / "augment.py"),
            "--clean-jsonl",
            str(clean_path),
            "--base-jsonl",
            str(base_path),
            str(clean_path),
            "--output-jsonl",
            str(output_path),
            "--fraction",
            "1.0",
            "--min-remaining-turns",
            "4",
            "--max-drop-frac",
            "0.5",
            "--id-suffix",
            "_phrase_mid",
            "--seed",
            "23",
        ],
        env=env,
    )
    row_count = sum(1 for _ in output_path.open(encoding="utf-8"))
    if row_count != config["data"]["expected_rows"]:
        raise RuntimeError(f"expected {config['data']['expected_rows']} rows, got {row_count}")
    return output_path


def prepare_model(task_root: Path, config: dict) -> Path:
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=config["model"]["repo"],
            revision=config["model"]["revision"],
        )
    )
    asset_dir = task_root / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    link = asset_dir / "base"
    if not link.exists():
        link.symlink_to(snapshot, target_is_directory=True)
    return link


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    task_root = Path(__file__).resolve().parent.parent
    for name in ("assets", "cache", "checkpoints", "control", "data", "env", "logs", "outputs", "tmp"):
        (task_root / name).mkdir(parents=True, exist_ok=True)

    os.environ["XDG_CACHE_HOME"] = str(task_root / "cache" / "xdg")
    os.environ["PIP_CACHE_DIR"] = str(task_root / "cache" / "pip")
    os.environ["TMPDIR"] = str(task_root / "tmp")
    site_packages = install_dependencies(task_root)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(site_packages), str(task_root / "code" / "moss"), env.get("PYTHONPATH", "")]
    )
    sys.path.insert(0, str(site_packages))

    train_data = prepare_data(task_root, config, env)
    base_model = prepare_model(task_root, config)
    training = config["training"]
    run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(training["num_processes"]),
            str(task_root / "code" / "moss" / "moss_tts_realtime" / "finetuning" / "sft.py"),
            "--model-path",
            str(base_model),
            "--codec-path",
            str(task_root / "assets" / "codec"),
            "--train-jsonl",
            str(train_data),
            "--output-dir",
            str(task_root / "outputs" / "model"),
            "--per-device-batch-size",
            str(training["per_device_batch_size"]),
            "--gradient-accumulation-steps",
            str(training["gradient_accumulation_steps"]),
            "--learning-rate",
            str(training["learning_rate"]),
            "--num-epochs",
            str(training["num_epochs"]),
            "--num-workers",
            str(training["num_workers"]),
            "--mixed-precision",
            training["mixed_precision"],
            "--attn-implementation",
            training["attn_implementation"],
            "--seed",
            str(training["seed"]),
        ],
        env=env,
    )
    (task_root / "control" / "training_done").touch()
    while not (task_root / "control" / "stop").exists():
        time.sleep(30)


if __name__ == "__main__":
    main()
