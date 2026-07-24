#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

PINNED_SPEECH_LATENCY_REVISION = "d0041438abf097a1ec3055e7f09656ad6302f672"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SEGALE alignment for one ACL6060 run.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--speech-latency-repo", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-lang", choices=["zh", "de", "ja"], default="")
    parser.add_argument("--segmenter", choices=["spacy", "ersatz"], default="spacy")
    parser.add_argument("--embedding-model", default="sentence-transformers/LaBSE")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="count", default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected-revision", default=PINNED_SPEECH_LATENCY_REVISION)
    return parser.parse_args()


def git_revision(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def input_fingerprint(output_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in ["ref.jsonl", "hyp.jsonl"]:
        digest.update(name.encode("utf-8"))
        digest.update((output_dir / name).read_bytes())
    return digest.hexdigest()


def load_segale_module(repo: Path) -> ModuleType:
    module_path = repo / "src" / "alignment" / "segale.py"
    if not module_path.exists():
        raise FileNotFoundError(module_path)
    sys.path.insert(0, str(repo / "SEGALE"))
    spec = importlib.util.spec_from_file_location("acl6060_speech_latency_segale", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_alignment(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir or (args.run_dir / "segale_alignment")
    input_summary_path = output_dir / "input_summary.json"
    if not input_summary_path.exists():
        raise FileNotFoundError(
            f"{input_summary_path} is missing; run build_acl6060_segale_inputs.py first"
        )
    input_summary = json.loads(input_summary_path.read_text(encoding="utf-8"))
    target_lang = args.target_lang or str(input_summary["target_lang"])
    fingerprint = input_fingerprint(output_dir)
    aligned_path = output_dir / "hyp" / f"aligned_{args.segmenter}_hyp.jsonl"
    summary_path = output_dir / "alignment_summary.json"
    if args.resume and aligned_path.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("input_fingerprint") == fingerprint:
            return summary

    revision = git_revision(args.speech_latency_repo)
    if args.expected_revision and revision != args.expected_revision:
        raise RuntimeError(
            f"Speech-to-Speech-Latency revision mismatch: {revision} != {args.expected_revision}"
        )
    if shutil.which("vecalign") is None:
        raise FileNotFoundError(
            "vecalign is not installed; install <speech-latency-repo>/SEGALE "
            "in the active environment"
        )

    module = load_segale_module(args.speech_latency_repo)
    module.step2_segale(
        system_file=str(output_dir / "hyp.jsonl"),
        ref_file=str(output_dir / "ref.jsonl"),
        segmenter=args.segmenter,
        task_lang=target_lang,
        proc_device=args.device,
        verbose=args.verbose,
        max_size=args.max_size,
        embedding_model=args.embedding_model,
        seed=args.seed,
    )
    if not aligned_path.exists():
        raise FileNotFoundError(aligned_path)
    aligned_rows = sum(
        1 for line in aligned_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    summary: dict[str, object] = {
        "alignment_backend": "SEGALE",
        "speech_latency_repo": "https://github.com/SakaiXue6666/Speech-to-Speech-Latency",
        "speech_latency_revision": revision,
        "segmenter": args.segmenter,
        "target_lang": target_lang,
        "embedding_model": args.embedding_model,
        "device": args.device,
        "max_size": args.max_size,
        "seed": args.seed,
        "input_fingerprint": fingerprint,
        "aligned_rows": aligned_rows,
        "aligned_jsonl": str(aligned_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(run_alignment(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
