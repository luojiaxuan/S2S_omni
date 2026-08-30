#!/usr/bin/env python3
"""Run the frozen four-cell codec-context cascade comparison."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


CELL_QUEUES = (("c0", "c3"), ("c1", "c2"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", required=True, type=Path)
    parser.add_argument("--asr-port", type=int, default=48573)
    parser.add_argument("--gpu-count", type=int, choices=(1, 2), default=2)
    parser.add_argument("--moss-tts-root", type=Path, default=None)
    parser.add_argument("--segale-python", default="/data/venvs/segale_eval2/bin/python")
    parser.add_argument("--speech-latency-repo", type=Path, default=None)
    return parser.parse_args()


def run(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(command) + "\n")
        handle.flush()
        subprocess.run(
            command,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )


def snapshot(cache_root: Path, slug: str, revision_prefix: str = "") -> Path:
    candidates = sorted((cache_root / slug / "snapshots").glob(f"{revision_prefix}*"))
    if not candidates:
        raise FileNotFoundError(f"no cached snapshot for {slug}@{revision_prefix}")
    if revision_prefix and len(candidates) != 1:
        raise RuntimeError(f"ambiguous cached snapshot for {slug}@{revision_prefix}: {candidates}")
    return candidates[-1]


def split_summaries(task_root: Path, cell: str, talks: list[int]) -> None:
    rows = [
        json.loads(line)
        for line in (task_root / "input" / f"{cell}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    summaries = [
        json.loads(line)
        for line in (task_root / "output" / f"{cell}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != len(talks) or len(summaries) != len(talks):
        raise RuntimeError(
            f"{cell}: expected {len(talks)} rows and summaries, got {len(rows)} and {len(summaries)}"
        )
    for talk, row, summary in zip(talks, rows, summaries, strict=True):
        if summary["row_id"] != row["row_id"]:
            raise RuntimeError(f"{cell}/talk{talk}: row order mismatch")
        path = task_root / "output" / cell / f"talk{talk}.summary.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False) + "\n", encoding="utf-8")


def synthesize_queue(
    task_root: Path,
    queue: tuple[str, ...],
    logical_gpu: int,
    model_path: Path,
    codec_path: Path,
    talks: list[int],
    moss_tts_root: Path,
) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(logical_gpu)
    env["TORCHDYNAMO_DISABLE"] = "1"
    for cell in queue:
        done = task_root / "state" / f"{cell}.done"
        if done.exists():
            continue
        output_dir = task_root / "output" / cell
        output_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                sys.executable,
                str(task_root / "code" / "w0.py"),
                "--model-path",
                str(model_path),
                "--codec-path",
                str(codec_path),
                "--moss-tts-root",
                str(moss_tts_root),
                "--fixed-ref",
                str(task_root / "input" / "ref.wav"),
                "--rows-jsonl",
                str(task_root / "input" / f"{cell}.jsonl"),
                "--out-dir",
                str(output_dir),
                "--summary-jsonl",
                str(task_root / "output" / f"{cell}.jsonl"),
                "--device",
                "cuda",
                "--min-runaway-floor-s",
                "15",
                "--sliding-window",
                "11",
                "--soft-reset-keep",
                "3",
                "--continuous-codec-context",
                "--log-every",
                "50",
            ],
            task_root / "logs" / f"{cell}.log",
            env,
        )
        split_summaries(task_root, cell, talks)
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text("done\n", encoding="utf-8")


def wait_for_server(port: int, process: subprocess.Popen, timeout_s: float = 600.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ASR server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3):
                return
        except Exception:
            time.sleep(2)
    raise TimeoutError("ASR server did not become healthy")


def score_cell(
    task_root: Path,
    cell: str,
    dataset_root: Path,
    port: int,
) -> None:
    run_dir = task_root / "result" / cell
    done = task_root / "state" / f"{cell}.asr.done"
    if done.exists():
        return
    run(
        [
            sys.executable,
            str(task_root / "code" / "w1.py"),
            "--rows-dir",
            str(task_root / "input" / cell),
            "--wav-dir",
            str(task_root / "output" / cell),
            "--run-dir",
            str(run_dir),
            "--dataset-root",
            str(dataset_root),
            "--transcribe-helper",
            str(task_root / "code" / "w2.py"),
            "--base-url",
            f"http://127.0.0.1:{port}",
            "--label",
            cell,
        ],
        task_root / "logs" / f"{cell}.asr.log",
    )
    done.write_text("done\n", encoding="utf-8")


def align_cell(
    task_root: Path,
    cell: str,
    dataset_root: Path,
    segale_python: str,
    speech_latency_repo: Path,
    logical_gpu: int,
) -> dict:
    run_dir = task_root / "result" / cell
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(logical_gpu)
    log_path = task_root / "logs" / f"{cell}.align.log"
    run(
        [
            segale_python,
            str(task_root / "code" / "w3.py"),
            "--run-dir",
            str(run_dir),
            "--dataset-root",
            str(dataset_root),
        ],
        log_path,
        env,
    )
    run(
        [
            segale_python,
            str(task_root / "code" / "w4.py"),
            "--run-dir",
            str(run_dir),
            "--speech-latency-repo",
            str(speech_latency_repo),
            "--target-lang",
            "zh",
            "--device",
            "cuda",
        ],
        log_path,
        env,
    )
    summary_path = run_dir / "bleu_summary.json"
    run(
        [
            segale_python,
            str(task_root / "code" / "w5.py"),
            "--run-dir",
            str(run_dir),
            "--output-jsonl",
            str(run_dir / "xcomet_input.jsonl"),
            "--summary-json",
            str(summary_path),
            "--bleu-tokenizer",
            "zh",
        ],
        log_path,
        env,
    )
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    task_root = args.task_root.resolve()
    config = json.loads((task_root / "config.json").read_text(encoding="utf-8"))
    talks = [
        int(value)
        for value in config.get("talks", config.get("frozen_inputs", {}).get("talks", []))
    ]
    if not talks:
        raise ValueError("config does not define talks")
    moss_tts_root = args.moss_tts_root or (task_root / "resources" / "m0")
    speech_latency_repo = args.speech_latency_repo or (task_root / "resources" / "r0")
    cache_root = Path("/root/.cache/huggingface/hub")
    model_path = snapshot(
        cache_root,
        "models--gavinlaw--moss-tts-realtime-infinisst-en-zh-v7-traj",
        "1947001d",
    )
    codec_path = snapshot(cache_root, "models--OpenMOSS-Team--MOSS-Audio-Tokenizer")
    dataset_root = snapshot(cache_root, "datasets--gavinlaw--rasst-main-result-data")
    resolved = {
        "model_path": str(model_path),
        "codec_path": str(codec_path),
        "dataset_root": str(dataset_root),
    }
    (task_root / "resolved_resources.json").write_text(
        json.dumps(resolved, indent=2) + "\n", encoding="utf-8"
    )

    cell_queues = CELL_QUEUES if args.gpu_count == 2 else (("c0", "c1", "c2", "c3"),)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.gpu_count) as pool:
        futures = [
            pool.submit(
                synthesize_queue,
                task_root,
                queue,
                logical_gpu,
                model_path,
                codec_path,
                talks,
                moss_tts_root,
            )
            for logical_gpu, queue in enumerate(cell_queues)
        ]
        for future in futures:
            future.result()

    asr_log = (task_root / "logs" / "asr-server.log").open("a", encoding="utf-8")
    asr_env = os.environ.copy()
    asr_env["CUDA_VISIBLE_DEVICES"] = "0"
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            "Qwen/Qwen3-ASR-1.7B",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.asr_port),
            "--mem-fraction-static",
            "0.45",
        ],
        stdout=asr_log,
        stderr=subprocess.STDOUT,
        env=asr_env,
    )
    try:
        wait_for_server(args.asr_port, server)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(score_cell, task_root, cell, dataset_root, args.asr_port)
                for cell in ("c0", "c1", "c2", "c3")
            ]
            for future in futures:
                future.result()
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        asr_log.close()

    summaries = {
        cell: align_cell(
            task_root,
            cell,
            dataset_root,
            args.segale_python,
            speech_latency_repo,
            args.gpu_count - 1,
        )
        for cell in ("c0", "c1", "c2", "c3")
    }
    result = {
        "cells": config["cells"],
        "resolved_resources": resolved,
        "summaries": summaries,
    }
    (task_root / "result" / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
