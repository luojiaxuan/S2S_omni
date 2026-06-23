#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_omni_codec_pairs import (  # noqa: E402
    SYSTEM_PROMPT,
    USER_PROMPT,
    append_jsonl,
    clean_completion,
    iter_input_records,
    read_existing_ids,
    rewrite_audio_uri,
    sanitize_name,
    validate_pair,
    write_wav,
)
from s2s_omni.audio import load_audio_span  # noqa: E402
from s2s_omni.codec_data import base_id_from_id  # noqa: E402

CAPTURING_CODE2WAV_FACTORY = (
    "s2s_omni.sglang_code2wav_capture.create_capturing_code2wav_scheduler"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Omni wav/code pairs through the sglang-omni speech pipeline."
    )
    parser.add_argument("--input", required=True, help="Split manifest JSONL or raw GigaSpeech TSV.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--speaker", default="Ethan")
    parser.add_argument("--sglang-omni-root", default=os.environ.get("SGLANG_OMNI_ROOT", ""))
    parser.add_argument("--relay-backend", default="shm", choices=["shm", "nixl"])
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--hop-length", type=int, default=1920)
    parser.add_argument("--codec-frame-rate", type=float, default=12.5)
    parser.add_argument("--num-quantizers", type=int, default=16)
    parser.add_argument("--codebook-size", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--gpu-thinker", type=int, default=0)
    parser.add_argument("--gpu-talker", type=int, default=None)
    parser.add_argument("--gpu-code2wav", type=int, default=None)
    parser.add_argument("--gpu-image-encoder", type=int, default=None)
    parser.add_argument("--gpu-audio-encoder", type=int, default=None)
    parser.add_argument("--thinker-tp-size", type=int, default=1)
    parser.add_argument("--gpu-thinker-tp", default=None)
    parser.add_argument("--colocated", action="store_true")
    parser.add_argument("--colocated-memory-profile", default="h200", choices=["h200", "h20", "none"])
    parser.add_argument("--enable-partial-start", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--partial-start-min-chunks", type=int, default=5)
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument("--thinker-mem-fraction-static", type=float, default=None)
    parser.add_argument("--talker-mem-fraction-static", type=float, default=None)
    parser.add_argument("--thinker-max-seq-len", type=int, default=8192)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-rejected", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-rejected-artifacts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-rms", type=float, default=1e-4)
    parser.add_argument("--max-length-delta-s", type=float, default=0.08)
    parser.add_argument("--audio-prefix-map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def _set_stage_gpu(config: Any, stage_name: str, gpu_id: int | list[int]) -> None:
    for stage in config.stages:
        if stage.name == stage_name:
            stage.gpu = gpu_id
            return
    raise ValueError(f"stage {stage_name!r} not found")


def _set_stage_tp_size(config: Any, stage_name: str, tp_size: int) -> None:
    for stage in config.stages:
        if stage.name == stage_name:
            stage.tp_size = tp_size
            stage.parallelism.tp = tp_size
            return
    raise ValueError(f"stage {stage_name!r} not found")


def _update_stage_factory_args(
    config: Any,
    stage_name: str,
    *,
    updates: dict[str, object] | None = None,
    server_arg_updates: dict[str, object] | None = None,
) -> None:
    for stage in config.stages:
        if stage.name != stage_name:
            continue
        factory_args = dict(stage.factory_args or {})
        if updates:
            factory_args.update(updates)
        if server_arg_updates:
            overrides = dict(factory_args.get("server_args_overrides") or {})
            overrides.update(server_arg_updates)
            factory_args["server_args_overrides"] = overrides
        stage.factory_args = factory_args
        return
    raise ValueError(f"stage {stage_name!r} not found")


def _set_stage_factory(config: Any, stage_name: str, factory: str) -> None:
    for stage in config.stages:
        if stage.name == stage_name:
            stage.factory = factory
            return
    raise ValueError(f"stage {stage_name!r} not found")


def _set_stage_total_gpu_fraction(config: Any, stage_name: str, fraction: float) -> None:
    for stage in config.stages:
        if stage.name == stage_name:
            stage.runtime.resources.total_gpu_memory_fraction = float(fraction)
            return
    raise ValueError(f"stage {stage_name!r} not found")


def _parse_gpu_list(spec: str, expected: int) -> list[int]:
    values = [int(piece.strip()) for piece in spec.split(",") if piece.strip()]
    if len(values) != expected:
        raise ValueError(f"expected {expected} GPU ids, got {values}")
    if len(set(values)) != len(values):
        raise ValueError(f"GPU ids must be distinct, got {values}")
    return values


def ensure_sglang_omni_root(args: argparse.Namespace) -> None:
    if not args.sglang_omni_root:
        return
    root = str(Path(args.sglang_omni_root).expanduser().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _audio_payload_to_float32(data: dict[str, Any]) -> np.ndarray:
    raw = data.get("audio_waveform")
    if raw is None:
        raw = data.get("audio_data") or data.get("audio")
    if raw is None:
        raise RuntimeError("code2wav result did not contain audio")
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().float().numpy()
    elif isinstance(raw, (bytes, bytearray, memoryview)):
        dtype = np.dtype(data.get("audio_waveform_dtype", "float32"))
        raw = np.frombuffer(raw, dtype=dtype).copy()
        shape = data.get("audio_waveform_shape")
        if shape:
            raw = raw.reshape(shape)
    audio = np.asarray(raw, dtype=np.float32).reshape(-1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


def _write_source_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        path,
        np.asarray(audio, dtype=np.float32).reshape(-1),
        sample_rate,
        subtype="PCM_16",
    )


def _stage_data(result: Any, stage_name: str) -> dict[str, Any]:
    if not isinstance(result, dict) or stage_name not in result:
        raise RuntimeError(f"pipeline result missing stage {stage_name!r}")
    payload = result[stage_name]
    data = getattr(payload, "data", payload)
    if not isinstance(data, dict):
        raise RuntimeError(f"stage {stage_name!r} returned non-dict data")
    return data


def build_sglang_config(args: argparse.Namespace) -> Any:
    ensure_sglang_omni_root(args)

    from sglang_omni.models.qwen3_omni.config import (  # noqa: PLC0415
        MIN_PARTIAL_START_CHUNKS,
        Qwen3OmniSpeechColocatedPipelineConfig,
        Qwen3OmniSpeechPipelineConfig,
    )

    enable_partial_start = (
        not args.colocated
        if args.enable_partial_start is None
        else bool(args.enable_partial_start)
    )
    if enable_partial_start and args.partial_start_min_chunks < MIN_PARTIAL_START_CHUNKS:
        raise ValueError(
            f"--partial-start-min-chunks must be >= {MIN_PARTIAL_START_CHUNKS}"
        )

    gpu_talker = args.gpu_talker if args.gpu_talker is not None else (args.gpu_thinker if args.colocated else 1)
    gpu_code2wav = args.gpu_code2wav if args.gpu_code2wav is not None else (args.gpu_thinker if args.colocated else 0)
    gpu_image_encoder = args.gpu_image_encoder if args.gpu_image_encoder is not None else args.gpu_thinker
    gpu_audio_encoder = args.gpu_audio_encoder if args.gpu_audio_encoder is not None else args.gpu_thinker

    if args.colocated and len({args.gpu_thinker, gpu_talker, gpu_code2wav, gpu_image_encoder, gpu_audio_encoder}) != 1:
        raise ValueError("--colocated requires all Qwen3-Omni stages on the same GPU")

    config_cls = (
        Qwen3OmniSpeechColocatedPipelineConfig
        if args.colocated
        else Qwen3OmniSpeechPipelineConfig
    )
    config = config_cls(model_path=args.model, relay_backend=args.relay_backend)

    _set_stage_gpu(config, "image_encoder", gpu_image_encoder)
    _set_stage_gpu(config, "audio_encoder", gpu_audio_encoder)
    if args.thinker_tp_size > 1:
        if not args.gpu_thinker_tp:
            raise ValueError("--thinker-tp-size > 1 requires --gpu-thinker-tp")
        _set_stage_tp_size(config, "thinker", args.thinker_tp_size)
        _set_stage_gpu(config, "thinker", _parse_gpu_list(args.gpu_thinker_tp, args.thinker_tp_size))
        _update_stage_factory_args(
            config,
            "thinker",
            server_arg_updates={"disable_custom_all_reduce": True},
        )
    else:
        _set_stage_gpu(config, "thinker", args.gpu_thinker)
    _set_stage_gpu(config, "talker_ar", gpu_talker)
    _set_stage_gpu(config, "code2wav", gpu_code2wav)
    _set_stage_factory(config, "code2wav", CAPTURING_CODE2WAV_FACTORY)
    if args.colocated and args.colocated_memory_profile != "none":
        profiles = {
            "h200": {
                "image_encoder": 0.017,
                "audio_encoder": 0.017,
                "thinker": 0.769,
                "talker_ar": 0.123,
                "code2wav": 0.014,
            },
            "h20": {
                "image_encoder": 0.025,
                "audio_encoder": 0.025,
                "thinker": 0.75,
                "talker_ar": 0.12,
                "code2wav": 0.02,
            },
        }
        for stage_name, fraction in profiles[args.colocated_memory_profile].items():
            _set_stage_total_gpu_fraction(config, stage_name, fraction)

    thinker_mem_fraction = (
        args.thinker_mem_fraction_static
        if args.thinker_mem_fraction_static is not None
        else args.mem_fraction_static
    )
    talker_mem_fraction = (
        args.talker_mem_fraction_static
        if args.talker_mem_fraction_static is not None
        else args.mem_fraction_static
    )
    if thinker_mem_fraction is not None:
        _update_stage_factory_args(
            config,
            "thinker",
            server_arg_updates={"mem_fraction_static": thinker_mem_fraction},
        )
    if talker_mem_fraction is not None:
        _update_stage_factory_args(
            config,
            "talker_ar",
            server_arg_updates={"mem_fraction_static": talker_mem_fraction},
        )
    if args.thinker_max_seq_len:
        updates = {"thinker_max_seq_len": int(args.thinker_max_seq_len)}
        _update_stage_factory_args(config, "thinker", updates=updates)
        _update_stage_factory_args(config, "preprocessing", updates=updates)
    partial_updates: dict[str, object] = {"enable_partial_start": enable_partial_start}
    if enable_partial_start:
        partial_updates["partial_start_min_chunks"] = int(args.partial_start_min_chunks)
    _update_stage_factory_args(config, "talker_ar", updates=partial_updates)
    return config


class SglangOmniPairGenerator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.runner = None

    async def start(self) -> None:
        if self.runner is not None:
            return
        ensure_sglang_omni_root(self.args)
        from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner  # noqa: PLC0415

        config = build_sglang_config(self.args)
        self.runner = MultiProcessPipelineRunner(config)
        await self.runner.start(timeout=self.args.startup_timeout)

    async def stop(self) -> None:
        if self.runner is not None:
            await self.runner.stop()
            self.runner = None

    async def generate_one(
        self,
        record: dict[str, Any],
        codes_path: Path,
        source_wav_path: Path,
    ) -> dict[str, Any]:
        if self.runner is None:
            raise RuntimeError("generator is not started")
        from sglang_omni.proto import OmniRequest  # noqa: PLC0415

        audio_uri = str(record.get("audio_path") or record.get("source_audio") or "")
        if not audio_uri:
            raise ValueError("record is missing audio_path/source_audio")
        resolved_audio_uri = rewrite_audio_uri(audio_uri, self.args.audio_prefix_map)
        source_audio, _ = load_audio_span(
            resolved_audio_uri,
            target_sample_rate=self.args.audio_sample_rate,
        )
        _write_source_wav(source_wav_path, source_audio, self.args.audio_sample_rate)
        sample_id = str(record["id"])
        request_id = f"codec-pair-{sanitize_name(sample_id)}-{time.time_ns()}"
        request = OmniRequest(
            inputs={
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT},
                ],
                "audios": [str(source_wav_path)],
                "audio_target_sr": self.args.audio_sample_rate,
            },
            params={
                "stream": False,
                "max_new_tokens": self.args.max_new_tokens,
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "top_k": self.args.top_k,
                "repetition_penalty": self.args.repetition_penalty,
                "seed": self.args.seed,
                "speaker": self.args.speaker,
            },
            metadata={
                "output_modalities": ["text", "audio"],
                "code2wav_codes_path": str(codes_path),
            },
        )
        result = await asyncio.wait_for(
            self.runner.coordinator.submit(request_id, request),
            timeout=self.args.timeout,
        )
        decode_data = _stage_data(result, "decode")
        code2wav_data = _stage_data(result, "code2wav")
        generated_text = clean_completion(str(decode_data.get("text") or ""))
        wav = _audio_payload_to_float32(code2wav_data)
        sidecar_path = Path(str(code2wav_data.get("codec_codes_path") or codes_path))
        if not sidecar_path.exists():
            raise RuntimeError(f"missing captured code sidecar: {sidecar_path}")
        codes = np.load(sidecar_path).astype(np.int16, copy=False)
        return {
            "generated_text": generated_text,
            "wav": wav,
            "codes": codes,
            "resolved_source_audio": resolved_audio_uri,
            "sglang_source_wav_path": str(source_wav_path),
            "sglang_request_id": request_id,
        }


def select_records(args: argparse.Namespace, accepted_path: Path, rejected_path: Path) -> list[dict[str, Any]]:
    done_ids = set()
    if args.resume:
        done_ids |= read_existing_ids(accepted_path)
        done_ids |= read_existing_ids(rejected_path)
    wanted_ids = set(args.ids or [])
    selected: list[dict[str, Any]] = []
    eligible_index = 0
    for record in iter_input_records(args.input):
        sample_id = str(record.get("id") or "")
        if not sample_id:
            continue
        if wanted_ids and sample_id not in wanted_ids:
            continue
        if sample_id in done_ids:
            continue
        if eligible_index % args.num_shards == args.shard_index:
            selected.append(record)
            if args.max_records > 0 and len(selected) >= args.max_records:
                break
        eligible_index += 1
    return selected


async def main_async(args: argparse.Namespace) -> None:
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")

    output_dir = Path(args.output_dir)
    wav_dir = output_dir / "wav"
    codes_dir = output_dir / "codes"
    source_wav_dir = output_dir / "source_wav"
    accepted_path = output_dir / "pairs.jsonl"
    rejected_path = output_dir / "pairs_rejected.jsonl"
    if not args.resume:
        for path in [accepted_path, rejected_path]:
            if path.exists():
                path.unlink()

    selected = select_records(args, accepted_path, rejected_path)
    if not selected:
        print(json.dumps({"selected": 0}, ensure_ascii=False, indent=2))
        return

    generator = SglangOmniPairGenerator(args)
    await generator.start()
    accepted = 0
    rejected = 0
    try:
        for index, record in enumerate(selected, start=1):
            sample_id = str(record["id"])
            safe_id = sanitize_name(sample_id)
            wav_path = wav_dir / f"{safe_id}.wav"
            codes_path = codes_dir / f"{safe_id}.npy"
            source_wav_path = source_wav_dir / f"{safe_id}.wav"
            row: dict[str, Any] = {
                "id": sample_id,
                "base_id": record.get("base_id") or base_id_from_id(sample_id),
                "source_audio": record.get("audio_path") or record.get("source_audio"),
                "source_text": record.get("source_text"),
                "reference_translation": record.get("reference_translation"),
                "src_lang": record.get("src_lang", "en"),
                "tgt_lang": record.get("tgt_lang", "zh"),
                "model": args.model,
                "speaker": args.speaker,
                "serving_framework": "sglang-omni",
                "sample_rate": args.sample_rate,
                "hop_length": args.hop_length,
                "codec_frame_rate": args.codec_frame_rate,
                "generation_params": {
                    "stream": False,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "repetition_penalty": args.repetition_penalty,
                    "seed": args.seed,
                    "relay_backend": args.relay_backend,
                    "colocated": args.colocated,
                    "thinker_tp_size": args.thinker_tp_size,
                },
                "source_record": record,
            }
            try:
                result = await generator.generate_one(
                    record,
                    codes_path,
                    source_wav_path,
                )
                reject_reasons = validate_pair(result, args)
                codes = result.pop("codes")
                wav = result.pop("wav")
                row.update(result)
                row["codec_num_quantizers"] = int(codes.shape[0]) if codes.ndim == 2 else None
                row["codec_frames"] = int(codes.shape[1]) if codes.ndim == 2 else None
                row["code_shape"] = [int(v) for v in codes.shape]
                row["code_min"] = int(codes.min()) if codes.size else None
                row["code_max"] = int(codes.max()) if codes.size else None
                row["duration_s"] = round(float(wav.shape[0]) / float(args.sample_rate), 6)
                row["expected_duration_s"] = (
                    round(float(codes.shape[1] * args.hop_length) / float(args.sample_rate), 6)
                    if codes.ndim == 2
                    else None
                )
                if not reject_reasons:
                    write_wav(wav_path, wav, args.sample_rate)
                    codes_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(codes_path, codes)
                    row["wav_path"] = str(wav_path)
                    row["codes_path"] = str(codes_path)
                    row["accepted"] = True
                    row["reject_reasons"] = []
                else:
                    row["accepted"] = False
                    row["reject_reasons"] = reject_reasons
                    if not args.keep_rejected_artifacts and codes_path.exists():
                        codes_path.unlink()
            except Exception as exc:
                row["accepted"] = False
                row["reject_reasons"] = [f"exception:{type(exc).__name__}"]
                row["error"] = str(exc)
                if not args.keep_rejected_artifacts and codes_path.exists():
                    codes_path.unlink()

            if row.get("accepted"):
                append_jsonl(accepted_path, row)
                accepted += 1
            else:
                rejected += 1
                if args.save_rejected:
                    append_jsonl(rejected_path, row)
            if args.log_every > 0 and index % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "processed": index,
                            "accepted": accepted,
                            "rejected": rejected,
                            "last_id": sample_id,
                            "last_reject_reasons": row.get("reject_reasons"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        await generator.stop()

    summary = {
        "input": args.input,
        "output_dir": str(output_dir),
        "selected_this_run": len(selected),
        "accepted_this_run": accepted,
        "rejected_this_run": rejected,
        "accepted_total": len(read_existing_ids(accepted_path)),
        "rejected_total": len(read_existing_ids(rejected_path)),
        "model": args.model,
        "speaker": args.speaker,
        "serving_framework": "sglang-omni",
        "sample_rate": args.sample_rate,
        "hop_length": args.hop_length,
        "codec_frame_rate": args.codec_frame_rate,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    (output_dir / f"summary_shard{args.shard_index}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
