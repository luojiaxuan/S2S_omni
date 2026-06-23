#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.audio import load_audio_span
from s2s_omni.codec_data import base_id_from_id
from s2s_omni.io import read_jsonl


SYSTEM_PROMPT = (
    "You are a speech-to-speech translation engine. Translate the user's English "
    "speech into natural spoken Chinese. Output only the Chinese translation."
)
USER_PROMPT = "Translate this English speech into natural spoken Chinese and speak the translation."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Omni self-domain wav/code pairs from speech-to-speech translation."
    )
    parser.add_argument("--input", required=True, help="Split manifest JSONL or raw GigaSpeech TSV.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--speaker", default="Ethan")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--hop-length", type=int, default=1920)
    parser.add_argument("--codec-frame-rate", type=float, default=12.5)
    parser.add_argument("--num-quantizers", type=int, default=16)
    parser.add_argument("--codebook-size", type=int, default=2048)
    parser.add_argument("--thinker-max-new-tokens", type=int, default=256)
    parser.add_argument("--thinker-do-sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--talker-max-new-tokens", type=int, default=4096)
    parser.add_argument("--talker-do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--talker-temperature", type=float, default=0.9)
    parser.add_argument("--talker-top-k", type=int, default=50)
    parser.add_argument("--talker-top-p", type=float, default=1.0)
    parser.add_argument("--talker-repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-rejected", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-rms", type=float, default=1e-4)
    parser.add_argument("--max-length-delta-s", type=float, default=0.08)
    parser.add_argument("--audio-prefix-map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180] or "sample"


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")
        handle.flush()


def read_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = str(row.get("id") or "")
            if sample_id:
                out.add(sample_id)
    return out


def rewrite_audio_uri(uri: str, prefix_maps: list[str]) -> str:
    for item in prefix_maps:
        if "=" not in item:
            raise ValueError(f"invalid --audio-prefix-map {item!r}; expected OLD=NEW")
        old, new = item.split("=", 1)
        if old and uri.startswith(old):
            return new + uri[len(old) :]
    return uri


def parse_audio_span(uri: str) -> tuple[str, int | None, int | None]:
    parts = uri.rsplit(":", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return parts[0], int(parts[1]), int(parts[2])
    return uri, None, None


def tsv_row_to_record(row: dict[str, str]) -> dict[str, Any] | None:
    if row.get("src_lang") != "en" or row.get("tgt_lang") != "zh":
        return None
    source_text = (row.get("src_text") or "").strip()
    target_text = (row.get("tgt_text") or "").strip()
    if not source_text or not target_text:
        return None
    sample_id = str(row.get("id") or "")
    audio_uri = row.get("audio") or ""
    _path, _offset, span_frames = parse_audio_span(audio_uri)
    if not sample_id or span_frames is None:
        return None
    return {
        "id": sample_id,
        "base_id": base_id_from_id(sample_id),
        "audio_path": audio_uri,
        "source_audio": audio_uri,
        "source_text": source_text,
        "reference_translation": target_text,
        "src_lang": "en",
        "tgt_lang": "zh",
        "duration_s": round(float(span_frames) / 16000.0, 6),
        "n_frames": int(row.get("n_frames") or span_frames),
        "speaker": row.get("speaker"),
        "metadata": {"source": "raw_gigaspeech_tsv"},
    }


def iter_input_records(path: str) -> Iterable[dict[str, Any]]:
    input_path = Path(path)
    if input_path.suffix == ".tsv":
        csv.field_size_limit(sys.maxsize)
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                record = tsv_row_to_record(row)
                if record is not None:
                    yield record
        return
    for record in read_jsonl(input_path):
        if "audio_path" not in record and "source_audio" in record:
            record["audio_path"] = record["source_audio"]
        if "base_id" not in record and record.get("id"):
            record["base_id"] = base_id_from_id(str(record["id"]))
        yield record


def messages_for_record(record: dict[str, Any], audio_uri: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_uri},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]


def clean_completion(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    for token in ["<|im_end|>", "<|endoftext|>", "<think>", "</think>"]:
        text = text.replace(token, "")
    for prefix in ["Translation:", "Answer:", "译文：", "译文:", "翻译：", "翻译:"]:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return " ".join(text.split())


def audio_to_float32(audio: Any) -> np.ndarray:
    if audio is None:
        raise RuntimeError("model did not return audio")
    if hasattr(audio, "detach"):
        audio = audio.reshape(-1).detach().float().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return np.clip(audio, -1.0, 1.0)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="PCM_16")


class Code2WavCapture:
    def __init__(self, code2wav: Any) -> None:
        self.code2wav = code2wav
        self.original = code2wav.chunked_decode
        self.codes = None
        self.decoded_wav = None
        code2wav.chunked_decode = self._wrapped

    def _wrapped(self, codes: Any, *args: Any, **kwargs: Any) -> Any:
        self.codes = codes.detach().cpu().clone()
        wav = self.original(codes, *args, **kwargs)
        self.decoded_wav = wav.detach().cpu().clone()
        return wav

    def clear(self) -> None:
        self.codes = None
        self.decoded_wav = None

    def restore(self) -> None:
        self.code2wav.chunked_decode = self.original


class OmniPairGenerator:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        self.torch = torch
        self.args = args
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(args.model, trust_remote_code=True)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            args.model,
            trust_remote_code=True,
            dtype="auto",
            device_map=args.device_map,
        )
        self.model.eval()
        if not getattr(self.model, "has_talker", False):
            self.model.enable_talker()
        self.capture = Code2WavCapture(self.model.code2wav)

    def generate_one(self, record: dict[str, Any]) -> dict[str, Any]:
        audio_uri = str(record.get("audio_path") or record.get("source_audio") or "")
        if not audio_uri:
            raise ValueError("record is missing audio_path/source_audio")
        resolved_audio_uri = rewrite_audio_uri(audio_uri, self.args.audio_prefix_map)
        source_audio, _ = load_audio_span(
            resolved_audio_uri,
            target_sample_rate=self.args.audio_sample_rate,
        )
        messages = messages_for_record(record, resolved_audio_uri)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=text,
            audio=[source_audio],
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        device = getattr(self.model, "device", next(self.model.parameters()).device)
        inputs = inputs.to(device)
        float_dtype = getattr(self.model, "dtype", None)
        if float_dtype is not None:
            for key in ["input_features", "pixel_values", "pixel_values_videos"]:
                if key in inputs:
                    inputs[key] = inputs[key].to(dtype=float_dtype)

        self.capture.clear()
        with self.torch.inference_mode():
            text_ids, audio = self.model.generate(
                **inputs,
                speaker=self.args.speaker,
                use_audio_in_video=False,
                return_audio=True,
                thinker_max_new_tokens=self.args.thinker_max_new_tokens,
                thinker_do_sample=self.args.thinker_do_sample,
                talker_max_new_tokens=self.args.talker_max_new_tokens,
                talker_do_sample=self.args.talker_do_sample,
                talker_temperature=self.args.talker_temperature,
                talker_top_k=self.args.talker_top_k,
                talker_top_p=self.args.talker_top_p,
                talker_repetition_penalty=self.args.talker_repetition_penalty,
            )
        sequences = text_ids.sequences if hasattr(text_ids, "sequences") else text_ids
        generated = sequences[:, inputs["input_ids"].shape[1] :]
        generated_text = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        if self.capture.codes is None:
            raise RuntimeError("code2wav.chunked_decode was not called; no talker codes captured")
        codes = self.capture.codes
        if codes.ndim == 3 and codes.shape[0] == 1:
            codes = codes[0]
        wav = audio_to_float32(audio)
        return {
            "generated_text": clean_completion(generated_text),
            "wav": wav,
            "codes": codes.numpy().astype(np.int16, copy=False),
            "resolved_source_audio": resolved_audio_uri,
        }


def validate_pair(
    result: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    reasons: list[str] = []
    generated_text = str(result.get("generated_text") or "").strip()
    wav = np.asarray(result.get("wav"), dtype=np.float32).reshape(-1)
    codes = np.asarray(result.get("codes"))
    if not generated_text:
        reasons.append("empty_generated_text")
    if codes.ndim != 2:
        reasons.append(f"bad_code_rank:{codes.shape}")
    elif codes.shape[0] != args.num_quantizers:
        reasons.append(f"bad_num_quantizers:{codes.shape}")
    else:
        code_min = int(codes.min()) if codes.size else -1
        code_max = int(codes.max()) if codes.size else -1
        if code_min < 0 or code_max >= args.codebook_size:
            reasons.append(f"code_out_of_range:{code_min}:{code_max}")
    if wav.size == 0 or not np.isfinite(wav).all():
        reasons.append("bad_wav")
    else:
        rms = float(np.sqrt(np.mean(np.square(wav.astype(np.float64)))))
        if rms < args.min_rms:
            reasons.append(f"low_rms:{rms:.6g}")
    if codes.ndim == 2 and wav.size:
        expected_samples = int(codes.shape[1]) * int(args.hop_length)
        delta_s = abs(wav.size - expected_samples) / float(args.sample_rate)
        if delta_s > args.max_length_delta_s:
            reasons.append(f"wav_code_length_mismatch:{delta_s:.6f}s")
    return reasons


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")

    output_dir = Path(args.output_dir)
    wav_dir = output_dir / "wav"
    codes_dir = output_dir / "codes"
    accepted_path = output_dir / "pairs.jsonl"
    rejected_path = output_dir / "pairs_rejected.jsonl"
    if not args.resume:
        for path in [accepted_path, rejected_path]:
            if path.exists():
                path.unlink()
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

    if not selected:
        print(json.dumps({"selected": 0, "done": len(done_ids)}, ensure_ascii=False, indent=2))
        return

    generator = OmniPairGenerator(args)
    accepted = 0
    rejected = 0
    for index, record in enumerate(selected, start=1):
        sample_id = str(record["id"])
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
            "sample_rate": args.sample_rate,
            "hop_length": args.hop_length,
            "codec_frame_rate": args.codec_frame_rate,
            "generation_params": {
                "thinker_max_new_tokens": args.thinker_max_new_tokens,
                "thinker_do_sample": args.thinker_do_sample,
                "talker_max_new_tokens": args.talker_max_new_tokens,
                "talker_do_sample": args.talker_do_sample,
                "talker_temperature": args.talker_temperature,
                "talker_top_k": args.talker_top_k,
                "talker_top_p": args.talker_top_p,
                "talker_repetition_penalty": args.talker_repetition_penalty,
            },
            "source_record": record,
        }
        try:
            result = generator.generate_one(record)
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
                wav_path = wav_dir / f"{sanitize_name(sample_id)}.wav"
                codes_path = codes_dir / f"{sanitize_name(sample_id)}.npy"
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
        except Exception as exc:
            row["accepted"] = False
            row["reject_reasons"] = [f"exception:{type(exc).__name__}"]
            row["error"] = str(exc)

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


if __name__ == "__main__":
    main()
