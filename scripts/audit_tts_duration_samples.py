#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_jsonl, write_jsonl
from s2s_omni.metrics import unit_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize sample text pairs and audit speech duration.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tts-requests", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--speaker", default="Ethan")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-reference-units", type=int, default=90)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--talker-max-new-tokens", type=int, default=4096)
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def clean_completion(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    for token in ["<|im_end|>", "<|endoftext|>", "<think>", "</think>"]:
        text = text.replace(token, "")
    for prefix in ["朗读：", "朗读:", "文本：", "文本:", "Read:", "Output:"]:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return " ".join(text.split())


def audio_to_float32(audio: Any) -> np.ndarray:
    if audio is None:
        raise RuntimeError("model did not return audio")
    if hasattr(audio, "detach"):
        audio = audio.reshape(-1).detach().float().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = np.nan_to_num(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return np.clip(audio, -1.0, 1.0)


def select_samples(manifest_path: str, tts_path: str, num_samples: int, seed: int, min_reference_units: int) -> list[dict[str, Any]]:
    tts_by_id = {row["id"]: row for row in read_jsonl(tts_path)}
    candidates: list[dict[str, Any]] = []
    for row in read_jsonl(manifest_path):
        metadata = row.get("metadata") or {}
        if metadata.get("policy_sample_kind") != "compression":
            continue
        tts = tts_by_id.get(row["id"])
        if not tts:
            continue
        reference = row.get("reference_translation") or ""
        compressed = tts.get("target_text") or row.get("compressed_translation") or ""
        ref_units = unit_count(reference, row.get("tgt_lang") or "zh")
        cmp_units = unit_count(compressed, row.get("tgt_lang") or "zh")
        if ref_units < min_reference_units or cmp_units <= 0 or cmp_units >= ref_units:
            continue
        candidates.append(
            {
                "id": row["id"],
                "src_lang": row.get("src_lang"),
                "tgt_lang": row.get("tgt_lang"),
                "source_text": row.get("source_text"),
                "reference_text": reference,
                "compressed_text": compressed,
                "reference_units": ref_units,
                "compressed_units": cmp_units,
                "target_duration_budget_s": tts.get("target_duration_budget_s"),
                "estimated_compressed_speech_s": tts.get("estimated_target_speech_s"),
            }
        )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:num_samples]


class TalkerSynthesizer:
    def __init__(self, model_name: str, device_map: str, speaker: str, sample_rate: int, max_new_tokens: int, talker_max_new_tokens: int) -> None:
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        self.torch = torch
        self.speaker = speaker
        self.sample_rate = sample_rate
        self.max_new_tokens = max_new_tokens
        self.talker_max_new_tokens = talker_max_new_tokens
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype="auto",
            device_map=device_map,
        )
        self.model.eval()

    def synthesize(self, text: str) -> tuple[str, np.ndarray]:
        messages = [
            {
                "role": "system",
                "content": "You are a text-to-speech engine. Repeat the user's text exactly and speak it aloud. Do not add any other words.",
            },
            {"role": "user", "content": text},
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=prompt, return_tensors="pt", padding=True, use_audio_in_video=False)
        device = getattr(self.model, "device", next(self.model.parameters()).device)
        inputs = inputs.to(device)
        with self.torch.inference_mode():
            text_ids, audio = self.model.generate(
                **inputs,
                thinker_return_dict_in_generate=True,
                thinker_max_new_tokens=self.max_new_tokens,
                thinker_do_sample=False,
                speaker=self.speaker,
                use_audio_in_video=False,
                return_audio=True,
                talker_max_new_tokens=self.talker_max_new_tokens,
                talker_do_sample=True,
                talker_temperature=0.9,
                talker_top_k=50,
                talker_top_p=1.0,
            )
        sequences = text_ids.sequences if hasattr(text_ids, "sequences") else text_ids
        generated = sequences[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return clean_completion(response), audio_to_float32(audio)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="PCM_16")
    return round(float(len(audio)) / float(sample_rate), 3)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = select_samples(
        args.manifest,
        args.tts_requests,
        args.num_samples,
        args.seed,
        args.min_reference_units,
    )
    if not samples:
        raise SystemExit("no samples selected")

    synth = TalkerSynthesizer(
        args.model,
        args.device_map,
        args.speaker,
        args.sample_rate,
        args.max_new_tokens,
        args.talker_max_new_tokens,
    )
    rows: list[dict[str, Any]] = []
    for sample in samples:
        row = dict(sample)
        for kind, text_key in [("reference", "reference_text"), ("compressed", "compressed_text")]:
            generated_text, audio = synth.synthesize(sample[text_key])
            wav_path = output_dir / f"{sample['id']}__{kind}.wav"
            duration_s = write_wav(wav_path, audio, args.sample_rate)
            row[f"{kind}_generated_text"] = generated_text
            row[f"{kind}_audio_path"] = str(wav_path)
            row[f"{kind}_duration_s"] = duration_s
            row[f"{kind}_generated_units"] = unit_count(generated_text, sample.get("tgt_lang") or "zh")
        ref_d = row["reference_duration_s"]
        cmp_d = row["compressed_duration_s"]
        row["actual_duration_ratio"] = round(cmp_d / ref_d, 6) if ref_d else None
        row["unit_ratio"] = round(row["compressed_units"] / row["reference_units"], 6)
        rows.append(row)
        print(json.dumps({k: row[k] for k in ["id", "reference_units", "compressed_units", "reference_duration_s", "compressed_duration_s", "actual_duration_ratio"]}, ensure_ascii=False), flush=True)

    write_jsonl(output_dir / "duration_audit.jsonl", rows)
    ratios = [row["actual_duration_ratio"] for row in rows if row.get("actual_duration_ratio") is not None]
    summary = {
        "count": len(rows),
        "mean_actual_duration_ratio": round(sum(ratios) / len(ratios), 6) if ratios else None,
        "all_compressed_shorter": all(row["compressed_duration_s"] < row["reference_duration_s"] for row in rows),
        "speaker": args.speaker,
        "model": args.model,
        "sample_rate": args.sample_rate,
        "rows": rows,
    }
    (output_dir / "duration_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
