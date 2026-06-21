#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from s2s_omni.metrics import heuristic_score_sample
from s2s_omni.prompts import SYSTEM_COMPRESSION, build_compression_user_prompt
from s2s_omni.schema import S2SSample


DEFAULT_IDS = [
    "AUD0000000094_565__speed_2",
    "AUD0000000120_686__speed_2",
    "AUD0000000113_1386__speed_1.7",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate base vs Evo speech samples with frozen Qwen3-Omni talker."
    )
    parser.add_argument("--input", required=True, help="Teacher requests or sample JSONL.")
    parser.add_argument("--adapter", required=True, help="Thinker PEFT adapter directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--ids", nargs="*", default=DEFAULT_IDS)
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--speaker", default="Ethan")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--thinker-max-new-tokens", type=int, default=160)
    parser.add_argument("--talker-max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--talker-do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Qwen3-Omni talker sampling. Deterministic talker decode can fail to stop.",
    )
    parser.add_argument("--talker-temperature", type=float, default=0.9)
    parser.add_argument("--talker-top-k", type=int, default=50)
    parser.add_argument("--talker-top-p", type=float, default=1.0)
    parser.add_argument("--skip-base", action="store_true")
    return parser.parse_args()


def load_samples(path: str, ids: list[str], num_samples: int) -> list[S2SSample]:
    selected: list[S2SSample] = []
    wanted = set(ids)
    for record in read_jsonl(path):
        payload = record.get("sample") if isinstance(record.get("sample"), dict) else record
        sample = S2SSample.from_dict(payload)
        if wanted and sample.id not in wanted:
            continue
        selected.append(sample)
    if not selected and wanted:
        raise SystemExit(f"none of the requested ids were found: {', '.join(ids)}")
    if num_samples > 0:
        selected = selected[:num_samples]
    return selected


def messages_for_sample(sample: S2SSample) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_COMPRESSION.strip()},
        {
            "role": "user",
            "content": build_compression_user_prompt(sample, include_reference=False),
        },
    ]


def clean_completion(text: str) -> str:
    text = (text or "").strip()
    for token in ["<|im_end|>", "<|endoftext|>"]:
        text = text.replace(token, "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<think>", "").replace("</think>", "")
    text = text.strip().strip("`").strip()
    for prefix in [
        "Compressed translation:",
        "Translation:",
        "Answer:",
        "译文：",
        "译文:",
        "压缩译文：",
        "压缩译文:",
    ]:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return " ".join(text.split())


def _adapter_model_path(adapter_dir: str) -> Path:
    path = Path(adapter_dir) / "adapter_model.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"adapter weights not found: {path}")
    return path


def _module_name_from_lora_key(key: str) -> str:
    name = key.removeprefix("base_model.model.")
    name = name.removesuffix(".lora_A.weight")
    return f"thinker.{name}"


def merge_lora_into_full_omni(model: Any, adapter_dir: str) -> dict[str, Any]:
    import torch
    from safetensors.torch import safe_open

    adapter_path = _adapter_model_path(adapter_dir)
    adapter_config = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    rank = int(adapter_config.get("r", 16))
    alpha = int(adapter_config.get("lora_alpha", 32))
    scaling = alpha / rank

    modules = dict(model.named_modules())
    merged = 0
    skipped: list[str] = []
    with safe_open(adapter_path, framework="pt", device="cpu") as tensors:
        keys = set(tensors.keys())
        for key in sorted(keys):
            if not key.endswith(".lora_A.weight"):
                continue
            module_name = _module_name_from_lora_key(key)
            b_key = key.replace(".lora_A.weight", ".lora_B.weight")
            module = modules.get(module_name)
            if module is None or b_key not in keys or not hasattr(module, "weight"):
                skipped.append(module_name)
                continue

            weight = module.weight
            device = weight.device
            a = tensors.get_tensor(key).to(device=device, dtype=torch.float32)
            b = tensors.get_tensor(b_key).to(device=device, dtype=torch.float32)
            delta = torch.matmul(b, a) * scaling
            if tuple(delta.shape) != tuple(weight.shape):
                skipped.append(module_name)
                continue
            weight.data.add_(delta.to(dtype=weight.dtype))
            merged += 1

    return {
        "merged": merged,
        "skipped": len(skipped),
        "skipped_examples": skipped[:8],
        "adapter_path": str(adapter_path),
    }


def audio_to_float32(audio: Any) -> np.ndarray:
    if audio is None:
        raise RuntimeError("model did not return audio")
    if hasattr(audio, "detach"):
        audio = audio.reshape(-1).detach().float().cpu().numpy()
    else:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = np.nan_to_num(audio.astype(np.float32))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return np.clip(audio, -1.0, 1.0)


class OmniSpeechGenerator:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration
        from transformers import Qwen3OmniMoeProcessor

        torch.manual_seed(260621)
        self.args = args
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            args.model,
            trust_remote_code=True,
        )
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            args.model,
            trust_remote_code=True,
            dtype="auto",
            device_map=args.device_map,
        )
        self.model.eval()
        self.adapter_merged = False

    def merge_adapter(self) -> dict[str, Any]:
        if self.adapter_merged:
            return {"merged": 0, "already_merged": True}
        stats = merge_lora_into_full_omni(self.model, self.args.adapter)
        self.adapter_merged = True
        return stats

    def generate(self, sample: S2SSample) -> tuple[str, np.ndarray]:
        import torch

        text = self.processor.apply_chat_template(
            messages_for_sample(sample),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        device = getattr(self.model, "device", next(self.model.parameters()).device)
        inputs = inputs.to(device)
        with torch.inference_mode():
            text_ids, audio = self.model.generate(
                **inputs,
                thinker_return_dict_in_generate=True,
                thinker_max_new_tokens=self.args.thinker_max_new_tokens,
                thinker_do_sample=False,
                speaker=self.args.speaker,
                use_audio_in_video=False,
                return_audio=True,
                talker_max_new_tokens=self.args.talker_max_new_tokens,
                talker_do_sample=self.args.talker_do_sample,
                talker_temperature=self.args.talker_temperature,
                talker_top_k=self.args.talker_top_k,
                talker_top_p=self.args.talker_top_p,
            )
        sequences = text_ids.sequences if hasattr(text_ids, "sequences") else text_ids
        generated = sequences[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return clean_completion(response), audio_to_float32(audio)


def write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="PCM_16")
    return round(float(len(audio)) / float(sample_rate), 3)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    samples = load_samples(args.input, args.ids, args.num_samples)
    generator = OmniSpeechGenerator(args)

    rows: list[dict[str, Any]] = []
    for sample in samples:
        row: dict[str, Any] = {
            "id": sample.id,
            "source_text": sample.source_text,
            "reference_translation": sample.reference_translation,
            "budget_chars": sample.target.max_target_chars,
            "target_duration_s": sample.target.max_target_duration_s,
        }
        if not args.skip_base:
            base_text, base_audio = generator.generate(sample)
            base_path = output_dir / f"{sample.id}__base.wav"
            row["base_audio"] = str(base_path)
            row["base_duration_s"] = write_audio(base_path, base_audio, args.sample_rate)
            row["base_prediction"] = base_text
            row.update(
                {
                    f"base_{k}": v
                    for k, v in heuristic_score_sample(sample, base_text).items()
                    if k != "id"
                }
            )
            print(json.dumps({"id": sample.id, "phase": "base", "text": base_text}, ensure_ascii=False), flush=True)
        rows.append(row)

    merge_stats = generator.merge_adapter()
    print(json.dumps({"manual_lora_merge": merge_stats}, ensure_ascii=False), flush=True)

    for row, sample in zip(rows, samples):
        evo_text, evo_audio = generator.generate(sample)
        evo_path = output_dir / f"{sample.id}__evo.wav"
        row["evo_audio"] = str(evo_path)
        row["evo_duration_s"] = write_audio(evo_path, evo_audio, args.sample_rate)
        row["evo_prediction"] = evo_text
        row.update(
            {
                f"evo_{k}": v
                for k, v in heuristic_score_sample(sample, evo_text).items()
                if k != "id"
            }
        )
        print(json.dumps({"id": sample.id, "phase": "evo", "text": evo_text}, ensure_ascii=False), flush=True)

    write_jsonl(output_dir / "manifest.jsonl", rows)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "adapter": args.adapter,
                "speaker": args.speaker,
                "sample_rate": args.sample_rate,
                "generation": {
                    "thinker_max_new_tokens": args.thinker_max_new_tokens,
                    "talker_max_new_tokens": args.talker_max_new_tokens,
                    "talker_do_sample": args.talker_do_sample,
                    "talker_temperature": args.talker_temperature,
                    "talker_top_k": args.talker_top_k,
                    "talker_top_p": args.talker_top_p,
                },
                "samples": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
