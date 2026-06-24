#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_jsonl
from s2s_omni.omni_talker import patch_qwen3_omni_no_split, prepare_talker_condition, soft_code2wav
from s2s_omni.rasst import write_mono_wav
from scripts.train_qwen3_omni_softwav_lora import (
    load_processor_inputs,
    messages_with_audio_content,
    require_finite,
    wav_only_logits_by_q,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Qwen3-Omni wav/text predictions for RASST soft-wav manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--adapter")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--speaker", default="Ethan")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--thinker-decode-mode", choices=["hf-generate", "manual-greedy"], default="hf-generate")
    parser.add_argument("--thinker-max-new-tokens", type=int, default=128)
    parser.add_argument("--thinker-do-sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--talker-max-new-tokens", type=int, default=1024)
    parser.add_argument("--talker-do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--talker-temperature", type=float, default=0.9)
    parser.add_argument("--talker-top-k", type=int, default=50)
    parser.add_argument("--talker-top-p", type=float, default=1.0)
    parser.add_argument("--audio-mode", choices=["native", "manual-st-argmax"], default="native")
    parser.add_argument("--manual-codec-frame-rate", type=float, default=12.5)
    parser.add_argument("--manual-chars-per-second", type=float, default=4.5)
    parser.add_argument("--manual-min-frames", type=int, default=4)
    parser.add_argument("--manual-max-frames", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def _adapter_model_path(adapter_dir: str) -> Path:
    path = Path(adapter_dir) / "adapter_model.safetensors"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _module_name_from_lora_key(key: str) -> str:
    return key.removeprefix("base_model.model.").removesuffix(".lora_A.weight")


def merge_lora_into_omni(model: Any, adapter_dir: str) -> dict[str, Any]:
    import torch
    from safetensors.torch import safe_open

    adapter_path = _adapter_model_path(adapter_dir)
    adapter_config = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    rank = int(adapter_config.get("r", 16))
    alpha = int(adapter_config.get("lora_alpha", 32))
    scaling = alpha / rank
    modules = dict(model.named_modules())
    merged = 0
    skipped = []
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
            a = tensors.get_tensor(key).to(device=weight.device, dtype=torch.float32)
            b = tensors.get_tensor(b_key).to(device=weight.device, dtype=torch.float32)
            delta = torch.matmul(b, a) * scaling
            if tuple(delta.shape) != tuple(weight.shape):
                skipped.append(module_name)
                continue
            weight.data.add_(delta.to(dtype=weight.dtype))
            merged += 1
    return {
        "adapter_path": str(adapter_path),
        "merged": merged,
        "skipped": len(skipped),
        "skipped_examples": skipped[:20],
    }


def audio_to_numpy(audio: Any) -> np.ndarray:
    if audio is None:
        return np.zeros((0,), dtype=np.float32)
    if hasattr(audio, "detach"):
        audio = audio.reshape(-1).detach().float().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return np.clip(audio, -1.0, 1.0)


class OmniManifestGenerator:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        torch.manual_seed(260623)
        self.torch = torch
        self.args = args
        patch_qwen3_omni_no_split(Qwen3OmniMoeForConditionalGeneration)
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(args.model, trust_remote_code=True)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            args.model,
            trust_remote_code=True,
            dtype="auto",
            device_map=args.device_map,
        ).eval()
        if not getattr(self.model, "has_talker", False):
            self.model.enable_talker()
        self.merge_stats = None
        if args.adapter:
            self.merge_stats = merge_lora_into_omni(self.model, args.adapter)
            print(json.dumps({"manual_lora_merge": self.merge_stats}, ensure_ascii=False), flush=True)

    def _prepare_prompt_inputs(self, record: dict[str, Any]) -> tuple[Any, int]:
        from qwen_omni_utils import process_mm_info

        messages = messages_with_audio_content(record)
        prompt_messages = messages[:-1]
        text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        audios, images, videos = process_mm_info(prompt_messages, use_audio_in_video=False)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
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
        return inputs, int(inputs["input_ids"].shape[1])

    def _decode_generated_text(self, text_ids: Any, prompt_len: int) -> tuple[str, dict[str, Any]]:
        sequences = text_ids.sequences if hasattr(text_ids, "sequences") else text_ids
        generated = sequences[:, prompt_len:]
        ids = [int(token_id) for token_id in generated[0].detach().cpu().tolist()]
        tokenizer = self.processor.tokenizer
        vocab_size = int(len(tokenizer))
        eos_token_ids = tokenizer.eos_token_id
        if eos_token_ids is None:
            eos_set: set[int] = set()
        elif isinstance(eos_token_ids, int):
            eos_set = {int(eos_token_ids)}
        else:
            eos_set = {int(token_id) for token_id in eos_token_ids}
        valid_ids = []
        invalid_ids = []
        for token_id in ids:
            if token_id in eos_set:
                break
            if 0 <= token_id < vocab_size:
                valid_ids.append(token_id)
            else:
                invalid_ids.append(token_id)
        text = tokenizer.decode(
            valid_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        stats = {
            "raw_new_tokens": len(ids),
            "decoded_text_tokens": len(valid_ids),
            "invalid_text_token_count": len(invalid_ids),
            "invalid_text_token_examples": invalid_ids[:8],
        }
        if ids:
            stats["raw_text_token_min"] = min(ids)
            stats["raw_text_token_max"] = max(ids)
        return text, stats

    def generate_text_only(self, record: dict[str, Any]) -> str:
        if self.args.thinker_decode_mode == "manual-greedy":
            return self.generate_text_manual_greedy(record)
        inputs, prompt_len = self._prepare_prompt_inputs(record)
        with self.torch.inference_mode():
            result = self.model.generate(
                **inputs,
                speaker=self.args.speaker,
                use_audio_in_video=False,
                return_audio=False,
                thinker_return_dict_in_generate=True,
                thinker_max_new_tokens=self.args.thinker_max_new_tokens,
                thinker_do_sample=self.args.thinker_do_sample,
            )
        text_ids = result[0] if isinstance(result, tuple) else result
        text, stats = self._decode_generated_text(text_ids, prompt_len)
        self.last_text_decode_stats = stats
        return text

    def generate_text_manual_greedy(self, record: dict[str, Any]) -> str:
        inputs, _prompt_len = self._prepare_prompt_inputs(record)
        tokenizer = self.processor.tokenizer
        vocab_size = int(len(tokenizer))
        eos_token_id = tokenizer.eos_token_id
        if not isinstance(eos_token_id, int):
            eos_token_id = None
        suppress_ids = {
            int(token_id)
            for token_id in getattr(tokenizer, "all_special_ids", [])
            if int(token_id) != eos_token_id
        }
        thinker_keys = {
            "input_ids",
            "input_features",
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "attention_mask",
            "feature_attention_mask",
            "audio_feature_lengths",
            "video_second_per_grid",
        }
        generated_ids: list[int] = []
        with self.torch.inference_mode():
            for _step in range(int(self.args.thinker_max_new_tokens)):
                kwargs = {key: value for key, value in inputs.items() if key in thinker_keys}
                kwargs["use_audio_in_video"] = False
                kwargs["return_dict"] = True
                kwargs["use_cache"] = False
                outputs = self.model.thinker(**kwargs)
                scores = outputs.logits[:, -1, :].float()
                require_finite("manual greedy thinker scores", scores)
                if vocab_size < scores.shape[-1]:
                    scores[:, vocab_size:] = -float("inf")
                for token_id in suppress_ids:
                    if 0 <= token_id < scores.shape[-1]:
                        scores[:, token_id] = -float("inf")
                next_token = int(scores.argmax(dim=-1)[0].detach().cpu())
                if eos_token_id is not None and next_token == eos_token_id:
                    break
                generated_ids.append(next_token)
                token_tensor = self.torch.tensor(
                    [[next_token]],
                    dtype=inputs["input_ids"].dtype,
                    device=inputs["input_ids"].device,
                )
                inputs["input_ids"] = self.torch.cat([inputs["input_ids"], token_tensor], dim=1)
                if "attention_mask" in inputs:
                    mask_tensor = self.torch.ones(
                        (inputs["attention_mask"].shape[0], 1),
                        dtype=inputs["attention_mask"].dtype,
                        device=inputs["attention_mask"].device,
                    )
                    inputs["attention_mask"] = self.torch.cat([inputs["attention_mask"], mask_tensor], dim=1)
        text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        self.last_text_decode_stats = {
            "thinker_decode_mode": self.args.thinker_decode_mode,
            "raw_new_tokens": len(generated_ids),
            "decoded_text_tokens": len(generated_ids),
            "invalid_text_token_count": 0,
            "invalid_text_token_examples": [],
        }
        if generated_ids:
            self.last_text_decode_stats["raw_text_token_min"] = min(generated_ids)
            self.last_text_decode_stats["raw_text_token_max"] = max(generated_ids)
        return text

    def manual_frame_count(self, text: str) -> int:
        chars = len("".join(str(text or "").split()))
        if chars <= 0:
            return 0
        seconds = chars / max(float(self.args.manual_chars_per_second), 1.0e-6)
        frames = int(np.ceil(seconds * float(self.args.manual_codec_frame_rate)))
        return max(int(self.args.manual_min_frames), min(int(self.args.manual_max_frames), frames))

    def render_predicted_text(self, record: dict[str, Any], text: str) -> tuple[np.ndarray, dict[str, Any]]:
        if not str(text or "").strip():
            return np.zeros((0,), dtype=np.float32), {
                "audio_mode": self.args.audio_mode,
                "generated_codec_frames": 0,
                "manual_chars_per_second": self.args.manual_chars_per_second,
            }
        render_record = dict(record)
        render_messages = [dict(message) for message in render_record.get("messages") or []]
        if render_messages and render_messages[-1].get("role") == "assistant":
            render_messages[-1]["content"] = text
        else:
            render_messages.append({"role": "assistant", "content": text})
        render_record["messages"] = render_messages
        render_record["target_text"] = text
        frames = self.manual_frame_count(text)
        with self.torch.inference_mode():
            inputs, _prompt_len, _target_token_len = load_processor_inputs(
                self.processor,
                render_record,
                self.model,
            )
            condition = prepare_talker_condition(
                self.model,
                inputs,
                speaker=self.args.speaker,
                detach_thinker=True,
            )
            require_finite("manual talker condition inputs_embeds", condition.inputs_embeds)
            require_finite("manual talker condition trailing_text_hidden", condition.trailing_text_hidden)
            logits_by_q = wav_only_logits_by_q(
                self.model,
                condition,
                frames,
                temperature=1.0,
                mode="st_argmax",
                detach_frame_feedback=True,
            )
            wav = soft_code2wav(
                self.model.code2wav,
                logits_by_q,
                temperature=1.0,
                mode="st_argmax",
            )
        return audio_to_numpy(wav), {
            "audio_mode": self.args.audio_mode,
            "generated_codec_frames": frames,
            "manual_chars_per_second": self.args.manual_chars_per_second,
            "manual_codec_frame_rate": self.args.manual_codec_frame_rate,
        }

    def generate_one(self, record: dict[str, Any]) -> tuple[str, np.ndarray, dict[str, Any]]:
        if self.args.audio_mode == "manual-st-argmax":
            text_out = self.generate_text_only(record)
            wav, debug = self.render_predicted_text(record, text_out)
            debug.update(getattr(self, "last_text_decode_stats", {}))
            return text_out, wav, debug
        inputs, prompt_len = self._prepare_prompt_inputs(record)
        with self.torch.inference_mode():
            text_ids, audio = self.model.generate(
                **inputs,
                speaker=self.args.speaker,
                use_audio_in_video=False,
                return_audio=True,
                thinker_return_dict_in_generate=True,
                thinker_max_new_tokens=self.args.thinker_max_new_tokens,
                thinker_do_sample=self.args.thinker_do_sample,
                talker_max_new_tokens=self.args.talker_max_new_tokens,
                talker_do_sample=self.args.talker_do_sample,
                talker_temperature=self.args.talker_temperature,
                talker_top_k=self.args.talker_top_k,
                talker_top_p=self.args.talker_top_p,
            )
        text_out, decode_stats = self._decode_generated_text(text_ids, prompt_len)
        return text_out, audio_to_numpy(audio), {"audio_mode": self.args.audio_mode, **decode_stats}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
        handle.flush()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    wav_dir = output_dir / "wav"
    pred_path = output_dir / "predictions.jsonl"
    if pred_path.exists():
        pred_path.unlink()
    records = read_jsonl(args.manifest)
    if args.max_records > 0:
        records = records[: args.max_records]
    generator = OmniManifestGenerator(args)
    for index, record in enumerate(records, start=1):
        row = {"id": record.get("id"), "target_text": record.get("target_text")}
        try:
            text, wav, debug = generator.generate_one(record)
            wav_path = wav_dir / f"{record['id']}.wav"
            write_mono_wav(wav_path, wav, args.sample_rate)
            row.update(
                {
                    "prediction_text": text,
                    "generated_wav_path": str(wav_path),
                    "generated_duration_s": round(float(wav.size) / float(args.sample_rate), 6),
                    "accepted": True,
                    **debug,
                }
            )
        except Exception as exc:
            row.update(
                {
                    "accepted": False,
                    "error": str(exc),
                    "reject_reasons": [f"exception:{type(exc).__name__}"],
                }
            )
        append_jsonl(pred_path, row)
        if args.log_every > 0 and index % args.log_every == 0:
            print(json.dumps({"processed": index, "last_id": record.get("id")}, ensure_ascii=False), flush=True)
    print(json.dumps({"output": str(pred_path), "records": len(records)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
