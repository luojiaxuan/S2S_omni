#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from s2s_omni.io import read_jsonl
from s2s_omni.prompts import SYSTEM_COMPRESSION, build_compression_user_prompt
from s2s_omni.schema import S2SSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Qwen3-Omni-compatible talker code labels from fixed target text "
            "under a target speech-duration budget."
        )
    )
    parser.add_argument("--input", nargs="+", required=True, help="Sample/label/SFT JSONL files.")
    parser.add_argument(
        "--sample-manifest",
        help="Optional manifest with audio_path and budgets, keyed by id, for message-only SFT JSONL.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--speaker", default="Ethan")
    parser.add_argument(
        "--audio-prefix-map",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Rewrite audio URI prefixes before reading, useful when host mounts differ.",
    )
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--codec-frame-rate", type=float, default=12.5)
    parser.add_argument("--budget-margin-frames", type=int, default=12)
    parser.add_argument("--min-duration-ratio", type=float, default=0.35)
    parser.add_argument("--max-duration-ratio", type=float, default=1.1)
    parser.add_argument("--eos-boost", type=float, default=4.0)
    parser.add_argument("--talker-do-sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--talker-temperature", type=float, default=0.9)
    parser.add_argument("--talker-top-k", type=int, default=50)
    parser.add_argument("--talker-top-p", type=float, default=1.0)
    parser.add_argument("--talker-repetition-penalty", type=float, default=1.05)
    parser.add_argument(
        "--code-predictor-do-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sample residual codec groups inside the Omni talker. Greedy is safer for labels.",
    )
    parser.add_argument("--decode-audio", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-rejected", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def target_from_messages(messages: list[dict[str, Any]]) -> str:
    if not messages or messages[-1].get("role") != "assistant":
        return ""
    content = messages[-1].get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return " ".join(parts).strip()
    return ""


def load_sample_manifest(path: str | None) -> dict[str, S2SSample]:
    if not path:
        return {}
    samples: dict[str, S2SSample] = {}
    for record in read_jsonl(path):
        sample = S2SSample.from_dict(record.get("sample") or record)
        samples[sample.id] = sample
    return samples


def iter_training_items(
    paths: Iterable[str],
    sample_manifest: dict[str, S2SSample],
) -> Iterable[tuple[S2SSample, str, dict[str, Any]]]:
    for path in paths:
        for record in read_jsonl(path):
            sample_payload = record.get("sample") if isinstance(record.get("sample"), dict) else None
            target_text = str(record.get("compressed_translation") or "").strip()
            if sample_payload is not None:
                sample = S2SSample.from_dict(sample_payload)
                if target_text:
                    sample.compressed_translation = target_text
                target_text = target_text or sample.preferred_target_text()
                yield sample, target_text, record
                continue

            messages = record.get("messages")
            if isinstance(messages, list):
                sample_id = str(record.get("id") or "")
                sample = sample_manifest.get(sample_id)
                if sample is None:
                    continue
                target_text = target_from_messages(messages)
                if target_text:
                    sample.compressed_translation = target_text
                yield sample, target_text or sample.preferred_target_text(), record
                continue

            sample = S2SSample.from_dict(record)
            target_text = target_text or sample.preferred_target_text()
            yield sample, target_text, record


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value[:180] or "sample"


def rewrite_audio_uri(uri: str, prefix_maps: list[str]) -> str:
    for item in prefix_maps:
        if "=" not in item:
            raise ValueError(f"invalid --audio-prefix-map {item!r}; expected OLD=NEW")
        old, new = item.split("=", 1)
        if old and uri.startswith(old):
            return new + uri[len(old) :]
    return uri


def module_device(module: Any) -> Any:
    return next(module.parameters()).device


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")
        handle.flush()


def read_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = str(record.get("id") or "")
            if sample_id:
                ids.add(sample_id)
    return ids


def messages_for_forced_target(sample: S2SSample, target_text: str) -> list[dict[str, Any]]:
    user_text = build_compression_user_prompt(sample, include_reference=False)
    if sample.audio_path:
        user_content: str | list[dict[str, str]] = [
            {"type": "audio", "audio": sample.audio_path},
            {"type": "text", "text": user_text},
        ]
    else:
        user_content = user_text
    return [
        {"role": "system", "content": SYSTEM_COMPRESSION.strip()},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": target_text},
    ]


class EosBudgetLogitsProcessor:
    def __init__(
        self,
        eos_token_id: int,
        min_new_tokens: int,
        budget_new_tokens: int,
        eos_boost: float,
    ) -> None:
        self.eos_token_id = eos_token_id
        self.min_new_tokens = max(0, min_new_tokens)
        self.budget_new_tokens = max(self.min_new_tokens, budget_new_tokens)
        self.eos_boost = eos_boost
        self.step = 0

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if self.step < self.min_new_tokens:
            scores[:, self.eos_token_id] = scores[:, self.eos_token_id] - 1.0e4
        elif self.step >= self.budget_new_tokens and self.eos_boost:
            scores[:, self.eos_token_id] = scores[:, self.eos_token_id] + self.eos_boost
        self.step += 1
        return scores


class ConstrainedTalkerCodeGenerator:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration
        from transformers import Qwen3OmniMoeProcessor

        torch.manual_seed(260622)
        self.torch = torch
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
        if not getattr(self.model, "has_talker", False):
            self.model.enable_talker()
        if not args.code_predictor_do_sample:
            self._patch_code_predictor_greedy_decode()

    def _patch_code_predictor_greedy_decode(self) -> None:
        original_generate = self.model.talker.code_predictor.generate

        def generate_without_sampling(*call_args: Any, **kwargs: Any) -> Any:
            kwargs["do_sample"] = False
            kwargs["remove_invalid_values"] = True
            kwargs["renormalize_logits"] = True
            return original_generate(*call_args, **kwargs)

        self.model.talker.code_predictor.generate = generate_without_sampling

    def generate_one(self, sample: S2SSample, target_text: str) -> dict[str, Any]:
        torch = self.torch
        if not sample.audio_path:
            raise ValueError("sample is missing audio_path")
        audio_uri = rewrite_audio_uri(sample.audio_path, self.args.audio_prefix_map)
        audio, _ = load_audio_span(audio_uri, target_sample_rate=self.args.audio_sample_rate)
        messages = messages_for_forced_target(sample, target_text)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        inputs = self.processor(
            text=text,
            audio=[audio],
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

        budget_s = sample.target.max_target_duration_s or sample.timing.src_duration_s
        if budget_s is None:
            budget_s = max(1.0, len(target_text) / 5.0)
        budget_frames = max(1, int(round(float(budget_s) * self.args.codec_frame_rate)))
        min_frames = max(1, int(round(budget_frames * self.args.min_duration_ratio)))
        max_new_tokens = max(min_frames + 1, budget_frames + self.args.budget_margin_frames)

        with torch.inference_mode():
            talker_input = self._prepare_talker_inputs(inputs)
            talker_result = self._generate_talker(
                talker_input,
                min_frames=min_frames,
                budget_frames=budget_frames,
                max_new_tokens=max_new_tokens,
            )
            codes = self._extract_codes(talker_result)
            wav = None
            if self.args.decode_audio:
                wav = self.model.code2wav.chunked_decode(
                    codes,
                    chunk_size=300,
                    left_context_size=25,
                ).float()

        frames = int(codes.shape[-1])
        duration_s = frames / float(self.args.codec_frame_rate)
        accepted = duration_s <= float(budget_s) * self.args.max_duration_ratio
        reject_reasons = [] if accepted else ["over_duration_budget"]
        return {
            "codes": codes.detach().cpu().to(torch.int16).numpy(),
            "wav": None if wav is None else wav.reshape(-1).detach().cpu().float().numpy(),
            "codec_frames": frames,
            "codec_duration_s": round(duration_s, 3),
            "budget_duration_s": round(float(budget_s), 3),
            "budget_frames": budget_frames,
            "min_frames": min_frames,
            "max_new_tokens": max_new_tokens,
            "accepted": accepted,
            "reject_reasons": reject_reasons,
        }

    def _prepare_talker_inputs(self, inputs: Any) -> dict[str, Any]:
        torch = self.torch
        model = self.model
        input_ids = inputs["input_ids"]
        thinker_kwargs = {
            key: value
            for key, value in inputs.items()
            if key
            in {
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
        }
        thinker_kwargs["use_audio_in_video"] = False
        thinker_kwargs["output_hidden_states"] = True
        thinker_kwargs["return_dict"] = True
        thinker_kwargs["use_cache"] = False
        thinker_outputs = model.thinker(**thinker_kwargs)
        thinker_embed = thinker_outputs.hidden_states[0]
        thinker_hidden = thinker_outputs.hidden_states[
            model.config.talker_config.accept_hidden_layer
        ]

        im_starts = torch.nonzero(input_ids[0] == model.config.im_start_token_id).flatten()
        im_start_indexes = torch.cat(
            (
                im_starts,
                torch.tensor([input_ids.shape[-1]], device=input_ids.device, dtype=input_ids.dtype),
            ),
            dim=-1,
        )
        multimodal_mask = (
            (input_ids == model.config.thinker_config.audio_token_id)
            | (input_ids == model.config.thinker_config.image_token_id)
            | (input_ids == model.config.thinker_config.video_token_id)
        ).to(input_ids.device)

        speaker_id = model.config.talker_config.speaker_id.get(self.args.speaker.lower())
        if speaker_id is None:
            raise NotImplementedError(f"Speaker {self.args.speaker} not implemented")
        text_projection_device = module_device(model.talker.text_projection)
        thinker_embedding = model.thinker.get_input_embeddings()
        thinker_embedding_device = module_device(thinker_embedding)
        talker_special_tokens = torch.tensor(
            [
                [
                    model.config.tts_bos_token_id,
                    model.config.tts_eos_token_id,
                    model.config.tts_pad_token_id,
                ]
            ],
            device=thinker_embedding_device,
            dtype=input_ids.dtype,
        )
        tts_bos_embed, tts_eos_embed, tts_pad_embed = (
            model.talker.text_projection(
                thinker_embedding(talker_special_tokens).to(text_projection_device)
            )
            .chunk(3, dim=1)
        )

        talker_input_embeds = []
        talker_input_ids = []
        trailing_text_hidden = None
        for idx in range(len(im_start_indexes) - 1):
            im_start_index = int(im_start_indexes[idx].item())
            segment_end_index = int(im_start_indexes[idx + 1].item())
            role_token = input_ids[0][im_start_index + 1]
            if role_token == model.config.system_token_id:
                continue
            if role_token == model.config.user_token_id:
                user_part = self._get_talker_user_parts(
                    im_start_index,
                    segment_end_index,
                    multimodal_mask,
                    thinker_hidden,
                    thinker_embed,
                )
                talker_input_embeds.append(user_part)
                talker_input_ids.append(input_ids[:, im_start_index:segment_end_index])
                continue
            if role_token == model.config.assistant_token_id:
                assistant_embeds, assistant_ids, trailing_text_hidden = self._get_talker_assistant_parts(
                    im_start_index,
                    segment_end_index,
                    speaker_id,
                    thinker_embed,
                    tts_pad_embed,
                    tts_bos_embed,
                    tts_eos_embed,
                )
                talker_input_embeds.append(assistant_embeds)
                talker_input_ids.append(assistant_ids)
                continue
            raise AssertionError("Expected role id after <|im_start|>")

        if trailing_text_hidden is None:
            raise RuntimeError("failed to locate assistant segment for talker conditioning")
        talker_input_id = torch.cat(
            [ids.to(text_projection_device) for ids in talker_input_ids],
            dim=1,
        )
        return {
            "inputs_embeds": torch.cat(
                [embed.to(text_projection_device) for embed in talker_input_embeds],
                dim=1,
            ),
            "trailing_text_hidden": trailing_text_hidden,
            "tts_pad_embed": tts_pad_embed,
            "talker_input_ids": talker_input_id,
            "attention_mask": torch.ones_like(talker_input_id, dtype=torch.long),
        }

    def _get_talker_user_parts(
        self,
        im_start_index: int,
        segment_end_index: int,
        multimodal_mask: Any,
        thinker_hidden: Any,
        thinker_embed: Any,
    ) -> Any:
        torch = self.torch
        model = self.model
        text_projection_device = module_device(model.talker.text_projection)
        user_talker_part = torch.empty(
            (
                1,
                segment_end_index - im_start_index,
                model.config.talker_config.text_config.hidden_size,
            ),
            device=text_projection_device,
            dtype=model.talker.dtype,
        )
        user_mm_mask = multimodal_mask[:, im_start_index:segment_end_index].to(text_projection_device)
        if user_mm_mask.any():
            hidden_mm = thinker_hidden[:, im_start_index:segment_end_index].to(text_projection_device)
            user_talker_part[user_mm_mask] = model.talker.hidden_projection(
                hidden_mm[user_mm_mask]
            )
        text_embed = thinker_embed[:, im_start_index:segment_end_index].to(text_projection_device)
        user_talker_part[~user_mm_mask] = model.talker.text_projection(text_embed[~user_mm_mask])
        return user_talker_part

    def _get_talker_assistant_parts(
        self,
        im_start_index: int,
        segment_end_index: int,
        speaker_id: int,
        thinker_embed: Any,
        tts_pad_embed: Any,
        tts_bos_embed: Any,
        tts_eos_embed: Any,
    ) -> tuple[Any, Any, Any]:
        torch = self.torch
        model = self.model
        text_projection_device = module_device(model.talker.text_projection)
        codec_embedding = model.talker.get_input_embeddings()
        codec_embedding_device = module_device(codec_embedding)

        assistant_hidden = model.talker.text_projection(
            thinker_embed[:, im_start_index:segment_end_index].to(text_projection_device)
        )
        assistant_text_hidden = torch.cat(
            (
                assistant_hidden[:, :3],
                tts_pad_embed.expand(-1, 4, -1),
                tts_bos_embed,
                assistant_hidden[:, 3:4],
            ),
            dim=1,
        )
        codec_special_tokens = torch.tensor(
            [
                [
                    model.config.talker_config.codec_nothink_id,
                    model.config.talker_config.codec_think_bos_id,
                    model.config.talker_config.codec_think_eos_id,
                    speaker_id,
                    model.config.talker_config.codec_pad_id,
                    model.config.talker_config.codec_bos_id,
                ]
            ],
            device=codec_embedding_device,
            dtype=torch.long,
        )
        assistant_codec_hidden = torch.cat(
            (
                torch.zeros(
                    (1, 3, model.config.talker_config.text_config.hidden_size),
                    device=codec_embedding_device,
                    dtype=model.talker.dtype,
                ),
                codec_embedding(codec_special_tokens),
            ),
            dim=1,
        ).to(text_projection_device)
        trailing_text_hidden = torch.cat((assistant_hidden[:, 4:], tts_eos_embed), dim=1)
        input_ids = torch.full(
            (1, assistant_text_hidden.shape[1]),
            fill_value=model.config.tts_pad_token_id,
            dtype=torch.long,
            device=text_projection_device,
        )
        return assistant_text_hidden + assistant_codec_hidden, input_ids, trailing_text_hidden

    def _generate_talker(
        self,
        talker_input: dict[str, Any],
        min_frames: int,
        budget_frames: int,
        max_new_tokens: int,
    ) -> Any:
        from transformers import LogitsProcessorList

        model = self.model
        suppressed_tokens = [
            token_id
            for token_id in range(
                model.config.talker_config.text_config.vocab_size - 1024,
                model.config.talker_config.text_config.vocab_size,
            )
            if token_id != model.config.talker_config.codec_eos_token_id
        ]
        logits_processor = LogitsProcessorList(
            [
                EosBudgetLogitsProcessor(
                    eos_token_id=model.config.talker_config.codec_eos_token_id,
                    min_new_tokens=min_frames,
                    budget_new_tokens=budget_frames,
                    eos_boost=self.args.eos_boost,
                )
            ]
        )
        return model.talker.generate(
            **talker_input,
            max_new_tokens=max_new_tokens,
            do_sample=self.args.talker_do_sample,
            top_k=self.args.talker_top_k,
            top_p=self.args.talker_top_p,
            temperature=self.args.talker_temperature,
            eos_token_id=model.config.talker_config.codec_eos_token_id,
            forced_eos_token_id=model.config.talker_config.codec_eos_token_id,
            repetition_penalty=self.args.talker_repetition_penalty,
            suppress_tokens=suppressed_tokens,
            logits_processor=logits_processor,
            remove_invalid_values=True,
            renormalize_logits=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    def _extract_codes(self, talker_result: Any) -> Any:
        torch = self.torch
        hidden_codes = [hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None]
        if not hidden_codes:
            raise RuntimeError("talker did not return hidden-state codec labels")
        return torch.stack(hidden_codes, dim=1).transpose(1, 2).contiguous()


def write_wav(path: Path, wav: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.nan_to_num(np.asarray(wav, dtype=np.float32).reshape(-1))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    sf.write(path, np.clip(audio, -1.0, 1.0), sample_rate, subtype="PCM_16")


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")

    output_dir = Path(args.output_dir)
    codes_dir = output_dir / "codes"
    wav_dir = output_dir / "wav"
    accepted_path = output_dir / "talker_code_labels.jsonl"
    rejected_path = output_dir / "talker_code_labels_rejected.jsonl"
    if not args.resume:
        for path in [accepted_path, rejected_path]:
            if path.exists():
                path.unlink()
    existing_accepted = read_existing_ids(accepted_path) if args.resume else set()
    existing_rejected = read_existing_ids(rejected_path) if args.resume else set()
    done_ids = existing_accepted | existing_rejected

    sample_manifest = load_sample_manifest(args.sample_manifest)
    wanted = set(args.ids or [])
    selected = []
    eligible_index = 0
    for sample, target_text, source_record in iter_training_items(args.input, sample_manifest):
        if wanted and sample.id not in wanted:
            continue
        if not target_text.strip():
            continue
        if eligible_index % args.num_shards == args.shard_index:
            if sample.id not in done_ids:
                selected.append((sample, target_text, source_record))
                if args.max_records > 0 and len(selected) >= args.max_records:
                    break
        eligible_index += 1
    if not selected and not done_ids:
        raise SystemExit("no records selected")

    accepted_count = 0
    rejected_count = 0
    generator = ConstrainedTalkerCodeGenerator(args) if selected else None
    for index, (sample, target_text, source_record) in enumerate(selected, start=1):
        row: dict[str, Any] = {
            "id": sample.id,
            "target_text": target_text,
            "source_audio": sample.audio_path,
            "resolved_source_audio": (
                rewrite_audio_uri(sample.audio_path, args.audio_prefix_map)
                if sample.audio_path
                else None
            ),
            "sample": sample.to_dict(),
            "speaker": args.speaker,
            "model": args.model,
            "source_record_keys": sorted(source_record.keys()),
        }
        try:
            result = generator.generate_one(sample, target_text)
            codes_name = f"{sanitize_name(sample.id)}.npy"
            codes_path = codes_dir / codes_name
            codes_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(codes_path, result.pop("codes"))
            row.update(result)
            row["codes_path"] = str(codes_path)
            row["codec_num_quantizers"] = 16
            if row.get("wav") is not None:
                wav_path = wav_dir / f"{sanitize_name(sample.id)}.wav"
                write_wav(wav_path, row.pop("wav"), args.sample_rate)
                row["wav_path"] = str(wav_path)
            else:
                row.pop("wav", None)
        except Exception as exc:
            row.update(
                {
                    "accepted": False,
                    "reject_reasons": [f"exception:{type(exc).__name__}"],
                    "error": str(exc),
                }
            )
        if row.get("accepted"):
            append_jsonl(accepted_path, row)
            accepted_count += 1
        else:
            if args.save_rejected:
                append_jsonl(rejected_path, row)
            rejected_count += 1
        if args.log_every > 0 and index % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "processed": index,
                        "accepted": accepted_count,
                        "rejected": rejected_count,
                        "existing_accepted": len(existing_accepted),
                        "existing_rejected": len(existing_rejected),
                        "last_id": sample.id,
                        "last_duration_s": row.get("codec_duration_s"),
                        "last_budget_s": row.get("budget_duration_s"),
                        "last_reject_reasons": row.get("reject_reasons", []),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = {
        "input": args.input,
        "sample_manifest": args.sample_manifest,
        "model": args.model,
        "speaker": args.speaker,
        "selected_this_run": len(selected),
        "accepted_this_run": accepted_count,
        "rejected_this_run": rejected_count,
        "existing_accepted": len(existing_accepted),
        "existing_rejected": len(existing_rejected),
        "accepted_total": len(existing_accepted) + accepted_count,
        "rejected_total": len(existing_rejected) + rejected_count,
        "output": str(accepted_path),
        "rejected_output": str(rejected_path),
        "codec_frame_rate": args.codec_frame_rate,
        "budget_margin_frames": args.budget_margin_frames,
        "min_duration_ratio": args.min_duration_ratio,
        "max_duration_ratio": args.max_duration_ratio,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
