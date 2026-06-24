#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_jsonl
from s2s_omni.omni_talker import (
    logits_to_code_embedding,
    multi_resolution_stft_loss,
    patch_qwen3_omni_no_split,
    prepare_talker_condition,
    soft_code2wav,
    waveform_l1_loss,
)
from s2s_omni.rasst import load_mono_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-Omni thinker+talker LoRA SFT with soft wav loss.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--dev-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--speaker", default="Ethan")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-dev-records", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-history-chunks", type=int, default=0)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--train-scope", choices=["all", "thinker", "talker"], default="all")
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
    )
    parser.add_argument("--text-ce-weight", type=float, default=1.0)
    parser.add_argument("--codec-ce-weight", type=float, default=1.0)
    parser.add_argument("--wav-l1-weight", type=float, default=0.2)
    parser.add_argument("--stft-weight", type=float, default=0.5)
    parser.add_argument("--eos-ce-weight", type=float, default=0.1)
    parser.add_argument("--nonempty-eos-avoid-weight", type=float, default=0.0)
    parser.add_argument("--code2wav-grad-mode", choices=["st_argmax", "soft"], default="st_argmax")
    parser.add_argument("--detach-wav-only-frame-feedback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detach-talker-condition-thinker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--soft-temperature-start", type=float, default=1.2)
    parser.add_argument("--soft-temperature-end", type=float, default=0.7)
    parser.add_argument("--max-codec-frames", type=int, default=96)
    parser.add_argument("--code2wav-hop-length", type=int, default=1920)
    parser.add_argument("--target-sample-rate", type=int, default=24000)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=260623)
    parser.add_argument("--resume-from")
    parser.add_argument("--tokenize-smoke", action="store_true")
    return parser.parse_args()


def make_scheduler(optimizer: Any, warmup_steps: int, total_steps: int):
    from torch.optim.lr_scheduler import LambdaLR

    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def trim_streaming_history(record: dict[str, Any], max_history_chunks: int) -> dict[str, Any]:
    if max_history_chunks <= 0:
        return record
    messages = list(record.get("messages") or [])
    audios = list(record.get("audios") or [])
    if not messages or messages[-1].get("role") != "assistant":
        return record
    system_messages = []
    rest = messages
    if messages[0].get("role") == "system":
        system_messages = [messages[0]]
        rest = messages[1:]
    pairs = []
    for index in range(0, len(rest), 2):
        if index + 1 >= len(rest):
            return record
        user_msg = rest[index]
        assistant_msg = rest[index + 1]
        if user_msg.get("role") != "user" or assistant_msg.get("role") != "assistant":
            return record
        pairs.append((user_msg, assistant_msg))
    if len(pairs) <= max_history_chunks:
        return record
    keep_pairs = pairs[-max_history_chunks:]
    trimmed = dict(record)
    trimmed_messages = list(system_messages)
    for user_msg, assistant_msg in keep_pairs:
        trimmed_messages.extend([user_msg, assistant_msg])
    trimmed["messages"] = trimmed_messages
    trimmed["audios"] = audios[-max_history_chunks:]
    return trimmed


def messages_with_audio_content(record: dict[str, Any]) -> list[dict[str, Any]]:
    audios = list(record.get("audios") or [])
    audio_index = 0
    out: list[dict[str, Any]] = []
    for message in record["messages"]:
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "user" and "<audio>" in content:
            if audio_index >= len(audios):
                raise ValueError(f"{record.get('id')}: not enough audios for user messages")
            text_part = content.replace("<audio>", "").strip()
            items: list[dict[str, str]] = [{"type": "audio", "audio": str(audios[audio_index])}]
            if text_part:
                items.append({"type": "text", "text": text_part})
            out.append({"role": "user", "content": items})
            audio_index += 1
        else:
            out.append({"role": role, "content": content})
    if audio_index != len(audios):
        raise ValueError(f"{record.get('id')}: unused audios {len(audios) - audio_index}")
    return out


def split_prompt_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("training record messages must end with assistant")
    return messages[:-1]


def load_processor_inputs(processor: Any, record: dict[str, Any], model: Any) -> tuple[Any, Any, int]:
    from qwen_omni_utils import process_mm_info

    record = trim_streaming_history(record, getattr(load_processor_inputs, "max_history_chunks", 0))
    messages = messages_with_audio_content(record)
    prompt_messages = split_prompt_messages(messages)
    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    prompt_audios, prompt_images, prompt_videos = process_mm_info(
        prompt_messages,
        use_audio_in_video=False,
    )
    inputs = processor(
        text=full_text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    prompt_inputs = processor(
        text=prompt_text,
        audio=prompt_audios,
        images=prompt_images,
        videos=prompt_videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    device = getattr(model, "device", next(model.parameters()).device)
    inputs = inputs.to(device)
    float_dtype = getattr(model, "dtype", None)
    if float_dtype is not None:
        for key in ["input_features", "pixel_values", "pixel_values_videos"]:
            if key in inputs:
                inputs[key] = inputs[key].to(dtype=float_dtype)
    target_text = str(record.get("target_text") or "")
    target_token_len = len(
        processor.tokenizer(target_text, add_special_tokens=False)["input_ids"]
    )
    return inputs, prompt_inputs["input_ids"].shape[1], target_token_len


def thinker_text_loss(model: Any, inputs: Any, prompt_len: int, target_token_len: int) -> Any:
    torch = _torch()
    input_len = inputs["input_ids"].shape[1]
    target_start = min(prompt_len, input_len)
    target_end = min(input_len, target_start + max(0, target_token_len))
    target_ids_cpu = (
        inputs["input_ids"][:, target_start:target_end]
        .detach()
        .cpu()
        .long()
    )
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
    kwargs = {key: value for key, value in inputs.items() if key in thinker_keys}
    kwargs["use_audio_in_video"] = False
    kwargs["return_dict"] = True
    outputs = model.thinker(**kwargs)
    shift_logits = outputs.logits[:, :-1].contiguous()
    shift_logits = torch.nan_to_num(
        shift_logits.float(),
        nan=0.0,
        posinf=1.0e4,
        neginf=-1.0e4,
    )
    labels = torch.full(
        inputs["input_ids"].shape,
        fill_value=-100,
        dtype=torch.long,
        device=shift_logits.device,
    )
    if target_ids_cpu.numel():
        labels[:, target_start:target_end] = target_ids_cpu.to(shift_logits.device)
    shift_labels = labels[:, 1:].contiguous()
    valid = (shift_labels != -100) & (shift_labels >= 0) & (shift_labels < shift_logits.shape[-1])
    if not bool(valid.any()):
        return shift_logits.new_zeros(())
    safe_labels = shift_labels.masked_fill(~valid, -100)
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        safe_labels.view(-1),
        ignore_index=-100,
    )
    if not bool(torch.isfinite(loss).detach().cpu()):
        return shift_logits.new_zeros(())
    return loss


def teacher_frame_embeds(model: Any, condition: Any, codes: Any) -> Any:
    torch = _torch()
    batch, quantizers, frames = codes.shape
    text_projection_device = condition.inputs_embeds.device
    primary_embedding = model.talker.get_input_embeddings()
    residual_embeddings = model.talker.code_predictor.get_input_embeddings()
    pieces = []
    for frame in range(frames):
        q0_ids = safe_code_ids(
            codes[:, 0, frame : frame + 1],
            primary_embedding.num_embeddings,
            module_device(primary_embedding),
        )
        q0 = primary_embedding(q0_ids)
        q0 = q0.to(text_projection_device)
        frame_embed = q0
        for q in range(1, quantizers):
            q_ids = safe_code_ids(
                codes[:, q, frame : frame + 1],
                residual_embeddings[q - 1].num_embeddings,
                module_device(residual_embeddings[q - 1]),
            )
            emb = residual_embeddings[q - 1](
                q_ids
            )
            frame_embed = frame_embed + emb.to(text_projection_device)
        if frame < condition.trailing_text_hidden.shape[1]:
            text_hidden = condition.trailing_text_hidden[:, frame : frame + 1].to(text_projection_device)
        else:
            text_hidden = condition.tts_pad_embed.to(text_projection_device)
        pieces.append(frame_embed + text_hidden)
    return torch.cat(pieces, dim=1) if pieces else condition.inputs_embeds[:, :0]


def talker_primary_forward(model: Any, condition: Any, codes: Any) -> tuple[Any, Any]:
    torch = _torch()
    frames = codes.shape[-1]
    if frames <= 0:
        out = model.talker(
            inputs_embeds=condition.inputs_embeds,
            attention_mask=condition.attention_mask,
            talker_input_ids=condition.talker_input_ids,
            trailing_text_hidden=condition.trailing_text_hidden,
            tts_pad_embed=condition.tts_pad_embed,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        return out.logits[:, -1:, :], out.hidden_states[0][-1][:, -1:, :]

    prev_frame_embeds = teacher_frame_embeds(model, condition, codes[:, :, :-1])
    inputs_embeds = torch.cat((condition.inputs_embeds, prev_frame_embeds), dim=1)
    pad_ids = torch.full(
        (codes.shape[0], max(0, frames - 1)),
        fill_value=model.config.tts_pad_token_id,
        dtype=condition.talker_input_ids.dtype,
        device=condition.talker_input_ids.device,
    )
    talker_input_ids = torch.cat((condition.talker_input_ids, pad_ids), dim=1)
    attention_mask = torch.ones_like(talker_input_ids, dtype=torch.long)
    out = model.talker(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        talker_input_ids=talker_input_ids,
        trailing_text_hidden=condition.trailing_text_hidden,
        tts_pad_embed=condition.tts_pad_embed,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    prefix_last = condition.inputs_embeds.shape[1] - 1
    positions = torch.arange(prefix_last, prefix_last + frames, device=out.logits.device)
    logits = out.logits.index_select(1, positions)
    hidden = out.hidden_states[0][-1].index_select(1, positions)
    return logits, hidden


def residual_logits_for_frame(model: Any, hidden: Any, q_codes: Any) -> list[Any]:
    torch = _torch()
    logits = []
    primary_embedding = model.talker.get_input_embeddings()
    q0_embed = model.talker.get_input_embeddings()(
        safe_code_ids(q_codes[:, 0:1], primary_embedding.num_embeddings, module_device(primary_embedding))
    ).to(hidden.device)
    out = model.talker.code_predictor(
        inputs_embeds=torch.cat((hidden, q0_embed), dim=1),
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    logits.append(out.logits[:, -1, :])
    past = out.past_key_values
    for q in range(1, q_codes.shape[1] - 1):
        out = model.talker.code_predictor(
            input_ids=safe_code_ids(
                q_codes[:, q : q + 1],
                model.config.talker_config.code_predictor_config.vocab_size,
                module_device(model.talker.code_predictor),
            ),
            past_key_values=past,
            generation_steps=q,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        logits.append(out.logits[:, -1, :])
        past = out.past_key_values
    return logits


def safe_code_ids(ids: Any, vocab_size: int, device: Any) -> Any:
    ids = ids.detach().long().to(device)
    valid = (ids >= 0) & (ids < int(vocab_size))
    return ids.masked_fill(~valid, 0)


def cross_entropy_ignore_invalid(logits: Any, labels: Any) -> Any:
    torch = _torch()
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1.0e4, neginf=-1.0e4)
    labels = labels.detach().cpu().long().to(logits.device)
    valid = (labels >= 0) & (labels < logits.shape[-1])
    if not bool(valid.any()):
        return logits.new_zeros(())
    safe_labels = labels.masked_fill(~valid, -100)
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        safe_labels.reshape(-1),
        ignore_index=-100,
    )


def add_loss_term(total: Any | None, term: Any) -> Any:
    if total is None:
        return term
    return total.to(term.device) + term


def codec_losses_and_logits(model: Any, condition: Any, codes: Any, max_frames: int) -> tuple[Any, Any]:
    torch = _torch()
    if codes.ndim != 3:
        raise ValueError(f"expected codes [B,Q,T], got {tuple(codes.shape)}")
    if max_frames > 0 and codes.shape[-1] > max_frames:
        codes = codes[:, :, :max_frames]
    primary_logits, primary_hidden = talker_primary_forward(model, condition, codes)
    primary_loss = cross_entropy_ignore_invalid(
        primary_logits,
        codes[:, 0, : primary_logits.shape[1]],
    )
    all_logits = [primary_logits]
    residual_loss = primary_loss.new_tensor(0.0)
    residual_count = 0
    residual_by_q: list[list[Any]] = [[] for _ in range(codes.shape[1] - 1)]
    for frame in range(primary_logits.shape[1]):
        frame_codes = codes[:, :, frame]
        frame_hidden = primary_hidden[:, frame : frame + 1]
        frame_residual_logits = residual_logits_for_frame(model, frame_hidden, frame_codes)
        for q, q_logits in enumerate(frame_residual_logits, start=1):
            residual_loss = residual_loss + cross_entropy_ignore_invalid(q_logits, frame_codes[:, q])
            residual_by_q[q - 1].append(q_logits.unsqueeze(1))
            residual_count += 1
    if residual_count:
        residual_loss = residual_loss / residual_count

    for q_logits in residual_by_q:
        all_logits.append(torch.cat(q_logits, dim=1))
    return primary_loss + residual_loss, all_logits


def residual_logits_for_frame_st(
    model: Any,
    hidden: Any,
    primary_logits: Any,
    *,
    temperature: float,
    mode: str,
) -> list[Any]:
    code_predictor = model.talker.code_predictor
    predictor_device = module_device(code_predictor)
    codebook = int(model.config.code2wav_config.codebook_size)
    primary_embedding = model.talker.get_input_embeddings()
    q0_embed = logits_to_code_embedding(
        primary_logits,
        primary_embedding.weight,
        vocab=min(codebook, primary_logits.shape[-1], primary_embedding.weight.shape[0]),
        temperature=temperature,
        mode=mode,
    ).to(predictor_device)
    out = code_predictor(
        inputs_embeds=torch_cat_device([hidden, q0_embed], predictor_device),
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    logits = [out.logits[:, -1, :]]
    past = out.past_key_values
    for q in range(1, int(model.config.code2wav_config.num_quantizers) - 1):
        prev_logits = logits[-1].unsqueeze(1)
        prev_ids = safe_code_ids(
            prev_logits.argmax(dim=-1),
            model.config.talker_config.code_predictor_config.vocab_size,
            predictor_device,
        )
        out = code_predictor(
            input_ids=prev_ids,
            past_key_values=past,
            generation_steps=q,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        logits.append(out.logits[:, -1, :])
        past = out.past_key_values
    return logits


def frame_embed_from_logits(
    model: Any,
    condition: Any,
    primary_logits: Any,
    residual_logits: list[Any],
    frame_index: int,
    *,
    temperature: float,
    mode: str,
) -> Any:
    text_projection_device = condition.inputs_embeds.device
    codebook = int(model.config.code2wav_config.codebook_size)
    primary_embedding = model.talker.get_input_embeddings()
    frame_embed = logits_to_code_embedding(
        primary_logits,
        primary_embedding.weight,
        vocab=min(codebook, primary_logits.shape[-1], primary_embedding.weight.shape[0]),
        temperature=temperature,
        mode=mode,
    ).to(text_projection_device)
    residual_embeddings = model.talker.code_predictor.get_input_embeddings()
    for q, q_logits in enumerate(residual_logits, start=1):
        q_logits = q_logits.unsqueeze(1) if q_logits.ndim == 2 else q_logits
        q_embedding = residual_embeddings[q - 1]
        frame_embed = frame_embed + logits_to_code_embedding(
            q_logits,
            q_embedding.weight,
            vocab=min(codebook, q_logits.shape[-1], q_embedding.weight.shape[0]),
            temperature=temperature,
            mode=mode,
        ).to(text_projection_device)
    if frame_index < condition.trailing_text_hidden.shape[1]:
        text_hidden = condition.trailing_text_hidden[:, frame_index : frame_index + 1].to(
            text_projection_device
        )
    else:
        text_hidden = condition.tts_pad_embed.to(text_projection_device)
    return frame_embed + text_hidden


def wav_only_logits_by_q(
    model: Any,
    condition: Any,
    frames: int,
    *,
    temperature: float,
    mode: str,
    detach_frame_feedback: bool,
) -> list[Any]:
    torch = _torch()
    frames = max(1, int(frames))
    quantizers = int(model.config.code2wav_config.num_quantizers)
    by_q: list[list[Any]] = [[] for _ in range(quantizers)]
    prev_embeds: list[Any] = []
    for frame in range(frames):
        if prev_embeds:
            generated_embeds = torch.cat(prev_embeds, dim=1)
            inputs_embeds = torch.cat((condition.inputs_embeds, generated_embeds), dim=1)
            pad_ids = torch.full(
                (condition.talker_input_ids.shape[0], len(prev_embeds)),
                fill_value=model.config.tts_pad_token_id,
                dtype=condition.talker_input_ids.dtype,
                device=condition.talker_input_ids.device,
            )
            talker_input_ids = torch.cat((condition.talker_input_ids, pad_ids), dim=1)
        else:
            inputs_embeds = condition.inputs_embeds
            talker_input_ids = condition.talker_input_ids
        out = model.talker(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones_like(talker_input_ids, dtype=torch.long),
            talker_input_ids=talker_input_ids,
            trailing_text_hidden=condition.trailing_text_hidden,
            tts_pad_embed=condition.tts_pad_embed,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        primary_logits = out.logits[:, -1:, :]
        hidden = out.hidden_states[0][-1][:, -1:, :]
        residual_logits = residual_logits_for_frame_st(
            model,
            hidden,
            primary_logits,
            temperature=temperature,
            mode=mode,
        )
        by_q[0].append(primary_logits)
        for q, q_logits in enumerate(residual_logits, start=1):
            by_q[q].append(q_logits.unsqueeze(1))
        next_frame_embed = frame_embed_from_logits(
            model,
            condition,
            primary_logits,
            residual_logits,
            frame,
            temperature=temperature,
            mode=mode,
        )
        if detach_frame_feedback:
            next_frame_embed = next_frame_embed.detach()
        prev_embeds.append(next_frame_embed)
    return [torch.cat(items, dim=1) for items in by_q]


def torch_cat_device(tensors: list[Any], device: Any) -> Any:
    torch = _torch()
    return torch.cat([tensor.to(device) for tensor in tensors], dim=1)


def frames_for_target_wav(target_wav: Any, args: argparse.Namespace) -> int:
    frames = int(math.ceil(float(target_wav.shape[-1]) / float(max(1, args.code2wav_hop_length))))
    if args.max_codec_frames > 0:
        frames = min(frames, int(args.max_codec_frames))
    return max(1, frames)


def eos_loss_for_empty(model: Any, condition: Any) -> Any:
    torch = _torch()
    logits, _hidden = talker_primary_forward(
        model,
        condition,
        torch.empty(
            (1, model.config.code2wav_config.num_quantizers, 0),
            dtype=torch.long,
            device=condition.inputs_embeds.device,
        ),
    )
    eos_id = int(model.config.talker_config.codec_eos_token_id)
    if not bool(torch.isfinite(logits).all().detach().cpu()):
        return logits.new_zeros(())
    if eos_id < 0 or eos_id >= logits.shape[-1]:
        return logits.new_zeros(())
    label = torch.tensor(
        [eos_id],
        dtype=torch.long,
        device=logits.device,
    )
    return torch.nn.functional.cross_entropy(logits[:, -1, :], label)


def eos_avoid_loss_for_nonempty(model: Any, primary_logits: Any) -> Any:
    torch = _torch()
    eos_id = int(model.config.talker_config.codec_eos_token_id)
    if eos_id < 0 or eos_id >= primary_logits.shape[-1]:
        return primary_logits.new_zeros(())
    finite = torch.isfinite(primary_logits).all()
    if not bool(finite.detach().cpu()):
        return primary_logits.new_zeros(())
    probs = torch.softmax(primary_logits.float(), dim=-1)[..., eos_id]
    probs = probs.clamp(min=0.0, max=1.0 - 1e-6)
    return -torch.log1p(-probs).mean()


def load_codes(path: str | Path, device: Any | None = None) -> Any:
    torch = _torch()
    codes = np.load(path)
    if codes.ndim == 3 and codes.shape[0] == 1:
        codes = codes[0]
    if codes.ndim != 2:
        raise ValueError(f"{path}: expected 2-D codes, got {codes.shape}")
    tensor = torch.from_numpy(codes.astype(np.int64, copy=False)).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def load_target_wav(path: str | Path, sample_rate: int, device: Any) -> Any:
    torch = _torch()
    wav, _ = load_mono_audio(path, target_sample_rate=sample_rate)
    return torch.from_numpy(wav).reshape(1, 1, -1).to(device)


def compute_record_loss(
    *,
    model: Any,
    processor: Any,
    record: dict[str, Any],
    args: argparse.Namespace,
    global_step: int,
    total_steps: int,
) -> dict[str, Any]:
    model = unwrap_peft_model(model)
    inputs, prompt_len, target_token_len = load_processor_inputs(processor, record, model)
    torch = _torch()
    if args.text_ce_weight > 0:
        if args.train_scope == "talker":
            with torch.no_grad():
                text_loss = thinker_text_loss(model, inputs, prompt_len, target_token_len)
        else:
            text_loss = thinker_text_loss(model, inputs, prompt_len, target_token_len)
    else:
        text_loss = torch.zeros((), dtype=torch.float32, device=inputs["input_ids"].device)
    condition = prepare_talker_condition(
        model,
        inputs,
        speaker=args.speaker,
        detach_thinker=args.detach_talker_condition_thinker,
    )
    text_ce_weight_for_total = 0.0 if args.train_scope == "talker" else args.text_ce_weight
    total = None
    if text_ce_weight_for_total:
        total = add_loss_term(total, text_ce_weight_for_total * text_loss)
    metrics: dict[str, Any] = {"text_loss": float(text_loss.detach().float().cpu())}

    has_codes = bool(record.get("target_codes_path"))
    has_wav = bool(record.get("target_wav_path"))
    if record.get("expected_empty_target") or not has_wav:
        eos_loss = eos_loss_for_empty(model, condition)
        total = add_loss_term(total, args.eos_ce_weight * eos_loss)
        metrics["eos_loss"] = float(eos_loss.detach().float().cpu())
        metrics["codec_loss"] = None
        metrics["wav_l1"] = None
        metrics["stft_loss"] = None
        if total is None:
            total = torch.zeros((), dtype=torch.float32, device=inputs["input_ids"].device)
        return {"loss": total, "metrics": metrics}

    progress = min(1.0, max(0.0, global_step / float(max(1, total_steps))))
    temperature = args.soft_temperature_start + progress * (
        args.soft_temperature_end - args.soft_temperature_start
    )
    target_wav = load_target_wav(
        record["target_wav_path"],
        args.target_sample_rate,
        condition.inputs_embeds.device if total is None else total.device,
    )
    if has_codes:
        codes = load_codes(record["target_codes_path"])
        codec_loss, logits_by_q = codec_losses_and_logits(model, condition, codes, args.max_codec_frames)
        total = add_loss_term(total, args.codec_ce_weight * codec_loss)
        metrics["codec_loss"] = float(codec_loss.detach().float().cpu())
        metrics["wav_only_frames"] = 0.0
    else:
        target_frames = frames_for_target_wav(target_wav, args)
        logits_by_q = wav_only_logits_by_q(
            model,
            condition,
            target_frames,
            temperature=temperature,
            mode=args.code2wav_grad_mode,
            detach_frame_feedback=args.detach_wav_only_frame_feedback,
        )
        metrics["codec_loss"] = None
        metrics["wav_only_frames"] = float(target_frames)
    pred_wav = soft_code2wav(
        model.code2wav,
        logits_by_q,
        temperature=temperature,
        mode=args.code2wav_grad_mode,
    )
    if args.nonempty_eos_avoid_weight > 0:
        eos_avoid = eos_avoid_loss_for_nonempty(model, logits_by_q[0])
        eos_avoid_component = args.nonempty_eos_avoid_weight * eos_avoid
        total = add_loss_term(total, eos_avoid_component)
        metrics["eos_avoid_loss"] = float(eos_avoid.detach().float().cpu())
        metrics["eos_avoid_component"] = float(eos_avoid_component.detach().float().cpu())
    target_wav = target_wav.to(pred_wav.device)
    wav_l1 = waveform_l1_loss(pred_wav, target_wav)
    stft = multi_resolution_stft_loss(pred_wav, target_wav)
    wav_l1_component = args.wav_l1_weight * wav_l1
    stft_component = args.stft_weight * stft
    total = add_loss_term(total, wav_l1_component)
    total = add_loss_term(total, stft_component)
    metrics["wav_l1"] = float(wav_l1.detach().float().cpu())
    metrics["stft_loss"] = float(stft.detach().float().cpu())
    metrics["wav_l1_component"] = float(wav_l1_component.detach().float().cpu())
    metrics["stft_loss_component"] = float(stft_component.detach().float().cpu())
    metrics["soft_temperature"] = temperature
    metrics["code2wav_grad_mode_st_argmax"] = 1.0 if args.code2wav_grad_mode == "st_argmax" else 0.0
    metrics["total_loss"] = float(total.detach().float().cpu())
    return {"loss": total, "metrics": metrics}


def save_adapter(model: Any, output_dir: Path, step: int, metrics: dict[str, Any]) -> None:
    step_dir = output_dir / f"checkpoint-{step}"
    step_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(step_dir)
    (step_dir / "trainer_state.json").write_text(
        json.dumps({"step": step, "metrics": metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "latest_checkpoint.txt").write_text(str(step_dir), encoding="utf-8")


def trainable_summary(model: Any) -> dict[str, Any]:
    trainable = 0
    total = 0
    for _name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return {"trainable": trainable, "total": total, "ratio": trainable / max(1, total)}


def unwrap_peft_model(model: Any) -> Any:
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def parameter_allowed_by_scope(name: str, scope: str) -> bool:
    if scope == "all":
        return True
    if scope == "talker":
        return ".talker." in name or name.startswith("base_model.model.talker.")
    if scope == "thinker":
        return ".thinker." in name or name.startswith("base_model.model.thinker.")
    raise ValueError(f"unsupported train scope: {scope}")


def main() -> None:
    args = parse_args()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    patch_qwen3_omni_no_split(Qwen3OmniMoeForConditionalGeneration)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    records = read_jsonl(args.train_manifest)
    if args.max_records > 0:
        records = records[: args.max_records]
    if not records:
        raise SystemExit("train manifest is empty")
    load_processor_inputs.max_history_chunks = args.max_history_chunks

    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model, trust_remote_code=True)
    kwargs: dict[str, Any] = {"trust_remote_code": True, "dtype": "auto"}
    if args.device_map:
        kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    base_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(args.model, **kwargs)
    if not getattr(base_model, "has_talker", False):
        base_model.enable_talker()
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
    for param in base_model.code2wav.parameters():
        param.requires_grad = False

    lora_targets = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_targets,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    for name, param in model.named_parameters():
        if ".code2wav." in name or name.startswith("base_model.model.code2wav."):
            param.requires_grad = False
        elif param.requires_grad and not parameter_allowed_by_scope(name, args.train_scope):
            param.requires_grad = False
        elif param.requires_grad and param.is_floating_point() and param.dtype != torch.float32:
            param.data = param.data.float()
    print(json.dumps({"trainable": trainable_summary(model)}, ensure_ascii=False), flush=True)

    if args.tokenize_smoke:
        first = records[0]
        inputs, prompt_len, target_token_len = load_processor_inputs(
            processor,
            first,
            unwrap_peft_model(model),
        )
        print(
            json.dumps(
                {
                    "id": first.get("id"),
                    "input_ids": list(inputs["input_ids"].shape),
                    "prompt_len": prompt_len,
                    "target_token_len": target_token_len,
                    "audios": len(first.get("audios") or []),
                    "target_text": first.get("target_text"),
                    "has_target_codes": bool(first.get("target_codes_path")),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(records) / max(1, args.gradient_accumulation_steps))
    total_steps = args.max_steps or max(1, int(math.ceil(args.epochs * steps_per_epoch)))
    scheduler = make_scheduler(optimizer, args.warmup_steps, total_steps)
    model.train()
    global_step = 0
    micro_step = 0
    running: dict[str, float] = {}
    running_count = 0

    while global_step < total_steps:
        random.shuffle(records)
        for record in records:
            result = compute_record_loss(
                model=model,
                processor=processor,
                record=record,
                args=args,
                global_step=global_step,
                total_steps=total_steps,
            )
            result_loss_value = float(result["loss"].detach().float().cpu())
            if not bool(torch.isfinite(result["loss"]).all().detach().cpu()):
                skip_row = {
                    "skipped_nonfinite_loss": True,
                    "id": record.get("id"),
                    **{
                        key: value
                        for key, value in result["metrics"].items()
                        if value is None or isinstance(value, (int, float, str))
                    },
                }
                print(json.dumps(skip_row, ensure_ascii=False), flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue
            if not result["loss"].requires_grad:
                skip_row = {
                    "skipped_no_grad_loss": True,
                    "id": record.get("id"),
                    **{
                        key: value
                        for key, value in result["metrics"].items()
                        if value is None or isinstance(value, (int, float, str))
                    },
                }
                print(json.dumps(skip_row, ensure_ascii=False), flush=True)
                continue
            loss = result["loss"] / max(1, args.gradient_accumulation_steps)
            loss.backward()
            micro_step += 1
            for key, value in result["metrics"].items():
                if value is None:
                    continue
                running[key] = running.get(key, 0.0) + float(value)
            running["loss"] = running.get("loss", 0.0) + result_loss_value
            running_count += 1
            if micro_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    log_row = {
                        "step": global_step,
                        "lr": scheduler.get_last_lr()[0],
                        **{
                            key: round(value / max(1, running_count), 6)
                            for key, value in sorted(running.items())
                        },
                    }
                    print(json.dumps(log_row, ensure_ascii=False), flush=True)
                    running = {}
                    running_count = 0
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_adapter(model, output_dir, global_step, {"kind": "periodic"})
                if global_step >= total_steps:
                    break
        if micro_step % args.gradient_accumulation_steps != 0 and global_step < total_steps:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

    save_adapter(model, output_dir, global_step, {"kind": "final"})
    model.save_pretrained(output_dir / "final")
    print(json.dumps({"done": True, "global_step": global_step}, ensure_ascii=False), flush=True)


def module_device(module: Any) -> Any:
    return next(module.parameters()).device


def _torch() -> Any:
    import torch

    return torch


if __name__ == "__main__":
    main()
