from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any


QWEN3_OMNI_NO_SPLIT_MODULES = [
    "Qwen3OmniMoeThinkerTextDecoderLayer",
    "Qwen3OmniMoeAudioEncoderLayer",
    "Qwen3OmniMoeVisionBlock",
    "Qwen3OmniMoeTalkerDecoderLayer",
    "Qwen3OmniMoeTalkerCodePredictorDecoderLayer",
    "Qwen3OmniMoeCode2WavTransformerLayer",
    "Qwen3OmniMoeCode2WavDecoderBlock",
]


def patch_qwen3_omni_no_split(model_cls: Any) -> None:
    existing = list(getattr(model_cls, "_no_split_modules", []) or [])
    model_cls._no_split_modules = list(dict.fromkeys(existing + QWEN3_OMNI_NO_SPLIT_MODULES))


def module_device(module: Any) -> Any:
    return next(module.parameters()).device


@dataclass
class TalkerCondition:
    inputs_embeds: Any
    talker_input_ids: Any
    attention_mask: Any
    trailing_text_hidden: Any
    tts_pad_embed: Any


class Code2WavCapture:
    def __init__(self, code2wav: Any) -> None:
        self.code2wav = code2wav
        self.original = code2wav.chunked_decode
        self.codes = None
        self.wav = None
        code2wav.chunked_decode = self._wrapped

    def _wrapped(self, codes: Any, *args: Any, **kwargs: Any) -> Any:
        self.codes = codes.detach().cpu().clone()
        wav = self.original(codes, *args, **kwargs)
        self.wav = wav.detach().cpu().clone()
        return wav

    def clear(self) -> None:
        self.codes = None
        self.wav = None

    def restore(self) -> None:
        self.code2wav.chunked_decode = self.original


def prepare_talker_condition(
    model: Any,
    inputs: Any,
    use_audio_in_video: bool = False,
    speaker: str = "Ethan",
    detach_thinker: bool = False,
) -> TalkerCondition:
    torch = _torch()
    input_ids = inputs["input_ids"]
    source_input_ids = input_ids.detach().clone()
    im_start_positions = torch.nonzero(
        source_input_ids[0] == model.config.im_start_token_id,
        as_tuple=True,
    )[0]
    im_start_indexes = [int(pos) for pos in im_start_positions.detach().cpu().tolist()]
    im_start_indexes.append(int(source_input_ids.shape[-1]))
    segment_roles = []
    for idx in range(len(im_start_indexes) - 1):
        im_start = im_start_indexes[idx]
        segment_end = im_start_indexes[idx + 1]
        if im_start < 0 or im_start + 1 >= source_input_ids.shape[-1] or segment_end <= im_start:
            raise RuntimeError(f"invalid talker segment bounds: {im_start}:{segment_end}")
        segment_roles.append(int(source_input_ids[0, im_start + 1].detach().cpu().item()))
    multimodal_mask = (
        (source_input_ids == model.config.thinker_config.audio_token_id)
        | (source_input_ids == model.config.thinker_config.image_token_id)
        | (source_input_ids == model.config.thinker_config.video_token_id)
    ).to(source_input_ids.device)

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
    thinker_kwargs = {key: value for key, value in inputs.items() if key in thinker_keys}
    thinker_kwargs["use_audio_in_video"] = use_audio_in_video
    thinker_kwargs["output_hidden_states"] = True
    thinker_kwargs["return_dict"] = True
    thinker_kwargs["use_cache"] = False
    context = torch.no_grad() if detach_thinker else nullcontext()
    with context:
        thinker_outputs = model.thinker(**thinker_kwargs)
    thinker_embed = thinker_outputs.hidden_states[0].to(input_ids.device)
    thinker_hidden = thinker_outputs.hidden_states[model.config.talker_config.accept_hidden_layer].to(
        input_ids.device
    )
    if detach_thinker:
        thinker_embed = thinker_embed.detach()
        thinker_hidden = thinker_hidden.detach()

    special_tokens = torch.tensor(
        [[model.config.tts_bos_token_id, model.config.tts_eos_token_id, model.config.tts_pad_token_id]],
        device=model.thinker.device,
        dtype=input_ids.dtype,
    )
    text_projection_device = module_device(model.talker.text_projection)
    tts_bos_embed, tts_eos_embed, tts_pad_embed = (
        model.talker.text_projection(model.thinker.get_input_embeddings()(special_tokens))
        .to(text_projection_device)
        .chunk(3, dim=1)
    )

    speaker_id = _speaker_id(model, speaker)
    talker_embeds = []
    talker_ids = []
    trailing_text_hidden = None
    for idx in range(len(im_start_indexes) - 1):
        im_start = im_start_indexes[idx]
        segment_end = im_start_indexes[idx + 1]
        role_token = segment_roles[idx]
        if role_token == model.config.system_token_id:
            continue
        if role_token == model.config.user_token_id:
            talker_embeds.append(
                _talker_user_part(
                    model,
                    im_start,
                    segment_end,
                    multimodal_mask,
                    thinker_hidden,
                    thinker_embed,
                )
            )
            talker_ids.append(source_input_ids[:, im_start:segment_end])
            continue
        if role_token == model.config.assistant_token_id and idx == len(im_start_indexes) - 2:
            assistant_part, assistant_ids, trailing_text_hidden = _talker_assistant_part(
                model,
                im_start,
                segment_end,
                speaker_id,
                thinker_embed,
                tts_pad_embed,
                tts_bos_embed,
                tts_eos_embed,
            )
            talker_embeds.append(assistant_part)
            talker_ids.append(assistant_ids)
            continue
        if role_token == model.config.assistant_token_id:
            continue
        raise AssertionError("expected role token after <|im_start|>")

    if trailing_text_hidden is None:
        raise RuntimeError("failed to locate assistant segment for talker conditioning")
    talker_input_ids = torch.cat([ids.to(input_ids.device) for ids in talker_ids], dim=1)
    inputs_embeds = torch.cat([emb.to(input_ids.device) for emb in talker_embeds], dim=1)
    return TalkerCondition(
        inputs_embeds=inputs_embeds,
        talker_input_ids=talker_input_ids,
        attention_mask=torch.ones_like(talker_input_ids, dtype=torch.long),
        trailing_text_hidden=trailing_text_hidden,
        tts_pad_embed=tts_pad_embed,
    )


def _speaker_id(model: Any, speaker: str) -> int:
    speaker_id = model.config.talker_config.speaker_id.get(speaker.lower())
    if speaker_id is None:
        raise NotImplementedError(f"speaker {speaker!r} is not implemented by this model")
    return int(speaker_id)


def _talker_user_part(
    model: Any,
    im_start_index: int,
    segment_end_index: int,
    multimodal_mask: Any,
    thinker_hidden: Any,
    thinker_embed: Any,
) -> Any:
    torch = _torch()
    text_projection_device = module_device(model.talker.text_projection)
    part = torch.empty(
        (
            1,
            segment_end_index - im_start_index,
            model.config.talker_config.text_config.hidden_size,
        ),
        device=text_projection_device,
        dtype=model.talker.dtype,
    )
    mm_mask = multimodal_mask[:, im_start_index:segment_end_index].to(text_projection_device)
    if mm_mask.any():
        hidden_mm = thinker_hidden[:, im_start_index:segment_end_index].to(text_projection_device)
        part[mm_mask] = model.talker.hidden_projection(hidden_mm[mm_mask])
    text_embed = thinker_embed[:, im_start_index:segment_end_index].to(text_projection_device)
    part[~mm_mask] = model.talker.text_projection(text_embed[~mm_mask])
    return part


def _talker_assistant_part(
    model: Any,
    im_start_index: int,
    segment_end_index: int,
    speaker_id: int,
    thinker_embed: Any,
    tts_pad_embed: Any,
    tts_bos_embed: Any,
    tts_eos_embed: Any,
) -> tuple[Any, Any, Any]:
    torch = _torch()
    text_projection_device = module_device(model.talker.text_projection)
    codec_embedding = model.talker.get_input_embeddings()
    codec_embedding_device = module_device(codec_embedding)
    tts_pad_embed = tts_pad_embed.to(text_projection_device)
    tts_bos_embed = tts_bos_embed.to(text_projection_device)
    tts_eos_embed = tts_eos_embed.to(text_projection_device)

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
    invalid = (codec_special_tokens < 0) | (codec_special_tokens >= codec_embedding.num_embeddings)
    if bool(invalid.any()):
        bad_ids = codec_special_tokens[invalid].detach().cpu().tolist()
        raise RuntimeError(
            f"invalid codec special token ids {bad_ids} for embedding size {codec_embedding.num_embeddings}"
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


def generate_talker_codes(
    model: Any,
    condition: TalkerCondition,
    *,
    max_new_tokens: int = 4096,
    min_new_tokens: int | None = None,
    do_sample: bool = False,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    suppress_primary_special: bool = False,
    primary_from_sequences: bool = False,
) -> Any:
    torch = _torch()
    from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

    suppressed_tokens = [
        token_id
        for token_id in range(
            model.config.talker_config.text_config.vocab_size - 1024,
            model.config.talker_config.text_config.vocab_size,
        )
        if token_id != model.config.talker_config.codec_eos_token_id
    ]
    suppressed_tensor = torch.tensor(suppressed_tokens, dtype=torch.long)

    class _SuppressPrimaryCodecTokens(LogitsProcessor):
        def __call__(self, input_ids: Any, scores: Any) -> Any:
            tokens = suppressed_tensor.to(scores.device)
            scores = scores.clone()
            scores.index_fill_(1, tokens, -float("inf"))
            return scores

    min_new_tokens_value = int(min_new_tokens or 0)

    class _SuppressEosUntilMinNewTokens(LogitsProcessor):
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, input_ids: Any, scores: Any) -> Any:
            self.calls += 1
            if min_new_tokens_value <= 0 or self.calls > min_new_tokens_value:
                return scores
            scores = scores.clone()
            scores[:, model.config.talker_config.codec_eos_token_id] = -float("inf")
            return scores

    original_code_predictor_generate = model.talker.code_predictor.generate

    def code_predictor_generate_greedy(*args: Any, **kwargs: Any) -> Any:
        kwargs["do_sample"] = False
        kwargs["remove_invalid_values"] = True
        kwargs["renormalize_logits"] = True
        return original_code_predictor_generate(*args, **kwargs)

    logits_processor = LogitsProcessorList()
    if suppress_primary_special:
        logits_processor.append(_SuppressPrimaryCodecTokens())
    if min_new_tokens_value > 0:
        logits_processor.append(_SuppressEosUntilMinNewTokens())
    model.talker.code_predictor.generate = code_predictor_generate_greedy
    if hasattr(model.talker, "rope_deltas"):
        model.talker.rope_deltas = None
    try:
        generate_kwargs = {
            "inputs_embeds": condition.inputs_embeds,
            "trailing_text_hidden": condition.trailing_text_hidden,
            "tts_pad_embed": condition.tts_pad_embed,
            "talker_input_ids": condition.talker_input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": model.config.talker_config.codec_eos_token_id,
            "repetition_penalty": repetition_penalty,
            "suppress_tokens": suppressed_tokens,
            "logits_processor": logits_processor,
            "output_hidden_states": True,
            "return_dict_in_generate": True,
            "remove_invalid_values": True,
            "renormalize_logits": True,
        }
        if do_sample:
            generate_kwargs.update(
                {
                    "top_k": top_k,
                    "top_p": top_p,
                    "temperature": temperature,
                }
            )
        result = model.talker.generate(**generate_kwargs)
    finally:
        model.talker.code_predictor.generate = original_code_predictor_generate
        if hasattr(model.talker, "rope_deltas"):
            model.talker.rope_deltas = None
    hidden_codes = [hid[-1] for hid in result.hidden_states if hid[-1] is not None]
    if not hidden_codes:
        raise RuntimeError("talker did not return hidden-state codec labels")
    codes = (
        torch.stack(hidden_codes, dim=1)
        .transpose(1, 2)
        .to(result.hidden_states[-1][-1].device)
        .contiguous()
    )
    if primary_from_sequences:
        primary = result.sequences[:, : codes.shape[-1]].to(codes.device)
        if primary.shape[-1] == codes.shape[-1]:
            codes[:, 0, :] = primary
    return codes


def soft_code2wav(
    code2wav: Any,
    logits: Any,
    temperature: float = 1.0,
    mode: str = "soft",
) -> Any:
    """Differentiable code2wav path from codec logits.

    Args:
        logits: Either a tensor shaped [batch, quantizers, frames, vocab] or a
            list of per-quantizer tensors shaped [batch, frames, vocab].
    """

    torch = _torch()
    if isinstance(logits, (list, tuple)):
        logits_by_q = list(logits)
        quantizers = len(logits_by_q)
    else:
        if logits.ndim != 4:
            raise ValueError(f"expected logits [B,Q,T,V], got {tuple(logits.shape)}")
        quantizers = logits.shape[1]
        logits_by_q = [logits[:, q] for q in range(quantizers)]
    if quantizers != code2wav.config.num_quantizers:
        raise ValueError(f"expected {code2wav.config.num_quantizers} quantizers, got {quantizers}")
    codebook = int(code2wav.config.codebook_size)
    temp = max(float(temperature), 1.0e-6)
    emb_weight = code2wav.code_embedding.weight
    pieces = []
    for q, q_logits_raw in enumerate(logits_by_q):
        offset = q * codebook
        vocab = min(q_logits_raw.shape[-1], emb_weight.shape[0] - offset)
        if vocab <= 0:
            raise ValueError(f"quantizer {q} has no valid code2wav embedding range")
        pieces.append(
            logits_to_code_embedding(
                q_logits_raw,
                emb_weight,
                offset=offset,
                vocab=vocab,
                temperature=temp,
                mode=mode,
            )
        )
    hidden = torch.stack(pieces, dim=1).mean(1)
    hidden = code2wav.pre_transformer(inputs_embeds=hidden).last_hidden_state
    hidden = hidden.permute(0, 2, 1)
    for blocks in code2wav.upsample:
        for block in blocks:
            hidden = block(hidden)
    wav = hidden
    for block in code2wav.decoder:
        wav = block(wav)
    return wav.clamp(min=-1, max=1)


def logits_to_code_embedding(
    logits: Any,
    embedding_weight: Any,
    *,
    offset: int = 0,
    vocab: int | None = None,
    temperature: float = 1.0,
    mode: str = "soft",
) -> Any:
    torch = _torch()
    if mode not in {"soft", "st_argmax"}:
        raise ValueError(f"unsupported code embedding mode: {mode}")
    offset = int(offset)
    if vocab is None:
        vocab = min(logits.shape[-1], embedding_weight.shape[0] - offset)
    vocab = int(vocab)
    if vocab <= 0:
        raise ValueError("code embedding vocab must be positive")
    temp = max(float(temperature), 1.0e-6)
    weight = embedding_weight[offset : offset + vocab]
    probs = torch.softmax(torch.nan_to_num((logits[..., :vocab] / temp).float()), dim=-1)
    if mode == "st_argmax":
        hard = torch.nn.functional.one_hot(probs.argmax(dim=-1), num_classes=vocab).to(
            dtype=probs.dtype,
            device=probs.device,
        )
        probs = hard + probs - probs.detach()
    return torch.matmul(probs.to(weight.device, dtype=weight.dtype), weight)


def waveform_l1_loss(pred: Any, target: Any) -> Any:
    length = min(pred.shape[-1], target.shape[-1])
    if length <= 0:
        return pred.new_tensor(0.0)
    return (pred[..., :length] - target[..., :length]).abs().mean()


def multi_resolution_stft_loss(
    pred: Any,
    target: Any,
    fft_sizes: tuple[int, ...] = (512, 1024, 2048),
    hop_ratio: float = 0.25,
) -> Any:
    torch = _torch()
    length = min(pred.shape[-1], target.shape[-1])
    if length <= 0:
        return pred.new_tensor(0.0)
    pred = pred[..., :length].reshape(-1, length).float()
    target = target[..., :length].reshape(-1, length).float()
    loss = pred.new_tensor(0.0)
    used = 0
    for n_fft in fft_sizes:
        if length < n_fft:
            continue
        hop = max(1, int(n_fft * hop_ratio))
        window = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
        pred_mag = torch.stft(
            pred,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
        ).abs()
        target_mag = torch.stft(
            target,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
        ).abs()
        loss = loss + (torch.log1p(pred_mag) - torch.log1p(target_mag)).abs().mean()
        used += 1
    if used == 0:
        return waveform_l1_loss(pred, target)
    return loss / used


def _torch() -> Any:
    import torch

    return torch
