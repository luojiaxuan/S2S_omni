from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .codec_data import (
    align_wav_to_num_frames,
    base_id_from_id,
    load_mono_wav,
    resolve_manifest_path,
)
from .io import read_jsonl


@dataclass
class Wav2OmniCodecConfig:
    sample_rate: int = 24000
    hop_length: int = 1920
    num_quantizers: int = 16
    codebook_size: int = 2048
    hidden_size: int = 512
    conv_channels: tuple[int, ...] = (64, 128, 256)
    transformer_layers: int = 6
    transformer_heads: int = 8
    transformer_ffn_mult: int = 4
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Wav2OmniCodecConfig":
        payload = dict(data)
        if isinstance(payload.get("conv_channels"), list):
            payload["conv_channels"] = tuple(int(v) for v in payload["conv_channels"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["conv_channels"] = list(self.conv_channels)
        return payload


class OmniCodecPairDataset:
    def __init__(
        self,
        manifest: str | Path,
        sample_rate: int = 24000,
        hop_length: int = 1920,
        num_quantizers: int = 16,
        max_frames: int = 0,
        random_crop: bool = False,
        max_records: int = 0,
        seed: int = 1234,
    ) -> None:
        self.manifest = Path(manifest)
        records = read_jsonl(self.manifest)
        if max_records > 0:
            records = records[:max_records]
        self.records = records
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.num_quantizers = int(num_quantizers)
        self.max_frames = int(max_frames)
        self.random_crop = bool(random_crop)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        wav_path = resolve_manifest_path(record["wav_path"], self.manifest)
        codes_path = resolve_manifest_path(record["codes_path"], self.manifest)
        wav = load_mono_wav(wav_path, self.sample_rate)
        codes = np.load(codes_path).astype(np.int64, copy=False)
        if codes.ndim == 3 and codes.shape[0] == 1:
            codes = codes[0]
        if codes.ndim != 2:
            raise ValueError(f"{codes_path}: expected 2-D codes, got shape {codes.shape}")
        if codes.shape[0] != self.num_quantizers and codes.shape[1] == self.num_quantizers:
            codes = codes.T
        if codes.shape[0] != self.num_quantizers:
            raise ValueError(
                f"{codes_path}: expected {self.num_quantizers} quantizers, got {codes.shape}"
            )

        num_frames = int(codes.shape[1])
        wav = align_wav_to_num_frames(wav, num_frames, self.hop_length)
        if self.max_frames > 0 and num_frames > self.max_frames:
            max_start = num_frames - self.max_frames
            if self.random_crop:
                rng = random.Random(self.seed + index + random.randint(0, 2**20))
                frame_start = rng.randint(0, max_start)
            else:
                frame_start = max_start // 2
            frame_end = frame_start + self.max_frames
            codes = codes[:, frame_start:frame_end]
            wav = wav[frame_start * self.hop_length : frame_end * self.hop_length]
            num_frames = self.max_frames

        return {
            "id": str(record.get("id") or index),
            "record": record,
            "wav": wav,
            "codes": codes,
            "num_frames": num_frames,
        }


class OmniCodecCollator:
    def __init__(self, hop_length: int = 1920, ignore_index: int = -100) -> None:
        self.hop_length = int(hop_length)
        self.ignore_index = int(ignore_index)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_frames = max(int(feature["num_frames"]) for feature in features)
        max_samples = max_frames * self.hop_length
        num_quantizers = int(features[0]["codes"].shape[0])
        batch_size = len(features)
        wavs = torch.zeros((batch_size, max_samples), dtype=torch.float32)
        codes = torch.full(
            (batch_size, num_quantizers, max_frames),
            fill_value=self.ignore_index,
            dtype=torch.long,
        )
        frame_mask = torch.zeros((batch_size, max_frames), dtype=torch.bool)
        ids: list[str] = []
        records: list[dict[str, Any]] = []
        for row, feature in enumerate(features):
            num_frames = int(feature["num_frames"])
            num_samples = num_frames * self.hop_length
            wavs[row, :num_samples] = torch.from_numpy(feature["wav"][:num_samples])
            codes[row, :, :num_frames] = torch.from_numpy(feature["codes"][:, :num_frames])
            frame_mask[row, :num_frames] = True
            ids.append(str(feature["id"]))
            records.append(dict(feature["record"]))
        return {
            "ids": ids,
            "records": records,
            "wav": wavs,
            "codes": codes,
            "frame_mask": frame_mask,
        }


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for wav2codec training/eval") from exc
    return torch, nn, F


torch, nn, F = require_torch()


class Wav2OmniCodecModel(nn.Module):
    def __init__(self, config: Wav2OmniCodecConfig | dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = (
            config
            if isinstance(config, Wav2OmniCodecConfig)
            else Wav2OmniCodecConfig.from_dict(config or {})
        )

        conv_layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in self.config.conv_channels:
            conv_layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size=9, stride=4, padding=4),
                    nn.GroupNorm(1, out_ch),
                    nn.GELU(),
                ]
            )
            in_ch = out_ch
        conv_layers.extend(
            [
                nn.Conv1d(in_ch, self.config.hidden_size, kernel_size=7, stride=3, padding=3),
                nn.GroupNorm(1, self.config.hidden_size),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            ]
        )
        self.frame_encoder = nn.Sequential(*conv_layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_size,
            nhead=self.config.transformer_heads,
            dim_feedforward=self.config.hidden_size * self.config.transformer_ffn_mult,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.transformer_layers,
        )
        self.norm = nn.LayerNorm(self.config.hidden_size)
        self.heads = nn.ModuleList(
            [
                nn.Linear(self.config.hidden_size, self.config.codebook_size)
                for _ in range(self.config.num_quantizers)
            ]
        )

    def forward(self, wav: torch.Tensor, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        if wav.ndim != 2:
            raise ValueError(f"expected wav shape [B, samples], got {tuple(wav.shape)}")
        batch_size, num_samples = wav.shape
        num_frames = math.ceil(num_samples / self.config.hop_length)
        target_samples = num_frames * self.config.hop_length
        if target_samples != num_samples:
            wav = F.pad(wav, (0, target_samples - num_samples))
        frames = wav.reshape(batch_size, num_frames, self.config.hop_length)
        encoded = self.frame_encoder(frames.reshape(-1, 1, self.config.hop_length)).squeeze(-1)
        encoded = encoded.reshape(batch_size, num_frames, self.config.hidden_size)
        key_padding_mask = None
        if frame_mask is not None:
            key_padding_mask = ~frame_mask[:, :num_frames].bool()
        hidden = self.context_encoder(encoded, src_key_padding_mask=key_padding_mask)
        hidden = self.norm(hidden)
        logits = torch.stack([head(hidden) for head in self.heads], dim=1)
        return logits

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2),
            encoding="utf-8",
        )
        torch.save(self.state_dict(), output_dir / "model.pt")

    @classmethod
    def from_pretrained(cls, path: str | Path, map_location: str | None = "cpu") -> "Wav2OmniCodecModel":
        path = Path(path)
        config = Wav2OmniCodecConfig.from_dict(
            json.loads((path / "config.json").read_text(encoding="utf-8"))
        )
        model = cls(config)
        state = torch.load(path / "model.pt", map_location=map_location)
        model.load_state_dict(state)
        return model


def wav2codec_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    vocab_size = int(logits.shape[-1])
    return F.cross_entropy(
        logits.permute(0, 2, 1, 3).reshape(-1, vocab_size),
        labels.permute(0, 2, 1).reshape(-1),
        ignore_index=ignore_index,
    )


@torch.no_grad()
def codec_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    topk: Iterable[int] = (1, 5),
    ignore_index: int = -100,
) -> dict[str, Any]:
    labels = labels[:, :, : logits.shape[2]]
    valid = labels.ne(ignore_index)
    total = int(valid.sum().item())
    out: dict[str, Any] = {"valid_codes": total}
    if total == 0:
        return out

    pred = logits.argmax(dim=-1)
    correct = pred.eq(labels) & valid
    out["accuracy"] = round(float(correct.sum().item() / total), 6)
    per_quantizer = []
    for q in range(labels.shape[1]):
        q_valid = valid[:, q]
        denom = int(q_valid.sum().item())
        if denom == 0:
            per_quantizer.append(None)
        else:
            per_quantizer.append(round(float((correct[:, q] & q_valid).sum().item() / denom), 6))
    out["per_quantizer_accuracy"] = per_quantizer

    valid_frames = valid.all(dim=1)
    frame_total = int(valid_frames.sum().item())
    if frame_total:
        frame_correct = correct.all(dim=1) & valid_frames
        out["frame_accuracy"] = round(float(frame_correct.sum().item() / frame_total), 6)
        out["valid_frames"] = frame_total

    for k in topk:
        if k <= 1:
            continue
        top = logits.topk(k=min(int(k), logits.shape[-1]), dim=-1).indices
        top_correct = top.eq(labels.unsqueeze(-1)).any(dim=-1) & valid
        out[f"top{k}_accuracy"] = round(float(top_correct.sum().item() / total), 6)
    return out
