#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OmniCodecCollator = None
OmniCodecPairDataset = None
Wav2OmniCodecConfig = None
Wav2OmniCodecModel = None
wav2codec_loss = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train wav -> Omni codec inverter.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--dev-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--hop-length", type=int, default=1920)
    parser.add_argument("--num-quantizers", type=int, default=16)
    parser.add_argument("--codebook-size", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--transformer-layers", type=int, default=6)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-position-frames", type=int, default=2048)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--max-train-records", type=int, default=0)
    parser.add_argument("--max-dev-records", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=260623)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-max-batches", type=int, default=100)
    parser.add_argument("--overfit", action="store_true")
    return parser.parse_args()


def latest_state_dir(output_dir: Path) -> Path | None:
    latest = output_dir / "latest_checkpoint.txt"
    if latest.exists():
        candidate = Path(latest.read_text(encoding="utf-8").strip())
        if candidate.exists():
            return candidate
    states = sorted((output_dir / "checkpoints").glob("step_*"))
    return states[-1] if states else None


def make_scheduler(optimizer: Any, warmup_steps: int, total_steps: int):
    from torch.optim.lr_scheduler import LambdaLR

    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def make_count_tensor(logits: Any, labels: Any, topk: int = 5, ignore_index: int = -100) -> Any:
    import torch

    labels = labels[:, :, : logits.shape[2]]
    valid = labels.ne(ignore_index)
    pred = logits.argmax(dim=-1)
    correct = pred.eq(labels) & valid
    per_q_total = valid.sum(dim=(0, 2)).to(torch.float64)
    per_q_correct = correct.sum(dim=(0, 2)).to(torch.float64)
    valid_frames = valid.all(dim=1)
    frame_correct = correct.all(dim=1) & valid_frames
    top = logits.topk(k=min(topk, logits.shape[-1]), dim=-1).indices
    top_correct = top.eq(labels.unsqueeze(-1)).any(dim=-1) & valid
    return torch.cat(
        [
            valid.sum().reshape(1).to(torch.float64),
            correct.sum().reshape(1).to(torch.float64),
            valid_frames.sum().reshape(1).to(torch.float64),
            frame_correct.sum().reshape(1).to(torch.float64),
            top_correct.sum().reshape(1).to(torch.float64),
            per_q_total,
            per_q_correct,
        ]
    )


def counts_to_metrics(counts: Any, num_quantizers: int) -> dict[str, Any]:
    values = counts.detach().cpu().tolist()
    total, correct, frame_total, frame_correct, top5_correct = values[:5]
    offset = 5
    per_q_total = values[offset : offset + num_quantizers]
    per_q_correct = values[offset + num_quantizers : offset + 2 * num_quantizers]
    return {
        "valid_codes": int(total),
        "accuracy": round(correct / total, 6) if total else None,
        "top5_accuracy": round(top5_correct / total, 6) if total else None,
        "valid_frames": int(frame_total),
        "frame_accuracy": round(frame_correct / frame_total, 6) if frame_total else None,
        "per_quantizer_accuracy": [
            round(c / t, 6) if t else None for c, t in zip(per_q_correct, per_q_total)
        ],
    }


def evaluate(accelerator: Any, model: Any, loader: Any, max_batches: int, num_quantizers: int) -> dict[str, Any]:
    import torch

    model.eval()
    total_counts = None
    total_loss = torch.tensor(0.0, device=accelerator.device, dtype=torch.float64)
    total_weight = torch.tensor(0.0, device=accelerator.device, dtype=torch.float64)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if max_batches > 0 and batch_index > max_batches:
                break
            logits = model(batch["wav"], batch["frame_mask"])
            loss = wav2codec_loss(logits, batch["codes"])
            weight = batch["codes"].ne(-100).sum().to(torch.float64)
            counts = make_count_tensor(logits, batch["codes"])
            gathered_counts = accelerator.gather_for_metrics(counts.unsqueeze(0)).sum(dim=0)
            gathered_loss = accelerator.gather_for_metrics((loss.detach().to(torch.float64) * weight).reshape(1)).sum()
            gathered_weight = accelerator.gather_for_metrics(weight.reshape(1)).sum()
            total_counts = gathered_counts if total_counts is None else total_counts + gathered_counts
            total_loss = total_loss + gathered_loss
            total_weight = total_weight + gathered_weight
    model.train()
    metrics = counts_to_metrics(total_counts, num_quantizers) if total_counts is not None else {}
    metrics["loss"] = round(float((total_loss / total_weight).item()), 6) if total_weight.item() else None
    return metrics


def save_model(accelerator: Any, model: Any, output_dir: Path, step: int, metrics: dict[str, Any]) -> None:
    state_dir = output_dir / "checkpoints" / f"step_{step:08d}"
    accelerator.save_state(state_dir)
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(state_dir / "model")
        (state_dir / "trainer_state.json").write_text(
            json.dumps({"step": step, "metrics": metrics}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "latest_checkpoint.txt").write_text(str(state_dir), encoding="utf-8")


def main() -> None:
    args = parse_args()
    global OmniCodecCollator
    global OmniCodecPairDataset
    global Wav2OmniCodecConfig
    global Wav2OmniCodecModel
    global wav2codec_loss
    from s2s_omni.wav2codec import (
        OmniCodecCollator,
        OmniCodecPairDataset,
        Wav2OmniCodecConfig,
        Wav2OmniCodecModel,
        wav2codec_loss,
    )

    from accelerate import Accelerator
    import torch
    from torch.utils.data import DataLoader

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    accelerator.print(json.dumps(vars(args), ensure_ascii=False, indent=2))
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "train_args.json").write_text(
            json.dumps(vars(args), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()

    cfg = Wav2OmniCodecConfig(
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        num_quantizers=args.num_quantizers,
        codebook_size=args.codebook_size,
        hidden_size=args.hidden_size,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        dropout=args.dropout,
        max_position_frames=args.max_position_frames,
    )
    model = Wav2OmniCodecModel(cfg)
    train_dataset = OmniCodecPairDataset(
        args.train_manifest,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        num_quantizers=args.num_quantizers,
        max_frames=args.max_frames,
        random_crop=not args.overfit,
        max_records=args.max_train_records,
        seed=args.seed,
    )
    dev_manifest = args.train_manifest if args.overfit or not args.dev_manifest else args.dev_manifest
    dev_dataset = OmniCodecPairDataset(
        dev_manifest,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        num_quantizers=args.num_quantizers,
        max_frames=args.max_frames,
        random_crop=False,
        max_records=args.max_dev_records or (args.max_train_records if args.overfit else 0),
        seed=args.seed,
    )
    collator = OmniCodecCollator(hop_length=args.hop_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=not args.overfit,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    per_process_batches = math.ceil(
        len(train_dataset)
        / max(1, args.per_device_batch_size * max(1, accelerator.num_processes))
    )
    steps_per_epoch = math.ceil(per_process_batches / max(1, args.gradient_accumulation_steps))
    total_steps = args.max_steps or max(1, int(math.ceil(args.epochs * steps_per_epoch)))
    scheduler = make_scheduler(
        optimizer,
        args.warmup_steps * max(1, accelerator.num_processes),
        total_steps * max(1, accelerator.num_processes),
    )
    model, optimizer, train_loader, dev_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        dev_loader,
        scheduler,
    )

    resume_from = Path(args.resume_from) if args.resume_from else None
    if resume_from is None and args.resume:
        resume_from = latest_state_dir(output_dir)
    global_step = 0
    if resume_from is not None and resume_from.exists():
        accelerator.print(f"resuming from {resume_from}")
        accelerator.load_state(resume_from)
        try:
            state = json.loads((resume_from / "trainer_state.json").read_text(encoding="utf-8"))
            global_step = int(state.get("step") or 0)
        except FileNotFoundError:
            global_step = 0

    model.train()
    running_loss = 0.0
    running_items = 0
    while global_step < total_steps:
        for batch in train_loader:
            with accelerator.accumulate(model):
                logits = model(batch["wav"], batch["frame_mask"])
                loss = wav2codec_loss(logits, batch["codes"])
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                global_step += 1
                running_loss += float(accelerator.gather_for_metrics(loss.detach().reshape(1)).mean().item())
                running_items += 1
                if args.log_every > 0 and global_step % args.log_every == 0:
                    accelerator.print(
                        json.dumps(
                            {
                                "step": global_step,
                                "loss": round(running_loss / max(1, running_items), 6),
                                "lr": scheduler.get_last_lr()[0],
                            },
                            ensure_ascii=False,
                        )
                    )
                    running_loss = 0.0
                    running_items = 0
                if args.eval_every > 0 and global_step % args.eval_every == 0:
                    metrics = evaluate(
                        accelerator,
                        model,
                        dev_loader,
                        args.eval_max_batches,
                        args.num_quantizers,
                    )
                    accelerator.print(json.dumps({"step": global_step, "dev": metrics}, ensure_ascii=False))
                if args.save_every > 0 and global_step % args.save_every == 0:
                    save_model(accelerator, model, output_dir, global_step, {"kind": "periodic"})
                if global_step >= total_steps:
                    break

    final_metrics = evaluate(
        accelerator,
        model,
        dev_loader,
        args.eval_max_batches,
        args.num_quantizers,
    )
    accelerator.print(json.dumps({"step": global_step, "final_dev": final_metrics}, ensure_ascii=False))
    save_model(accelerator, model, output_dir, global_step, final_metrics)
    if accelerator.is_main_process:
        final_dir = output_dir / "final"
        accelerator.unwrap_model(model).save_pretrained(final_dir)
        (output_dir / "final_metrics.json").write_text(
            json.dumps(final_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
