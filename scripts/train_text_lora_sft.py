#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_jsonl
from s2s_omni.io import read_yaml


REQUIRED = ["torch", "transformers", "datasets", "peft", "accelerate", "qwen_omni_utils"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text-side LoRA SFT entrypoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-file", help="Override train_file from config.")
    parser.add_argument("--output-dir", help="Override output_dir from config.")
    parser.add_argument("--max-steps", type=int, help="Override max_steps from config.")
    parser.add_argument("--max-records", type=int, default=0, help="Use only the first N records.")
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        help="Override per_device_train_batch_size from config.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="Override gradient_accumulation_steps from config.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument(
        "--tokenize-smoke",
        action="store_true",
        help="Load processor/tokenizer and inspect the first training example without loading model weights.",
    )
    return parser.parse_args()


def missing_modules() -> list[str]:
    return [name for name in REQUIRED if importlib.util.find_spec(name) is None]


def load_processor(cfg: dict[str, Any]):
    backend = cfg.get("model_backend", "qwen3_omni")
    if backend in {"qwen3_omni", "qwen3_omni_thinker"}:
        from transformers import Qwen3OmniMoeProcessor

        processor = Qwen3OmniMoeProcessor.from_pretrained(
            cfg["model_name_or_path"],
            trust_remote_code=bool(cfg.get("trust_remote_code", True)),
        )
        tokenizer = processor.tokenizer
    else:
        from transformers import AutoTokenizer

        processor = None
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_name_or_path"],
            trust_remote_code=bool(cfg.get("trust_remote_code", True)),
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return processor, tokenizer


def apply_chat_template(processor: Any, tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if processor is not None and hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def apply_generation_prompt(
    processor: Any, tokenizer: Any, messages: list[dict[str, str]]
) -> str:
    if processor is not None and hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def split_messages(record: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"record {record.get('id')} is missing messages")
    if messages[-1].get("role") != "assistant":
        raise ValueError(f"record {record.get('id')} must end with assistant response")
    prefix = messages[:-1]
    return prefix, messages


def build_features(
    record: dict[str, Any],
    processor: Any,
    tokenizer: Any,
    max_seq_length: int,
    completion_only_loss: bool,
) -> dict[str, Any]:
    prefix_messages, full_messages = split_messages(record)
    prompt_text = apply_generation_prompt(processor, tokenizer, prefix_messages)
    full_text = apply_chat_template(processor, tokenizer, full_messages)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_len = min(len(prompt_ids), len(full_ids))
    if len(full_ids) > max_seq_length:
        overflow = len(full_ids) - max_seq_length
        full_ids = full_ids[-max_seq_length:]
        prompt_len = min(len(full_ids), max(0, prompt_len - overflow))

    labels = list(full_ids)
    if completion_only_loss:
        for idx in range(prompt_len):
            labels[idx] = -100
    attention_mask = [1] * len(full_ids)
    return {
        "id": record.get("id"),
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_tokens": prompt_len,
        "target_tokens": sum(label != -100 for label in labels),
        "prompt_text": prompt_text,
        "full_text": full_text,
    }


class CompressionSFTDataset:
    def __init__(
        self,
        records: list[dict[str, Any]],
        processor: Any,
        tokenizer: Any,
        max_seq_length: int,
        completion_only_loss: bool,
    ) -> None:
        self.features = [
            build_features(record, processor, tokenizer, max_seq_length, completion_only_loss)
            for record in records
        ]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.features[index]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["labels"],
        }


class DataCollatorForCompletionOnly:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def load_model(cfg: dict[str, Any]):
    backend = cfg.get("model_backend", "qwen3_omni")
    kwargs = {
        "trust_remote_code": bool(cfg.get("trust_remote_code", True)),
        "dtype": "auto",
    }
    if cfg.get("device_map"):
        kwargs["device_map"] = cfg["device_map"]
    if cfg.get("attn_implementation"):
        kwargs["attn_implementation"] = cfg["attn_implementation"]
    if backend == "qwen3_omni":
        from transformers import Qwen3OmniMoeForConditionalGeneration

        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            cfg["model_name_or_path"], **kwargs
        )
    elif backend == "qwen3_omni_thinker":
        from transformers import AutoConfig, Qwen3OmniMoeThinkerForConditionalGeneration

        full_config = AutoConfig.from_pretrained(
            cfg["model_name_or_path"],
            trust_remote_code=bool(cfg.get("trust_remote_code", True)),
        )
        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            cfg["model_name_or_path"],
            config=full_config.thinker_config,
            **kwargs,
        )
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(cfg["model_name_or_path"], **kwargs)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if bool(cfg.get("gradient_checkpointing", True)) and hasattr(
        model, "gradient_checkpointing_enable"
    ):
        model.gradient_checkpointing_enable()
    return model


def make_peft_config(cfg: dict[str, Any]):
    from peft import LoraConfig

    lora_cfg = cfg.get("lora", {})
    return LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules", [])),
        task_type="CAUSAL_LM",
    )


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    if args.train_file:
        cfg["train_file"] = args.train_file
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.per_device_train_batch_size is not None:
        cfg["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.gradient_accumulation_steps is not None:
        cfg["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    missing = missing_modules()
    if args.check_env or args.dry_run:
        print("config:")
        for key in ["model_name_or_path", "train_file", "output_dir", "max_seq_length"]:
            print(f"  {key}: {cfg.get(key)}")
        print("missing training modules:", ", ".join(missing) if missing else "none")
        if args.dry_run:
            return
    if missing:
        raise SystemExit(
            "Missing training dependencies: "
            + ", ".join(missing)
            + "\nInstall in an isolated venv: pip install transformers datasets peft accelerate qwen-omni-utils"
        )

    train_file = Path(cfg["train_file"])
    if not train_file.exists():
        raise SystemExit(f"train_file does not exist: {train_file}")

    processor, tokenizer = load_processor(cfg)
    records = read_jsonl(train_file)
    if args.max_records > 0:
        records = records[: args.max_records]
    if not records:
        raise SystemExit(f"train_file has no records: {train_file}")

    dataset = CompressionSFTDataset(
        records,
        processor=processor,
        tokenizer=tokenizer,
        max_seq_length=int(cfg.get("max_seq_length", 4096)),
        completion_only_loss=bool(cfg.get("completion_only_loss", True)),
    )
    first = dataset.features[0]
    if args.tokenize_smoke:
        print(
            json.dumps(
                {
                    "id": first["id"],
                    "records": len(records),
                    "input_tokens": len(first["input_ids"]),
                    "prompt_tokens_masked": first["prompt_tokens"],
                    "target_tokens_trained": first["target_tokens"],
                    "tokenizer": tokenizer.__class__.__name__,
                    "pad_token": tokenizer.pad_token,
                    "eos_token": tokenizer.eos_token,
                    "prompt_preview": first["prompt_text"][:500],
                    "full_preview": first["full_text"][:700],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    from peft import get_peft_model
    from transformers import Trainer, TrainingArguments

    model = load_model(cfg)
    peft_config = make_peft_config(cfg)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    args_train = TrainingArguments(
        output_dir=cfg["output_dir"],
        run_name=cfg.get("run_name"),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 16)),
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
        max_steps=int(cfg.get("max_steps", -1)),
        logging_steps=int(cfg.get("logging_steps", 5)),
        save_steps=int(cfg.get("save_steps", 200)),
        save_total_limit=int(cfg.get("save_total_limit", 3)),
        bf16=bool(cfg.get("bf16", True)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        remove_unused_columns=False,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=dataset,
        data_collator=DataCollatorForCompletionOnly(
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        ),
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    main()
