from __future__ import annotations

from typing import Iterable

from .prompts import sft_messages
from .schema import S2SSample


def to_sft_record(sample: S2SSample, format_name: str = "messages") -> dict:
    messages = sft_messages(sample)
    if format_name == "messages":
        return {"id": sample.id, "messages": messages}
    if format_name == "prompt_completion":
        prompt = "\n\n".join(
            [
                f"System:\n{messages[0]['content']}",
                f"User:\n{messages[1]['content']}",
                "Assistant:\n",
            ]
        )
        return {"id": sample.id, "prompt": prompt, "completion": messages[2]["content"]}
    raise ValueError(f"unknown SFT format: {format_name}")


def iter_sft_records(
    samples: Iterable[S2SSample], format_name: str = "messages"
) -> Iterable[dict]:
    for sample in samples:
        if not (sample.compressed_translation or sample.reference_translation):
            continue
        yield to_sft_record(sample, format_name=format_name)
