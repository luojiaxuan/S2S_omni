from __future__ import annotations

import json

from .schema import S2SSample


SYSTEM_COMPRESSION = """You are a professional simultaneous interpreter.
Your task is meaning-preserving compression for streaming speech translation.
Preserve all core claims, entities, numbers, terminology, negation, modality,
speaker intent, and causal or contrastive relations. Compress or omit only
repetitions, fillers, self-repairs, redundant modifiers, and non-critical
examples. Do not add information.
Use natural modern spoken language that a TTS system can read aloud smoothly.
For Chinese, avoid classical/literary shorthand such as 其, 之, 乃, 遂, 皆,
足证, or telegraphic noun piles. Prefer clear contemporary wording even if it
costs a few more characters.
"""


def build_compression_user_prompt(sample: S2SSample, include_reference: bool = False) -> str:
    target = sample.target
    constraints = {
        "mode": target.mode,
        "target_ratio": target.target_ratio,
        "max_target_chars": target.max_target_chars,
        "max_target_words": target.max_target_words,
        "max_target_duration_s": target.max_target_duration_s,
        "max_target_wpm": target.max_target_wpm,
        "max_end_lag_s": target.max_end_lag_s,
        "must_keep_terms": sample.must_keep_terms,
        "core_meanings": sample.core_meanings,
    }
    parts = [
        f"Source language: {sample.src_lang}",
        f"Target language: {sample.tgt_lang}",
        "Source transcript:",
        sample.source_text,
    ]
    if include_reference and sample.reference_translation:
        parts.extend(["Reference translation:", sample.reference_translation])
    if target.max_target_chars is not None:
        parts.extend(
            [
                "Hard length budget:",
                (
                    f"Use no more than {target.max_target_chars} target-language CJK "
                    "characters. Count every Chinese/Japanese/Korean character."
                ),
            ]
        )
    elif target.max_target_words is not None:
        parts.extend(
            [
                "Hard length budget:",
                f"Use no more than {target.max_target_words} target-language words.",
            ]
        )
    parts.extend(
        [
            "Style requirements:",
            (
                "Use natural modern spoken target-language phrasing suitable for "
                "speech output. Do not use archaic/classical compression or "
                "telegram-like fragments."
            ),
            "Compression constraints as JSON:",
            json.dumps(constraints, ensure_ascii=False),
            "Return only the compressed target-language translation.",
        ]
    )
    return "\n".join(parts)


JUDGE_SYSTEM = """You are an expert evaluator for streaming speech-to-speech translation
and simultaneous interpreting. Judge whether a candidate is an interpreter-style
compressed translation, not a literal full translation. Do not penalize the
candidate merely for using different wording or omitting non-critical detail.
Reward faithful compression when all core meaning is preserved and the output
would be easy to listen to in real time. Penalize critical omissions,
hallucinations, changed numbers, changed entities, changed negation/modality,
broken discourse relations, and wording that would force unnatural speech rate.
Return strict JSON only.
"""


def build_judge_user_prompt(sample: S2SSample, candidate: str) -> str:
    payload = {
        "src_lang": sample.src_lang,
        "tgt_lang": sample.tgt_lang,
        "source_text": sample.source_text,
        "reference_translation": sample.reference_translation,
        "candidate_translation": candidate,
        "core_meanings": sample.core_meanings,
        "must_keep_terms": sample.must_keep_terms,
        "compression_target": sample.target.to_dict(),
        "evaluation_task": (
            "Evaluate semantic compression for streaming S2S. The candidate may be "
            "shorter than the reference. Score whether it preserves the source's "
            "core meaning under the target duration/length budget."
        ),
        "rubric": {
            "semantic_core_preservation": "0..5",
            "critical_omission": "0..5 where 5 means severe critical omission",
            "hallucination": "0..5 where 5 means severe unsupported addition",
            "term_number_faithfulness": "0..5",
            "compression_appropriateness": "0..5",
            "listenability": "0..5, including concise phrasing and low cognitive load",
            "reference_alignment": "0..5, allowing paraphrase and safe compression",
            "overall": "0..5",
            "pass_sft_gate": "boolean; true only if suitable as a supervised target",
        },
        "required_json_keys": [
            "semantic_core_preservation",
            "critical_omission",
            "hallucination",
            "term_number_faithfulness",
            "compression_appropriateness",
            "listenability",
            "reference_alignment",
            "overall",
            "pass_sft_gate",
            "rationale",
            "missing_core_meanings",
            "unsafe_omissions",
            "acceptable_omissions",
            "hallucinated_content",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def sft_messages(sample: S2SSample) -> list[dict[str, str]]:
    answer = sample.compressed_translation or sample.reference_translation
    if not answer:
        raise ValueError(f"sample {sample.id} has no supervised target")
    return [
        {"role": "system", "content": SYSTEM_COMPRESSION.strip()},
        {"role": "user", "content": build_compression_user_prompt(sample, include_reference=False)},
        {"role": "assistant", "content": answer},
    ]
