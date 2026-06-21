from __future__ import annotations

from typing import Any

from .llm_client import ChatClient, extract_json_object
from .prompts import JUDGE_SYSTEM, build_judge_user_prompt
from .schema import S2SSample


def judge_sample(
    sample: S2SSample,
    candidate: str,
    client: ChatClient,
    temperature: float = 0.0,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM.strip()},
        {"role": "user", "content": build_judge_user_prompt(sample, candidate)},
    ]
    raw = client.chat(
        messages,
        temperature=temperature,
        max_tokens=1200,
        response_format={"type": "json_object"},
    )
    data = extract_json_object(raw)
    data["judge_raw"] = raw
    return data


def combine_judge_and_heuristics(
    heuristic: dict[str, Any], judge: dict[str, Any] | None
) -> dict[str, Any]:
    if not judge:
        return heuristic
    row = dict(heuristic)
    for key, value in judge.items():
        row[f"judge_{key}"] = value
    overall = judge.get("overall")
    if isinstance(overall, (int, float)):
        row["judge_overall_0_1"] = float(overall) / 5.0
    return row
