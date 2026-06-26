#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.floras_live import (
    FLORES_REPO,
    FlorasSelection,
    download_file,
    env_api_key,
    export_floras_audio,
    hf_resolve_url,
    load_parquet_rows,
    parse_float,
    parse_speeds,
    proportional_sentence_rows,
    sanitize_id,
    source_text_without_lang_tag,
    split_sentences,
)
from s2s_omni.io import write_jsonl
from s2s_omni.llm_client import ChatClient, extract_json_object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FLORAS ZH<->EN live S2S manifests.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--repo-id", default=FLORES_REPO)
    parser.add_argument("--speeds", default="1.0,1.5,2.0")
    parser.add_argument("--samples-per-direction", type=int, default=1)
    parser.add_argument("--min-duration-s", type=float, default=600.0)
    parser.add_argument("--max-shards-per-split", type=int, default=0)
    parser.add_argument("--stop-after-candidates-per-direction", type=int, default=32)
    parser.add_argument("--multilingual-shards", nargs="*", default=[])
    parser.add_argument("--monolingual-shards", nargs="*", default=[])
    parser.add_argument("--reference-backend", choices=["openai", "none"], default="openai")
    parser.add_argument("--reference-model", default=os.environ.get("S2S_REFERENCE_MODEL", "gpt-5-mini"))
    parser.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--allow-missing-reference", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-every-shard", type=int, default=10)
    return parser.parse_args()


def shard_specs(
    cache_dir: Path,
    repo_id: str,
    split_dir: str,
    shard_count: int,
    explicit: list[str],
    max_shards: int,
) -> list[dict[str, Any]]:
    if explicit:
        return [{"path": Path(path), "remote_url": "", "remote_path": str(path)} for path in explicit]
    limit = shard_count if max_shards <= 0 else min(shard_count, max_shards)
    indices = list(range(shard_count))
    if split_dir == "multilingual":
        indices = list(reversed(indices))
    indices = indices[:limit]
    specs = []
    for idx in indices:
        remote_path = f"{split_dir}/test-{idx:05d}-of-{shard_count:05d}.parquet"
        local_path = cache_dir / split_dir / Path(remote_path).name
        specs.append(
            {
                "path": local_path,
                "remote_url": hf_resolve_url(repo_id, remote_path),
                "remote_path": remote_path,
            }
        )
    return specs


def ensure_shard(spec: dict[str, Any]) -> Path:
    path = Path(spec["path"])
    remote_url = str(spec.get("remote_url") or "")
    if remote_url:
        return download_file(remote_url, path)
    return path


def candidate_from_row(
    row: dict[str, Any],
    *,
    direction: str,
    source_wav_dir: Path,
    min_duration_s: float,
    source_shard: Path,
    require_translation: bool,
) -> dict[str, Any] | None:
    source_text = source_text_without_lang_tag(str(row.get("text") or ""))
    translation = str(row.get("translation") or "").strip()
    if require_translation and not translation:
        return None
    if not source_text:
        return None
    row_id = str(row.get("id") or "")
    safe = sanitize_id(f"{direction}_{row_id}")
    wav_path = source_wav_dir / f"{safe}.wav"
    try:
        duration_s = export_floras_audio(row, wav_path)
    except Exception as exc:
        print(json.dumps({"rejected": row_id, "reason": f"audio_export_failed:{exc}"}), flush=True)
        return None
    if duration_s < min_duration_s:
        return None
    return {
        "id": row_id,
        "language": row.get("language"),
        "score": parse_float(row.get("score")),
        "source_duration_s": round(duration_s, 6),
        "source_wav_path": str(wav_path),
        "source_transcript": source_text,
        "target_reference_text": translation,
        "summary": str(row.get("summary") or ""),
        "source_shard": str(source_shard),
    }


def collect_candidates(
    shard_list: list[dict[str, Any]],
    *,
    direction: str,
    language: str,
    output_dir: Path,
    min_duration_s: float,
    require_translation: bool,
    log_every: int,
    stop_after: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    source_wav_dir = output_dir / "source_wav"
    for shard_index, spec in enumerate(shard_list, start=1):
        shard = ensure_shard(spec)
        rows = load_parquet_rows(shard)
        for row in rows:
            if str(row.get("language") or "") != language:
                continue
            candidate = candidate_from_row(
                row,
                direction=direction,
                source_wav_dir=source_wav_dir,
                min_duration_s=min_duration_s,
                source_shard=shard,
                require_translation=require_translation,
            )
            if candidate is not None:
                candidates.append(candidate)
                if stop_after > 0 and len(candidates) >= stop_after:
                    candidates.sort(key=lambda item: (item["score"], item["id"]))
                    return candidates
        if log_every > 0 and shard_index % log_every == 0:
            print(
                json.dumps(
                    {
                        "direction": direction,
                        "scanned_shards": shard_index,
                        "candidates": len(candidates),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    candidates.sort(key=lambda item: (item["score"], item["id"]))
    return candidates


def translate_reference(
    client: ChatClient,
    source_text: str,
    target_language: str,
    *,
    sentence_mode: bool = False,
) -> Any:
    if sentence_mode:
        source_sentences = split_sentences(source_text, "en")
        user = {
            "source_language": "English",
            "target_language": target_language,
            "sentences": source_sentences,
            "instruction": "Translate each sentence naturally and faithfully. Return JSON only.",
        }
        schema = '{"translations":["..."]}'
    else:
        user = {
            "source_language": "English",
            "target_language": target_language,
            "text": source_text,
            "instruction": "Translate the full transcript naturally and faithfully for speech evaluation. Return JSON only.",
        }
        schema = '{"translation":"..."}'
    content = json.dumps(user, ensure_ascii=False)
    messages = [
        {
            "role": "system",
            "content": f"You are a professional translation engine. Return JSON only with schema {schema}",
        },
        {"role": "user", "content": content},
    ]
    raw = client.chat(messages, temperature=0.0, max_tokens=8192, response_format={"type": "json_object"})
    data = extract_json_object(raw)
    if sentence_mode:
        values = data.get("translations")
        if not isinstance(values, list):
            raise ValueError("sentence translation response missing translations list")
        return [str(item).strip() for item in values]
    return str(data.get("translation") or "").strip()


def build_selection(
    candidate: dict[str, Any],
    *,
    direction: str,
    target_lang: str,
    reference_kind: str,
    translated_sentences: list[str] | None,
) -> FlorasSelection:
    source_sentences = split_sentences(candidate["source_transcript"], candidate["language"])
    sentence_rows = proportional_sentence_rows(
        source_sentences,
        float(candidate["source_duration_s"]),
        target_lang,
        translated_sentences=translated_sentences,
    )
    return FlorasSelection(
        direction=direction,
        source_lang=str(candidate["language"]),
        target_lang=target_lang,
        id=str(candidate["id"]),
        score=float(candidate["score"]),
        source_duration_s=float(candidate["source_duration_s"]),
        source_wav_path=str(candidate["source_wav_path"]),
        source_transcript=str(candidate["source_transcript"]),
        target_reference_text=str(candidate["target_reference_text"]),
        summary=str(candidate.get("summary") or ""),
        source_shard=str(candidate["source_shard"]),
        reference_kind=reference_kind,
        target_sentences=sentence_rows,
    )


def selection_to_dict(selection: FlorasSelection) -> dict[str, Any]:
    return {
        "sample_id": sanitize_id(f"{selection.direction}_{selection.id}"),
        "direction": selection.direction,
        "source_lang": selection.source_lang,
        "target_lang": selection.target_lang,
        "floras_id": selection.id,
        "score": selection.score,
        "source_duration_s": round(selection.source_duration_s, 6),
        "source_wav_path": selection.source_wav_path,
        "source_transcript": selection.source_transcript,
        "target_reference_text": selection.target_reference_text,
        "summary": selection.summary,
        "source_shard": selection.source_shard,
        "reference_kind": selection.reference_kind,
        "target_sentences": selection.target_sentences,
    }


def expand_runs(selection_rows: list[dict[str, Any]], speeds: list[float]) -> list[dict[str, Any]]:
    runs = []
    for row in selection_rows:
        for speed in speeds:
            run = dict(row)
            run["speed_factor"] = speed
            run["run_id"] = f"{row['sample_id']}__speed_{speed:g}"
            runs.append(run)
    return runs


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    speeds = parse_speeds(args.speeds)

    multilingual = shard_specs(
        cache_dir,
        args.repo_id,
        "multilingual",
        114,
        args.multilingual_shards,
        args.max_shards_per_split,
    )
    monolingual = shard_specs(
        cache_dir,
        args.repo_id,
        "monolingual",
        14,
        args.monolingual_shards,
        args.max_shards_per_split,
    )

    zh_candidates = collect_candidates(
        multilingual,
        direction="zh-en",
        language="zh",
        output_dir=output_dir,
        min_duration_s=args.min_duration_s,
        require_translation=True,
        log_every=args.log_every_shard,
        stop_after=args.stop_after_candidates_per_direction,
    )
    en_candidates = collect_candidates(
        monolingual,
        direction="en-zh",
        language="en",
        output_dir=output_dir,
        min_duration_s=args.min_duration_s,
        require_translation=False,
        log_every=args.log_every_shard,
        stop_after=args.stop_after_candidates_per_direction,
    )
    if len(zh_candidates) < args.samples_per_direction:
        raise SystemExit(f"not enough zh->en candidates with translation: {len(zh_candidates)}")
    if len(en_candidates) < args.samples_per_direction:
        raise SystemExit(f"not enough en->zh candidates: {len(en_candidates)}")

    client: ChatClient | None = None
    if args.reference_backend == "openai":
        client = ChatClient(
            base_url=args.openai_base_url.rstrip("/"),
            api_key=env_api_key(),
            model=args.reference_model,
            timeout_s=180.0,
        )

    selections = []
    for candidate in zh_candidates[: args.samples_per_direction]:
        translated_sentences = None
        if client is not None:
            translated_sentences = translate_reference(
                client,
                candidate["source_transcript"],
                "English",
                sentence_mode=True,
            )
        selections.append(
            build_selection(
                candidate,
                direction="zh-en",
                target_lang="en",
                reference_kind="floras_x_en_translation",
                translated_sentences=translated_sentences,
            )
        )

    for candidate in en_candidates[: args.samples_per_direction]:
        reference_kind = "missing"
        translated_sentences = None
        if client is not None:
            candidate["target_reference_text"] = translate_reference(
                client,
                candidate["source_transcript"],
                "Chinese",
                sentence_mode=False,
            )
            translated_sentences = translate_reference(
                client,
                candidate["source_transcript"],
                "Chinese",
                sentence_mode=True,
            )
            reference_kind = f"llm_generated:{args.reference_model}"
        if not candidate.get("target_reference_text") and not args.allow_missing_reference:
            raise SystemExit("en->zh reference is empty; set OPENAI_API_KEY or use --allow-missing-reference")
        selections.append(
            build_selection(
                candidate,
                direction="en-zh",
                target_lang="zh",
                reference_kind=reference_kind,
                translated_sentences=translated_sentences,
            )
        )

    selection_rows = [selection_to_dict(selection) for selection in selections]
    for row in selection_rows:
        row["reference_backend"] = args.reference_backend
        row["reference_model"] = args.reference_model if args.reference_backend != "none" else None
        row["reference_prompt_version"] = "floras_live_reference_v1"
    run_rows = expand_runs(selection_rows, speeds)
    write_jsonl(output_dir / "selected_samples.jsonl", selection_rows)
    write_jsonl(output_dir / "live_runs.jsonl", run_rows)
    (output_dir / "manifest_meta.json").write_text(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "speeds": speeds,
                "samples_per_direction": args.samples_per_direction,
                "min_duration_s": args.min_duration_s,
                "reference_backend": args.reference_backend,
                "reference_model": args.reference_model if args.reference_backend != "none" else None,
                "selected_samples": len(selection_rows),
                "runs": len(run_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"selected_samples": len(selection_rows), "runs": len(run_rows)}, indent=2))


if __name__ == "__main__":
    main()
