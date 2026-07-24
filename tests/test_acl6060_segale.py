from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


segale_inputs = load_script(
    "acl6060_segale_inputs",
    ROOT / "scripts/build_acl6060_segale_inputs.py",
)
xcomet_input = load_script(
    "acl6060_segale_xcomet_input",
    ROOT / "scripts/build_acl6060_xcomet_input.py",
)
xcomet_runner = load_script(
    "acl6060_segale_xcomet_runner",
    ROOT / "scripts/run_acl6060_xcomet_xl.py",
)


def test_prediction_units_preserve_original_character_spans() -> None:
    units, spans = segale_inputs.prediction_units_with_spans("你 好。", "zh")
    assert units == ["你", "好", "。"]
    assert spans == [(0, 1), (2, 3), (3, 4)]

    units, spans = segale_inputs.prediction_units_with_spans("Hallo,  Welt!", "de")
    assert units == ["Hallo,", "Welt!"]
    assert spans == [(0, 6), (8, 13)]


def test_segale_input_builder_groups_gold_segments_by_document(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    data_dir = tmp_path / "data"
    run_dir.mkdir()
    data_dir.mkdir()
    (data_dir / "source.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    (data_dir / "ref.txt").write_text("一\n二\n三\n", encoding="utf-8")
    (data_dir / "audio.yaml").write_text(
        "- wav: path/a.wav\n  offset: 0.0\n  duration: 1.0\n"
        "- wav: path/a.wav\n  offset: 1.0\n  duration: 2.0\n"
        "- wav: path/b.wav\n  offset: 0.0\n  duration: 3.0\n",
        encoding="utf-8",
    )
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "source_text_file": str(data_dir / "source.txt"),
                "ref_file": str(data_dir / "ref.txt"),
                "audio_yaml": str(data_dir / "audio.yaml"),
                "target_lang": "zh",
                "speed_factor": 2.0,
            }
        ),
        encoding="utf-8",
    )
    instances = [
        {
            "index": 0,
            "source": ["/audio/a.wav"],
            "prediction": "甲乙",
            "delays": [1, 2],
            "elapsed": [2, 3],
        },
        {
            "index": 1,
            "source": ["/audio/b.wav"],
            "prediction": "丙",
            "delays": [1],
            "elapsed": [2],
        },
    ]
    (run_dir / "instances.log").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in instances) + "\n",
        encoding="utf-8",
    )

    output_dir = run_dir / "segale_alignment"
    summary = segale_inputs.build_segale_inputs(run_dir, output_dir)
    ref_rows = segale_inputs.read_jsonl(output_dir / "ref.jsonl")
    hyp_rows = segale_inputs.read_jsonl(output_dir / "hyp.jsonl")
    assert summary["source_segments"] == 3
    assert [row["seg_id"] for row in ref_rows] == [1, 2, 3]
    assert hyp_rows == [
        {"src": "one two", "tgt": "甲乙", "sys_id": "run", "doc_id": "a.wav", "seg_id": 0},
        {"src": "three", "tgt": "丙", "sys_id": "run", "doc_id": "b.wav", "seg_id": 1},
    ]
    scaled_yaml = (output_dir / "audio.scaled.basename.yaml").read_text(encoding="utf-8")
    assert "offset: 0.5" in scaled_yaml
    assert "duration: 1.0" in scaled_yaml


def test_segale_null_alignments_are_fixed_to_zero() -> None:
    config = {
        "provider": "provider",
        "target_lang": "zh",
        "chunk_ms": 960,
        "speed_factor": 1.0,
    }
    segments = [
        {
            "doc_id": "a.wav",
            "seg_id": 1,
            "src": "source",
            "ref": "参考",
            "tgt": "假设",
            "src_ref_ids": [0],
            "mt_indices": [0],
        },
        {
            "doc_id": "a.wav",
            "seg_id": 2,
            "src": "",
            "ref": "",
            "tgt": "额外内容",
            "src_ref_ids": [],
            "mt_indices": [1],
        },
        {
            "doc_id": "a.wav",
            "seg_id": 3,
            "src": "missing source content",
            "ref": "缺失内容",
            "tgt": "",
            "src_ref_ids": [1],
            "mt_indices": [],
        },
    ]
    rows = xcomet_input.build_xcomet_rows(Path("/run"), segments, config)
    assert [row["null_alignment_type"] for row in rows] == [
        "",
        "over_translation",
        "under_translation",
    ]
    assert [row["fixed_xcomet_xl_score"] for row in rows] == [None, 0.0, 0.0]

    scored = xcomet_runner.attach_scores(
        rows,
        model_scores=[0.9],
        model_name="Unbabel/XCOMET-XL",
        mode="source_hypothesis_reference",
    )
    assert [row["xcomet_xl_score"] for row in scored] == [0.9, 0.0, 0.0]
    assert xcomet_runner.arithmetic_mean(scored, "xcomet_xl_score") == 0.3


def test_comet_prediction_load_compatibility_preserves_explicit_weight_mode(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    fake_torch = SimpleNamespace(
        load=lambda *args, **kwargs: calls.append(kwargs) or kwargs.get("weights_only")
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    xcomet_runner.configure_comet_prediction_load_compatibility()
    assert fake_torch.load("prediction.pt") is False
    assert fake_torch.load("checkpoint.pt", weights_only=True) is True
    assert calls == [{"weights_only": False}, {"weights_only": True}]


def test_segale_alignment_requires_complete_source_and_target_coverage() -> None:
    complete = [
        {"doc_id": "a", "src_ref_ids": [1], "mt_indices": [0]},
        {"doc_id": "a", "src_ref_ids": [], "mt_indices": [1]},
        {"doc_id": "a", "src_ref_ids": [2], "mt_indices": []},
    ]
    xcomet_input.validate_alignment_coverage(complete, 2)
