import tempfile
import unittest
from pathlib import Path

from scripts.build_floras_qe_inputs import make_segment_rows
from s2s_omni.floras_qe import attach_qe_scores, load_qe_scores, row_key


class FlorasQETest(unittest.TestCase):
    def test_row_key_canonicalizes_historical_dashboard_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            score_path = Path(tmp) / "qe_scores.jsonl"
            score_path.write_text(
                '{"qe_row_key":"run1||seed||960||seed_960","run_id":"run1","model":"seed","chunk_ms":960,'
                '"eval_label":"seed_960","xcomet_qe_score":0.5,"metricx_qe_score":9.0,'
                '"hypothesis_chars":12}\n',
                encoding="utf-8",
            )
            scores = load_qe_scores(score_path)
        row = {
            "run_id": "run1",
            "compare_backend": "seed_ast_chunk960",
            "compare_chunk_ms": 960,
            "eval_label": "seed_ast_chunk960",
            "candidate_text": "hello world!",
        }
        self.assertEqual(row_key(row), "run1||seed||960||seed_960")
        attached = attach_qe_scores(row, scores)
        self.assertEqual(attached["xcomet_qe_score"], 0.5)
        self.assertEqual(attached["metricx_qe_score"], 9.0)

    def test_row_key_canonicalizes_openai_chunk_alias(self) -> None:
        row = {
            "run_id": "run1",
            "compare_backend": "chatgpt",
            "compare_chunk_ms": 1920,
            "eval_label": "openai_chunk1920",
        }
        self.assertEqual(row_key(row), "run1||chatgpt||1920||openai_1920")

    def test_row_key_preserves_variant_labels(self) -> None:
        mixed = {
            "run_id": "run1",
            "compare_backend": "kit",
            "compare_chunk_ms": 1920,
            "eval_label": "kit_mixed_high_quality_target_asr",
        }
        online = {
            "run_id": "run1",
            "compare_backend": "kit",
            "compare_chunk_ms": 1920,
            "eval_label": "kit_online_low_latency_target_asr",
        }
        self.assertNotEqual(row_key(mixed), row_key(online))
        self.assertEqual(
            row_key(mixed),
            "run1||kit||1920||kit_mixed_high_quality_target_asr",
        )

    def test_attach_marks_stale_hypothesis_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            score_path = Path(tmp) / "qe_scores.jsonl"
            score_path.write_text(
                '{"qe_row_key":"run1||chatgpt||960||openai_960","run_id":"run1","model":"chatgpt","chunk_ms":960,'
                '"eval_label":"openai_960","xcomet_qe_score":0.5,"metricx_qe_score":9.0,'
                '"hypothesis_chars":4}\n',
                encoding="utf-8",
            )
            scores = load_qe_scores(score_path)
        row = {
            "run_id": "run1",
            "compare_backend": "chatgpt",
            "compare_chunk_ms": 960,
            "eval_label": "openai_960",
            "candidate_text": "changed text",
        }
        attached = attach_qe_scores(row, scores)
        self.assertEqual(attached["qe_hypothesis_chars_mismatch"]["expected"], 4)
        self.assertEqual(attached["qe_hypothesis_chars_mismatch"]["actual"], 12)

    def test_reference_anchor_segment_builder(self) -> None:
        rows = make_segment_rows(
            key="ref_anchor_short",
            run_id="run1",
            model="reference_anchor",
            chunk_ms=None,
            eval_label="gpt_reference_anchor",
            speed_factor=1.0,
            source_text="one two three four five six",
            hypothesis_text="一 二 三 四 五 六",
            max_source_chars=8,
            max_hypothesis_chars=4,
        )
        self.assertGreater(len(rows), 1)
        self.assertEqual(rows[0]["qe_row_key"], "ref_anchor_short")
        self.assertEqual(rows[0]["model"], "reference_anchor")
        self.assertEqual(rows[0]["reference"], "")
        self.assertEqual(rows[-1]["segment_count"], len(rows))


if __name__ == "__main__":
    unittest.main()
