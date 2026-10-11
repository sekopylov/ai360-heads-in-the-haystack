"""Artifact checks for the real Ministral detection smoke (no GPU needed)."""
import json
import tempfile
import unittest
from pathlib import Path

from datasphere.validate_ministral_smoke import HEADS, METRICS, validate


class MinistralSmokeValidationTests(unittest.TestCase):
    def fixture(self, root):
        detection = root / "detection"
        results = detection / "results"
        aggregation = detection / "aggregation"
        results.mkdir(parents=True)
        aggregation.mkdir()
        manifest = {"complete": True, "adapter": "ministral3_8b_no_thinking",
                    "total_cases": 3, "completed_cases": 3, "lengths": [1000],
                    "depths": [50], "attention_scope": "all_decode_tokens",
                    "retrieval_metrics": sorted(METRICS), "run_id": "fixture"}
        (detection / "run.json").write_text(json.dumps(manifest))
        for i in range(1, 4):
            row = {"case_id": f"detect-{i}", "generated_token_ids": [7, 2],
                   "model_response": "answer", "final_answer_started": True,
                   "finish_reason": "eos", "score": 100.0,
                   "experiment": {"run_id": "fixture",
                       "retrieval_scores": {m: dict.fromkeys(HEADS, 0.1) for m in METRICS},
                       "retrieval_qualifying_steps": {"needle_attention_mass_v1": 1}}}
            (results / f"detect-{i}_results.json").write_text(json.dumps(row))
        files = {m: f"{m}.json" for m in METRICS}
        (aggregation / "run.json").write_text(json.dumps({"source_run_id": "fixture",
            "selected_case_count": 3, "head_score_files": files}))
        for m, name in files.items():
            (aggregation / name).write_text(json.dumps({h: [0.1] * 3 for h in HEADS}))
        return detection

    def test_valid_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            self.assertTrue(validate(root)["passed"])
            self.assertTrue((root / "smoke_validation.json").is_file())

    def test_missing_head_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detection = self.fixture(root)
            path = detection / "results/detect-1_results.json"
            row = json.loads(path.read_text())
            del row["experiment"]["retrieval_scores"]["legacy"]["0-0"]
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "1088 heads"):
                validate(root)

    def test_no_observed_mass_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detection = self.fixture(root)
            for path in (detection / "results").glob("*.json"):
                row = json.loads(path.read_text())
                row["experiment"]["retrieval_qualifying_steps"]["needle_attention_mass_v1"] = 0
                path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "No positive"):
                validate(root)
