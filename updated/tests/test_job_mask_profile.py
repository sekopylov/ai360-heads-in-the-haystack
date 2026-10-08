import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "retrieval_job_mask_tests", Path(__file__).resolve().parents[1] / "datasphere" / "job.py"
)
job = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = job
SPEC.loader.exec_module(job)


class JobMaskProfileTests(unittest.TestCase):
    def test_bottom_strategy_and_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scores = root / 'scores.json'
            scores.write_text('{}')
            argv = ['job.py', '--profile', 'mask', '--output-root', str(root / 'out'),
                    '--mask-data', str(root), '--head-scores', str(scores),
                    '--device-map', 'cpu', '--lengths', '8000,30000', '--depths', '15,45,75',
                    '--context-count', '2', '--topks', '4,8', '--random-repeats', '2',
                    '--mask-selections', 'top,bottom,random']
            with patch.object(sys, 'argv', argv), patch.object(job, 'run_command') as run:
                job.main()
            self.assertEqual(len(run.call_args_list), 18)
            self.assertEqual(run.call_args_list[0].kwargs['progress'].total, 108)
            bottom = [c for c in run.call_args_list if 'bottom' in c.args]
            self.assertEqual(len(bottom), 4)
            self.assertTrue(all('--head-selection' in c.args for c in bottom))

    def test_mask_reuses_full_masking_loop_without_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scores = root / "scores.json"
            scores.write_text(json.dumps({"0-0": [1.0]}))
            (root / "needles.jsonl").write_text('{}\n{}\n{}\n')
            output = root / "output"
            arguments = ["job.py", "--output-root", str(output),
                         "--head-scores", str(scores), "--mask-data", str(root),
                         "--device-map", "cpu", "--lengths", "4000,8000",
                         "--depths", "15,45,75", "--topks", "1,2",
                         "--context-count", "2", "--random-repeats", "3",
                         "--context-seed", "100", "--seed", "42"]
            calls = {}
            for profile in ("mask", "full"):
                argv = arguments + ["--profile", profile]
                if profile == "full":
                    argv += ["--detection-data", str(root)]
                with patch.object(sys, "argv", argv), patch.object(job, "run_command") as run:
                    self.assertEqual(job.main(), 0)
                calls[profile] = run.call_args_list
            self.assertEqual(calls["full"][0].args[0], "retrieval_head_detection.py")
            self.assertEqual(calls["full"][1].args[0], "aggregate_head_scores.py")
            self.assertEqual([call.args for call in calls["mask"]],
                             [call.args for call in calls["full"][2:]])
            self.assertEqual(len(calls["mask"]), 18)
            self.assertEqual(calls["mask"][0].kwargs["progress"].total, 108)
            self.assertEqual(calls["full"][0].kwargs["progress"].total, 126)
            self.assertTrue(all(call.args[0] == "needle_in_haystack_with_mask.py"
                                for call in calls["mask"]))

    def test_mask_validates_inputs_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            base = ["job.py", "--profile", "mask", "--output-root", str(output)]
            cases = [([], "--mask-data"),
                     (["--mask-data", str(root)], "--head-scores"),
                     (["--mask-data", str(root), "--head-scores", str(root / "missing")],
                      "Ranking file does not exist")]
            for extra, error in cases:
                with self.subTest(error=error), patch.object(sys, "argv", base + extra):
                    with self.assertRaisesRegex(SystemExit, error):
                        job.main()
            self.assertFalse(output.exists())

    def test_detection_profiles_still_require_detection_data(self):
        for profile in ("smoke", "full"):
            with self.subTest(profile=profile), patch.object(sys, "argv", [
                    "job.py", "--profile", profile, "--output-root", "/tmp/unused-job-test"]):
                with self.assertRaisesRegex(SystemExit, "--detection-data"):
                    job.main()
