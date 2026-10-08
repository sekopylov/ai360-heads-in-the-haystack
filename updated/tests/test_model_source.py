import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from retrieval_heads.model_source import resolve_model_source


class ModelSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def checkpoint(self, path):
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}")
        (path / "model.safetensors").write_bytes(b"test fixture")
        return path

    def test_direct_directory(self):
        local = self.checkpoint(self.root / "checkpoint")
        self.assertEqual(resolve_model_source(str(local)), (str(local), True))

    def test_ordered_search_and_preserved_hub_id(self):
        first = self.checkpoint(self.root / "first/Qwen/Qwen3.5-0.8B")
        self.checkpoint(self.root / "second/Qwen/Qwen3.5-0.8B")
        self.assertEqual(resolve_model_source("Qwen/Qwen3.5-0.8B", [str(self.root / "missing"),
                         str(self.root / "first"), str(self.root / "second")]), (str(first), True))

    def test_project_environment_does_not_add_implicit_search(self):
        self.checkpoint(self.root / "models/Qwen/Qwen3.5-0.8B")
        with patch.dict(os.environ, {"DS_PROJECT_HOME": str(self.root)}):
            self.assertEqual(resolve_model_source("Qwen/Qwen3.5-0.8B"),
                             ("Qwen/Qwen3.5-0.8B", False))

    def test_only_explicit_roots_are_searched(self):
        self.checkpoint(self.root / "models/Qwen/Qwen3.5-0.8B")
        with patch.dict(os.environ, {"DS_PROJECT_HOME": str(self.root)}):
            self.assertEqual(resolve_model_source("Qwen/Qwen3.5-0.8B", [str(self.root / "other")]),
                             ("Qwen/Qwen3.5-0.8B", False))

    def test_missing_checkpoint_falls_back_to_hub(self):
        self.assertEqual(resolve_model_source("Qwen/Qwen3.5-0.8B", [str(self.root)]),
                         ("Qwen/Qwen3.5-0.8B", False))

    def test_missing_direct_path_does_not_become_hub_id(self):
        with self.assertRaises(FileNotFoundError):
            resolve_model_source(str(self.root / "missing"))

    def test_incomplete_local_checkpoint_fails(self):
        incomplete = self.root / "Qwen/Qwen3.5-0.8B"
        incomplete.mkdir(parents=True)
        (incomplete / "config.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "weights missing"):
            resolve_model_source("Qwen/Qwen3.5-0.8B", [str(self.root)])

    def test_sharded_weights(self):
        local = self.root / "sharded"
        local.mkdir()
        (local / "config.json").write_text("{}")
        (local / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"a": "part1.safetensors", "b": "part2.safetensors"}}))
        (local / "part1.safetensors").write_bytes(b"test fixture")
        with self.assertRaisesRegex(ValueError, "shards missing"):
            resolve_model_source(str(local))
        (local / "part2.safetensors").write_bytes(b"test fixture")
        self.assertEqual(resolve_model_source(str(local)), (str(local), True))


if __name__ == "__main__":
    unittest.main()
