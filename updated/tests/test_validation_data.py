import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from retrieval_heads.experiment.data import ContextBuilder, load_validation_cases
from retrieval_heads.experiment.runner import ExperimentRunner
from retrieval_heads.experiment.types import ExperimentCase
from retrieval_heads.models.base import Prompt


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


class ValidationDataTests(unittest.TestCase):
    def write_case(self, root: Path, directory: str, case_id: str) -> Path:
        case = root / directory
        case.mkdir()
        (case / "corpus.txt").write_text(
            "A long fictional corpus. " * 20,
            encoding="utf-8",
        )
        (case / "needle.json").write_text(json.dumps({
            "case_id": case_id,
            "needle": "A deliberately strange fact.",
            "question": "What was the strange fact?",
            "expected_answer": "a strange fact",
        }), encoding="utf-8")
        return case

    def test_loads_sorted_validation_case_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_case(root, "z-case", "case-z")
            self.write_case(root, "a-case", "case-a")
            (root / "README.md").write_text("ignored", encoding="utf-8")

            cases = load_validation_cases(root)

            self.assertEqual([case.case_id for case in cases], ["case-a", "case-z"])
            self.assertEqual(cases[0].haystack_dir, root / "a-case")
            self.assertEqual(cases[0].expected_answer, "a strange fact")

    def test_seeded_random_window_is_paired_and_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            case_dir = Path(directory)
            corpus = "".join(chr(0x100 + index) for index in range(300))
            (case_dir / "corpus.txt").write_text(corpus, encoding="utf-8")
            case = ExperimentCase(
                "case", "[NEEDLE]", "question", "NEEDLE", case_dir,
            )

            def build(seed: int, length: int = 100) -> str:
                builder = ContextBuilder(
                    CharacterTokenizer(),
                    max_context_length=100,
                    period_tokens=[ord(".")],
                    final_context_length_buffer=0,
                    context_seed=seed,
                    random_start=True,
                )
                return builder.build(case, context_length=length, depth_percent=100)

            first = build(42)
            self.assertEqual(first, build(42))
            self.assertNotEqual(first, build(43))
            self.assertEqual(len(first), 100)
            self.assertTrue(first.endswith(case.needle))
            self.assertIn(first[:-len(case.needle)], corpus)

    def test_random_window_requires_a_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            case_dir = Path(directory)
            (case_dir / "corpus.txt").write_text("Text. " * 100, encoding="utf-8")
            case = ExperimentCase("case", "needle", "question", "answer", case_dir)
            builder = ContextBuilder(
                CharacterTokenizer(), max_context_length=50,
                period_tokens=[ord(".")], final_context_length_buffer=0,
                random_start=True,
            )
            with self.assertRaisesRegex(ValueError, "context seed"):
                builder.build(case, context_length=50, depth_percent=100)

    def test_masking_preparation_does_not_require_a_needle_locator(self):
        case = ExperimentCase(
            "case", "needle text", "question", "answer not token-aligned", Path("unused")
        )
        model = SimpleNamespace(
            encode_prompt=lambda context, question: Prompt(None, [1, 2, 3])
        )
        builder = SimpleNamespace(build=lambda *args, **kwargs: "context with needle text")
        runner = ExperimentRunner(model, builder, None)

        prepared = runner.prepare(case, context_length=100, depth_percent=45)

        self.assertIsNone(prepared.needle_span)
        self.assertEqual(prepared.context, "context with needle text")


if __name__ == "__main__":
    unittest.main()
