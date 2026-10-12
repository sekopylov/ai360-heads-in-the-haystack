import unittest
from pathlib import Path

import torch
import yaml

from retrieval_heads.attention import AttentionController, AttentionRequest
from retrieval_heads.attention.flash import validate_flash_request
from retrieval_heads.attention.types import AttentionStep
from retrieval_heads.experiment.scoring import create_retrieval_collector
from retrieval_heads.experiment.types import NeedleSpan


class QuestionMassTests(unittest.TestCase):
    def test_config_preserves_previous_glm_protocol(self):
        root = Path(__file__).resolve().parents[1] / 'colab'
        old = yaml.safe_load((root / 'glm4-6v-flash-no-thinking-qrscore-survey.yaml').read_text())
        new = yaml.safe_load((root / 'glm4-6v-flash-no-thinking-question-mass-survey.yaml').read_text())
        self.assertEqual({k: v for k, v in old.items() if k != 'retrieval_metrics'},
                         {k: v for k, v in new.items() if k != 'retrieval_metrics'})
        self.assertEqual(new['retrieval_metrics'], ['query_retrieval_score_v1',
                         'needle_attention_mass_v1', 'question_attention_mass_v1'])

    def test_compact_question_capture_and_cleanup(self):
        seen = []
        observer = type('Observer', (), {'on_step': lambda _, step: seen.append(step)})()
        controller = AttentionController((0,))
        controller.start(AttentionRequest(question_span=(2, 4)), observer)
        controller.begin_step()
        controller.record(0, torch.tensor([[[[.1, .2, .3, .4]]]]))
        controller.end_step(1)
        torch.testing.assert_close(seen[0].question_attention_mass[0], torch.tensor([.7]))
        self.assertEqual(seen[0].layers, {})
        self.assertEqual(controller._question_mass, {})
        controller.begin_step()
        with self.assertRaisesRegex(RuntimeError, 'did not report'):
            controller.end_step(1)
        controller.finish()
        self.assertEqual(controller._question_mass, {})

    def test_gate_uses_needle_tokens_not_question_tokens(self):
        collector = create_retrieval_collector('question_attention_mass_v1',
            eligible_heads=((0, 0),), prompt_token_ids=[11, 22, 33, 44],
            needle_span=NeedleSpan(0, 2), question_span=NeedleSpan(2, 4))
        for token, mass in [(11, .6), (33, .99), (22, .2)]:
            collector.on_step(AttentionStep(0, token,
                question_attention_mass={0: torch.tensor([mass])}))
        self.assertAlmostEqual(collector.scores['0-0'], .4, places=6)
        self.assertEqual(collector.qualifying_steps, 2)

    def test_full_capture_fallback_uses_question_span(self):
        collector = create_retrieval_collector('question_attention_mass_v1',
            eligible_heads=((0, 0),), prompt_token_ids=[11, 22, 33, 44],
            needle_span=NeedleSpan(0, 2), question_span=NeedleSpan(2, 4))
        collector.on_step(AttentionStep(0, 11, layers={0: torch.tensor([[.1, .2, .3, .4]])}))
        self.assertAlmostEqual(collector.scores['0-0'], .7, places=6)

    def test_flash_decode_rejects_question_capture(self):
        with self.assertRaisesRegex(ValueError, 'without attention capture'):
            validate_flash_request(AttentionRequest(question_span=(1, 2)))
