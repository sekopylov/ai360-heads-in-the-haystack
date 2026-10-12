import unittest
from pathlib import Path
import torch
import yaml

from colab.launch_glm_full_needle_experiment import bind_ranking
from colab.mask_worker import generation_count
from retrieval_heads.attention import AttentionController, AttentionRequest
from retrieval_heads.attention.types import AttentionStep
from retrieval_heads.attention.flash import validate_flash_request
from retrieval_heads.experiment.scoring import create_retrieval_collector
from retrieval_heads.experiment.types import NeedleSpan


class FullNeedleMassTests(unittest.TestCase):
    def test_full_target_same_old_gate(self):
        kwargs = dict(eligible_heads=((0, 0),), prompt_token_ids=[10, 11, 22, 33],
                      needle_span=NeedleSpan(1, 3))
        old = create_retrieval_collector('needle_attention_mass_v1', **kwargs)
        full = create_retrieval_collector('full_needle_attention_mass_v1',
                                        full_needle_span=NeedleSpan(0, 3), **kwargs)
        for token in (11, 10, 22):
            step = AttentionStep(0, token, layers={0: torch.tensor([[.4, .2, .3, .1]])})
            old.on_step(step)
            full.on_step(step)
        self.assertEqual(full.qualifying_steps, 2)  # Full-only token 10 remains excluded.
        self.assertEqual(old.qualifying_steps, full.qualifying_steps)
        self.assertAlmostEqual(old.scores['0-0'], .5, places=6)
        self.assertAlmostEqual(full.scores['0-0'], .9, places=6)

    def test_compact_old_full_and_question_capture(self):
        seen = []
        observer = type('Observer', (), {'on_step': lambda _, step: seen.append(step)})()
        controller = AttentionController((0,))
        controller.start(AttentionRequest(needle_span=(1,3),full_needle_span=(0,3),question_span=(3,4)),observer)
        controller.begin_step()
        controller.record(0,torch.tensor([[[[.4,.2,.3,.1]]]]))
        controller.end_step(11)
        self.assertEqual(seen[0].layers,{})
        torch.testing.assert_close(seen[0].needle_attention_mass[0],torch.tensor([.5]))
        torch.testing.assert_close(seen[0].full_needle_attention_mass[0],torch.tensor([.9]))
        self.assertEqual(controller._full_needle_mass,{})
        controller.finish()
        with self.assertRaises(ValueError):
            validate_flash_request(AttentionRequest(full_needle_span=(0,3)))

    def test_workflow_pins_validated_run_and_counts(self):
        root = Path(__file__).resolve().parents[1]/'colab'
        survey = yaml.safe_load((root/'glm4-6v-flash-no-thinking-full-needle-mass-survey.yaml').read_text())
        mask = yaml.safe_load((root/'glm4-6v-flash-no-thinking-paul-graham-full-needle-mass-top-mask.yaml').read_text())
        self.assertEqual(generation_count(mask),660)
        self.assertEqual(mask['topks'],[5,10,15,20,40])
        run = {**survey,'complete':True,'completed_cases':27,'run_id':'fixture'}
        validation = dict(passed=True,cases=27,head_count=1280)
        resolved = bind_ranking(mask,survey,run,validation,{'config':survey})
        self.assertEqual(resolved['ranking_source_run_id'],'fixture')
        self.assertIsNone(mask['ranking_source_run_id'])
        with self.assertRaises(ValueError):
            bind_ranking(mask,survey,{**run,'complete':False},validation,{'config':survey})
