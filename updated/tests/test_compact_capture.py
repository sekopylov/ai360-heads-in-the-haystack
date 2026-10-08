import unittest
import torch

from retrieval_heads.attention import AttentionController, AttentionRequest


class CompactCaptureTests(unittest.TestCase):
    def test_mass_without_full_vectors_and_top1_together(self):
        for capture in ['none', 'top1', 'full']:
            seen = []
            observer = type('Observer', (), {'on_step': lambda self, step: seen.append(step)})()
            controller = AttentionController((0,))
            controller.start(AttentionRequest(capture=capture, needle_span=(1, 3)), observer)
            controller.begin_step()
            probabilities = torch.tensor([[[[.1, .2, .3, .4]], [[.4, .3, .2, .1]]]])
            controller.record(0, probabilities)
            controller.end_step(11)
            torch.testing.assert_close(seen[0].needle_attention_mass[0], torch.tensor([.5, .5]))
            self.assertEqual(seen[0].needle_attention_mass[0].numel(), 2)
            if capture == 'none':
                self.assertEqual(seen[0].layers, {})
            elif capture == 'top1':
                self.assertEqual(seen[0].layers[0].tolist(), [3, 0])
            self.assertEqual(controller._layers, {})
            self.assertEqual(controller._needle_mass, {})
            controller.finish()

    def test_gate_applies_to_compact_mass_and_missing_layer_is_checked(self):
        controller = AttentionController((0,))
        seen = []
        observer = type('Observer', (), {'on_step': lambda self, step: seen.append(step)})()
        controller.start(AttentionRequest(needle_span=(0, 1)), observer)
        controller.begin_step()
        controller.record(0, torch.ones(1, 1, 1, 1))
        controller.end_step(11, publish=False)
        self.assertEqual(seen, [])
        controller.begin_step()
        with self.assertRaisesRegex(RuntimeError, 'did not report'):
            controller.end_step(11)
        controller.finish()
