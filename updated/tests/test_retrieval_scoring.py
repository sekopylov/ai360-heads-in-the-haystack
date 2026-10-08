import unittest

import torch

from retrieval_heads.attention.types import AttentionStep
from retrieval_heads.experiment.scoring import (
    LegacyRetrievalScoreCollector, MultisetRetrievalScoreCollector, NeedleAttentionMassCollector,
    available_retrieval_metrics, create_retrieval_collector,
    select_retrieval_metrics, retrieval_capture_requirements,
)
from retrieval_heads.experiment.types import NeedleSpan


class MultisetScoreTests(unittest.TestCase):
    def collector(self):
        return MultisetRetrievalScoreCollector(eligible_heads=((0, 0), (0, 1)),
                                       prompt_token_ids=[99, 11, 11, 22, 99],
                                       needle_span=NeedleSpan(1, 4))

    def step(self, collector, token, positions):
        collector.on_step(AttentionStep(index=0, token_id=token,
                                        layers={0: torch.tensor(positions)}))

    def test_duplicate_quota_per_head_and_exact_maximum(self):
        collector = self.collector()
        for _ in range(10):
            # Repeated hits at the same position may use the token's multiset quota.
            self.step(collector, 11, [1, 2])
        self.assertEqual(collector.scores, {"0-0": 2 / 3, "0-1": 2 / 3})
        for _ in range(10):
            self.step(collector, 22, [3, 3])
        self.assertEqual(collector.scores, {"0-0": 1.0, "0-1": 1.0})

    def test_mismatch_and_positions_outside_needle_do_not_use_quota(self):
        collector = self.collector()
        self.step(collector, 11, [0, 3])  # outside; mismatched prompt token
        self.step(collector, 22, [5, -1])
        self.assertEqual(collector.scores, {"0-0": 0.0, "0-1": 0.0})
        self.step(collector, 11, [1, 2])
        self.assertEqual(collector.scores, {"0-0": 1 / 3, "0-1": 1 / 3})

    def test_full_probabilities_and_independent_head_budgets(self):
        collector = self.collector()
        collector.on_step(AttentionStep(index=0, token_id=11, layers={0: torch.tensor([
            [0., 1., 0., 0., 0.], [0., 0., 0., 1., 0.]])}))
        self.assertEqual(collector.scores, {"0-0": 1 / 3, "0-1": 0.0})
        self.step(collector, 11, [0, 2])
        self.assertEqual(collector.scores, {"0-0": 1 / 3, "0-1": 1 / 3})

    def test_invalid_span_is_rejected(self):
        for span in (NeedleSpan(1, 1), NeedleSpan(-1, 2), NeedleSpan(0, 8)):
            with self.assertRaises(ValueError):
                MultisetRetrievalScoreCollector(eligible_heads=((0, 0),), prompt_token_ids=[11], needle_span=span)

    def test_legacy_preserves_uncapped_repeated_hits(self):
        collector = LegacyRetrievalScoreCollector(eligible_heads=((0, 0), (0, 1)),
                    prompt_token_ids=[99, 11, 11, 22, 99], needle_span=NeedleSpan(1, 4))
        for _ in range(12):
            self.step(collector, 11, [1, 0])
        self.assertAlmostEqual(collector.scores["0-0"], 4.0)
        self.assertEqual(collector.scores["0-1"], 0.0)
        self.assertEqual(collector.metric_name, "legacy")

    def test_unknown_metric_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown retrieval metric"):
            create_retrieval_collector("unknown", eligible_heads=((0, 0),), prompt_token_ids=[11],
                                      needle_span=NeedleSpan(0, 1))

    def test_factory_selects_independent_implementations(self):
        self.assertEqual(set(available_retrieval_metrics()), {"legacy", "needle_token_multiset_v1", "needle_attention_mass_v1"})
        for name, cls in (("legacy", LegacyRetrievalScoreCollector),
                          ("needle_token_multiset_v1", MultisetRetrievalScoreCollector)):
            collector = create_retrieval_collector(name, eligible_heads=((0, 0),),
                        prompt_token_ids=[11], needle_span=NeedleSpan(0, 1))
            self.assertIsInstance(collector, cls)
            collector.on_step(AttentionStep(index=0, token_id=11, layers={0: torch.tensor([0])}))
            collector.on_step(AttentionStep(index=1, token_id=11, layers={0: torch.tensor([0])}))
            self.assertEqual(collector.scores["0-0"], 2.0 if name == "legacy" else 1.0)


class AttentionMassTests(unittest.TestCase):
    def make_collector(self):
        return create_retrieval_collector("needle_attention_mass_v1",
            eligible_heads=((0, 0), (0, 1)), prompt_token_ids=[99, 11, 11, 22, 99],
            needle_span=NeedleSpan(1, 4))

    def test_conditional_mean_includes_repeats_and_zero_mass(self):
        c = self.make_collector()
        self.assertIsInstance(c, NeedleAttentionMassCollector)
        for token, masses in [(11, [.8, .2]), (99, [1., 1.]), (11, [.4, .6]), (22, [0., .4])]:
            c.on_step(AttentionStep(index=0, token_id=token,
                                   needle_attention_mass={0: torch.tensor(masses)}))
        self.assertAlmostEqual(c.scores['0-0'], .4, places=6)
        self.assertAlmostEqual(c.scores['0-1'], .4, places=6)
        self.assertEqual(c._counts, {'0-0': 3, '0-1': 3})

    def test_full_distribution_sums_entire_span_not_matching_positions(self):
        c = self.make_collector()
        c.on_step(AttentionStep(index=0, token_id=22, layers={0: torch.tensor([
            [.6, .2, .1, .1, 0.], [.1, .2, .3, .4, 0.]])}))
        self.assertAlmostEqual(c.scores['0-0'], .4, places=6)
        self.assertAlmostEqual(c.scores['0-1'], .9, places=6)

    def test_no_matching_generated_tokens_is_zero_and_top1_rejected(self):
        c = self.make_collector()
        c.on_step(AttentionStep(index=0, token_id=99))
        self.assertEqual(c.scores, {'0-0': 0., '0-1': 0.})
        with self.assertRaisesRegex(ValueError, 'not top1'):
            c.on_step(AttentionStep(index=1, token_id=11, layers={0: torch.tensor([1, 2])}))

    def test_selection_requirements_and_validation(self):
        single, names = select_retrieval_metrics(None, 'needle_attention_mass_v1,legacy')
        self.assertIsNone(single)
        self.assertEqual(retrieval_capture_requirements(names), (True, True))
        self.assertEqual(retrieval_capture_requirements(['needle_attention_mass_v1']), (False, True))
        self.assertEqual(select_retrieval_metrics(None, None), ('needle_token_multiset_v1', ['needle_token_multiset_v1']))
        for primary, multiple in [(None, ''), (None, 'legacy,legacy'), (None, 'unknown'), ('legacy', 'needle_attention_mass_v1')]:
            with self.assertRaises(ValueError):
                select_retrieval_metrics(primary, multiple)


if __name__ == "__main__":
    unittest.main()
