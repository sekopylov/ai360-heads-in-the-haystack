import json
from pathlib import Path
import tempfile
import unittest
import yaml
from colab.mask_worker import ROOT
from colab.mask_resume_plan import resume_plan, condition_command
from retrieval_heads.experiment.storage import result_stem


class FastResumePlanTests(unittest.TestCase):
    def config(self):
        config = yaml.safe_load((ROOT / 'colab/qwen3-8b-no-thinking-paul-graham-qrscore-mask.yaml').read_text())
        config.update(lengths=[1000, 30000], depths=[0, 100], topks=[1])
        return config

    def make_block(self, output, config, name='baseline', complete=True):
        folder = output / 'evaluation/context-42/evaluation' / name
        (folder / 'results').mkdir(parents=True)
        k = 0 if name == 'baseline' else 1
        run = dict(kind='head_masking', complete=complete, condition=name, adapter=config['adapter'],
                   model='/model', model_version='Qwen3-8B', lengths=config['lengths'], depths=config['depths'],
                   context_seed=42, max_new_tokens=256, prefill_attention='sdpa_flash', decode_attention='sdpa_flash',
                   mask_mode='legacy_uniform', randomized_per_case=False, context_window='seeded_contiguous_offset',
                   case_ids=[config['mask_case_id']], head_selection='top' if k else 'baseline',
                   blocked_heads=[[0, 0]] if k else [], head_scores_file='/ranking' if k else None,
                   completed_cases=4)
        (folder / 'run.json').write_text(json.dumps(run))
        for length in config['lengths']:
            for depth in config['depths']:
                name = result_stem('Qwen3-8B', length, depth, case_id=config['mask_case_id'])+'_results.json'
                # Planning must use atomic manifest + filenames, NOT parse each result/tokenize prompts.
                (folder / 'results' / name).write_text('not parsed in planning')
        return folder

    def test_complete_block_is_skipped_and_missing_conditions_are_planned(self):
        with tempfile.TemporaryDirectory() as d:
            output, config = Path(d), self.config()
            self.make_block(output, config)
            plan = resume_plan(config, output, '/model', '/ranking', [((0, 0), .9), ((0, 1), .1)])
            self.assertTrue(plan[0]['complete'])
            self.assertEqual(plan[0]['remaining'], 0)
            self.assertEqual(sum(p['remaining'] for p in plan), 8)
            self.assertFalse(plan[1]['complete'])

    def test_missing_file_or_changed_manifest_rejected(self):
        for change in ('file', 'backend', 'heads'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                output, config = Path(d), self.config()
                folder = self.make_block(output, config, name='top1')
                if change == 'file':
                    next((folder/'results').iterdir()).unlink()
                else:
                    path = folder / 'run.json'
                    run = json.loads(path.read_text())
                    run['decode_attention' if change == 'backend' else 'blocked_heads'] = 'wrong'
                    path.write_text(json.dumps(run))
                with self.assertRaises(ValueError):
                    resume_plan(config, output, '/model', '/ranking', [((0, 0), .9)])

    def test_partial_block_keeps_full_grid_and_original_seed_policy(self):
        config = self.config()
        args = condition_command(config, ROOT, '/model', '/ranking', '/output',
                                 dict(k=100, selection='top'))
        self.assertIn('--resume', args)
        self.assertNotIn('--seed', args)
        self.assertEqual(args[args.index('--lengths')+1], '1000,30000')
        self.assertEqual(args[args.index('--max-new-tokens')+1], '256')
