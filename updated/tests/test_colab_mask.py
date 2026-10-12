from pathlib import Path
import shlex
import unittest
import yaml

from colab.mask_worker import command, ROOT, generation_count, expected_head_keys, model_directory


class ColabMaskTests(unittest.TestCase):
    def test_ministral_matches_glm_top_only_protocol_with_own_ranking(self):
        config = yaml.safe_load((ROOT / 'colab/ministral3-8b-no-thinking-paul-graham-qrscore-top-mask.yaml').read_text())
        glm = yaml.safe_load((ROOT / 'colab/glm4-6v-flash-no-thinking-paul-graham-qrscore-top-mask.yaml').read_text())
        for key in ('lengths', 'depths', 'topks', 'mask_selections', 'max_new_tokens',
                    'context_seed', 'seed', 'dtype', 'prefill_attention', 'decode_attention'):
            self.assertEqual(config[key], glm[key])
        self.assertEqual(config['adapter'], 'ministral3_8b_no_thinking')
        self.assertEqual(config['ranking_metric'], 'query_retrieval_score_v1')
        self.assertEqual(config['ranking_source'], 'ministral3-8b-no-thinking-qrscore-survey')
        self.assertTrue(config['ranking_all_cases'])
        self.assertEqual(generation_count(config), 990)
        self.assertEqual(len(expected_head_keys(config['adapter'])), 1088)
        self.assertEqual(model_directory(config), '/content/models/Ministral-3-8B-Instruct-2512-BF16')

    def test_glm_top_only_qrscore_protocol(self):
        config = yaml.safe_load((ROOT / 'colab/glm4-6v-flash-no-thinking-paul-graham-qrscore-top-mask.yaml').read_text())
        qwen = yaml.safe_load((ROOT / 'colab/qwen3-8b-no-thinking-paul-graham-qrscore-mask.yaml').read_text())
        for key in ('lengths', 'depths', 'topks', 'max_new_tokens', 'context_seed', 'seed'):
            self.assertEqual(config[key], qwen[key])
        self.assertEqual(config['mask_selections'], ['top'])
        self.assertEqual(config['adapter'], 'glm4_6v_flash_no_thinking')
        self.assertEqual(config['dtype'], 'bfloat16')
        self.assertEqual(config['ranking_metric'], 'query_retrieval_score_v1')
        self.assertEqual(generation_count(config), 990)
        self.assertEqual(generation_count(qwen), 1870)
        self.assertEqual(len(expected_head_keys(config['adapter'])), 1280)
        self.assertEqual(len(expected_head_keys(qwen['adapter'])), 1152)
        self.assertEqual(model_directory(config), '/content/models/GLM-4.6V-Flash')
        self.assertEqual(model_directory(qwen), '/content/models/Qwen3-8B')

    def test_same_grid_and_masking_protocol_as_glm(self):
        config = yaml.safe_load((ROOT / 'colab/qwen3-8b-no-thinking-paul-graham-qrscore-mask.yaml').read_text())
        old = yaml.safe_load((ROOT / 'datasphere/glm4-6v-flash-no-thinking-paul-graham-multiset-mask.yaml').read_text())
        reference = shlex.split(old['cmd'])
        args = command(config, Path('/model'), Path('/ranking'), Path('/output'))
        for flag in ('--lengths', '--depths', '--topks', '--mask-selections', '--context-count',
                     '--context-seed', '--seed', '--max-new-tokens', '--mask-case-id',
                     '--prefill-attention', '--decode-attention', '--attention-scope'):
            self.assertEqual(args[args.index(flag) + 1], reference[reference.index(flag) + 1], flag)
        self.assertEqual(args[args.index('--adapter') + 1], 'qwen3_8b_no_thinking')
        self.assertEqual(args[args.index('--dtype') + 1], 'float16')
        self.assertEqual(config['ranking_metric'], 'query_retrieval_score_v1')
        self.assertTrue(config['ranking_all_cases'])
        self.assertEqual(len(config['lengths']) * len(config['depths']) *
                         (1 + 2 * len(config['topks'])), 1870)
