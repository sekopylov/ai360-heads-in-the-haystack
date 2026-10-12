"""Model-config-driven validation for dense Qwen/Ministral/GLM QRScore surveys."""
import json
import math


def validate_survey(output, config, checkpoint_config):
    source = json.loads((output / 'detection/run.json').read_text())
    if not source['complete'] or source['adapter'] != config['adapter']:
        raise ValueError('Expected a complete survey with the configured adapter')
    text = checkpoint_config.get('text_config', checkpoint_config)
    if text.get('model_type') not in ('qwen3', 'ministral3', 'glm4v_text'):
        raise ValueError('Validator supports dense Qwen3, Ministral3 and GLM4v text decoders')
    heads = {f'{layer}-{head}' for layer in range(text['num_hidden_layers'])
             for head in range(text['num_attention_heads'])}
    expected = {(case, length, depth) for case in ('detect-1', 'detect-2', 'detect-3')
                for length in config['lengths'] for depth in config['depths']}
    rows = [json.loads(path.read_text()) for path in (output / 'detection/results').glob('*_results.json')]
    actual = {(r['case_id'], r['context_length'], r['depth_percent']) for r in rows}
    if actual != expected or len(rows) != len(expected) or source['completed_cases'] != len(expected):
        raise ValueError('Survey grid incomplete or duplicated')
    for row in rows:
        experiment = row['experiment']
        if experiment['run_id'] != source['run_id']:
            raise ValueError('Result belongs to another run')
        tokens = row['generated_token_ids']
        if not tokens or len(tokens) > config['max_new_tokens']:
            raise ValueError('Invalid generation length')
        if '<think>' in row['raw_model_response'] or '</think>' in row['raw_model_response']:
            raise ValueError('Unexpected thinking in no-thinking survey')
        query, gold = experiment['qrscore_query_span'], experiment['qrscore_gold_span']
        if not 0 <= gold[0] < gold[1] <= query[0] < query[1]:
            raise ValueError('Invalid QRScore spans')
        if 'full_needle_attention_mass_v1' in config['retrieval_metrics']:
            if experiment.get('full_needle_attention_span') != gold:
                raise ValueError('Full needle decode span differs from QRScore gold span')
            counts = experiment['retrieval_qualifying_steps']
            if counts['full_needle_attention_mass_v1'] != counts['needle_attention_mass_v1']:
                raise ValueError('Full and old needle mass must use identical decode steps')
        if 'question_attention_mass_v1' in config['retrieval_metrics']:
            if experiment.get('question_attention_span') != query:
                raise ValueError('Decode question span differs from QRScore question span')
            counts = experiment['retrieval_qualifying_steps']
            if counts['question_attention_mass_v1'] != counts['needle_attention_mass_v1']:
                raise ValueError('Question and needle mass must use identical decode steps')
        for metric in config['retrieval_metrics']:
            scores = experiment['retrieval_scores'][metric]
            if set(scores) != heads or any(not math.isfinite(v) or not 0 <= v <= 1.000001
                                           for v in scores.values()):
                raise ValueError('Invalid or incomplete head scores')
    aggregation = json.loads((output / 'detection/aggregation/run.json').read_text())
    if (aggregation['source_run_id'] != source['run_id'] or
            aggregation['selected_case_count'] != len(expected) or
            not aggregation['filters']['all_cases']):
        raise ValueError('QRScore aggregation must include all survey cases')
    return {'passed': True, 'kind': 'qrscore_detection_survey', 'cases': len(rows),
            'head_count': len(heads), 'adapter': config['adapter'],
            'metrics': config['retrieval_metrics'],
            'mean_answer_score': sum(r['score'] for r in rows) / len(rows),
            'eos_cases': sum(r['finish_reason'] == 'eos' for r in rows)}
