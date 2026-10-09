import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("mask_analysis", Path(__file__).parents[1] / "analyze_mask_runs.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_all_details_allow_reordering_and_punctuation():
    result = analysis.evidence('Inside a silver thimble: **velvet-pelican-47**.', 'paul-graham')
    assert result['level'] == 'all_details'


def test_partial_and_absent_are_distinct():
    assert analysis.evidence('He had a compass.', 'eugene-onegin')['level'] == 'partial'
    assert analysis.evidence('He had a bottle.', 'eugene-onegin')['level'] == 'no_anchor'


def test_evidence_requires_local_cooccurrence():
    text = 'green glass compass ' + 'other ' * 70 + 'nearest violin'
    assert analysis.evidence(text, 'eugene-onegin')['level'] == 'partial'


def test_evidence_is_lexical_not_entailment():
    assert analysis.evidence('NOT a porcelain beetle with nine blue dots.', 'hero-of-our-time')['level'] == 'all_details'


def test_repetition_unique_and_periodic_tokens():
    unique = analysis.repetition(list(range(100)), 'distinct words')
    assert unique['repeat_fraction'] == 0
    assert not unique['strong_repeat']
    repeated = analysis.repetition(list(range(40)) * 20, 'same words ' * 50)
    assert repeated['strong_repeat']
    assert repeated['repeat_fraction'] > .9
    assert repeated['dominant_first_token'] == 0
    assert repeated['dominant_second_token'] == 40


def test_short_text_and_numeric_nine():
    assert analysis.repetition([], '')['repeat_fraction'] == 0
    assert analysis.evidence('a porcelain beetle with 9 blue dots', 'hero-of-our-time')['level'] == 'all_details'
