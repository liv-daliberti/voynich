"""Synthetic fixtures to prevent label leakage and catch score regressions."""
from experiments.phonotactics.evaluate import (
    degree_length_matched_auc, glyph_bigram_model, make_rows, metrics,
)
from collections import Counter


def test_bigram_only_uses_training_vocabulary():
    model = glyph_bigram_model(Counter({'daiin': 3, 'dain': 2}))
    assert model('daiin') > model('qxz')


def test_candidates_are_not_observed_training_words():
    train = [{'words': ['dain', 'daiin', 'dain', 'dain']},
             {'words': ['dan', 'dain', 'dol']}]
    future = [{'words': ['dain', 'doin', 'dos', 'daiin']}]
    rows, info = make_rows(train, future, min_degree=1)
    words = {row['word'] for row in rows}
    assert not words.intersection({'dain', 'daiin', 'dan', 'dol'})
    assert info['new_future_types'] == 2
    assert all(row['attested_later'] == (row['word'] in {'doin', 'dos'}) for row in rows)


def test_stratified_comparison_controls_degree_and_length():
    items = [
        {'degree': 3, 'glyph_length': 4, 'attested_later': 1, 'bigram': -.1},
        {'degree': 3, 'glyph_length': 4, 'attested_later': 0, 'bigram': -.9},
        {'degree': 5, 'glyph_length': 6, 'attested_later': 1, 'bigram': -.8},
        {'degree': 5, 'glyph_length': 6, 'attested_later': 0, 'bigram': -.8},
    ]
    result = degree_length_matched_auc(items)
    assert result['positive_negative_pairs'] == 2
    assert result['pair_weighted_auc'] == 0.75
    assert metrics([0, 1], [.1, .9])['roc_auc'] == 1.0
