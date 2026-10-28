"""English controls and slot-filling invariants on tiny artificial corpora."""
from collections import Counter

from experiments.english_controls.benchmark import (
    character_edits, clean_gutenberg, english_features, evaluate_sample,
)
from experiments.context.slot_filling import (
    evaluate_cloze, graph_stats, shuffle_within_group_position,
)


def test_english_character_edits_do_not_combine_ch_and_sh():
    proposals = character_edits('chat', tuple('abcdefghst'))
    assert 'hat' in proposals
    assert 'shat' in proposals
    assert 'cat' in proposals
    assert 'chat' not in proposals


def test_gutenberg_trim_and_normalization():
    raw = ('Gutenberg boilerplate\nChapter 1\nDon\'t take the book. '
           'He walked!\n*** END OF THIS PROJECT GUTENBERG EBOOK\n '
           'Not in the body.')
    assert clean_gutenberg(raw, 'austen') == [
        'don', 't', 'take', 'the', 'book', 'he', 'walked']


def test_attestation_labels_cannot_leak_into_train_features():
    train = ['cat', 'cot', 'cut', 'cat', 'cot', 'cut']
    a, summary, _, _ = english_features(train, ['cit'], min_degree=2)
    b, _, _, _ = english_features(train, ['cit', 'citrus'], min_degree=2)
    by_a = {r['word']: r for r in a}
    by_b = {r['word']: r for r in b}
    assert by_a['cit']['attested_later'] == 1
    assert by_a['cit']['degree'] >= 2
    assert summary['new_future_types'] == 1
    assert [(v['degree'], v['bigram']) for v in a] == [
        (v['degree'], v['bigram']) for v in b]
    assert by_b['cit']['attested_later'] == 1


def test_shuffling_preserves_topic_and_position_marginals():
    sequences = [
        ('folio1', ['cat', 'sat', 'on', 'mat']),
        ('folio1', ['dog', 'sat', 'by', 'gate']),
        ('folio2', ['this', 'is', 'another', 'line']),
    ]
    null = shuffle_within_group_position(sequences, 25)
    for group in ('folio1', 'folio2'):
        for position in (0, -1):
            original = Counter(w[position] for grp, w in sequences if grp == group)
            shuffled = Counter(w[position] for grp, w in null if grp == group)
            assert original == shuffled
    assert graph_stats(null)[0] == graph_stats(sequences)[0]


def test_sparse_cloze_returns_explicit_no_tasks():
    sequences = [('folio1', ['the', 'cat', 'sat']) for _ in range(20)]
    report, tasks, edges = evaluate_cloze(sequences, eva=False, negatives=2)
    assert report['novel_cloze_tasks'] == 0
    assert report['metrics']['cloze_graph']['pairwise_accuracy'] is None
    assert edges['the', 'cat'] > 0


def test_masks_include_deleted_training_forms():
    words = ['cat','cot','cut','bat','bot','but','car','bar','far','fat']
    train = (words * 13)[:140]
    future = ['cit','bit','fit','bet','sat','hat','hot','hit','pat','pot']
    summary, rows, masks = evaluate_sample(train + future, mask_trials=4,
                                           top_k=12)
    assert len(masks) == 4
    assert all(m['deleted_training'] + m['later_attested'] +
               m['unattested'] <= 12 for m in masks)
    assert any(m['deleted_training'] > 0 for m in masks)
