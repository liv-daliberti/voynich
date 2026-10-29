"""Small fixtures for local copying, reverse direction and leakage controls."""
from experiments.local_copy.evaluate import neighborhood_map, measure_page, shuffled_folio, shuffled_line_order, run_corpus
from experiments.higher_order_direction.evaluate import (posbin,ConditionalNgram,blocked_folds,one_corpus)
from experiments.robustness75.robustness import glyphs


def test_edit_is_not_exact_copy():
    lines=[('p1',['chor','daiin']),('p1',['shor','daiin']),('p1',['or','chol'])]
    graph=neighborhood_map(lines,True)
    assert 'shor' in graph['chor']
    assert 'chor' not in graph['chor']
    report=measure_page([words for _,words in lines],graph)
    assert report[('all','previous_line_same_column')][2] >= 1
    assert report[('all','previous_line_same_column')][1] >= 1


def test_page_position_shuffle_preserves_type_counts_and_line_lengths():
    from collections import Counter
    lines=[['the','red','cat'],['the','blue','dog'],
           ['a','small','cat'],['a','large','dog']]
    shuffled=shuffled_folio(lines,17)
    assert [len(x) for x in shuffled]==[len(x) for x in lines]
    assert Counter(w for line in shuffled for w in line)==Counter(
        w for line in lines for w in line)
    assert Counter(line[0] for line in shuffled)==Counter(line[0] for line in lines)
    assert Counter(line[-1] for line in shuffled)==Counter(line[-1] for line in lines)


def test_one_edit_local_copy_synthetic_control():
    # Repeated source -> edited variants on the following physical line.
    lines=[('p1',['chor','daiin','shol']),
           ('p1',['shor','dain','chol']),
           ('p1',['chor','daiin','shol']),
           ('p1',['shor','dain','chol'])]*4
    result=run_corpus(lines,True,permutations=8)
    assert result['tokens']==48
    assert result['metrics']['all']['previous_line_same_column']['one_edit']['observed_rate']>0
    assert 'all:vertical_one_line:one_edit' in result['directional_contrasts']


def test_independent_folds_never_share_folios():
    seq=[(f'folio{j}',[f'w{j}','word','word','end']) for j in range(12)]
    allheld=[]
    for _,train,test in blocked_folds(seq,4):
        a={g for g,_ in train}
        b={g for g,_ in test}
        assert not (a&b)
        allheld.extend(b)
    assert len(set(allheld))==12


def test_position_prior_and_higher_order_normalized():
    seq=[('f1',['the','cat','sat','now']),
         ('f2',['the','dog','ran','now'])]*8
    model=ConditionalNgram(seq,alpha=8,max_vocab=20)
    for mode in ('unigram','position','bigram','trigram','fourgram'):
        history=['the']
        assert abs(sum(model.probability(w,history,1,4,mode)
                       for w in model.vocab)-1)<1e-8
    assert posbin(0,5)=='first'
    assert posbin(4,5)=='last'


def test_higher_order_fold_reports_include_directional_gain():
    seq=[]
    for folio in range(8):
        seq.extend([(f'f{folio}',['the','cat','sat','now']),
                    (f'f{folio}',['the','dog','ran','now'])])
    result=one_corpus(seq,folds=4,max_vocab=20)
    assert result['aggregate']['fourgram']['folios']==8
    assert 'reverse_specific_gain' in result['aggregate']['fourgram']


def test_shuffling_whole_lines_preserves_internal_order():
    lines=[['a','foo','bar'],['b','baz','bat'],['c','can','far']]
    shifted=shuffled_line_order(lines,11)
    assert sorted(map(tuple,shifted))==sorted(map(tuple,lines))
    assert [len(x) for x in shifted]==[len(x) for x in lines]
