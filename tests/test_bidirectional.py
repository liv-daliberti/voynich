import numpy as np
import pytest
from experiments.latent_grammar.word_lattice import WordLattice
from experiments.bidirectional.bidirectional import (
    CachedWalk, backwards, evaluate, fused, bins, select_fusion
)


def corpus():
    # Position 2 is the target, with different predictive evidence on each side.
    return [
        ("p1",["the","red","bird","flies","away"]),
        ("p1",["the","blue","bird","sings","away"]),
        ("p2",["a","small","dog","runs","fast"]),
        ("p2",["a","large","dog","rests","fast"]),
    ] * 12


def test_reverse_models_share_word_ids_and_normalized_distributions():
    sample=corpus()
    forward=WordLattice(sample,max_vocab=20,smoothing=2)
    backward=WordLattice(backwards(sample),max_vocab=20,smoothing=2)
    assert forward.vocab==backward.vocab
    fw,bw=CachedWalk(forward),CachedWalk(backward)
    line=sample[0][1]
    for i in range(len(line)):
        a=fw.distribution(line[:i],i)
        j=len(line)-1-i
        b=bw.distribution(list(reversed(line))[:j],j)
        assert abs(a.sum()-1)<1e-9
        assert abs(b.sum()-1)<1e-9
        for weight in (0.,.25,.5,.75,1.):
            c=fused(a,b,forward.q,weight)
            assert abs(c.sum()-1)<1e-9
            assert np.all(c>0)


def test_fixed_target_reveals_without_looking_ahead():
    sample=corpus()
    model=WordLattice(sample,max_vocab=20,smoothing=3)
    walk=CachedWalk(model)
    line=sample[0][1]
    for target in range(len(line)):
        for revealed in range(target+1):
            a=walk.distribution(line[:revealed],target)
            b=model.forecast(line[:revealed],target)
            assert np.allclose(a,b)
    with pytest.raises(ValueError):
        walk.distribution(line[:3],1)


def test_all_positions_are_evaluated_and_stratified():
    sample=corpus()
    fm=WordLattice(sample[:20],max_vocab=20,smoothing=3)
    bm=WordLattice(backwards(sample[:20]),max_vocab=20,smoothing=3)
    subset=sample[20:25]
    result=evaluate(fm,bm,subset,max_lines=5,seed=123)
    assert result["evaluated_positions"]==25
    for label in ("forward","backward","both"):
        assert result["metrics"]["all"][label]["n"]==25
    assert result["metrics"]["region:first"]["forward"]["n"]==5
    assert result["metrics"]["region:last"]["backward"]["n"]==5
    assert "0" in result["fixed_target_reveal_curves"]["from_start"]
    assert "0" in result["fixed_target_reveal_curves"]["from_end"]


def test_training_only_fusion_tuning():
    sample=corpus()
    gamma,grid=select_fusion(sample,2,2,max_vocab=20)
    assert gamma in (0.,.25,.5,.75,1.)
    assert len(grid)==5
    assert bins(0,5)=="first"
    assert bins(4,5)=="last"


def test_synthetic_right_context_predicts_center_word():
    # A center word is independent of its preceding marker, but exactly
    # determined by the right-hand marker. Reverse prediction should win.
    synthetic=(
        [("x",["start","alpha","endinga"])]*70 +
        [("x",["start","omega","endingo"])]*70)
    fm=WordLattice(synthetic,max_vocab=20,smoothing=.5)
    bm=WordLattice(backwards(synthetic),max_vocab=20,smoothing=.5)
    f=CachedWalk(fm)
    b=CachedWalk(bm)
    word="alpha"
    forward_prob=f.distribution(["start"],1)[fm.index[word]]
    backward_prob=b.distribution(["endinga"],1)[bm.index[word]]
    assert backward_prob > forward_prob
