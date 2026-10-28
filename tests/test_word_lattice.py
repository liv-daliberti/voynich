import numpy as np
from experiments.latent_grammar.word_lattice import (
    WordLattice, evaluate, aggregate_by_gap, tune_smoothing
)


def corpus():
    return [
        ("block1", ["the","cat","sleeps","at","night","now","soon"]),
        ("block1", ["the","dog","runs","at","noon","now","soon"]),
        ("block2", ["a","cat","sleeps","at","night","now","soon"]),
        ("block2", ["a","dog","runs","at","noon","now","soon"]),
    ]*16


def test_fixed_target_matches_exact_matrix_power():
    m=WordLattice(corpus(),max_vocab=20,smoothing=2.)
    import numpy as np
    I=np.eye(m.V)
    M=np.array([m.forward(I[i]) for i in range(m.V)])
    prior=m.start_probs
    for T in (1,3,5):
        assert np.allclose(m.forecast([],T),prior@np.linalg.matrix_power(M,T))
    for t in (1,2,3):
        prefix=corpus()[0][1][:t]
        p=np.zeros(m.V);p[m.index[prefix[-1]]]=1.
        assert np.allclose(m.forecast(prefix,5),
                           p@np.linalg.matrix_power(M,6-t))


def test_lattice_mass_and_unknown_are_preserved():
    m=WordLattice(corpus(),max_vocab=4)
    for prefix in ([],["the"],["the","some-unseen-word"]):
        prediction=m.forecast(prefix,5)
        assert abs(float(prediction.sum())-1)<1e-9
        assert np.all(prediction>0)
    layers=m.lattice(["the"],4)
    assert len(layers)==5
    assert all(abs(x["total_mass"]-1)<1e-8 for x in layers)


def test_fixed_target_evaluation_is_frozen_per_target():
    m=WordLattice(corpus()[:32],max_vocab=20)
    result=evaluate(m,corpus()[32:],max_targets=12)
    assert result["targets"]==12
    assert result["by_target_and_prefix"]["T3_t0"]["n"]>0
    assert result["by_target_and_prefix"]["T3_t3"]["n"]>0
    assert aggregate_by_gap(result)["1"]["n"]==12


def test_smoothing_selection_never_uses_outer_test():
    x=corpus()
    a, scores=tune_smoothing(x[:40], max_vocab=20)
    b, scores2=tune_smoothing(x[:40], max_vocab=20)
    assert (a,scores)==(b,scores2)
    assert a in (.5,2,8,32,128)
