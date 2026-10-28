import numpy as np
from experiments.latent_grammar.fixed_target import (
    DiscreteHMM, evaluate_fixed_target, copy_edit_lines, summarize_last_step
)


def training():
    return [
        ("g1", ["the", "cat", "sat", "on", "mats"]),
        ("g1", ["the", "dog", "ran", "to", "town"]),
        ("g2", ["a", "cat", "ran", "to", "town"]),
        ("g2", ["a", "dog", "sat", "on", "mats"]),
    ] * 8


def test_all_state_paths_are_marginalized_and_normalized():
    m = DiscreteHMM(states=3, max_vocab=20).fit(training(), iterations=3)
    for target in (3, 4):
        for prefix_size in range(target + 1):
            prefix = training()[0][1][:prefix_size]
            probs, state = m.fixed_target(prefix, target)
            assert abs(sum(probs) - 1) < 1e-9
            assert abs(sum(state) - 1) < 1e-9
            assert np.all(probs > 0)
    lattice = m.lattice(["the"], 4)
    assert abs(lattice["last_state_mass"] - 1) < 1e-9
    for edge in lattice["flows"]:
        assert abs(np.array(edge["edges"]).sum() - 1) < 1e-9


def test_exact_fixed_target_matches_brute_force_enumeration():
    m = DiscreteHMM(states=2, max_vocab=20).fit(training(), iterations=2)
    m.A = np.array([[.7, .3], [.2, .8]])
    m.pi = np.array([.6, .4])
    m.B = np.array([np.roll(np.arange(1, m.V+1), 0),
                    np.roll(np.arange(1, m.V+1), 1)], dtype=float)
    m.B /= m.B.sum(axis=1, keepdims=True)
    p, state = m.fixed_target([], 2)
    brute = np.zeros(m.K)
    for i in range(m.K):
        for j in range(m.K):
            for k in range(m.K):
                brute[k] += m.pi[i] * m.A[i,j] * m.A[j,k]
    assert np.allclose(brute, state)
    assert np.allclose(p, brute @ m.B)


def test_training_only_alignment_preserves_target_emissions():
    model = DiscreteHMM(states=3).fit(training(), iterations=2)
    donor = DiscreteHMM(states=3, seed=8).fit(training(), iterations=2)
    hybrid, mapping = model.transplant_transition(donor)
    assert np.allclose(hybrid.B, model.B)
    assert np.allclose(hybrid.pi, model.pi)
    assert np.allclose(hybrid.A.sum(axis=1), np.ones(3))
    assert len(set(mapping["donor_state_permutation"])) == 3


def test_fixed_target_traces_and_copy_edit_sample():
    m = DiscreteHMM(states=3).fit(training()[:16], iterations=2)
    r = evaluate_fixed_target(m, training()[16:], max_targets=4)
    assert r["n_targets"] == 4
    assert r["by_target_and_prefix"]["T3_seen0"]["n"] >= 1
    assert r["by_target_and_prefix"]["T3_seen3"]["n"] >= 1
    assert summarize_last_step(r)["n"] == 4
    gen = copy_edit_lines("#Header\n#Statistics\npchal shal qokal\n\nchol shol qokey\n")
    assert len(gen) == 2
