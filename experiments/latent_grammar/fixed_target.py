#!/usr/bin/env python3
"""Exact probabilistic state lattice, fixed-future-target prediction and structural transfer.

A discrete latent-class HMM is learned separately on each training corpus.
Forward filtering marginalizes *all* intermediate hidden state paths: this is
not beam search, and retained path probability equals 1 (up to roundoff).
No latent state or transferred link is a deciphered translation.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.context.slot_filling import (
    voynich_lines, english_sentences, prefix_budget, split_sequences,
    shuffle_within_group_position,
)


def copy_edit_lines(text):
    """Published Timm/Schinner generator's sample output, NOT our replica."""
    lines = []
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words = re.findall(r"[a-z]+", line.lower())
        if len(words) >= 3:
            lines.append((f"generated_block{i//30}", words))
    return lines


class DiscreteHMM:
    def __init__(self, states=6, max_vocab=450, seed=20261028):
        self.K, self.max_vocab, self.seed = states, max_vocab, seed

    def _ids(self, words):
        return np.fromiter((self.to_id.get(w, 0) for w in words),
                           dtype=np.int32, count=len(words))

    def fit(self, sequences, iterations=5, max_train_tokens=15500):
        freq = Counter(w for _, words in sequences for w in words)
        vocab = ["<UNK>"] + [w for w, _ in freq.most_common(self.max_vocab)]
        self.vocab = vocab
        self.to_id = {w: i for i, w in enumerate(vocab)}
        self.V = len(vocab)
        samples, n = [], 0
        for _, words in sequences:
            if n + len(words) > max_train_tokens:
                break
            obs = self._ids(words)
            if len(obs) >= 2:
                samples.append(obs)
                n += len(obs)
        if not samples:
            raise ValueError("No training samples")
        self.train_tokens = n
        self.train_sequences = len(samples)
        count = np.zeros(self.V)
        for s in samples:
            count += np.bincount(s, minlength=self.V)
        base = (count + .25) / (count.sum() + .25 * self.V)
        rng = np.random.default_rng(self.seed)
        self.pi = rng.dirichlet(np.ones(self.K))
        self.A = rng.dirichlet(np.ones(self.K) * 3, size=self.K)
        self.B = np.array([base * rng.lognormal(0, 1.0, self.V)
                           for _ in range(self.K)])
        self.B /= self.B.sum(axis=1, keepdims=True)
        history = []
        for _ in range(iterations):
            pi_count = np.full(self.K, .05)
            A_count = np.full((self.K, self.K), .3)
            B_count = np.tile(base[None, :], (self.K, 1)) * 2
            loglike = 0.
            for obs in samples:
                alpha, beta, scales, ll = self.forward_backward(obs)
                loglike += ll
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True)
                pi_count += gamma[0]
                nxt = self.B[:, obs[1:]].T * beta[1:]
                xi = (alpha[:-1, :, None] * self.A[None, :, :] *
                      nxt[:, None, :] / scales[1:, None, None])
                xi /= xi.sum(axis=(1, 2), keepdims=True)
                A_count += xi.sum(axis=0)
                for state in range(self.K):
                    np.add.at(B_count[state], obs, gamma[:, state])
            self.pi = pi_count / pi_count.sum()
            self.A = A_count / A_count.sum(axis=1, keepdims=True)
            self.B = B_count / B_count.sum(axis=1, keepdims=True)
            history.append(float(loglike / n))
        self.training_nll_trace = [-round(v, 5) for v in history]
        return self

    def forward_backward(self, obs):
        L = len(obs)
        e = self.B[:, obs].T
        alpha = np.zeros((L, self.K))
        scales = np.zeros(L)
        prior = self.pi.copy()
        for t in range(L):
            if t:
                prior = alpha[t - 1] @ self.A
            u = prior * e[t]
            scales[t] = max(u.sum(), 1e-300)
            alpha[t] = u / scales[t]
        beta = np.ones((L, self.K))
        for t in range(L - 2, -1, -1):
            beta[t] = self.A @ (e[t + 1] * beta[t + 1])
            beta[t] /= scales[t + 1]
        return alpha, beta, scales, float(np.log(scales).sum())

    def prefix_posterior(self, words):
        alpha = None
        for t, word in enumerate(words):
            prior = self.pi if t == 0 else alpha @ self.A
            weighted = prior * self.B[:, self.to_id.get(word, 0)]
            alpha = weighted / max(weighted.sum(), 1e-300)
        return alpha

    def fixed_target(self, prefix, target_index):
        """target_index zero-based; all prefix words occur strictly before T."""
        if len(prefix) > target_index:
            raise ValueError("Target must be unrevealed")
        if not prefix:
            state = self.pi.copy()
            steps = target_index
        else:
            state = self.prefix_posterior(prefix)
            steps = target_index - len(prefix) + 1
        for _ in range(steps):
            state = state @ self.A
        probs = state @ self.B
        assert abs(probs.sum() - 1) < 1e-8
        return probs, state

    def lattice(self, prefix, target_index, max_layers=5):
        """Exact state-flow DAG from prefix to fixed target, not pruned beams."""
        if not prefix:
            state = self.pi.copy()
            remaining = target_index
            first_position = 0
        else:
            state = self.prefix_posterior(prefix)
            remaining = target_index - len(prefix) + 1
            first_position = len(prefix) - 1
        if remaining > max_layers:
            raise ValueError("Example lattice exceeds display depth")
        layers = [{"position": first_position, "state_prob": state.tolist()}]
        flows = []
        for j in range(remaining):
            edge_flow = state[:, None] * self.A
            state = edge_flow.sum(axis=0)
            flows.append({"position": first_position + j,
                          "edges": edge_flow.tolist()})
            layers.append({"position": first_position + j + 1,
                           "state_prob": state.tolist()})
        final_words = layers[-1]["state_prob"]
        preds = np.asarray(final_words) @ self.B
        best = np.argsort(-preds)[:8]
        return {"layers": layers, "flows": flows,
                "last_state_mass": float(sum(final_words)),
                "top_target_words": [{"word": self.vocab[i],
                                      "probability": round(float(preds[i]), 6)}
                                     for i in best]}

    def state_features(self):
        """Permutation-equivariant descriptors, no lexical cross-corpus anchor."""
        stationary = self.pi.copy()
        for _ in range(200):
            stationary = stationary @ self.A
        ent = -(self.B * np.log(np.clip(self.B, 1e-300, None))).sum(1)
        ent /= math.log(max(self.V, 2))
        return np.stack([self.pi, stationary, np.diag(self.A), ent,
                         -(self.A * np.log(self.A)).sum(1) /
                         math.log(max(self.K, 2))], axis=1)

    def transplant_transition(self, donor):
        """Training-only Hungarian feature matching. English word meanings do
        NOT determine state mapping, and Voynich emissions remain Voynich."""
        if donor.K != self.K:
            raise ValueError("Same number of states required")
        a, b = self.state_features(), donor.state_features()
        scale = np.maximum(np.std(np.vstack([a, b]), axis=0), .15)
        costs = (((a[:, None, :] - b[None, :, :]) / scale) ** 2).sum(2)
        r, col = linear_sum_assignment(costs)
        perm = col[np.argsort(r)]
        new = DiscreteHMM(self.K, self.max_vocab, self.seed)
        new.vocab, new.to_id, new.V = self.vocab, self.to_id, self.V
        new.pi, new.B, new.A = self.pi.copy(), self.B.copy(), donor.A[np.ix_(perm, perm)].copy()
        return new, {"mean_cost": round(float(costs[r, col].mean()), 5),
                     "donor_state_permutation": perm.tolist()}


def evaluate_fixed_target(model, sequences, max_targets=130, seed=20261028):
    tasks = []
    for group, line in sequences:
        for T in (3, 5, 7, 9):
            if T < len(line):
                tasks.append((group, line, T))
    rng = random.Random(seed)
    rng.shuffle(tasks)
    tasks = tasks[:max_targets]
    totals = defaultdict(lambda: {"n": 0, "nll": 0., "entropy": 0.,
                                  "seen_n": 0, "seen_nll": 0.,
                                  "information_gain": 0., "brier": 0.})
    examples = []
    for group, line, T in tasks:
        true_id = model.to_id.get(line[T], 0)
        base_prob = None
        trace = []
        for t in range(T + 1):
            if t > T:
                break
            # t strictly before zero-indexed target position T.
            probs, state = model.fixed_target(line[:t], T)
            p = float(probs[true_id])
            h = float(-np.sum(probs * np.log(np.clip(probs, 1e-300, None))))
            key = (T, t, T - t)
            record = totals[key]
            record["n"] += 1
            record["nll"] += -math.log(max(p, 1e-300))
            record["entropy"] += h
            record["brier"] += float(np.square(probs).sum() - 2 * p + 1)
            if true_id:
                record["seen_n"] += 1
                record["seen_nll"] += -math.log(max(p, 1e-300))
            if base_prob is None:
                base_prob = p
            record["information_gain"] += math.log(max(p, 1e-300)) - math.log(
                max(base_prob, 1e-300))
            if len(examples) < 4:
                top_ids = np.argsort(-probs)[:5]
                trace.append({"revealed_prefix": t, "gap": T - t,
                              "true_word_prob": round(p, 6),
                              "entropy_nats": round(h, 5),
                              "state_probs_at_target": [round(float(x), 5) for x in state],
                              "predicted": [[model.vocab[i], round(float(probs[i]), 6)]
                                            for i in top_ids]})
        if len(examples) < 4:
            examples.append({"group": group, "target_index": T,
                             "target": line[T], "is_seen": bool(true_id),
                             "prefix_and_target": line[:T + 1],
                             "trace": trace,
                             "exact_state_lattice": model.lattice(
                                 line[:max(0, T - 3)], T)})
    # Explicitly group by fixed target index T and revealed prefix count t.
    reported = {}
    for (T, t, gap), v in sorted(totals.items()):
        reported[f"T{T}_seen{t}"] = {"T": T, "revealed": t,
            "remaining_gap": gap, "n": v["n"],
            "mean_nll": round(v["nll"] / v["n"], 5),
            "mean_entropy": round(v["entropy"] / v["n"], 5),
            "mean_information_gain_vs_no_prefix": round(
                v["information_gain"] / v["n"], 5),
            "mean_brier": round(v["brier"] / v["n"], 5),
            "seen_token_count": v["seen_n"],
            "seen_only_nll": round(v["seen_nll"] / v["seen_n"], 5)
            if v["seen_n"] else None}
    return {"n_targets": len(tasks), "by_target_and_prefix": reported,
            "examples": examples}


def summarize_last_step(result):
    recs = [v for v in result["by_target_and_prefix"].values()
            if v["remaining_gap"] == 1]
    if not recs:
        return None
    n = sum(x["n"] for x in recs)
    return {"n": n, "nll": round(sum(r["mean_nll"] * r["n"] for r in recs) / n, 5),
            "entropy": round(sum(r["mean_entropy"] * r["n"] for r in recs) / n, 5),
            "information_gain": round(sum(
                r["mean_information_gain_vs_no_prefix"] * r["n"]
                for r in recs) / n, 5)}


def run(zl, austen, darwin, generated, output, states=6, em_iterations=5,
        max_targets=130):
    raw_zl = Path(zl).read_text(encoding="utf-8-sig")
    corpora = {"Voynich_A": voynich_lines(raw_zl, "A"),
               "Voynich_B": voynich_lines(raw_zl, "B")}
    for name, path in (("austen", austen), ("darwin", darwin)):
        sentences = english_sentences(Path(path).read_text(encoding="utf-8-sig"), name)
        corpora[name] = prefix_budget(sentences, 22843)
    if generated:
        corpora["TimmSchinner_sample"] = copy_edit_lines(
            Path(generated).read_text(encoding="utf-8-sig"))
    fitted = {}
    summary = {}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, lines in corpora.items():
        train, test = split_sequences(lines)
        model = DiscreteHMM(states=states, seed=20261028).fit(
            train, iterations=em_iterations)
        fitted[name] = model
        real = evaluate_fixed_target(model, test, max_targets)
        shuffled = DiscreteHMM(states=states, seed=20261028).fit(
            shuffle_within_group_position(train, 20261028), iterations=em_iterations)
        null = evaluate_fixed_target(shuffled, test, max_targets)
        info = {"training_tokens": model.train_tokens,
                "training_sequences": model.train_sequences,
                "training_negative_loglik_trace": model.training_nll_trace,
                "vocabulary_size_with_unknown": model.V,
                "target_count": real["n_targets"],
                "observed": summarize_last_step(real),
                "position_shuffled": summarize_last_step(null),
                "latent_transition": model.A.round(5).tolist(),
                "state_features": model.state_features().round(5).tolist(),
                "states_top_words": [[model.vocab[i] for i in
                                     np.argsort(-model.B[k])[:8]]
                                    for k in range(states)]}
        summary[name] = info
        (output / f"{name}_traces.json").write_text(json.dumps(
            {"real": real, "shuffled": null}, indent=2) + "\n")
        print(name, json.dumps({"observed": info["observed"],
                                "shuffled": info["position_shuffled"]}), flush=True)
    # Does learning English dynamics help predict *Voynich* heldout forms?
    for target in ("Voynich_A", "Voynich_B"):
        train, test = split_sequences(corpora[target])
        transplanted = {}
        for donor_name in ("austen", "darwin", "TimmSchinner_sample"):
            if donor_name not in fitted:
                continue
            transferred, match = fitted[target].transplant_transition(
                fitted[donor_name])
            score = evaluate_fixed_target(transferred, test, max_targets)
            transplanted[donor_name] = {"alignment": match,
                                        "score": summarize_last_step(score)}
        summary[target]["structural_transfer"] = transplanted
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--zl", required=True)
    p.add_argument("--austen", required=True)
    p.add_argument("--darwin", required=True)
    p.add_argument("--generated", default=None)
    p.add_argument("--out", default="results/latent_grammar_lattice")
    p.add_argument("--states", type=int, default=6)
    p.add_argument("--em-iterations", type=int, default=5)
    p.add_argument("--max-targets", type=int, default=130)
    args = p.parse_args()
    run(args.zl, args.austen, args.darwin, args.generated, args.out,
        states=args.states, em_iterations=args.em_iterations,
        max_targets=args.max_targets)
