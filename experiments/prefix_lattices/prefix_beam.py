#!/usr/bin/env python3
"""Prefix distributions, branching sequence lattices and edit+transition ablations.

No translation or sentencehood claim: Voynich units are transcribed paragraph
LINES. All candidate strings, feature scores and vocabulary are training-only.

Fixed-horizon beams approximate high-scoring *paths*. They are not a probability
distribution over all possible continuations; use the normalized next-token
distribution for entropy and held-out likelihood.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.context.slot_filling import (
    voynich_lines, english_sentences, prefix_budget, split_sequences,
    shuffle_within_group_position, support, orthographic_model,
)

MODELS = ("unigram", "position", "bigram", "trigram", "variable",
          "edit_prior", "edit_context")


def pos_bucket(prefix_length: int) -> str:
    """Only information available at prediction time: NO oracle line length."""
    return "start" if prefix_length == 0 else (
        "early" if prefix_length <= 2 else "late")


class PrefixModel:
    """Interpolated finite-vocabulary word n-grams and graph novelty channel.

    N-gram smoothing is recursive Witten-Bell-like interpolation with a fixed
    pseudo-count, NOT an implementation of modified Kneser-Ney.
    """
    def __init__(self, sequences, eva=False, max_novel=128, novel_mass=.03,
                 unk_mass=.01, alpha=8.0):
        if not 0 <= novel_mass + unk_mass < 1:
            raise ValueError("Novel and unknown mass must total less than one")
        self.eva = eva
        self.alpha = alpha
        self.novel_mass = novel_mass
        self.unk_mass = unk_mass
        self.freq = Counter()
        self.positions = defaultdict(Counter)
        self.ngrams = {k: defaultdict(Counter) for k in (2, 3, 4)}
        self.contexts = {k: Counter() for k in (2, 3, 4)}
        for _, words in sequences:
            history = ["<s>"] * 3
            for i, word in enumerate(words):
                self.freq[word] += 1
                self.positions[pos_bucket(i)][word] += 1
                for k in (2, 3, 4):
                    context = tuple(history[-(k - 1):])
                    self.ngrams[k][context][word] += 1
                    self.contexts[k][context] += 1
                history.append(word)
        if not self.freq:
            raise ValueError("Nonempty training sequences required")
        self.vocab = tuple(sorted(self.freq))
        self.total = sum(self.freq.values())
        self.V = len(self.vocab)
        candidates = support(self.freq, eva) if max_novel else {}
        self.spelling = orthographic_model(self.freq, eva)
        ranked = sorted(candidates, key=lambda w: (
            -(math.log1p(len(candidates[w])) + 2 * self.spelling(w)), w))
        self.novel = tuple(ranked[:max_novel])
        self.neighbors = {w: candidates[w] for w in self.novel}
        self.novel_weights = self._normalized({
            w: math.log1p(len(self.neighbors[w])) + 2 * self.spelling(w)
            for w in self.novel})
        if not self.novel:
            self.novel_mass = 0.0
        self.known_mass = 1.0 - self.novel_mass - self.unk_mass

    @staticmethod
    def _normalized(log_weights):
        if not log_weights:
            return {}
        anchor = max(log_weights.values())
        weights = {k: math.exp(v - anchor) for k, v in log_weights.items()}
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def p_known(self, word, history, order, position=None):
        """Probability within observed training word types (sums to one)."""
        p = (self.freq[word] + .1) / (self.total + .1 * self.V)
        if order == 0:
            return p
        if order == -1:  # prefix-position-only baseline
            pc = self.positions[position]
            return (pc[word] + 20 * p) / (sum(pc.values()) + 20)
        for k in range(2, order + 1):
            padded = (["<s>"] * 3 + list(history))[-(k - 1):]
            context = tuple(padded)
            N = self.contexts[k][context]
            p = (self.ngrams[k][context][word] + self.alpha * p) / (
                N + self.alpha)
        return p

    def distribution(self, history, model="variable"):
        """Complete, normalized probability over fixed candidates + <UNK>."""
        if model not in MODELS:
            raise ValueError(model)
        order = {"unigram": 0, "position": -1, "bigram": 2,
                 "trigram": 3}.get(model, 4)
        position = pos_bucket(len(history))
        known = {w: self.p_known(w, history, order, position)
                 for w in self.vocab}
        dist = {w: p * self.known_mass for w, p in known.items()}
        if self.novel:
            if model == "edit_context":
                # A transfer kernel combining the edit graph with the
                # directed transition model. This is a hypothesis to test.
                # Use a *mean* neighbor conditional so high-degree forms do
                # not get a free bonus just for having more witnesses.
                scores = {}
                for w in self.novel:
                    witnesses = self.neighbors[w]
                    propagated = sum(known[n] for n in witnesses) / len(witnesses)
                    scores[w] = math.log(max(1e-300, propagated)) + math.log(
                        max(1e-300, self.novel_weights[w]))
                novel_weights = self._normalized(scores)
            else:
                novel_weights = self.novel_weights
            for w, p in novel_weights.items():
                dist[w] = p * self.novel_mass
        dist["<UNK>"] = self.unk_mass
        total = sum(dist.values())
        if abs(total - 1) > 1e-8:
            raise AssertionError(f"Not normalized: {total}")
        return dist


def entropy(dist):
    return -sum(p * math.log(p) for p in dist.values() if p)


def top(dist, k=5):
    return sorted(((w, p) for w, p in dist.items() if w != "<UNK>"),
                  key=lambda item: (-item[1], item[0]))[:k]


def diverse_beam(model: PrefixModel, prefix, horizon=3, width=6,
                 groups=3, expand=20, method="edit_context"):
    """Retain several *first-step-diverse* branches at every search depth.

    Fixed horizon means log-score comparisons are length matched. Branches
    represent best sequences, NOT calibrated probability mass estimates.
    """
    if width < 1 or groups < 1 or horizon < 1:
        raise ValueError("Positive beam settings required")
    # Branch roots are distinct next words. Keep approximately equal per root.
    roots = [w for w, _ in top(model.distribution(prefix, method), groups)]
    first = model.distribution(prefix, method)
    beams = [(root, (root,), math.log(first[root])) for root in roots]
    steps = [{"depth": 1, "paths": [
        {"words": list(tokens), "log_probability": round(score, 5)}
        for _, tokens, score in beams]}]
    for depth in range(2, horizon + 1):
        children = defaultdict(list)
        for root, tokens, logp in beams:
            predictions = top(model.distribution(
                list(prefix) + list(tokens), method), expand)
            for candidate, p in predictions:
                children[root].append((root, tokens + (candidate,),
                                       logp + math.log(p)))
        # Preserve multiple distinct roots rather than letting one root
        # consume the whole beam. Fill any remaining slots globally.
        keep = []
        quota = max(1, width // max(1, len(roots)))
        for root in roots:
            keep.extend(sorted(children[root],
                               key=lambda item: (-item[2], item[1]))[:quota])
        existing = {path[1] for path in keep}
        leftovers = sorted((path for all_paths in children.values()
                            for path in all_paths if path[1] not in existing),
                           key=lambda item: (-item[2], item[1]))
        keep.extend(leftovers[:max(0, width - len(keep))])
        beams = sorted(keep[:width], key=lambda item: (-item[2], item[1]))
        steps.append({"depth": depth, "paths": [
            {"words": list(tokens), "log_probability": round(logp, 5)}
            for _, tokens, logp in beams]})
    beams.sort(key=lambda item: (-item[2], item[1]))
    return {"prefix": list(prefix), "horizon": horizon, "beam_width": width,
            "diverse_first_roots": roots, "method": method, "steps": steps,
            "paths": [{"words": list(tokens),
                       "log_probability": round(logp, 5)}
                      for _, tokens, logp in beams],
            "path_probability_lower_bound": sum(
                math.exp(logp) for _, _, logp in beams)}


def score_events(fitted, sequences, max_events=900, seed=20261028):
    events = []
    for group, words in sequences:
        for i, word in enumerate(words):
            events.append((group, tuple(words[:i]), word, i))
    if len(events) > max_events:
        rng = random.Random(seed)
        events = sorted(rng.sample(events, max_events), key=lambda e: (
            e[0], e[3]))
    metrics = {m: defaultdict(float) for m in MODELS}
    traces = []
    for index, (group, prefix, truth, i) in enumerate(events):
        for name in MODELS:
            dist = fitted.distribution(prefix, name)
            p = dist.get(truth, dist["<UNK>"])
            guess = top(dist, 5)
            depth = min(i, 4)
            metric = metrics[name]
            metric["count"] += 1
            metric["nll"] += -math.log(max(p, 1e-300))
            metric["top1"] += int(bool(guess) and guess[0][0] == truth)
            metric["top5"] += int(any(w == truth for w, _ in guess))
            H = entropy(dist)
            metric["entropy"] += H
            metric[f"depth_{depth}_entropy"] += H
            metric["effective_candidates"] += math.exp(H)
            metric["top5_mass"] += sum(p for _, p in guess)
            metric[f"depth_{depth}_nll"] += -math.log(max(p, 1e-300))
            metric[f"depth_{depth}_count"] += 1
        if index < 10:
            dist = fitted.distribution(prefix, "variable")
            traces.append({"group": group, "prefix": list(prefix[-8:]),
                           "prefix_length": i, "truth": truth,
                           "seen_in_training": truth in fitted.freq,
                           "is_novel_candidate": truth in fitted.novel,
                           "entropy_nats": round(entropy(dist), 5),
                           "top5": [{"word": w, "probability": round(p, 5)}
                                    for w, p in top(dist)]})
    out = {}
    for name, sums in metrics.items():
        n = sums["count"]
        out[name] = {
            "evaluated_tokens": int(n),
            "mean_nll_nats": round(sums["nll"] / n, 5) if n else None,
            "top1": round(sums["top1"] / n, 5) if n else None,
            "top5": round(sums["top5"] / n, 5) if n else None,
            "mean_entropy_nats": round(sums["entropy"] / n, 5) if n else None,
            "mean_effective_candidate_count": round(sums["effective_candidates"] / n, 5) if n else None,
            "mean_entropy_by_prefix_depth": {
                str(k): round(sums[f"depth_{k}_entropy"] /
                              sums[f"depth_{k}_count"], 5)
                if sums[f"depth_{k}_count"] else None for k in range(5)},
            "mean_top5_mass": round(sums["top5_mass"] / n, 5) if n else None,
            "nll_by_prefix_depth": {
                str(k): round(sums[f"depth_{k}_nll"] /
                              sums[f"depth_{k}_count"], 5)
                if sums[f"depth_{k}_count"] else None
                for k in range(5)}}
    return out, traces, {
        "true_seen_rate": round(sum(t in fitted.freq for _, _, t, _ in events) /
                                len(events), 5) if events else None,
        "true_selected_novel_rate": round(sum(t in fitted.novel for _, _, t, _ in
                                              events) / len(events), 5)
                                    if events else None,
        "novel_candidates": len(fitted.novel)}


def run_corpus(sequences, eva=False, beam_examples=4,
               max_events=900, max_novel=128):
    train, heldout = split_sequences(sequences, ratio=.7)
    full = PrefixModel(train, eva=eva, max_novel=max_novel)
    shuffled = PrefixModel(shuffle_within_group_position(
        train, 20261028), eva=eva, max_novel=max_novel)
    scores, traces, coverage = score_events(full, heldout, max_events)
    shuffled_scores, _, _ = score_events(shuffled, heldout, max_events)
    examples = []
    for group, sentence in heldout:
        if len(sentence) < 5:
            continue
        prefix = sentence[:2]
        actual = sentence[2:5]
        result = diverse_beam(full, prefix, horizon=3)
        result.update({"group": group, "actual_continuation": actual,
                       "exact_path_in_beam": actual in [
                           p["words"] for p in result["paths"]]})
        examples.append(result)
        if len(examples) >= beam_examples:
            break
    return {
        "train_tokens": sum(len(s) for _, s in train),
        "test_tokens": sum(len(s) for _, s in heldout),
        "training_types": len(full.vocab),
        "candidate_universe_size": len(full.vocab) + len(full.novel) + 1,
        "novel_mass": full.novel_mass,
        "unknown_mass": full.unk_mass,
        "coverage": coverage,
        "heldout_metrics": scores,
        "shuffled_training_metrics": shuffled_scores,
        "prefix_traces": traces,
        "beam_examples": examples}


def run(zl_file, austen_file, darwin_file, out,
        budgets=(11063, 22843), max_events=900, max_novel=128):
    text_z = Path(zl_file).read_text(encoding="utf-8-sig")
    corpora = [(f"Voynich_{lang}", voynich_lines(text_z, lang), True)
               for lang in "AB"]
    for name, file in (("austen", austen_file), ("darwin", darwin_file)):
        sentences = english_sentences(Path(file).read_text(encoding="utf-8-sig"),
                                      name)
        for budget in budgets:
            corpora.append((f"{name}_{budget}",
                            prefix_budget(sentences, budget), False))
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    overview = {}
    for name, sequences, eva in corpora:
        data = run_corpus(sequences, eva, max_events=max_events,
                          max_novel=max_novel)
        (path / f"{name}_paths.json").write_text(
            json.dumps({"beam_examples": data.pop("beam_examples"),
                        "prefix_traces": data.pop("prefix_traces")}, indent=2)
            + "\n")
        overview[name] = data
        print(name, "variable NLL", data["heldout_metrics"]["variable"]["mean_nll_nats"],
              "vs shuffled", data["shuffled_training_metrics"]["variable"]["mean_nll_nats"],
              flush=True)
    (path / "summary.json").write_text(json.dumps(overview, indent=2) + "\n")
    return overview


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--zl", required=True)
    p.add_argument("--austen", required=True)
    p.add_argument("--darwin", required=True)
    p.add_argument("--out", default="results/prefix_lattices")
    p.add_argument("--budgets", type=int, nargs="+", default=[11063, 22843])
    p.add_argument("--max-events", type=int, default=900)
    p.add_argument("--max-novel", type=int, default=128)
    a = p.parse_args()
    run(a.zl, a.austen, a.darwin, a.out, a.budgets, a.max_events, a.max_novel)
