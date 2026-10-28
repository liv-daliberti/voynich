#!/usr/bin/env python3
"""Prospective Voynich word-form prediction, with a degree-matched control.

Evaluation is later *attestation*, never linguistic authenticity. Train-only
features are fit on a folio prefix. Inner fitting uses an earlier prefix.

Run from the repository root:
  python experiments/phonotactics/evaluate.py --zl data/ZL3b-n.txt \
      --it data/IT2a-n.txt --out results/phonotactics
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.robustness75.robustness import glyphs, one_edit_forms, parse


def glyph_bigram_model(counts: Counter[str], alpha: float = 0.5):
    """Token-weighted training-only bigram model over segmented EVA glyphs."""
    transitions = Counter()
    contexts = Counter()
    alphabet = {g for w in counts for g in glyphs(w)}
    for word, count in counts.items():
        seq = ('^',) + glyphs(word) + ('$',)
        for a, b in zip(seq, seq[1:]):
            transitions[a, b] += count
            contexts[a] += count
    width = len(alphabet) + 1  # end-of-word token

    def score(word: str) -> float:
        seq = ('^',) + glyphs(word) + ('$',)
        return sum(math.log((transitions[a, b] + alpha) /
                            (contexts[a] + width * alpha))
                   for a, b in zip(seq, seq[1:])) / (len(seq) - 1)
    return score


def make_rows(training_pages: list[dict], future_pages: list[dict],
              min_degree: int = 2, independently_attested: set[str] | None = None):
    """No future word, character or frequency contributes to any feature."""
    counts = Counter(w for page in training_pages for w in page['words'])
    future = {w for page in future_pages for w in page['words']} - counts.keys()
    alphabet = sorted({g for word in counts for g in glyphs(word)})
    degree = Counter()
    witness_strength = Counter()
    for w, n in counts.items():
        for proposal in one_edit_forms(glyphs(w), alphabet):
            degree[proposal] += 1
            witness_strength[proposal] += math.log1p(n)
    bigram = glyph_bigram_model(counts)
    rows = []
    for word, d in degree.items():
        if d < min_degree or word in counts:
            continue
        rows.append({'word': word, 'attested_later': int(word in future),
                     'degree': d, 'glyph_length': len(glyphs(word)),
                     'log_degree': math.log1p(d),
                     'bigram': bigram(word),
                     'mean_log_witness_frequency': witness_strength[word] / d,
                     'it_attested': int(word in independently_attested)
                     if independently_attested is not None else -1})
    rows.sort(key=lambda r: r['word'])
    return rows, {'training_types': len(counts),
                  'new_future_types': len(future),
                  'candidate_types': len(rows),
                  'future_types_reachable': sum(word in degree for word in future),
                  'candidate_positive_types': sum(r['attested_later'] for r in rows),
                  'alphabet_size': len(alphabet)}


def metrics(y, prediction):
    y = np.asarray(y, dtype=int)
    prediction = np.asarray(prediction, dtype=float)
    if len(set(y.tolist())) != 2:
        return {'roc_auc': None, 'average_precision': None}
    return {'roc_auc': round(float(roc_auc_score(y, prediction)), 6),
            'average_precision': round(float(average_precision_score(y, prediction)), 6)}


def degree_length_matched_auc(rows, score='bigram'):
    """Pair-weighted within-stratum concordance; ties score 0.5."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['degree'], row['glyph_length']].append(row)
    numer = 0.0
    denom = 0
    mixed = 0
    for group in groups.values():
        y = [r['attested_later'] for r in group]
        pos = sum(y)
        neg = len(y) - pos
        if not pos or not neg:
            continue
        pairs = pos * neg
        numer += roc_auc_score(y, [r[score] for r in group]) * pairs
        denom += pairs
        mixed += 1
    return {'pair_weighted_auc': round(float(numer / denom), 6) if denom else None,
            'positive_negative_pairs': denom, 'mixed_strata': mixed}


def fit_predict(inner, outer, columns):
    X_train = np.array([[r[c] for c in columns] for r in inner], dtype=float)
    y_train = np.array([r['attested_later'] for r in inner], dtype=int)
    if len(set(y_train.tolist())) < 2:
        raise ValueError('Inner evaluation needs both observed and unobserved forms')
    X_test = np.array([[r[c] for c in columns] for r in outer], dtype=float)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def run_language(zl_pages, it_pages, language, inner_train=.50,
                 outer_train=.70, min_degree=2):
    pages = [p for p in zl_pages if p['language'] == language]
    n = len(pages)
    early, cut = int(n * inner_train), int(n * outer_train)
    if not (0 < early < cut < n):
        raise ValueError('Need nonempty early training, inner test, and outer test')
    it = {w for p in it_pages if p['language'] == language for w in p['words']}
    inner, isummary = make_rows(pages[:early], pages[early:cut],
                                min_degree=min_degree)
    outer, osummary = make_rows(pages[:cut], pages[cut:],
                                min_degree=min_degree, independently_attested=it)
    y = [r['attested_later'] for r in outer]
    evals = {
        'raw_degree': metrics(y, [r['degree'] for r in outer]),
        'raw_bigram': metrics(y, [r['bigram'] for r in outer]),
        'raw_mean_witness_frequency': metrics(
            y, [r['mean_log_witness_frequency'] for r in outer]),
    }
    for label, columns in [
        ('trained_degree', ['log_degree']),
        ('trained_degree_length', ['log_degree', 'glyph_length']),
        ('trained_degree_length_bigram', ['log_degree', 'glyph_length', 'bigram']),
    ]:
        predicted = fit_predict(inner, outer, columns)
        evals[label] = metrics(y, predicted)
        # Store model predictions in CSV without using held-out labels for fitting.
        for row, score in zip(outer, predicted):
            row[label] = float(score)
    return {'language': language, 'n_folios': n,
            'inner_train_folios': early, 'outer_train_folios': cut,
            'inner': isummary, 'outer': osummary,
            'unattested_in_zl_but_in_it': sum(
                r['it_attested'] == 1 and r['attested_later'] == 0 for r in outer),
            'matched': degree_length_matched_auc(outer),
            'model_metrics': evals}, outer


def run(zl, it, out):
    zl_pages = parse(Path(zl).read_text(encoding='utf-8-sig'))
    it_pages = parse(Path(it).read_text(encoding='utf-8-sig'))
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for lang in 'AB':
        summary, rows = run_language(zl_pages, it_pages, lang)
        summaries.append(summary)
        with (output / f'candidates_{lang}.csv').open('w', newline='') as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(summary, indent=2))
    (output / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zl', required=True)
    parser.add_argument('--it', required=True)
    parser.add_argument('--out', default='results/phonotactics')
    a = parser.parse_args()
    run(a.zl, a.it, a.out)
