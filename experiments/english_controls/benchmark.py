#!/usr/bin/env python3
"""Austen/Darwin positive controls for Voynich unseen-form stability.

Requires locally acquired public-domain Gutenberg files (not bundled).
For English, edits are *characters*, not compound EVA glyphs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.robustness75.robustness import glyphs, one_edit_forms
from experiments.phonotactics.evaluate import (
    degree_length_matched_auc, fit_predict, metrics,
)


def clean_gutenberg(text: str, corpus: str) -> list[str]:
    """Strip front/back matter using document-specific starts, then tokenize.
    The normalization splits contractions and hyphenated compounds.
    """
    txt = text.replace('\r\n', '\n').lstrip('\ufeff')
    end = re.search(r'^\*{3}\s*END OF (?:THIS|THE) PROJECT GUTENBERG', txt, re.M | re.I)
    if end:
        txt = txt[:end.start()]
    if corpus == 'naibbe':
        return re.findall(r'[a-z]+', txt.lower())
    if corpus == 'austen':
        start = re.search(r'^Chapter\s+1\s*$', txt, re.M | re.I)
    elif corpus == 'darwin':
        # The first INTRODUCTION in the TOC is not the main text.
        matches = list(re.finditer(r'^INTRODUCTION\.\s*$', txt, re.M | re.I))
        start = matches[-1] if matches else None
    else:
        raise ValueError('corpus must be austen or darwin')
    if not start:
        raise ValueError('Could not locate text start; verify Gutenberg edition')
    return re.findall(r'[a-z]+', txt[start.end():].lower())


def character_edits(word: str, alphabet: tuple[str, ...]):
    """All unique one-character insertions, deletions and substitutions."""
    ans = set()
    for i in range(len(word)):
        ans.add(word[:i] + word[i + 1:])
        for char in alphabet:
            if char != word[i]:
                ans.add(word[:i] + char + word[i + 1:])
    for i in range(len(word) + 1):
        for char in alphabet:
            ans.add(word[:i] + char + word[i:])
    ans.discard('')
    ans.discard(word)
    return ans


def english_features(train: list[str], future: list[str], min_degree=2,
                     lexicon: set[str] | None = None, eva=False):
    freq = Counter(train)
    future_new = set(future) - freq.keys()
    units = glyphs if eva else tuple
    alphabet = tuple(sorted({c for w in freq for c in units(w)}))
    degree = Counter()
    strengths = Counter()
    neighbors = {}
    forms = sorted(freq)
    for wid, word in enumerate(forms):
        edits = (one_edit_forms(glyphs(word), list(alphabet)) if eva else
                 character_edits(word, alphabet))
        for candidate in edits:
            degree[candidate] += 1
            strengths[candidate] += math.log1p(freq[word])
            neighbors.setdefault(candidate, []).append(wid)

    transitions = Counter()
    contexts = Counter()
    for word, count in freq.items():
        seq = ('^',) + units(word) + ('
            transitions[a, b] += count
            contexts[a] += count
    vocab_size = len(alphabet) + 1

    def letter_bigram(word):
        seq = ('^',) + units(word) + ('

    rows = []
    for word, deg in degree.items():
        if deg < min_degree or word in freq:
            continue
        rows.append({'word': word, 'attested_later': int(word in future_new),
                     'degree': deg, 'glyph_length': len(units(word)),
                     'log_degree': math.log1p(deg),
                     'bigram': letter_bigram(word),
                     'mean_log_witness_frequency': strengths[word] / deg,
                     'lexicon_attested': int(word in lexicon) if lexicon is not None else -1})
    rows.sort(key=lambda r: r['word'])
    summary = {'train_tokens': len(train), 'train_types': len(freq),
               'future_tokens': len(future), 'new_future_types': len(future_new),
               'reachable_future_types': sum(w in degree for w in future_new),
               'candidate_types': len(rows),
               'candidate_positives': sum(r['attested_later'] for r in rows),
               'alphabet': list(alphabet)}
    return rows, summary, forms, neighbors


def evaluate_sample(tokens: list[str], lexicon: set[str] | None = None,
                    min_degree=2, mask_trials=500, seed=20261028,
                    top_k=500, eva=False):
    if len(tokens) < 100:
        raise ValueError('Control text too short')
    early = int(.5 * len(tokens))
    cut = int(.7 * len(tokens))
    inner, inner_summary, _, _ = english_features(
        tokens[:early], tokens[early:cut], min_degree, lexicon, eva=eva)
    outer, outer_summary, forms, neighbors = english_features(
        tokens[:cut], tokens[cut:], min_degree, lexicon, eva=eva)
    y = [r['attested_later'] for r in outer]
    results = {
        'raw_degree': metrics(y, [r['degree'] for r in outer]),
        'raw_bigram': metrics(y, [r['bigram'] for r in outer]),
        'matched_bigram': degree_length_matched_auc(outer),
    }
    if y and len(set(y)) == 2 and len({r['attested_later'] for r in inner}) == 2:
        for label, predictors in [
            ('degree_length', ['log_degree', 'glyph_length']),
            ('degree_length_bigram', ['log_degree', 'glyph_length', 'bigram']),
        ]:
            pred = fit_predict(inner, outer, predictors)
            for row, score in zip(outer, pred):
                row[label] = float(score)
            results[label] = metrics(y, pred)

    # Rank all mask-absent types, including deleted training words.
    rng = np.random.default_rng(seed)
    training_counts = Counter(tokens[:cut])
    form_ids = {w: i for i, w in enumerate(forms)}
    future_new = set(tokens[cut:]) - training_counts.keys()
    mask_vocab = sorted(w for w, witnesses in neighbors.items()
                        if len(witnesses) >= min_degree)
    ptr = np.zeros(len(mask_vocab) + 1, dtype=np.int64)
    flat_parts = []
    for i, word in enumerate(mask_vocab):
        ptr[i + 1] = ptr[i] + len(neighbors[word])
        flat_parts.extend(neighbors[word])
    flat = np.asarray(flat_parts, dtype=np.int32)
    self_ids = np.array([form_ids.get(w, -1) for w in mask_vocab], dtype=np.int32)
    kinds = np.array([0 if w in training_counts else 1 if w in future_new else 2
                      for w in mask_vocab], dtype=np.int8)
    top_hits = np.zeros(len(mask_vocab), dtype=np.int32)
    reach_hits = np.zeros(len(mask_vocab), dtype=np.int32)
    witness_sum = np.zeros(len(mask_vocab), dtype=np.float64)
    composition = []
    num_kept = int(round(len(forms) * .25))
    for trial in range(mask_trials):
        visible = np.zeros(len(forms), dtype=np.int8)
        visible[rng.choice(len(forms), size=num_kept, replace=False)] = 1
        support = np.add.reduceat(visible[flat], ptr[:-1])
        present = (self_ids >= 0) & (visible[self_ids.clip(min=0)] != 0)
        eligible = ~present
        reach_hits += ((support > 0) & eligible).astype(np.int32)
        witness_sum += support * eligible
        possible = np.flatnonzero((support > 0) & eligible)
        k = min(top_k, len(possible))
        if not k:
            composition.append({'trial': trial, 'deleted_training': 0,
                                'later_attested': 0, 'unattested': 0})
            continue
        ranking = support[possible] + rng.random(len(possible)) * .49
        chosen = possible[np.argpartition(ranking, -k)[-k:]]
        top_hits[chosen] += 1
        parts = np.bincount(kinds[chosen], minlength=3)
        composition.append({'trial': trial, 'deleted_training': int(parts[0]),
                            'later_attested': int(parts[1]),
                            'unattested': int(parts[2])})
    index = {w: i for i, w in enumerate(mask_vocab)}
    for row in outer:
        i = index[row['word']]
        row['top500_pct'] = round(100 * top_hits[i] / max(mask_trials, 1), 4)
        row['reachable_pct'] = round(100 * reach_hits[i] / max(mask_trials, 1), 4)
        row['mean_visible_witnesses'] = round(witness_sum[i] /
                                              max(mask_trials, 1), 4)
    results['deletion_masks'] = {
        'trials': mask_trials, 'survive_fraction': .25, 'top_k': top_k,
        'mean_deleted_training_topk': round(sum(c['deleted_training']
            for c in composition) / max(mask_trials, 1), 4),
        'mean_later_attested_topk': round(sum(c['later_attested']
            for c in composition) / max(mask_trials, 1), 4),
        'mean_unattested_topk': round(sum(c['unattested']
            for c in composition) / max(mask_trials, 1), 4)}
    if lexicon is not None:
        results['dictionary_coverage'] = {
            'lexicon_positive_candidates': sum(r['lexicon_attested'] for r in outer),
            'degree_roc_auc': metrics([r['lexicon_attested'] for r in outer],
                                      [r['degree'] for r in outer]),
            'bigram_roc_auc': metrics([r['lexicon_attested'] for r in outer],
                                      [r['bigram'] for r in outer]),
        }
    return {'inner': inner_summary, 'outer': outer_summary,
            'results': results}, outer, composition


def run(corpus_files, out, token_budgets=(11063, 22843),
        trials=500, lexicon_file=None):
    lexicon = None
    if lexicon_file:
        lexicon = {w.lower() for line in Path(lexicon_file).read_text().splitlines()
                   if (w := line.strip()).isascii() and w.isalpha()}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    reports = {}
    for label, path in corpus_files.items():
        eva = label == 'naibbe'
        tokens = clean_gutenberg(Path(path).read_text(encoding='utf-8-sig'), label)
        reports[label] = {}
        for budget in token_budgets:
            if len(tokens) < budget:
                raise ValueError(f'{label} has only {len(tokens)} words, not {budget}')
            summary, rows, masks = evaluate_sample(tokens[:budget], lexicon if not eva else None,
                                                    mask_trials=trials, eva=eva)
            reports[label][str(budget)] = summary
            prefix = f'{label}_{budget}'
            with (out / f'{prefix}_candidates.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with (out / f'{prefix}_mask_composition.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(masks[0]))
                writer.writeheader()
                writer.writerows(masks)
    (out / 'summary.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports, indent=2))
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--austen', required=True)
    parser.add_argument('--darwin', required=True)
    parser.add_argument('--naibbe', help='Optional EVA ciphertext control')
    parser.add_argument('--out', default='results/english_controls')
    parser.add_argument('--tokens', type=int, nargs='+', default=[11063, 22843])
    parser.add_argument('--trials', type=int, default=500)
    parser.add_argument('--lexicon', help='Optional independent wordlist, one word per line')
    a = parser.parse_args()
    files = {'austen': a.austen, 'darwin': a.darwin}
    if a.naibbe:
        files['naibbe'] = a.naibbe
    run(files, a.out, a.tokens, a.trials, a.lexicon)
,)
        for a, b in zip(seq, seq[1:]):
            transitions[a, b] += count
            contexts[a] += count
    vocab_size = len(alphabet) + 1

    def letter_bigram(word):
        seq = '^' + word + '$'
        return sum(math.log((transitions[a, b] + .5) /
                            (contexts[a] + .5 * vocab_size))
                   for a, b in zip(seq, seq[1:])) / (len(seq) - 1)

    rows = []
    for word, deg in degree.items():
        if deg < min_degree or word in freq:
            continue
        rows.append({'word': word, 'attested_later': int(word in future_new),
                     'degree': deg, 'glyph_length': len(word),
                     'log_degree': math.log1p(deg),
                     'bigram': letter_bigram(word),
                     'mean_log_witness_frequency': strengths[word] / deg,
                     'lexicon_attested': int(word in lexicon) if lexicon is not None else -1})
    rows.sort(key=lambda r: r['word'])
    summary = {'train_tokens': len(train), 'train_types': len(freq),
               'future_tokens': len(future), 'new_future_types': len(future_new),
               'reachable_future_types': sum(w in degree for w in future_new),
               'candidate_types': len(rows),
               'candidate_positives': sum(r['attested_later'] for r in rows),
               'alphabet': ''.join(alphabet)}
    return rows, summary, forms, neighbors


def evaluate_sample(tokens: list[str], lexicon: set[str] | None = None,
                    min_degree=2, mask_trials=500, seed=20261028,
                    top_k=500):
    if len(tokens) < 100:
        raise ValueError('Control text too short')
    early = int(.5 * len(tokens))
    cut = int(.7 * len(tokens))
    inner, inner_summary, _, _ = english_features(
        tokens[:early], tokens[early:cut], min_degree, lexicon)
    outer, outer_summary, forms, neighbors = english_features(
        tokens[:cut], tokens[cut:], min_degree, lexicon)
    y = [r['attested_later'] for r in outer]
    results = {
        'raw_degree': metrics(y, [r['degree'] for r in outer]),
        'raw_bigram': metrics(y, [r['bigram'] for r in outer]),
        'matched_bigram': degree_length_matched_auc(outer),
    }
    if min(y) != max(y) and len({r['attested_later'] for r in inner}) == 2:
        for label, predictors in [
            ('degree_length', ['log_degree', 'glyph_length']),
            ('degree_length_bigram', ['log_degree', 'glyph_length', 'bigram']),
        ]:
            pred = fit_predict(inner, outer, predictors)
            for row, score in zip(outer, pred):
                row[label] = float(score)
            results[label] = metrics(y, pred)

    # Rank all mask-absent types, including deleted training words.
    rng = np.random.default_rng(seed)
    training_counts = Counter(tokens[:cut])
    form_ids = {w: i for i, w in enumerate(forms)}
    future_new = set(tokens[cut:]) - training_counts.keys()
    mask_vocab = sorted(w for w, witnesses in neighbors.items()
                        if len(witnesses) >= min_degree)
    ptr = np.zeros(len(mask_vocab) + 1, dtype=np.int64)
    flat_parts = []
    for i, word in enumerate(mask_vocab):
        ptr[i + 1] = ptr[i] + len(neighbors[word])
        flat_parts.extend(neighbors[word])
    flat = np.asarray(flat_parts, dtype=np.int32)
    self_ids = np.array([form_ids.get(w, -1) for w in mask_vocab], dtype=np.int32)
    kinds = np.array([0 if w in training_counts else 1 if w in future_new else 2
                      for w in mask_vocab], dtype=np.int8)
    top_hits = np.zeros(len(mask_vocab), dtype=np.int32)
    reach_hits = np.zeros(len(mask_vocab), dtype=np.int32)
    witness_sum = np.zeros(len(mask_vocab), dtype=np.float64)
    composition = []
    num_kept = int(round(len(forms) * .25))
    for trial in range(mask_trials):
        visible = np.zeros(len(forms), dtype=np.int8)
        visible[rng.choice(len(forms), size=num_kept, replace=False)] = 1
        support = np.add.reduceat(visible[flat], ptr[:-1])
        present = (self_ids >= 0) & (visible[self_ids.clip(min=0)] != 0)
        eligible = ~present
        reach_hits += ((support > 0) & eligible).astype(np.int32)
        witness_sum += support * eligible
        possible = np.flatnonzero((support > 0) & eligible)
        k = min(top_k, len(possible))
        if not k:
            composition.append({'trial': trial, 'deleted_training': 0,
                                'later_attested': 0, 'unattested': 0})
            continue
        ranking = support[possible] + rng.random(len(possible)) * .49
        chosen = possible[np.argpartition(ranking, -k)[-k:]]
        top_hits[chosen] += 1
        parts = np.bincount(kinds[chosen], minlength=3)
        composition.append({'trial': trial, 'deleted_training': int(parts[0]),
                            'later_attested': int(parts[1]),
                            'unattested': int(parts[2])})
    index = {w: i for i, w in enumerate(mask_vocab)}
    for row in outer:
        i = index[row['word']]
        row['top500_pct'] = round(100 * top_hits[i] / max(mask_trials, 1), 4)
        row['reachable_pct'] = round(100 * reach_hits[i] / max(mask_trials, 1), 4)
        row['mean_visible_witnesses'] = round(witness_sum[i] /
                                              max(mask_trials, 1), 4)
    results['deletion_masks'] = {
        'trials': mask_trials, 'survive_fraction': .25, 'top_k': top_k,
        'mean_deleted_training_topk': round(sum(c['deleted_training']
            for c in composition) / max(mask_trials, 1), 4),
        'mean_later_attested_topk': round(sum(c['later_attested']
            for c in composition) / max(mask_trials, 1), 4),
        'mean_unattested_topk': round(sum(c['unattested']
            for c in composition) / max(mask_trials, 1), 4)}
    if lexicon is not None:
        results['dictionary_coverage'] = {
            'lexicon_positive_candidates': sum(r['lexicon_attested'] for r in outer),
            'degree_roc_auc': metrics([r['lexicon_attested'] for r in outer],
                                      [r['degree'] for r in outer]),
            'bigram_roc_auc': metrics([r['lexicon_attested'] for r in outer],
                                      [r['bigram'] for r in outer]),
        }
    return {'inner': inner_summary, 'outer': outer_summary,
            'results': results}, outer, composition


def run(corpus_files, out, token_budgets=(11063, 22843),
        trials=500, lexicon_file=None):
    lexicon = None
    if lexicon_file:
        lexicon = {w.lower() for line in Path(lexicon_file).read_text().splitlines()
                   if (w := line.strip()).isascii() and w.isalpha()}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    reports = {}
    for label, path in corpus_files.items():
        tokens = clean_gutenberg(Path(path).read_text(encoding='utf-8-sig'), label)
        reports[label] = {}
        for budget in token_budgets:
            if len(tokens) < budget:
                raise ValueError(f'{label} has only {len(tokens)} words, not {budget}')
            summary, rows, masks = evaluate_sample(tokens[:budget], lexicon,
                                                    mask_trials=trials)
            reports[label][str(budget)] = summary
            prefix = f'{label}_{budget}'
            with (out / f'{prefix}_candidates.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with (out / f'{prefix}_mask_composition.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(masks[0]))
                writer.writeheader()
                writer.writerows(masks)
    (out / 'summary.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports, indent=2))
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--austen', required=True)
    parser.add_argument('--darwin', required=True)
    parser.add_argument('--out', default='results/english_controls')
    parser.add_argument('--tokens', type=int, nargs='+', default=[11063, 22843])
    parser.add_argument('--trials', type=int, default=500)
    parser.add_argument('--lexicon', help='Optional independent wordlist, one word per line')
    a = parser.parse_args()
    run({'austen': a.austen, 'darwin': a.darwin}, a.out,
        a.tokens, a.trials, a.lexicon)
,)
        return sum(math.log((transitions[a, b] + .5) /
                            (contexts[a] + .5 * vocab_size))
                   for a, b in zip(seq, seq[1:])) / (len(seq) - 1)

    rows = []
    for word, deg in degree.items():
        if deg < min_degree or word in freq:
            continue
        rows.append({'word': word, 'attested_later': int(word in future_new),
                     'degree': deg, 'glyph_length': len(word),
                     'log_degree': math.log1p(deg),
                     'bigram': letter_bigram(word),
                     'mean_log_witness_frequency': strengths[word] / deg,
                     'lexicon_attested': int(word in lexicon) if lexicon is not None else -1})
    rows.sort(key=lambda r: r['word'])
    summary = {'train_tokens': len(train), 'train_types': len(freq),
               'future_tokens': len(future), 'new_future_types': len(future_new),
               'reachable_future_types': sum(w in degree for w in future_new),
               'candidate_types': len(rows),
               'candidate_positives': sum(r['attested_later'] for r in rows),
               'alphabet': ''.join(alphabet)}
    return rows, summary, forms, neighbors


def evaluate_sample(tokens: list[str], lexicon: set[str] | None = None,
                    min_degree=2, mask_trials=500, seed=20261028,
                    top_k=500):
    if len(tokens) < 100:
        raise ValueError('Control text too short')
    early = int(.5 * len(tokens))
    cut = int(.7 * len(tokens))
    inner, inner_summary, _, _ = english_features(
        tokens[:early], tokens[early:cut], min_degree, lexicon)
    outer, outer_summary, forms, neighbors = english_features(
        tokens[:cut], tokens[cut:], min_degree, lexicon)
    y = [r['attested_later'] for r in outer]
    results = {
        'raw_degree': metrics(y, [r['degree'] for r in outer]),
        'raw_bigram': metrics(y, [r['bigram'] for r in outer]),
        'matched_bigram': degree_length_matched_auc(outer),
    }
    if min(y) != max(y) and len({r['attested_later'] for r in inner}) == 2:
        for label, predictors in [
            ('degree_length', ['log_degree', 'glyph_length']),
            ('degree_length_bigram', ['log_degree', 'glyph_length', 'bigram']),
        ]:
            pred = fit_predict(inner, outer, predictors)
            for row, score in zip(outer, pred):
                row[label] = float(score)
            results[label] = metrics(y, pred)

    # Rank all mask-absent types, including deleted training words.
    rng = np.random.default_rng(seed)
    training_counts = Counter(tokens[:cut])
    form_ids = {w: i for i, w in enumerate(forms)}
    future_new = set(tokens[cut:]) - training_counts.keys()
    mask_vocab = sorted(w for w, witnesses in neighbors.items()
                        if len(witnesses) >= min_degree)
    ptr = np.zeros(len(mask_vocab) + 1, dtype=np.int64)
    flat_parts = []
    for i, word in enumerate(mask_vocab):
        ptr[i + 1] = ptr[i] + len(neighbors[word])
        flat_parts.extend(neighbors[word])
    flat = np.asarray(flat_parts, dtype=np.int32)
    self_ids = np.array([form_ids.get(w, -1) for w in mask_vocab], dtype=np.int32)
    kinds = np.array([0 if w in training_counts else 1 if w in future_new else 2
                      for w in mask_vocab], dtype=np.int8)
    top_hits = np.zeros(len(mask_vocab), dtype=np.int32)
    reach_hits = np.zeros(len(mask_vocab), dtype=np.int32)
    witness_sum = np.zeros(len(mask_vocab), dtype=np.float64)
    composition = []
    num_kept = int(round(len(forms) * .25))
    for trial in range(mask_trials):
        visible = np.zeros(len(forms), dtype=np.int8)
        visible[rng.choice(len(forms), size=num_kept, replace=False)] = 1
        support = np.add.reduceat(visible[flat], ptr[:-1])
        present = (self_ids >= 0) & (visible[self_ids.clip(min=0)] != 0)
        eligible = ~present
        reach_hits += ((support > 0) & eligible).astype(np.int32)
        witness_sum += support * eligible
        possible = np.flatnonzero((support > 0) & eligible)
        k = min(top_k, len(possible))
        if not k:
            composition.append({'trial': trial, 'deleted_training': 0,
                                'later_attested': 0, 'unattested': 0})
            continue
        ranking = support[possible] + rng.random(len(possible)) * .49
        chosen = possible[np.argpartition(ranking, -k)[-k:]]
        top_hits[chosen] += 1
        parts = np.bincount(kinds[chosen], minlength=3)
        composition.append({'trial': trial, 'deleted_training': int(parts[0]),
                            'later_attested': int(parts[1]),
                            'unattested': int(parts[2])})
    index = {w: i for i, w in enumerate(mask_vocab)}
    for row in outer:
        i = index[row['word']]
        row['top500_pct'] = round(100 * top_hits[i] / max(mask_trials, 1), 4)
        row['reachable_pct'] = round(100 * reach_hits[i] / max(mask_trials, 1), 4)
        row['mean_visible_witnesses'] = round(witness_sum[i] /
                                              max(mask_trials, 1), 4)
    results['deletion_masks'] = {
        'trials': mask_trials, 'survive_fraction': .25, 'top_k': top_k,
        'mean_deleted_training_topk': round(sum(c['deleted_training']
            for c in composition) / max(mask_trials, 1), 4),
        'mean_later_attested_topk': round(sum(c['later_attested']
            for c in composition) / max(mask_trials, 1), 4),
        'mean_unattested_topk': round(sum(c['unattested']
            for c in composition) / max(mask_trials, 1), 4)}
    if lexicon is not None:
        results['dictionary_coverage'] = {
            'lexicon_positive_candidates': sum(r['lexicon_attested'] for r in outer),
            'degree_roc_auc': metrics([r['lexicon_attested'] for r in outer],
                                      [r['degree'] for r in outer]),
            'bigram_roc_auc': metrics([r['lexicon_attested'] for r in outer],
                                      [r['bigram'] for r in outer]),
        }
    return {'inner': inner_summary, 'outer': outer_summary,
            'results': results}, outer, composition


def run(corpus_files, out, token_budgets=(11063, 22843),
        trials=500, lexicon_file=None):
    lexicon = None
    if lexicon_file:
        lexicon = {w.lower() for line in Path(lexicon_file).read_text().splitlines()
                   if (w := line.strip()).isascii() and w.isalpha()}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    reports = {}
    for label, path in corpus_files.items():
        tokens = clean_gutenberg(Path(path).read_text(encoding='utf-8-sig'), label)
        reports[label] = {}
        for budget in token_budgets:
            if len(tokens) < budget:
                raise ValueError(f'{label} has only {len(tokens)} words, not {budget}')
            summary, rows, masks = evaluate_sample(tokens[:budget], lexicon,
                                                    mask_trials=trials)
            reports[label][str(budget)] = summary
            prefix = f'{label}_{budget}'
            with (out / f'{prefix}_candidates.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with (out / f'{prefix}_mask_composition.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(masks[0]))
                writer.writeheader()
                writer.writerows(masks)
    (out / 'summary.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports, indent=2))
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--austen', required=True)
    parser.add_argument('--darwin', required=True)
    parser.add_argument('--out', default='results/english_controls')
    parser.add_argument('--tokens', type=int, nargs='+', default=[11063, 22843])
    parser.add_argument('--trials', type=int, default=500)
    parser.add_argument('--lexicon', help='Optional independent wordlist, one word per line')
    a = parser.parse_args()
    run({'austen': a.austen, 'darwin': a.darwin}, a.out,
        a.tokens, a.trials, a.lexicon)
,)
        for a, b in zip(seq, seq[1:]):
            transitions[a, b] += count
            contexts[a] += count
    vocab_size = len(alphabet) + 1

    def letter_bigram(word):
        seq = '^' + word + '$'
        return sum(math.log((transitions[a, b] + .5) /
                            (contexts[a] + .5 * vocab_size))
                   for a, b in zip(seq, seq[1:])) / (len(seq) - 1)

    rows = []
    for word, deg in degree.items():
        if deg < min_degree or word in freq:
            continue
        rows.append({'word': word, 'attested_later': int(word in future_new),
                     'degree': deg, 'glyph_length': len(word),
                     'log_degree': math.log1p(deg),
                     'bigram': letter_bigram(word),
                     'mean_log_witness_frequency': strengths[word] / deg,
                     'lexicon_attested': int(word in lexicon) if lexicon is not None else -1})
    rows.sort(key=lambda r: r['word'])
    summary = {'train_tokens': len(train), 'train_types': len(freq),
               'future_tokens': len(future), 'new_future_types': len(future_new),
               'reachable_future_types': sum(w in degree for w in future_new),
               'candidate_types': len(rows),
               'candidate_positives': sum(r['attested_later'] for r in rows),
               'alphabet': ''.join(alphabet)}
    return rows, summary, forms, neighbors


def evaluate_sample(tokens: list[str], lexicon: set[str] | None = None,
                    min_degree=2, mask_trials=500, seed=20261028,
                    top_k=500):
    if len(tokens) < 100:
        raise ValueError('Control text too short')
    early = int(.5 * len(tokens))
    cut = int(.7 * len(tokens))
    inner, inner_summary, _, _ = english_features(
        tokens[:early], tokens[early:cut], min_degree, lexicon)
    outer, outer_summary, forms, neighbors = english_features(
        tokens[:cut], tokens[cut:], min_degree, lexicon)
    y = [r['attested_later'] for r in outer]
    results = {
        'raw_degree': metrics(y, [r['degree'] for r in outer]),
        'raw_bigram': metrics(y, [r['bigram'] for r in outer]),
        'matched_bigram': degree_length_matched_auc(outer),
    }
    if min(y) != max(y) and len({r['attested_later'] for r in inner}) == 2:
        for label, predictors in [
            ('degree_length', ['log_degree', 'glyph_length']),
            ('degree_length_bigram', ['log_degree', 'glyph_length', 'bigram']),
        ]:
            pred = fit_predict(inner, outer, predictors)
            for row, score in zip(outer, pred):
                row[label] = float(score)
            results[label] = metrics(y, pred)

    # Rank all mask-absent types, including deleted training words.
    rng = np.random.default_rng(seed)
    training_counts = Counter(tokens[:cut])
    form_ids = {w: i for i, w in enumerate(forms)}
    future_new = set(tokens[cut:]) - training_counts.keys()
    mask_vocab = sorted(w for w, witnesses in neighbors.items()
                        if len(witnesses) >= min_degree)
    ptr = np.zeros(len(mask_vocab) + 1, dtype=np.int64)
    flat_parts = []
    for i, word in enumerate(mask_vocab):
        ptr[i + 1] = ptr[i] + len(neighbors[word])
        flat_parts.extend(neighbors[word])
    flat = np.asarray(flat_parts, dtype=np.int32)
    self_ids = np.array([form_ids.get(w, -1) for w in mask_vocab], dtype=np.int32)
    kinds = np.array([0 if w in training_counts else 1 if w in future_new else 2
                      for w in mask_vocab], dtype=np.int8)
    top_hits = np.zeros(len(mask_vocab), dtype=np.int32)
    reach_hits = np.zeros(len(mask_vocab), dtype=np.int32)
    witness_sum = np.zeros(len(mask_vocab), dtype=np.float64)
    composition = []
    num_kept = int(round(len(forms) * .25))
    for trial in range(mask_trials):
        visible = np.zeros(len(forms), dtype=np.int8)
        visible[rng.choice(len(forms), size=num_kept, replace=False)] = 1
        support = np.add.reduceat(visible[flat], ptr[:-1])
        present = (self_ids >= 0) & (visible[self_ids.clip(min=0)] != 0)
        eligible = ~present
        reach_hits += ((support > 0) & eligible).astype(np.int32)
        witness_sum += support * eligible
        possible = np.flatnonzero((support > 0) & eligible)
        k = min(top_k, len(possible))
        if not k:
            composition.append({'trial': trial, 'deleted_training': 0,
                                'later_attested': 0, 'unattested': 0})
            continue
        ranking = support[possible] + rng.random(len(possible)) * .49
        chosen = possible[np.argpartition(ranking, -k)[-k:]]
        top_hits[chosen] += 1
        parts = np.bincount(kinds[chosen], minlength=3)
        composition.append({'trial': trial, 'deleted_training': int(parts[0]),
                            'later_attested': int(parts[1]),
                            'unattested': int(parts[2])})
    index = {w: i for i, w in enumerate(mask_vocab)}
    for row in outer:
        i = index[row['word']]
        row['top500_pct'] = round(100 * top_hits[i] / max(mask_trials, 1), 4)
        row['reachable_pct'] = round(100 * reach_hits[i] / max(mask_trials, 1), 4)
        row['mean_visible_witnesses'] = round(witness_sum[i] /
                                              max(mask_trials, 1), 4)
    results['deletion_masks'] = {
        'trials': mask_trials, 'survive_fraction': .25, 'top_k': top_k,
        'mean_deleted_training_topk': round(sum(c['deleted_training']
            for c in composition) / max(mask_trials, 1), 4),
        'mean_later_attested_topk': round(sum(c['later_attested']
            for c in composition) / max(mask_trials, 1), 4),
        'mean_unattested_topk': round(sum(c['unattested']
            for c in composition) / max(mask_trials, 1), 4)}
    if lexicon is not None:
        results['dictionary_coverage'] = {
            'lexicon_positive_candidates': sum(r['lexicon_attested'] for r in outer),
            'degree_roc_auc': metrics([r['lexicon_attested'] for r in outer],
                                      [r['degree'] for r in outer]),
            'bigram_roc_auc': metrics([r['lexicon_attested'] for r in outer],
                                      [r['bigram'] for r in outer]),
        }
    return {'inner': inner_summary, 'outer': outer_summary,
            'results': results}, outer, composition


def run(corpus_files, out, token_budgets=(11063, 22843),
        trials=500, lexicon_file=None):
    lexicon = None
    if lexicon_file:
        lexicon = {w.lower() for line in Path(lexicon_file).read_text().splitlines()
                   if (w := line.strip()).isascii() and w.isalpha()}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    reports = {}
    for label, path in corpus_files.items():
        tokens = clean_gutenberg(Path(path).read_text(encoding='utf-8-sig'), label)
        reports[label] = {}
        for budget in token_budgets:
            if len(tokens) < budget:
                raise ValueError(f'{label} has only {len(tokens)} words, not {budget}')
            summary, rows, masks = evaluate_sample(tokens[:budget], lexicon,
                                                    mask_trials=trials)
            reports[label][str(budget)] = summary
            prefix = f'{label}_{budget}'
            with (out / f'{prefix}_candidates.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with (out / f'{prefix}_mask_composition.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(masks[0]))
                writer.writeheader()
                writer.writerows(masks)
    (out / 'summary.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports, indent=2))
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--austen', required=True)
    parser.add_argument('--darwin', required=True)
    parser.add_argument('--out', default='results/english_controls')
    parser.add_argument('--tokens', type=int, nargs='+', default=[11063, 22843])
    parser.add_argument('--trials', type=int, default=500)
    parser.add_argument('--lexicon', help='Optional independent wordlist, one word per line')
    a = parser.parse_args()
    run({'austen': a.austen, 'darwin': a.darwin}, a.out,
        a.tokens, a.trials, a.lexicon)
