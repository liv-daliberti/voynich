#!/usr/bin/env python3
"""Prospective unseen-word slot filling from training-only transition graphs.

A 'sentence' is a sentence in English, but a manuscript LINE in Voynich.
Evaluate these separately; boundary definitions are not equivalent.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.robustness75.robustness import glyphs, one_edit_forms
from experiments.english_controls.benchmark import character_edits


def voynich_lines(text: str, language: str):
    """Keep paragraph-line boundaries; strict folio language labels."""
    out = []
    folio = None
    lang = None
    for line in text.splitlines():
        head = re.match(r'^<([^.,>]+)>\s*<!\s*(.*?)>', line)
        if head:
            folio = head.group(1)
            m = re.search(r'\$L=([AB])(?:\s|$)', head.group(2))
            lang = m.group(1) if m else None
            continue
        row = re.match(r'^<([^>,]+),([^>]+)>\s*(.*)$', line)
        if not row or not re.search(r'[+@*=]P', row.group(2)) or lang != language:
            continue
        clean = re.sub(r'<![^>]*>|<%>|<\$>|\{[^}]*\}', '.', row.group(3))
        clean = clean.replace('<->', '.')
        words = [w for w in re.split(r'[.,\s]+', clean)
                 if re.fullmatch(r'[a-z]+', w)]
        if len(words) >= 3:
            out.append((folio, words))
    return out


def english_sentences(text: str, corpus: str):
    """Paragraph/sentence heuristic; keep original punctuation for boundaries."""
    txt = text.replace('\r\n', '\n').lstrip('\ufeff')
    end = re.search(r'^\*{3}\s*END OF (?:THIS|THE) PROJECT GUTENBERG',
                    txt, re.M | re.I)
    if end:
        txt = txt[:end.start()]
    if corpus == 'austen':
        start = re.search(r'^Chapter\s+1\s*$', txt, re.M | re.I)
    else:
        hits = list(re.finditer(r'^INTRODUCTION\.\s*$', txt, re.M | re.I))
        start = hits[-1] if hits else None
    if not start:
        raise ValueError('Unknown Gutenberg edition')
    txt = txt[start.end():]
    chunks = re.split(r'[.!?]+(?:[\x22\x27\u201d\u2019]*)\s+|\n\s*\n', txt)
    out = []
    block = tokens = 0
    for chunk in chunks:
        words = re.findall(r'[a-z]+', chunk.lower())
        if not 3 <= len(words) <= 100:
            continue
        if tokens >= 250:
            block += 1
            tokens = 0
        out.append((f'block{block}', words))
        tokens += len(words)
    return out


def prefix_budget(sequences, budget):
    """Take whole lines/sentences without passing the token budget."""
    result = []
    n = 0
    for group, words in sequences:
        if n + len(words) > budget:
            break
        result.append((group, words))
        n += len(words)
    return result


def split_sequences(seq, ratio=.7):
    target = sum(len(words) for _, words in seq) * ratio
    n = 0
    cut = 0
    for _, words in seq:
        if n >= target:
            break
        n += len(words)
        cut += 1
    return seq[:cut], seq[cut:]


def position(i, n):
    return 'start' if i == 0 else 'end' if i == n - 1 else 'middle'


def shuffle_within_group_position(seqs, seed):
    """Preserve folio/block and start/middle/end token multisets."""
    rng = random.Random(seed)
    buffers = defaultdict(list)
    refs = defaultdict(list)
    out = [(grp, list(words)) for grp, words in seqs]
    for si, (grp, words) in enumerate(seqs):
        for wi, word in enumerate(words):
            key = (grp, position(wi, len(words)))
            buffers[key].append(word)
            refs[key].append((si, wi))
    for key, tokens in buffers.items():
        rng.shuffle(tokens)
        for (si, wi), word in zip(refs[key], tokens):
            out[si][1][wi] = word
    return out


def graph_stats(seqs):
    frequency = Counter()
    prev_edge = Counter()
    pos_edge = Counter()
    n_tokens = 0
    for _, words in seqs:
        frequency.update(words)
        n_tokens += len(words)
        for i, word in enumerate(words):
            pos_edge[position(i, len(words)), word] += 1
            if i:
                prev_edge[words[i - 1], word] += 1
    return frequency, prev_edge, pos_edge, n_tokens


def support(train_words, eva):
    """Train-only one-edit neighborhoods for absent proposals."""
    train_vocab = set(train_words)
    alphabet = sorted(set(g for w in train_vocab for g in
                          (glyphs(w) if eva else tuple(w))))
    neighbors = defaultdict(list)
    for word in sorted(train_vocab):
        proposals = (one_edit_forms(glyphs(word), alphabet) if eva else
                     character_edits(word, tuple(alphabet)))
        for candidate in proposals:
            if candidate not in train_vocab:
                neighbors[candidate].append(word)
    return {w: ns for w, ns in neighbors.items() if len(ns) >= 2}


def orthographic_model(train_freq, eva):
    alphabet = {g for word in train_freq for g in
                (glyphs(word) if eva else tuple(word))}
    V = len(alphabet) + 1
    trans, context = Counter(), Counter()
    for word, f in train_freq.items():
        seq = ('^',) + (glyphs(word) if eva else tuple(word)) + ('$',)
        for a, b in zip(seq, seq[1:]):
            trans[a, b] += f
            context[a] += f

    def score(word):
        seq = ('^',) + (glyphs(word) if eva else tuple(word)) + ('$',)
        return sum(math.log((trans[a, b] + .5) /
                            (context[a] + .5 * V))
                   for a, b in zip(seq, seq[1:])) / (len(seq) - 1)
    return score


def neighbor_context(word, previous, following, pos, neighbors,
                     stats, prefix_only=False):
    """P(context | neighborhood(word)) relative to marginal context rate."""
    freq, edges, positions, total = stats
    witness = neighbors[word]
    F = sum(freq[n] for n in witness)
    V = len(freq) + 1
    alpha = .5
    left = sum(edges[previous, n] for n in witness)
    prior_left = (freq[previous] + alpha) / (total + alpha * V)
    left_score = math.log((left + alpha) / (F + alpha * V)) - math.log(prior_left)
    if prefix_only:
        return left_score
    right = sum(edges[n, following] for n in witness)
    prior_right = (freq[following] + alpha) / (total + alpha * V)
    right_score = math.log((right + alpha) / (F + alpha * V)) - math.log(prior_right)
    local_position = sum(positions[pos, n] for n in witness)
    pos_total = sum(positions[p, n] for p in ('start','middle','end') for n in witness)
    global_pos = sum(positions[pos, n] for n in freq)
    position_score = math.log((local_position + alpha) /
                              (pos_total + 3 * alpha)) - math.log(
                                  (global_pos + alpha) / (total + 3 * alpha))
    return left_score + right_score + position_score


def evaluate_cloze(sequences, eva, seed=20261028, negatives=9):
    train, test = split_sequences(sequences)
    stats = graph_stats(train)
    shuffled = graph_stats(shuffle_within_group_position(train, seed))
    candidates = support(stats[0], eva)
    score_spelling = orthographic_model(stats[0], eva)
    buckets = defaultdict(list)
    for word, ns in candidates.items():
        length = len(glyphs(word)) if eva else len(word)
        buckets[len(ns), length].append(word)
    training_words = set(stats[0])
    seen_heldout = set()
    rng = random.Random(seed)
    tasks = []
    for _, sentence in test:
        for i in range(1, len(sentence) - 1):
            target = sentence[i]
            # One task per truly new word type. Both neighbors must be
            # observed in training so prediction relies on learned context.
            if target in training_words or target not in candidates or target in seen_heldout:
                continue
            before, after = sentence[i - 1], sentence[i + 1]
            if before not in training_words or after not in training_words:
                continue
            key = (len(candidates[target]),
                   len(glyphs(target)) if eva else len(target))
            pool = buckets[key]
            if len(pool) < negatives + 1:
                continue
            distractors = rng.sample([w for w in pool if w != target], negatives)
            tasks.append({'word': target, 'before': before, 'after': after,
                          'position': position(i, len(sentence)),
                          'alternatives': [target] + distractors})
            seen_heldout.add(target)
    wins = Counter()
    top1 = Counter()
    cols = ('spelling', 'prefix_graph', 'cloze_graph', 'prefix_shuffled',
            'cloze_shuffled', 'spelling_plus_cloze',
            'spelling_plus_shuffled_cloze')
    for item in tasks:
        before, after, pos = item['before'], item['after'], item['position']
        predictions = defaultdict(list)
        for word in item['alternatives']:
            spell = score_spelling(word)
            prefix = neighbor_context(word, before, after, pos, candidates,
                                      stats, prefix_only=True)
            cloze = neighbor_context(word, before, after, pos, candidates,
                                     stats)
            null_prefix = neighbor_context(word, before, after, pos, candidates,
                                           shuffled, prefix_only=True)
            null_cloze = neighbor_context(word, before, after, pos, candidates,
                                          shuffled)
            for col, value in [('spelling', spell), ('prefix_graph', prefix),
                               ('cloze_graph', cloze),
                               ('prefix_shuffled', null_prefix),
                               ('cloze_shuffled', null_cloze),
                               ('spelling_plus_cloze', spell + cloze),
                               ('spelling_plus_shuffled_cloze', spell + null_cloze)]:
                predictions[col].append(value)
        for col in cols:
            arr = predictions[col]
            wins[col] += sum(int(arr[0] > v) + .5 * int(arr[0] == v)
                             for v in arr[1:]) / negatives
            top1[col] += int(arr[0] == max(arr)) / arr.count(max(arr))
    return {'training_tokens': stats[-1],
            'test_tokens': sum(len(s) for _, s in test),
            'training_types': len(training_words),
            'candidate_types': len(candidates),
            'novel_cloze_tasks': len(tasks),
            'negatives_per_task': negatives,
            'metrics': {key: {'pairwise_accuracy': round(wins[key] /
                     len(tasks), 6) if tasks else None,
                     'top1': round(top1[key] / len(tasks), 6) if tasks else None}
                        for key in cols}}, tasks, stats[1]



def evaluate_seen_control(sequences, eva, seed=20261028,
                          max_tasks=1000, negatives=9):
    """Sanity test: can local word order rank ALREADY OBSERVED word types?

    Distractors have identical unit length and similar token frequency.
    English should provide a detectable context-vs-shuffle signal. This is
    deliberately a distinct task from unseen-form neighbor transfer.
    """
    train, test = split_sequences(sequences)
    real = graph_stats(train)
    null = graph_stats(shuffle_within_group_position(train, seed))
    freq, _, _, _ = real
    V = len(freq)
    buckets = defaultdict(list)
    for word, count in freq.items():
        length = len(glyphs(word)) if eva else len(word)
        bucket = (length, int(math.log2(count + 1)))
        buckets[bucket].append(word)
    events = []
    for _, sentence in test:
        for i in range(1, len(sentence) - 1):
            before, word, after = sentence[i - 1:i + 2]
            if before not in freq or after not in freq or word not in freq:
                continue
            bucket = (len(glyphs(word)) if eva else len(word),
                      int(math.log2(freq[word] + 1)))
            if len(buckets[bucket]) < negatives + 1:
                continue
            events.append((before, word, after, position(i, len(sentence)),
                           bucket))
    rng = random.Random(seed)
    rng.shuffle(events)
    tasks = events[:max_tasks]
    wins = Counter()
    top1 = Counter()
    columns = ('frequency', 'observed_context', 'shuffled_context')

    def context_score(before, word, after, pos, stats):
        freq, edges, positions, _ = stats
        alpha = .5
        left = math.log((edges[before, word] + alpha) /
                        (freq[before] + alpha * V))
        right = math.log((edges[word, after] + alpha) /
                         (freq[word] + alpha * V))
        slot = math.log((positions[pos, word] + alpha) /
                        (freq[word] + 3 * alpha))
        return left + right + slot

    for before, correct, after, pos, bucket in tasks:
        negatives_set = rng.sample([w for w in buckets[bucket]
                                    if w != correct], negatives)
        alternatives = [correct] + negatives_set
        predictions = {
            'frequency': [math.log1p(freq[w]) for w in alternatives],
            'observed_context': [context_score(before, w, after, pos, real)
                                 for w in alternatives],
            'shuffled_context': [context_score(before, w, after, pos, null)
                                 for w in alternatives],
        }
        for col in columns:
            arr = predictions[col]
            wins[col] += sum(int(arr[0] > score) +
                             .5 * int(arr[0] == score)
                             for score in arr[1:]) / negatives
            top1[col] += int(arr[0] == max(arr)) / arr.count(max(arr))
    return {
        'task_count': len(tasks), 'negatives_per_task': negatives,
        'metrics': {key: {
            'pairwise_accuracy': round(wins[key] / len(tasks), 6)
                                 if tasks else None,
            'top1': round(top1[key] / len(tasks), 6)
                    if tasks else None} for key in columns}
    }

def run(zl, austen, darwin, out, token_budgets=(11063, 22843)):
    raw_zl = Path(zl).read_text(encoding='utf-8-sig')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    result = {}
    corpora = [(f'Voynich_{label}', voynich_lines(raw_zl, label), True)
               for label in 'AB']
    for label, file in [('austen', austen), ('darwin', darwin)]:
        sentences = english_sentences(Path(file).read_text(encoding='utf-8-sig'), label)
        for budget in token_budgets:
            corpora.append((f'{label}_{budget}',
                            prefix_budget(sentences, budget), False))
    for label, seq, eva in corpora:
        summary, tasks, edges = evaluate_cloze(seq, eva)
        summary['seen_word_control'] = evaluate_seen_control(seq, eva)
        result[label] = summary
        with (out / f'{label}_cloze_tasks.csv').open('w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(['word', 'before', 'after', 'position', 'alternatives'])
            for r in tasks:
                writer.writerow([r['word'], r['before'], r['after'],
                                 r['position'], '|'.join(r['alternatives'])])
        with (out / f'{label}_top_transitions.csv').open('w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(['previous', 'next', 'training_count'])
            for (a, b), count in edges.most_common(200):
                writer.writerow([a, b, count])
        with (out / f'{label}_transition_graph.dot').open('w') as fp:
            fp.write('digraph word_transitions {\n  rankdir=LR;\n')
            for (a, b), count in edges.most_common(80):
                # Tokens are lowercase alphabetic only; safe graph labels.
                fp.write(f'  "{a}" -> "{b}" [label="{count}", weight={count}];\n')
            fp.write('}\n')
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--zl', required=True)
    p.add_argument('--austen', required=True)
    p.add_argument('--darwin', required=True)
    p.add_argument('--out', default='results/context')
    a = p.parse_args()
    run(a.zl, a.austen, a.darwin, a.out)
