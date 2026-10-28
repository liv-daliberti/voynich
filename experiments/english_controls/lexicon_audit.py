#!/usr/bin/env python3
"""Post-hoc English lexical-attestation audit of 500-mask candidate CSVs.

Positive membership in a pronunciation lexicon is evidence of an established
spelling, not proof that a word fits the manuscript/book's context. Negative
membership is NOT evidence that an English string is not a word.
"""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict


def audit(candidate_file, lexicon, min_length=3, top=100):
    rows = []
    with Path(candidate_file).open(newline='') as f:
        for row in csv.DictReader(f):
            word = row['word'].lower()
            if len(word) < min_length or not word.isalpha():
                continue
            rows.append({'word': word, 'score': float(row['top500_pct']),
                         'later': int(row['attested_later']),
                         'known': int(word in lexicon)})
    if not rows:
        raise ValueError(f'No candidates in {candidate_file}')
    rows.sort(key=lambda row: (-row['score'], row['word']))
    chosen = rows[:top]
    return {'file': str(candidate_file), 'min_length': min_length,
            'candidate_types': len(rows), 'top_n': len(chosen),
            'lexicon_base_rate': sum(r['known'] for r in rows) / len(rows),
            'lexicon_top_n_rate': sum(r['known'] for r in chosen) / len(chosen),
            'later_attested_top_n': sum(r['later'] for r in chosen),
            'top_n_examples': [{'word': r['word'],
                                'nomination_pct': r['score'],
                                'lexicon_present': bool(r['known']),
                                'later_attested': bool(r['later'])}
                               for r in chosen[:20]]}


def run(files, lexicon_file, min_length=3, top=100):
    # One English entry per line, ASCII letters only, no pronunciations.
    lexicon = {w.lower() for line in Path(lexicon_file).read_text().splitlines()
               if (w := line.strip()).isascii() and w.isalpha()}
    return {'lexicon_size': len(lexicon),
            'warning': 'Absence from a lexicon does not imply an invalid word',
            'results': [audit(f, lexicon, min_length, top) for f in files]}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lexicon', required=True)
    ap.add_argument('--candidates', nargs='+', required=True)
    ap.add_argument('--min-length', type=int, default=3)
    ap.add_argument('--top', type=int, default=100)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    report = run(args.candidates, args.lexicon, args.min_length, args.top)
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
