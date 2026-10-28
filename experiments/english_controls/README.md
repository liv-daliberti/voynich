# Readable-language, cipher, and sentence-context controls

October 28, 2026. This is a reproducible extension of [75% robustness](../robustness75/README.md) and the [prospective glyph model](../phonotactics/README.md).

**Primary question:** Are edit-neighbor stability and glyph/letter-sequence constraints distinctive to Voynich, or are they also normal properties of written English?

**Secondary question:** Does the usage of a candidate's *observed training neighbors* predict where a previously unseen word could occur in a sentence/line, beyond spelling constraints?

## Source texts

- Austen, `Pride and Prejudice`, GITenberg 1342-0.txt, primary URL in [SOURCES](../../docs/SOURCES.md). Text begins at the first 'Chapter 1' and is cut before Gutenberg end matter.
- Darwin, `On the Origin of Species`, GITenberg 1228.txt, primary URL in [SOURCES](../../docs/SOURCES.md). The body begins at the final 'INTRODUCTION.' heading after the table of contents; end matter is removed.
- Optional cipher control: Naibbe `nathist_output_ciphertext.txt`, linked in SOURCES. Use **EVA glyphs**, not English letters, for its edits; the ciphertext is not a natural-language ground truth.
- Voynich ZL3b paragraph lines, strict per-folio A/B headers, EVA digraph and gallows compound units as in the original code.
- No entire transcriptions or books are committed; inputs must be downloaded locally.

English token normalization is lowercase `[a-z]+`, which splits apostrophes/hyphens. This should be treated as a controlled baseline, not a linguistically ideal tokenizer.

## Experiment 1: word-form prediction and 500 random deletion masks

```bash
pip install -r requirements.txt
python experiments/english_controls/benchmark.py \
  --austen data/1342-0.txt --darwin data/1228.txt \
  --naibbe data/nathist_output_ciphertext.txt \
  --tokens 11063 22843 --trials 500 \
  --out results/english_controls
```

Each corpus uses the first 11,063 or 22,843 tokens to approximately match earlier A/B token counts. Within the sample, 50% / 20% / 30% define the nested inner fit and outer train/test split. Edit candidates are generated using only the outer train vocabulary. The **500 deletion trials** remove 75% of *distinct training types*, then nominate top-500 supported mask-absent forms, including training words deleted by the mask (essential for replicating the Voynich experiment). Random ties are seeded.

Outputs: candidate CSVs with per-word support, 500-mask stability and logistic predictions; per-mask compositions (deleted training, later-attested, unattested); summary.json with candidate recall, ROC-AUC, average precision, matched degree+length concordance. Optional `--lexicon /path/to/words.txt` checks membership in a separately provided English lexicon; this is a deliberately imperfect reference (proper nouns, archaic forms, inflections, OCR artifacts).

### Initial exploratory screens (before the full Python rerun)

The following numbers were computed in a separate JS analysis of the original texts, using the same 70/30 prefix and character one-edit candidates. They represent **single-split screens**, not 500-mask output or verified results from this Python implementation.

| Corpus | Tokens | New candidates (degree>=2) | Held-out hits | Raw degree AUC | Raw letter-bigram AUC | Within-degree-and-length letter-bigram concordance |
|---|---:|---:|---:|---:|---:|---:|
| Austen | 11,063 | 22,725 | 42 | 0.703 | 0.765 | 0.756 |
| Austen | 22,843 | 38,280 | 61 | 0.714 | 0.784 | 0.748 |
| Darwin | 11,063 | 24,166 | 34 | 0.758 | 0.764 | 0.796 |
| Darwin | 22,843 | 39,096 | 62 | 0.701 | 0.739 | 0.811 |

Comparison: Voynich A (same earlier source-data screen) raw degree AUC 0.790, glyph-bigram AUC 0.831, matched concordance 0.879; B 0.785, 0.798, 0.885. Candidate base rates are different and sample definitions are not perfectly aligned: **these aren't calibrated cross-corpus effect sizes.**

## Experiment 2: line/sentence transition graph and unseen-form slot filling

```bash
python experiments/context/slot_filling.py \
  --zl data/ZL3b-n.txt \
  --austen data/1342-0.txt --darwin data/1228.txt \
  --out results/context
```

Training uses the first 70% of sentence/line sequences; evaluation uses the remainder. For Voynich, a sequence is an actual transcribed **paragraph line**, not an inferred sentence. English sequences are heuristically segmented sentences and assigned ~250-token pseudo-folio blocks for shuffle constraints. The two boundary definitions are not interchangeable.

Build a *directed, weighted* graph from each adjacent pair `previous -> next` in training sequences. Its raw top transitions are written to CSV. For candidates absent from train, transfer adjacent-context information through their one-edit neighbors: aggregate how often those neighbors occurred **after the known previous word**, **before the known following word**, and at sentence/line start/middle/end. Compare:

- Letter/glyph bigram spelling score.
- Left-prefix graph context alone.
- Left + right + position (cloze) graph context.
- Both graph scores with training tokens shuffled **within folio/pseudo-block and position bin** (a necessary position-preserving null).
- Spelling + actual context vs spelling + shuffled context.

The test takes the **first eligible occurrence of each genuinely new held-out candidate**, with previous and next tokens present in training. Rank the correct word against 9 randomly sampled forms with **exactly the same edit-graph degree and glyph/character length**. Report pairwise accuracy and top-1 rate; chance top-1 is 10%. Some unknown types are untestable and must be reported as selection limitations. Results are token-location-dependent conditional rankings, **not grammar or semantic decoding**.

### Required next experiments before publication

1. Re-run Python on original source and freeze SHA hashes; test results on independent book editions/transcriptions.
2. Cross-book English validation: train on Austen, evaluate in Darwin and reverse, so an unseen token isn't simply a story-specific name.
3. English independent wordlist and morphology checks to distinguish unattested but valid words from malformed proposals.
4. Multiple folio-blocked splits, cluster confidence intervals, permutation controls; match on frequency, initial/final units, glyph length, and degree.
5. Character trigrams, variable-order models, position templates, 2-edit candidates, MDL/finite-state generative baselines; then only a degree/position/morphology-matched graph/GNN ablation.
6. Determine if the prospective target is *a newly appearing type* or a **held-out token position**. These are distinct hypotheses and should not be collapsed.

Caveats: English plain text and Voynich EVA are not directly comparable on raw density; a 70/30 split by folio order isn't author chronology; lines aren't sentences; missing lexical attestations do not establish invalidity; a non-shuffled context effect could still reflect topic, copy-edit mechanisms or position.
