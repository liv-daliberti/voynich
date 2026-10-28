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

### Completed 500-mask Python replication

Reproduced in [GitHub Actions run 40781058172](https://github.com/liv-daliberti/voynich/actions/runs/40781058172), with raw source hashes and detailed candidate-level CSVs in its uploaded artifact.

**Correction:** a prior exploratory JavaScript screen counted certain duplicate edit constructions from the same source type as independent neighbors. The committed Python implementation correctly deduplicates proposals per source word. Its numbers supersede the preliminary English table. The earlier Voynich-only proposal generator already used set deduplication.

| Corpus | Sample tokens | Novel candidates (degree >= 2) | Later-attested candidates | Degree+length AUC | +bigram AUC | 500-mask mean later-attested top-500 |
|---|---:|---:|---:|---:|---:|---:|
| Austen | 11,063 | 13,590 | 42 | .704 | .810 | 3.436 |
| Austen | 22,843 | 23,410 | 59 | .690 | .810 | 3.816 |
| Darwin | 11,063 | 13,635 | 34 | .756 | .830 | 4.036 |
| Darwin | 22,843 | 22,956 | 62 | .630 | .796 | 4.050 |
| Naibbe (EVA) | 11,063 | 93,071 | 546 | .858 | .939 | 41.540 |
| Naibbe (EVA) | 22,843 | 155,148 | 697 | .920 | .959 | 33.876 |

*ROC-AUC is prospective attestation prediction, not probability of linguistic validity.* Naibbe is ciphertext, yet exhibits stronger performance in this setup than English, illustrating why these statistics do **not** establish natural-language syntax. Base rates, candidate pools, source lengths, and encoding differ greatly.

The full-source data hashes in the archived artifact are:
- Austen SHA256 `48e0522844402a86a3ea98f0947ba85ea54838db3c59022299298ed968a5a163`
- Darwin SHA256 `82ee88e28d5f4662d03d2bfd09330d79cf860c090930d1774e23aa1609bf3730`
- Naibbe SHA256 `9cdf2de12f371ac7efdb2e78713f229ada508286c1717758184238a59cd64326`
- ZL3b SHA256 `bf5b6d4ac1e3a51b1847a9c388318d609020441ccd56984c901c32b09beccafc`

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

### Completed unseen-form slot-filling result

For 172 Currier A and 334 Currier B distinct new-form contexts (first eligible occurrence per form), 9 distractors were matched on exact length and edit degree. On *pairwise accuracy*:

| Model | Voynich A | Voynich B |
|---|---:|---:|
| Spelling bigrams alone | .802 | .834 |
| Neighbor context: left only | .546 | .546 |
| Neighbor context: left + right + position | .612 | .552 |
| Neighbor context with position-preserving shuffles | .603 | .534 |
| Spelling + actual neighbor context | .773 | .743 |
| Spelling + shuffled neighbor context | .774 | .729 |

The neighbor-context score performs close to its shuffled control, and combining it with spelling lowers prediction performance relative to spelling alone. On *English unseen-form tests*, context transfer also fails to give a consistent improvement and only 24–54 eligible cases are available per sample. This is a **negative result for this specific neighbor-based estimator**, not a demonstration that Voynich has no syntax. The later added **seen-word positive control** separately checks whether ordinary English word-order information is detectable by the pipeline.

### Required next experiments before publication

1. Re-run Python on original source and freeze SHA hashes; test results on independent book editions/transcriptions.
2. Cross-book English validation: train on Austen, evaluate in Darwin and reverse, so an unseen token isn't simply a story-specific name.
3. English independent wordlist and morphology checks to distinguish unattested but valid words from malformed proposals.
4. Multiple folio-blocked splits, cluster confidence intervals, permutation controls; match on frequency, initial/final units, glyph length, and degree.
5. Character trigrams, variable-order models, position templates, 2-edit candidates, MDL/finite-state generative baselines; then only a degree/position/morphology-matched graph/GNN ablation.
6. Determine if the prospective target is *a newly appearing type* or a **held-out token position**. These are distinct hypotheses and should not be collapsed.

Caveats: English plain text and Voynich EVA are not directly comparable on raw density; a 70/30 split by folio order isn't author chronology; lines aren't sentences; missing lexical attestations do not establish invalidity; a non-shuffled context effect could still reflect topic, copy-edit mechanisms or position.
