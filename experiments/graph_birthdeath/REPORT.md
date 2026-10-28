# Exploratory graph and birth/death study, October 28, 2026

**Input:** ZL3b-n 2025, paragraph-only, strict per-folio Currier labels (unknown excluded), EVA units `ch, sh, ckh, cth, cph, cfh`. Never inherit language from previous page.

## All-pairs edit graph

| | Currier A | Currier B |
|---|---:|---:|
| Tokens | 11,063 | 22,843 |
| Types | 3,001 | 4,557 |
| One-glyph edit edges | 12,654 | 19,466 |

## Held-out context prediction, bits per context target, lower better

Five leaf-held-out folds, target is the last two EVA units of previous word / first two units of next word. Baseline uses word identity with first/last glyph and length-bin smoothing; graph model pools word-neighbor context; randomized graph preserves degree and morphological endpoint signatures. Inner-fold smoothing grid 0, 2, 6, 12, 24, 48 (48 chosen for every fold, suggesting larger grid needed).

| Model | A | B |
|---|---:|---:|
| Word+shape baseline | 4.0340 | 3.6239 |
| Matched rewired graph (nine nulls per fold) | 3.9209 | 3.5261 |
| Actual one-edit graph | **3.9137** | **3.5188** |

Actual graph gains about **0.0072 bits** per context target over morphology/degree-matched rewiring in each variety, favorable in all 5 folds. Very modest, exploratory, and **not evidence of a semantic lexicon**.

## Pseudo-dynamic nodes and edges in current foliation

Active words occur within a block of five nonoverlapping paragraph-bearing folio sides; edges active when both endpoint words present.

| | A | B |
|---|---:|---:|
| Mean active word types | 287.7 | 587.2 |
| Mean new appearances | 204.5 | 408.1 |
| Mean disappearances | 207.1 | 392.9 |
| Mean node Jaccard | 0.1642 | 0.1961 |
| Mean edge Jaccard | 0.1575 | 0.1940 |

A birth count was lower than the within-hand×section-shuffled null (210.85, exploratory one-sided permutation p=0.004, 250 permutations). But node-overlap did **not** pass the same null (p=0.155), and the ten-folio-window effect faded (birth p≈0.15). B showed no comparable persistence. The current folio order is not a verified chronological sequence, and a universal stationary birth/death mechanism is not supported.

## Graph robustness and missing types

10% of all observed word types removed at random: ~87.5% A and ~88.3% B of removed types have a surviving one-edit neighbor (40 trials). In a harder heldout missing-node test, generate all canonical single-EVA-edit variants from *surviving types only*. Rank up to 500 candidate types by number of surviving one-edit witnesses, tie-break by train-only glyph model; compare with glyph-likelihood ranking among identical candidates.

Top-500 missing-type recall, 3 independent draws: A **22.7%** (graph) versus **12.1%** (glyph); B **20.9%** versus **7.2%**. The deleted items are randomly removed **attested** forms, not unknown historical text or meanings.

## Limitations / falsification

- Most graph-smoothing benefit is shared by similarly constrained rewired graphs.
- The chosen graph smoothing at the edge of its grid indicates under-tuning.
- Frequency and graph degree are strongly linked; node deletion favors frequent morphologically central types.
- Births/deaths reflect sampling and hand/section transitions, not historical time.
- Direct independent-transcription and natural-language/cipher network replication remains pending.
- Original-data results computed in connected source-analysis environment; downloadable Python test fixtures passed, but no independent full-corpus Python rerun within offline runtime.

The reproducible Python modules are in the conversation study archive. Sources: [ZL3b-n](https://www.voynich.nu/data/ZL3b-n.txt), [Beinecke MS 408](https://collections.library.yale.edu/catalog/2002046).
