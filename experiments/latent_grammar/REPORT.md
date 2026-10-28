# Fixed-future-position and structural graph-alignment experiments

Exploratory research, 28 October 2026. **No decipherment, translation, or grammar-identification claim.**

## Reproduction

The full [source-data run 40785524667](https://github.com/liv-daliberti/voynich/actions/runs/40785524667) passed and archived summaries, word-lattice traces, HMM traces, cross-corpus graph matching, and source SHA256. Unit tests (including exact marginalization and positive-label controls) passed.

Five corpora: ZL3b Currier A/B paragraph lines, Austen and Darwin (first 22,843 words each), and the **published generated-text sample** from Timm and Schinner's self-citation generator (not a new replication of their generator).

## 1. Fixed future word: exact first-order word lattice

For each held-out line/sentence, choose target index `T∈{3,5,7,9}`, reveal positions `0,...,t-1` progressively, and score the *same* word `W_T`. All intermediate word-state paths are exactly marginalized using a normalized sparse, smoothed forward operator. **The 0-gap row means there are no unseen intermediate words and exactly one remaining transition**. The first-order model loses contextual memory rapidly by design.

300 held-out targets per corpus, same targets across actual/position-shuffled training controls. Top 700 training word types plus `<UNK>`; inner training validation selects smoothing from 0.5,2,8,32,128. These vocabulary and tuning choices differ from our earlier 160-event prefix experiment.

| Corpus | Selected smoothing | No-intermediate-word NLL (actual) | NLL after positional shuffle | Mean log-probability gain vs no prefix |
|---|---:|---:|---:|---:|
| Voynich A | 128 | 4.47440 | 4.50096 | +0.00532 |
| Voynich B | 128 | 4.60685 | 4.65153 | −0.01087 |
| Austen 23k | 32 | 4.36000 | 5.11462 | +0.42055 |
| Darwin 23k | 32 | 4.16285 | 4.80819 | +0.34396 |
| Published Timm/Schinner sample | 128 | 4.63151 | 4.67093 | −0.02221 |

Log probability gain = mean[log P(real W_T | revealed words) − log P(real W_T | empty prefix)]. The shuffled comparison uses separately fit, position-preserving shuffled training text, not a paired manual annotation baseline. Larger log-probability gains for English are consistent with substantial immediate predecessor information. Voynich gains are weak and the score can deteriorate. **Do not interpret this as Voynich having no grammar**, nor infer a causal mechanism from this one split.

For the **same fixed target index T=3**, mean NLL by observed prefix depth t:

| Corpus | t=0 | t=1 | t=2 | t=3 (immediate predecessor known) |
|---|---:|---:|---:|---:|
| Voynich A (101 target positions) | 4.48067 | 4.48069 | 4.48421 | 4.50298 |
| Voynich B (99) | 4.64400 | 4.64389 | 4.65082 | 4.67254 |
| Austen (77) | 4.79800 | 4.78833 | 4.69539 | 4.44507 |
| Darwin (77) | 4.43911 | 4.41570 | 4.30246 | 4.03157 |

Thus the true target gets more predictable as words immediately before it are revealed in English, but this does not hold with this first-order model on Voynich. Critically, this is a comparison of predictive distributions for **the same target position**, not different next-token positions.

## 2. Six-state hidden grammar and structural transfer

A separately fitted six-state unsupervised HMM computes the fixed-target distribution exactly over latent state paths. **Negative control result:** it barely detects real-vs-shuffled sequential dependencies even in English. For Austen held-out no-gap target NLL is 4.36033 real versus 4.36021 shuffled; for Darwin 4.30807 versus 4.30876. So this simple HMM is *not* a successful grammar induction model, and small gains after transplanting English transition matrices into a Voynich emission model cannot establish transferable grammar.

State alignment uses anonymous distributional descriptors only, with Hungarian matching. Without verified anchors or successful English positive controls, no semantics can be attached to the anonymous classes.

## 3. Whole-word context graph alignment with known English substitution keys

To test a meaning-free, structure-only analogy mechanism, first split each English control into disjoint halves, randomly rename every word in the second half with a fixed unique code, then attempt to recover the **true hidden identity** from frequency, line position, predecessor/successor entropy, and neighborhood graph patterns. No letters, word length or English meanings enter the matcher.

| Known renaming experiment | Left top types | Mappable gold types in other half's top 70 | Exact recovered | Frequency-rank-only recovered |
|---|---:|---:|---:|---:|
| Austen | 70 | 62 | 40 | 8 |
| Darwin | 70 | 55 | 32 | 6 |

This is a meaningful positive control for *this controlled identity matching setup*, not a general bilingual decipherment result. Both sides are from the same book, just renamed, and their underlying vocabulary overlaps strongly.

Apply analogous matching to Voynich, but **do not call the resulting pairs translations**. Transition-pattern cosine similarity after graph matching (higher is closer):

| Structural comparison | Observed Voynich order | Voynich position-preserving shuffle |
|---|---:|---:|
| Austen → Voynich A | 0.259 | 0.177 |
| Austen → Voynich B | 0.231 | 0.191 |
| Darwin → Voynich A | 0.273 | 0.220 |
| Darwin → Voynich B | 0.179 | 0.206 |

The actual structure scores higher than the shuffle in three out of four comparisons, but not Darwin→B. Because the matching is optimized on observed graph structure, nonzero similarity is *expected even for unrelated corpora*. This is insufficient to conclude that an English part-of-speech category has a Voynich counterpart, let alone that any word translates to an English word. Report match-stability across independent splits, multiple random seeds, other languages and procedural controls before interpretation.

## Priority next tests

1. Multiple folio-block/section folds and uncertainty intervals for **differences between real-vs-shuffled NLL**, not only point estimates.
2. Broader smoothing search in Voynich and generated sample: best alpha of 128 is the largest tried, suggesting the optimum may lie beyond the grid.
3. Compare against Brown/anchor-HMM/regularized HMM and class-based trigram models for **English-positive grammar induction**, before transferring states to Voynich.
4. For the future-target task, extend context from a first-order state to explicit pair-state trigrams, probabilistic suffix trees and morphology/position-aware finite-state models. Check held-out fixed-target log loss, not visual plausibility.
5. Test structural matching on independent Austen/Darwin windows and a **known artificial word-substitution cipher**, then on generated self-citation and cipher controls. Quantify stability/abstention and false matches.
6. A genuine Voynich→English translation hypothesis requires a bilingual anchor or independently verifiable constraints. Graph similarities alone cannot identify word meanings.

Primary sources: Bowern & Lindemann (2021); Timm & Schinner (2020); Conneau et al. (2018); Søgaard, Ruder & Vulić (2018); Stratos, Collins & Hsu (2016).
