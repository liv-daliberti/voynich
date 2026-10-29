# Bidirectional word prediction at every line position

Exploratory results, October 28, 2026. **No decipherment, syntax identification, or claim that Voynich was written backwards.**

## Protocol and replication

Training: first approximately 70% of complete paragraph lines in each Currier variety, or first 70% of English heuristic sentences (Austen/Darwin 22,843-token prefixes). Also evaluate the published Timm–Schinner generated sample. Each held-out corpus evaluates **every target word** in 50 deterministically sampled complete lines/sentences. Same 701-type train-only vocabulary (700 word types plus an unknown token) in both directions. Smoothing chosen separately by direction from an inner 0.5/2/8/32/128 grid. Bidirectional word-conditioned fusion strength selected on inner validation data, not held-out targets.

The first-order model marginalizes all unknown intervening words exactly, in both directions. A two-sided product-of-experts score is **normalized separately for each missing target**, and is a *cloze pseudo-likelihood*, not a joint whole-sentence likelihood. All-position NLLs are directly comparable as conditional word prediction scores, but two-sided scores have access to more observations.

The [successful source-data run](https://github.com/liv-daliberti/voynich/actions/runs/40793700976) includes 700 folio/pseudoblock-cluster bootstrap resamples and exports per-position scores and fixed-target reveal trajectories. A newer run also exports paired interior position examples for inspection. CI intervals below are preliminary, not guaranteed well calibrated with 12–25 distinct held-out blocks.

## Held-out conditional NLL (lower better)

| Corpus | Held-out positions | Known-word target rate | Forward | Backward | Two-sided |
|---|---:|---:|---:|---:|---:|
| Voynich A | 447 | .669 | 4.520 | 4.573 | **4.474** |
| Voynich B | 488 | .701 | 4.667 | 4.666 | **4.578** |
| Austen | 1,051 | .808 | 4.267 | 4.332 | **3.850** |
| Darwin | 1,816 | .782 | 4.228 | 4.235 | **3.822** |
| Published Timm/Schinner sample | 456 | .706 | 4.732 | 4.654 | **4.608** |

**No robust overall backward advantage.** Backward-minus-forward NLL point and 95% clustered bootstrap CI: A +.053 [−.036,+.132], B −.000 [−.078,+.078], Austen +.065 [−.011,+.142], Darwin +.008 [−.034,+.046], procedural sample −.078 [−.177,+.021]. Positive means forward wins.

## What word order adds over a position-preserving shuffle

| Corpus | Forward NLL improvement | Backward NLL improvement | Two-sided improvement | Backward-specific improvement (95% CI) |
|---|---:|---:|---:|---:|
| Voynich A | +.0505 | +.0307 | +.0428 | −.0198 [−.0486,+.0072] |
| Voynich B | +.0590 | +.0999 | +.0859 | +.0409 [−.0072,+.0902] |
| Austen | +.7695 | +.7539 | +1.1941 | −.0156 [−.0891,+.0511] |
| Darwin | +.6704 | +.6663 | +1.0627 | −.0041 [−.0490,+.0350] |
| Timm/Schinner sample | +.0002 | +.0269 | +.0158 | +.0268 [−.0045,+.0607] |

Improvements are NLL(shuffle)-NLL(real), evaluated on identical held-out positions. The *backward-specific improvement* is the difference of backward and forward gains, paired by the same held-out lines and resampled by manuscript folio/English pseudo-block. None of these directional differences excludes zero in this first run. However, relative to the shuffled baseline, **Currier B has a numerically larger right-context signal**. This deserves independent replication, not a conclusion of backward grammar.

## Position-specific pattern

Separate the first/last word from interior positions, because manuscript line boundaries and English sentences are not directly equivalent.

| Currier variety | Region | Forward NLL | Backward NLL | Two-sided NLL |
|---|---|---:|---:|---:|
| A | First word | 4.186 | 3.990 | 3.882 |
| A | Early interior | 4.393 | 4.446 | 4.379 |
| A | Middle | 4.503 | 4.532 | 4.481 |
| A | Late interior | 5.206 | 5.277 | 5.208 |
| A | Last word | 3.831 | 4.176 | 3.818 |
| B | First word | 3.299 | 3.652 | 3.240 |
| B | Early interior | 4.859 | 4.717 | 4.731 |
| B | Middle | 4.802 | 4.812 | 4.748 |
| B | Late interior | 5.027 | 5.068 | 4.991 |
| B | Last word | 4.340 | 4.183 | 4.083 |

These data show heterogeneous position effects, not a simple right-to-left hierarchy. See the source artifact `summary.json` for first ten exact distances from either boundary and `*_reveal.json` for fixed-target traces.

**Critical coverage warning:** About 19–33% of held-out tokens in these samples are assigned to the single `<UNK>` category. Results for known-word forms must be reported separately, and a future open-vocabulary model should combine an actual glyph/character distribution with positional transition constraints. The current models do NOT predict spelling identities for held-out OOV words.

## Falsification and next tests

1. Increase the number of **folio-/quire-blocked splits** and compare multiple random seeds, stratified by Currier variety, hand and section. Current clusters are few.
2. Compare matched *observed-context lengths* and run a directional **permutation test** for (backward shuffle benefit)-(forward shuffle benefit).
3. Use cross-section, near-margin and line-length controls to distinguish genuine syntactic dependence from ending fillers and start-of-line prefixes identified by Currier and reviewed by Bowern and Lindemann.
4. Upgrade short-memory first-order models to modified Kneser-Ney, variable-order ngrams, class-based/anchor-HMM grammar and normalized morphology. Compute exact first-order fixed-target marginals, then approximate or exact longer-order marginals explicitly.
5. Test known-cipher and self-citation procedural controls at several generator parameter settings, matched by length, number of lines and word coverage.
6. Compare left-only, right-only and both-sides predictions **within each line position and known-word subset**. For bidirectional models, evaluate held-out masked-word conditional NLL; do not call it whole-line likelihood.

Reference context: Currier (1976), *The line as functional entity*; Bowern & Lindemann (2021), *The Linguistics of the Voynich Manuscript*; Peters et al. (2018), *Deep contextualized word representations*; Timm & Schinner (2020), procedural self-citation text.
