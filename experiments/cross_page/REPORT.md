# Cross-page reconstruction: Voynich versus five new Timm–Schinner manuscripts

Reproduced October 28–29, 2026. This is **predictive evaluation of words absent from a source page**, not actual historical copying proof or decipherment.

## Software and provenance

- Full source ZL3b transcription, 197 paragraph-bearing folio sides (114 Currier A; 83 B). The within-variety page distance p+1 means *next folio assigned to that Currier variety*, not necessarily physically adjacent. The code ALSO computes literal physical-order all-folio distances, retaining observed A/B slots for permutations.
- Five **fresh** generated manuscripts using Torsten Timm's original public Java `text-generator.jar` at pinned upstream 2019-11-29 commit `a6ede2202dd7ad6285ce2c007bf22c2a0e7709b7`; identical glyph/morphology/source chooser/initial-line settings and seeds 19,23,29,31,37. Increase line-count setting from default 1,200 to 5,000 for a sufficiently long output; **not the literal original-length default**.
- The primary generated comparison repaginates the sequence to the *exact 197 ZL3b page token budgets*, copying observed A/B labels as page-size comparison strata only. These labels have no Currier linguistic meaning in generated text.
- An **essential sensitivity test** runs the same experiment on the generator's actual 29-line native pages (173 pages per seed).
- Public-domain Austen and Naibbe ciphertext are split into identical real-folio token budgets as additional controls.
- Each source page trains its own canonical one-EVA-glyph edit proposal model from ONLY its vocabulary/glyph alphabet. Novel candidate types need >=2 distinct one-edit neighbors, rank by support, top 50. Predictions and support are fixed before evaluating any target page.
- For p+d with d in 1,2,3,5,10,20,40, score the proportion of **target-page types absent on p** recovered, plus top-50 precision, coverage of candidate space, expected hits from uniform random same-pool ranking, lexical overlap and reverse-page placebo.
- Nulls: 300 permutations of page order within observed Currier strata, and a *paired same-target* comparison between source page one position earlier and a source at least 20 earlier chosen by nearest source type count (then token count). These are observational and imperfect controls.

Source code: `experiments/cross_page/reconstruct.py`. Run: [GitHub Actions](https://github.com/liv-daliberti/voynich/actions/runs/40799074915). Raw input hashes, original generator commit and final configs archived with the JSONs in separate workflow artifacts.

## Main distance comparison (same-variety indexes; 50 nominated forms per source)

**Novel-to-source target-form recall**, page gap 1 vs 20:

| Cohort | Lag 1 | Lag 20 | Lag-1 minus lag-20 |
|---|---:|---:|---:|
| Voynich A | 7.7403% | 6.8566% | +0.8837 pp |
| Voynich B | 5.3305% | 4.5138% | +0.8167 pp |
| Timm A-sized pseudo-pages, mean of 5 seeds | 7.4208% | 5.5067% | +1.9141 pp |
| Timm B-sized pseudo-pages, mean of 5 seeds | 5.2165% | 4.0520% | +1.1644 pp |
| Austen A-sized | 2.4772% | 2.8054% | -0.3282 pp |
| Naibbe A-sized | 4.5774% | 4.7229% | -0.1455 pp |

For the initial **lag 1** model, Voynich A true novel recall 0.077403 vs page-order shuffled 0.068840, B 0.053305 vs 0.046843. One-sided p=.003322 from 300 page-order permutations for each. This is evidence of **local folio-order predictability**, not a direct test of a physical writing sequence.

The generator's matched-pseudo-page recall decline appears across all five seed draws. Mean A-sized decay +1.914 pp (individual-seed range +1.384 to +2.528 pp); B-sized mean +1.164 pp (range +0.707 to +1.999 pp). Generator seed variation does not measure real-manuscript uncertainty.

## Critical pagination check

Using the **actual Timm generator page boundaries** (29 lines per page, 173 pages per seed) instead of matching Voynich token budgets:

- Novel-form recall lag 1: mean **3.9275%**;
- Novel-form recall lag 20: mean **3.7556%**;
- Lag 1 minus lag 20: only **+0.1719 pp**.

This is **much weaker** than its matched-folio pseudo-page decay. Page segmentation and source/target length are material confounds. The five generator outputs therefore show that copy/edit generation **can produce** such decay with one pagination choice, but have not established that the original method reproduces the precise Voynich profile under its native pagination.

## Same-target, size-matched far-source test

Evaluate precisely the same target page with (1) source one same-variety page earlier and (2) a prior source >=20 same-variety folio positions earlier selected to match source word-type count:

| Corpus | Same-target lag-1 vs matched-far top-50 precision gain |
|---|---:|
| Voynich A | +0.010213; naive target-bootstrap interval [-0.000537,+0.020963] |
| Voynich B | +0.002540; interval [-0.014286,+0.019048] |
| Timm A-sized 5-seed mean | +0.009787 |
| Timm B-sized 5-seed mean | +0.005206 |
| Austen A-sized | -0.006596 |
| Naibbe A-sized | -0.000426 |

The point estimates are compatible with a small local source benefit, but **both Voynich intervals cross zero**. The B source match has a mean source-vocabulary size difference of 33 types, indicating imperfect matching. Individual targets share folios, so these resampling intervals are optimistic, not publication-grade cluster inference.

## Why raw top-50 hits do NOT simply decrease with distance

Gold targets are defined as word types on the target page *absent from the source*. More distant pages usually have more such types. As a result, raw top-50 **precision may increase** at distance 20 even if the percentage of unseen forms recovered decreases. For example, Voynich B top-50 precision is 0.123415 at lag 1 but 0.137143 at lag 20. Never interpret this as greater distant-source predictive relevance. Report novel-form recall and candidate-set-adjusted random expectation, alongside paired same-target comparisons.

## Strength of evidence and next stage

- Strong: the generator is **authentic released software**, not our approximation; independent seeds produce reproducible local decay on matched pseudo-pages.
- Supportive, not decisive: Voynich A/B lag-1 recall exceeds their within-stratum page-order permutation null.
- Weak: ranking proximity over same-target size-matched distant sources; effect intervals include zero.
- Not shown: actual historical word copying, causal text-generation order, semantics or decipherment.
- Major open risk: generated page segmentation. Native and matched page structures differ, and real folio labels are only editorial-order observations.
- Next research: prespecify a distance-decay statistic such as within-target *conditional rank* adjusted for number of target novel forms; perform a **block/quire-aware** page-order shuffle preserving section and source-size bins; compare native generated page sizes with size-matched *real* subsets; test source retrieval within and across physical lines and differentiate horizontal/vertical proximity. Evaluate independent transcription and multiple original generator parameterizations.

Full individual lag curves, far matched comparisons, five generator configurations/hashes and native sensitivities are in the archived artifacts and consolidated result package.
