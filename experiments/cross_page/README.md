# Cross-page word-form reconstruction and Timm–Schinner generator ensemble

This experiment asks whether a **complete source page** predicts *new-to-that-source*
word forms on pages at distances 1, 2, 3, 5, 10, 20 and 40, and whether
the predictive relationship changes with source distance.

**Neither adjacent foliation nor source-to-target page order is guaranteed
historical composition order.** Scores measure editorial-order predictability
and cannot by themselves demonstrate self-citation or decipherment.

## Complete source-only pipeline

1. Parse 197 paragraph-bearing Currier A/B ZL3b folio sides (114 A, 83 B).
2. For page p, construct one canonical EVA-glyph edit proposal from every
   distinct *source-page* word; train alphabet and edit witness counts use
   **ONLY page p**. A target form cannot influence candidate generation.
   Require two distinct source witnesses per candidate.
3. Rank missing-from-source candidates by number of source witnesses; break
   ties deterministically by the spelling itself (there is no target label
   in this selection). Nominate 50 candidates per source page.
4. On target page q, define gold as **word types in q absent from p**. Record
   top-50 hits, precision per 50, recall over gold novel types, fraction of
   gold forms reachable in the source-generated candidate universe, and
   expected hits for 50 uniform random picks from the *same candidate set*.
5. Aggregate over sources and targets at each requested page distance,
   in original physical folio sequence and separately in same-Currier-variety
   subsequences. In the latter, p+1 denotes the **next same-variety folio**,
   not necessarily the immediately next physical folio.
6. Compare source p's predictions with those of later source pages under
   reversed order as a placebo.
7. Randomize page sequence 300 times within each Currier stratum, retaining
   each page's complete word inventory and source model. For physical
   foliation, shuffle only within original A/B positions to control for
   broad variety distribution.
8. Strong *same-target* control: for target q at least 20 same-variety
   positions from the beginning, compare q-1 as source with an earlier
   source >=20 folio positions away selected by nearest distinct-type
   count and then token count. The target and evaluator are identical.
   Compute paired top-50 precision and excess above same-candidate-set
   random expectation with exploratory bootstraps over targets.

## Original generator ensemble, NOT a surrogate

Use Torsten Timm's published Java `text-generator.jar` at upstream commit
`a6ede2202dd7ad6285ce2c007bf22c2a0e7709b7` (2019-11-29).

- Fixed published morphology rules, word source selection, glyph restrictions,
  initial line, suggestion parameters and pseudo-random implementation.
- Five new pseudo-random seeds: 19, 23, 29, 31 and 37.
- `text.lines_to_create` raised from published default 1,200 to **5,000**
  so every freshly generated output contains enough tokens for the 197
  matched page budgets. This is the *same software/method*, but not exactly
  the default-length configuration.
- Primary comparison: segment each generated text into the **exact ZL3b
  folio token budgets**, preserving the 197 A/B labels only as matched
  cohort-size strata, **not implying that generated pages have Currier
  A/B grammar**.
- Important sensitivity: also respect the generator's NATIVE
  `text.lines_per_page=29` boundaries and run the entire distance
  analysis on native generated pages; these should not be mixed with
  the matched-budget result.
- Every run saves the original generator jar SHA256, Git commit SHA, final
  configuration, source text SHA256 and summary output.
- Austen (public-domain *Pride and Prejudice*) and Naibbe ciphertext also
  receive identical matched page budgets and the exact same measurements.

## Critical measurement caveat

The number of words novel relative to a source page tends to grow with page
distance. Thus raw top-50 **hit precision is not necessarily monotone** with
distance even if source-derived novelty becomes less predictive! Report
**novel-type recall, reachable fraction, candidate-set-adjusted expected
hits and paired same-target distant controls** together. Candidate ranking
also prioritizes high-degree forms and can be dominated by frequent,
morphologically central words.

A copy/edit process can produce local similarity in both directions even
if the author physically copied prior words; comparing p to p+1 versus p-1
does not directly identify the causal direction. The generation mechanism
itself is established only for *synthetic Timm* output, not for Voynich.

Folio order, region, writing hand, section and transcription disagreements
must be controlled in subsequent blocked analyses. Five seeds are an
independent procedural ensemble, not 5 independent manuscripts.
Randomization p-values test page-order exchangeability conditional on
the specific observed page content, not a complete causal hypothesis.

## Reproduce

GitHub Actions: `.github/workflows/cross-page-reconstruction.yml`, with
eight separate jobs for real Voynich, Austen, Naibbe and five genuine
Timm–Schinner generated manuscripts. Data source hashes and model outputs
are archived as workflow artifacts.

Manual syntax:

```bash
python experiments/cross_page/reconstruct.py \
  --zl data/ZL3b-n.txt --mode voynich \
  --shuffles 300 --out results/cross_page/voynich.json

python experiments/cross_page/reconstruct.py \
  --zl data/ZL3b-n.txt --mode timm --seed 23 \
  --generated data/timm_seed23.txt --shuffles 300 \
  --out results/cross_page/timm23.json
```

Use script `tests/test_cross_page.py` for target-leakage, exact-lag,
same-target far-control and native pagination invariants.

See `REPORT.md` for completed comparison and results.
