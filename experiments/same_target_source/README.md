# Quire- and section-aware permutation controls

This experiment decomposes previously observed **nearby-page advantages**
into contributions explained by Currier variety, illustration/section,
quire, hand, and word-type-count matching.

The source-data input is the original ZL3b transcription, restricted to
197 A/B paragraph-bearing folio sides. The code fixes an important metadata
error in earlier exploratory work:

| ZL header | Correct meaning | Usage |
|---|---|---|
| `$L` | Currier A/B label | Base stratum |
| `$I` | Illustration/section category (H, B, S, ...) | Section stratum |
| `$Q` | Quire identifier (A, B, ...) | Quire stratum |
| `$H` | Hand | Writer/hand stratum |
| `$F` | Folio position within quire | **NOT section** |

The earlier `page_graphs/compare.py` mapped `section_code=$F`.
Any prior **section-stratified** comparisons depending on that mapping
are superseded by this correctly parsed study.

## Graph outcome

Create the one-EVA-glyph edit graph for each page using the full observed
vocabulary. For every pair of pages, compute **word-type Jaccard**,
one-edit-edge Jaccard and cosine similarity between the distributions of
page graph degrees (coarse unlabeled topology).

Within Currier A and B separately, compare mean graph similarity for pages
1–3 positions apart with pages 20+ positions apart. The test statistic is
near-minus-far. Note: the position index in this metric is **within each
Currier subsequence**, not necessarily physically adjacent folios.

The permutation null reassigns entire pages to positions, never altering
their internal words or graphs. Evaluate six nested exchangeability strata:
1. Currier variety.
2. Variety and section (`$I`).
3. Variety and quire (`$Q`).
4. Variety, section and quire.
5. Variety, section, quire and hand.
6. The preceding four plus approximate source-vocabulary-size tertiles
   **within** strata having six or more pages.

The script reports number of movable pages, group sizes, actual pages
changed per permutation, observed statistic, null mean and 95% null
quantiles, exploratory one-sided Monte-Carlo p-values. The first run used
120 permutations per null. A follow-up uses **2,000 repetitions** for the
more constrained quire-aware graph nulls to resolve the small B residual.

A quire-aware shuffle preserves between-quire shifts by design. Therefore,
if its null reproduces almost all of the observed near-far difference,
the naive page-order finding may be explained by **quire/section block
composition** rather than predictable differences between successive
pages *within* blocks.

## Same-target copying/reconstruction outcome

Use `experiments/same_target_source/evaluate.py` and its source-only
candidate model. For a fixed target page t, compare t-1 to an earlier
source >=20 same-variety folio positions away, matched to the near source
within 25% of distinct-type count (and closest token count). For both
sources, the **gold target set is identical**: words in the target page
absent from **both** sources. No target forms enter source selection or
candidate nomination.

Report delta raw target recall, delta coverage of source-generated
one-edit alternatives, delta recall above uniform-random ranker within
each source's candidate set, and rank percentile among target forms
proposed by both sources.

The candidate source match is repeated with three levels of strictness:
- same Currier variety only;
- same variety and section, with source t-1 and target section-matched;
- same variety, section and hand, with source t-1 and target matched.

For each of the six page-order nulls, recalculate the matched source
selection and compare the chance-adjusted recall delta with permutation
expectation. Report **number of eligible fixed targets** in real and
permuted orders: strict B strata can yield very few targets, invalidating
strong claims. Page-order permutations change target assignment to
positions, so they are not themselves a strict target-identity-fixed
permutation; the source-comparison within each observed order does keep
the exact target fixed.

## Run

```bash
python experiments/same_target_source/quire_section_nulls.py \
  --zl data/ZL3b-n.txt \
  --out results/quire_section/summary.json --permutations 120

python experiments/same_target_source/quire_section_nulls.py \
  --zl data/ZL3b-n.txt --graph-only --permutations 2000 \
  --nulls currier_quire currier_section_quire \
          currier_section_quire_hand \
          currier_section_quire_hand_size \
  --out results/quire_section/high_resolution_graph.json
```

GitHub Actions `.github/workflows/quire-section-nulls.yml` runs both,
records the original source SHA256, and uploads results. Synthetic tests
verify the correct header parser and ensure no page crosses a preserved
metadata grouping. A high p-value does **not** prove that ordering is
random, because the null may preserve quire boundaries and therefore
explain much of the original variation.

Folio order is not established production chronology. `$I` reflects
illustration category, not verified grammatical topic or author intent.
Only 197 A/B paragraph-bearing sides are evaluated; missing folios and
transcription ambiguities remain possible sources of bias. Results do
not identify a copy-edit mechanism or decipher any manuscript text.
