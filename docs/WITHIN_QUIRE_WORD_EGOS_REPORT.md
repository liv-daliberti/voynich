# Within-quire page prediction and word-centered edit graph evolution

Reproduced on 29 October 2026 from ZL3b EVA; manuscript foliation is **not** established historical writing order. All hypotheses and p-values are exploratory.

## 1. Within-quire *same-target* prediction

Protocol in `experiments/within_quire/predict.py`. For each target folio side with at least two earlier folios in the **same quire** (and Currier variety), compare (a) immediately previous same-quire folio versus (b) an earlier source at least two within-quire positions behind, chosen by nearest training type count, with <=30% relative source-size discrepancy. Both sources are evaluated against the **identical gold set of target types absent from BOTH sources**. Training-only edit-witness models rank up to 50 forms; report recall after subtracting chance expectation from a random ranker over each source's candidate pool, as well as raw recall, coverage, word/edge Jaccard and percentile ranking of jointly supported forms.

The comparison is run both with every eligible target and requiring near-source and target to share illustration category and hand. Randomization shuffles entire folios within a quire, then additionally within section and hand, then also within source-size tertiles (where grouping size permits). Confidence intervals are bootstrapped **by quire**, not treating word candidates as independent. The strongest null still moves about 141 of the 197 analyzed folio sides per permutation.

[Final source-data CI workflow (all jobs succeeded)](https://github.com/liv-daliberti/voynich/actions/runs/40808005107), with 350 Voynich and 200 Austen/Timm shuffles and per-target JSON. Matched Austen and a **fresh Timm/Schinner 5,000-line Java generator sample, seed 29** are repaginated to the 197 real folio token budgets and inherit the corresponding synthetic quire and section grouping solely as experimental controls. **They do not possess actual Voynich quire assignments.**

### Observed chance-adjusted near-vs-earlier-source recall advantage

| Cohort | Targets | Gain | Bootstrap 95% CI |
|---|---:|---:|---:|
| Voynich pooled | 142 | **+.008123** | [.003582, .013454] |
| Voynich Currier A | 82 | **+.011797** | [.005999, .018495] |
| Voynich Currier B | 60 | +.003102 | [-.005248, .008641] |
| Austen pseudo-quire pooled | 139 | +.000015 | [-.002800, .002729] |
| Timm pseudo-quire pooled | 139 | **+.008156** | [.002691, .014811] |

### Strongest order-null: preserve within-quire section, hand and source-size bin

| Cohort | Actual minus shuffled expectation | One-sided permutation p |
|---|---:|---:|
| Voynich pooled | +.008185 | .008547 |
| Voynich A | **+.011274** | .008547 |
| Voynich B | +.003964 | .113960 |
| Austen pooled | +.000200 | .452736 |
| Timm pooled | **+.007717** | .014925 |

With near-source and target additionally constrained to match section and hand: Voynich A +.011382 on 79 target folios (size-aware order-null p=.0114); Voynich B +.004609 on 52 target folios (p=.157). The effect is strongest and most convincing in A, **not B**.

These are hypothesis-generating findings. The permutation tests were selected after multiple exploratory model comparisons and are not corrected for multiple testing; a matched next-folio source can preserve vocabulary information without being the actual historical copied source. Timm's matching result shows a local copying mechanism *can produce* the pattern; it does not distinguish that mechanism conclusively from genre/topic persistence or another constrained generator.

## 2. Build a graph around EVERY word type

Implemented in `experiments/word_egos/evaluate.py`. Every observed word form is a center of a **global one-glyph edit graph** (canonical compound EVA units), with a one-hop ring and a two-hop ring; record the nodes and induced edges of the one-ring. The global graph uses ALL ZL3b pages, so it is *descriptive*, not a held-out prediction.

- **6,481 distinct observed word types** receive complete global edit-ego graph JSON records (`all_word_egos.jsonl`).
- **22,875 word-on-folio appearances** receive graph snapshots (`all_word_folio_egos.jsonl`): the actual neighboring words, between-neighbor edges, triangles and two-hop words visible on that folio.
- `word_summary.csv`, `word_ego_per_folio.csv`, and `selected_word_histories.json` contain per-type distributions and page trajectories.
- **25 Graphviz DOT files** are produced for frequent centers, including `daiin`, `chol`, `chedy`, `qokeey`. The downloadable standalone interactive HTML explorer allows browsing any of the 6,481 global centers and all 197 folio sides, including page-to-page shifts.
- A center need not be observed on a page for >=2 of its neighboring words to be present. We record this as **ghost support**, never asserting that the absent word is grammatical or was written. For example, `oor` occurs on only 2 folios but has >=2 observed one-edit neighbors on 161 other pages, demonstrating how permissive graph support can be.

Concrete word examples: `daiin` has 47 global one-edit neighbors, but averages 3.41 visible neighbors per folio where the center occurs; `chol` has 61 global neighbors with average 5.96 per observed folio; `chedy` has 65 global neighbors and average 9.84.

### Does the ego network change predictably within a quire?

For the **180 high-document-frequency / high-degree centers**, compare observed one-edit-neighborhood set Jaccard between pages 1–2 quire-local positions apart and pages >=3 positions apart, requiring the center to be present on both pages. To avoid having ubiquitous words dominate, estimate the nearby-minus-distant difference for each center and then average across centers. Compare with 350 permutations of whole pages, preserving quire, optionally section and hand.

| Outcome | Mean near-minus-far neighbor overlap |
|---|---:|
| Observed | **−.004096** |
| Within-quire order-shuffle mean | −.012404 |
| Observed **excess over shuffle** | **+.008308** |

One-sided p=.0342 (quire only), p=.0456 (quire + section), p=.0370 (quire + section + hand). These are uncorrected *word-center selection* and *page-order* comparisons and do not isolate causal copying or grammar. The negative raw near-minus-far value emphasizes that we are detecting a small reduction in local neighborhood discontinuity **relative to shuffle**, not straightforward increasing similarity of consecutive pages.

## 3. Research interpretation and limitations

- An informative prior-page word graph within the same quire is a stronger prospective statement than generic nearby-page word overlap. It remains conditional on word frequency, page-size matching, vocabulary variation, arbitrary page order and overlapping edit families.
- The strongest observed within-quire effect is in **Currier A**, even though earlier unadjusted page graph similarity was stronger in B. The two analyses ask different questions.
- A fresh real Timm Java generator sample replicates the pooled within-quire source-selection benefit, an essential non-semantic procedural positive control. One generated seed is insufficient to estimate generator variability for this new within-quire statistic. Repeat at least five fresh seeds.
- Austen shows some near-page word Jaccard advantage but **no meaningful within-quire source edit-rank advantage** under this setup. Spelling edit units differ between English letters and EVA.
- Per-word egos describe the entire observed vocabulary; global graph contains future information relative to an early folio. Do not reuse these graph snapshots as prospective features without recomputing from earlier training-only pages.
- The variance and null p-values for the 180 selected centers are not family-wise corrected. Center words are linked in the same vocabulary graph and are not independent observational units.
- Quire membership is a codicological grouping, not manuscript production chronology. Illustration category `$I`, quire `$Q` and hand `$H` are used correctly; `$F` is folio slot and not section.

The next strong test is per-center prediction on a held-out folio, modeling the likelihood of **which specific neighbor forms activate** from previous pages and comparing against a Timm five-seed ensemble, conditional on matching quire and section.
