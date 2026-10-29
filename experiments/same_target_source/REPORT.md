# Reassessment of folio-order structure after quire and section stratification

October 28–29 2026. Follow-up to the cross-page and independent page-graph studies. No writing chronology or decipherment claim.

**Critical metadata correction:** The ZL3b `$I` field represents illustration/section, `$Q` quire, `$H` hand. The `$F` field represents a folio position within a quire, and had been misnamed `section_code` in older exploratory code. This report uses `$I` for all section matches and supersedes the prior erroneous section-matched source results.

The [successful full-input source-data run](https://github.com/liv-daliberti/voynich/actions/runs/40805099037) used all 197 A/B paragraph-bearing folio sides, 120 whole-page permutations for each nested null, and frozen source SHA256. Source model and permutation tests passed. A follow-up **2,000-draw graph-only** quire-aware null is also being independently executed for higher-resolution p-values.

## 1. Near-vs-far per-page word-graph similarity

Test statistic: mean pairwise word-type Jaccard at variety-local position gap 1–3 minus mean at gap >=20.

| Preserved strata | Currier A null mean | Currier A one-sided p | Currier B null mean | Currier B one-sided p |
|---|---:|---:|---:|---:|
| Actual manuscript near-far statistic | **+.018074** | N/A | **+.035737** | N/A |
| Currier variety | +.000038 | .0083 | -.000209 | .0083 |
| Variety + section ($I) | +.006367 | .0083 | +.030064 | .0083 |
| Variety + quire ($Q) | +.016663 | .0826 | +.034904 | .1983 |
| Variety + section + quire | +.017086 | .1488 | +.034662 | .0909 |
| Variety + section + quire + hand | +.017006 | .1322 | +.034659 | .0992 |
| Variety + section + quire + hand + source-size tertile | +.017571 | .2810 | +.034237 | .0165 |

Interpretation: **much of the original nearby-folio advantage survives reordering WITHIN the same quires**. The observed near-far statistic is not evidence that fine-grained sequence order carries much information once quire block organization is held fixed. B retains a tiny apparent residual in the strictest null (~.00150), but p=.0165 from just 120 randomizations, amid many tested outcomes and stratum definitions; it is **not confirmatory**. The high-resolution 2,000-draw follow-up should be used for precise interpretation.

The parallel **edge Jaccard** and degree-histogram topology effects:
- Currier A edge near-far observed +.015857; strictest size-aware quire-null mean +.015279, p=.331.
- Currier B edge observed +.039098; strictest null mean +.036872, p=.0083.
- Currier B degree-histogram cosine near-far observed +.043381; strictest null mean +.044642, p=.769. The observed topology effect therefore does not survive quire-aware matching.

This is a crucial falsification of the naive claim that B's nearby-page *topology* continuity necessarily reflects an evolving grammar. Coarse word-graph shape is dominated by the quire/section composition in this test.

## 2. Does the actual preceding page predict the SAME target better?

Match the immediate previous same-Currier source page against a source >=20 same-variety positions earlier, chosen *without seeing target words*, within 25% of source distinct-type count and matched on section/hand if requested. The gold target vocabulary is held fixed and excludes both sources' already-observed forms. Candidate score is a difference in target novel-form recall ABOVE that predicted by a uniform ranker among each source's eligible generated candidates. Positive favors the immediately previous source page.

| Source matching protocol | Currier A target pages | Currier A observed gain | Currier B target pages | Currier B observed gain |
|---|---:|---:|---:|---:|
| Same section AND hand; nearby source also matches target | 74 | +.004958 | **5** | +.016771 |
| Same section; nearby source also matches target | 74 | +.004958 | 13 | +.011152 |
| Same Currier variety and similar source sizes only | 89 | +.007510 | 46 | +.002445 |

Strictest quire/section/hand/size conditional permutation expectations and one-sided p:
- A strict (74 targets): null mean -.000167; observed .004958; p=.107.
- B strict (**5 targets**): null mean .013351; observed .016771; p=.331.
- A section-only (74): null mean -.000167; p=.107.
- B section-only (13): null mean .005380; p=.157.
- A same-variety (89): null mean .003027; p=.0579.
- B same-variety (46): null mean .001824; p=.380.

**No robust fixed-target excess near-source predictive value appears under the strictest nulls.** Importantly, B's strict sample of five is far too small to support a directional copying inference. Distant source pages generally belong to *different* quires because of the >=20-page lag, so these are source-selection comparisons conditioned on a quire-aware **order-shuffle null**, not a direct comparison with a distant source in the same quire.

## 3. Permutation mobility and limitations

Every null keeps the original page contents/graph fixed and permutes only assignment to folio positions.

| Preserved metadata | Number of strata | Movable pages | Mean pages moved per permutation |
|---|---:|---:|---:|
| Currier | 2 | 197 | 195.0 |
| Currier + section | 9 | 196 | 187.7 |
| Currier + quire | 20 | 197 | 176.6 |
| Currier + section + quire | 25 | 194 | 172.5 |
| Currier + section + quire + hand | 28 | 192 | 169.2 |
| + within-stratum source-size tertiles where feasible | 56 | 192 | 141.7 |

The most constrained null still rearranges many pages, though largely within physically constrained blocks. It can retain quire-level shifts by design; failure to reject that null does not establish strict exchangeability of page order or rule out local copying.

These are exploratory tests over multiple outcomes and strata, without family-wise multiplicity correction. Quire and illustration/section are correlated and cannot be cleanly separated causally. The folio order is editorial, not verified production chronology. The reused similarity metric depends on page vocabulary frequency and graph connectivity and does not identify meaning or historical copying.

## Reproduction

Script `experiments/same_target_source/quire_section_nulls.py`; full matching source script `experiments/same_target_source/evaluate.py`; CI `.github/workflows/quire-section-nulls.yml`. Four synthetic unit tests verify the actual ZL metadata mapping, group-preserving permutations, size bins and graph null sample counts.

Main result: **the earlier near-vs-far effects largely reflect the manuscript's quire and section organization; evidence for residual within-quire local ordering or source copying remains weak**.
