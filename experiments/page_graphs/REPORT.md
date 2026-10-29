# Independent folio graph reconstruction versus matched English and cipher pseudo-pages

Exploratory source-data analysis, October 28–29, 2026. **No decipherment or manuscript-production chronology claim.**

## Completed reproduction

[Full 500-mask, 500-page-order-permutation GitHub Actions run](https://github.com/liv-daliberti/voynich/actions/runs/40797172800) succeeded for all three cohorts. The original ZL3b paragraph folios were grouped into 114 Currier A sides and 83 Currier B sides. Austen and the Naibbe ciphertext were each split into **197 synthetic pages matching the exact manuscript folio token counts**, in the same observed A/B sequence. Austen uses one alphabetic character as an edit unit; Naibbe and Voynich use compound EVA glyph segmentation. These controls are *not independent folios with authorship or production metadata*.

Every page is a separate canonical one-glyph edit graph, with per-page node types, graph edges, components, isolated fractions, degree histograms and pairwise lexical/edge overlap. For each page, 500 seeded deletion masks retain 25% of its distinct word types, nominate up to 50 candidate forms by surviving unique one-edit witnesses, and compare recovered *actual deleted* types and original graph edges with a random ranking from the **same generated candidate set**. All masked types are used only for *evaluation labels*. The list of possible glyph units is computed from the full page before masking; this is a generous/predeclared-alphabet version of the recovery test, not a strict unknown-symbol generalization test.

Each corpus also has **500 page-order permutations within A-sized and B-sized groups**, evaluating whether adjacent pages (1–3 apart) share more than distant pages (20+ apart). P-values are finite randomization-test resolutions and exploratory; pairs share pages and cannot be treated as independent draws.

## Local graph structure

| Cohort | Median types/page | Median edges/page | Median isolated fraction |
|---|---:|---:|---:|
| Voynich A | 69 | 62 | .305 |
| Voynich B | 160 | 263 | .209 |
| Austen A-sized | 63.5 | 15.5 | .718 |
| Austen B-sized | 163 | 55 | .695 |
| Naibbe A-sized | 75 | 33 | .477 |
| Naibbe B-sized | 209 | 194 | .348 |

Voynich pages have markedly denser one-edit graphs than Austen at comparable token budgets, while Naibbe is intermediate. Raw density is influenced by word-length distribution, vocabulary count and alphabet segmentation; this is not alone evidence for a particular generating mechanism.

## Nearby vs distant graph resemblance

**Word-type Jaccard near-minus-far** and page-order permutation p:

| Cohort | Near (1–3) | Far (20+) | Difference | p |
|---|---:|---:|---:|---:|
| Voynich A | .096713 | .078639 | +.018074 | .001996 |
| Voynich B | .137975 | .102238 | +.035737 | .001996 |
| Austen A-sized | .147246 | .124938 | +.022308 | .001996 |
| Austen B-sized | .165813 | .141394 | +.024419 | .001996 |
| Naibbe A-sized | .094075 | .092886 | +.001189 | .095808 |
| Naibbe B-sized | .124921 | .111737 | +.013185 | .001996 |

Both ordinary English prose and a known ciphertext can exhibit nearby-page vocabulary persistence. A near-minus-far effect **is not unique evidence of self-citation**.

**Degree-histogram topology cosine near-minus-far:** Voynich A +.003670 (p=.071856), Voynich B **+.043381** (p=.001996), Austen A-sized +.001690 (p=.003992), Austen B-sized −.000304 (p=.720559), Naibbe A-sized +.002618 (p=.021956), Naibbe B-sized +.018712 (p=.001996). These are similarities of coarse *unlabeled degree distributions*, not graph isomorphism or grammar correspondence. They remain sensitive to page size and sparse-node proportions.

## Can each page regenerate 75% of its hidden vocabulary?

Average **correctly recovered deleted types in the top 50** across 500 independent masks per page:

| Cohort | Hits among top 50 | Precision | Deleted-type recall | Expected random hits (same candidate pool) | Lift vs random |
|---|---:|---:|---:|---:|---:|
| Voynich A | 3.5165 | 7.03% | 6.29% | .3023 | 11.63x |
| Voynich B | 6.8781 | 13.76% | 5.20% | .3166 | 21.73x |
| Austen A-sized | .8261 | 1.65% | 1.56% | .0717 | 11.52x |
| Austen B-sized | 1.8019 | 3.60% | 1.61% | .0729 | 24.72x |
| Naibbe A-sized | 1.5812 | 3.16% | 2.47% | .1567 | 10.09x |
| Naibbe B-sized | 4.4780 | 8.96% | 2.99% | .2174 | 20.60x |

The ranking recovers more attested removed words in absolute terms from Voynich because its page graphs offer more structurally supported candidates. **Relative to a uniform random candidate ranking, lift is remarkably comparable in all three corpora** (roughly 10–12× for A-sized, 20–25× for B-sized). Thus the existence of a strong recovery lift is not peculiar to Voynich.

Among deleted types, the fraction with *any* surviving one-edit witness in the allowed candidate universe is: Voynich A 34.65%, B 42.47%; Austen A 11.10%, B 12.99%; Naibbe A 20.89%, B 30.70%. This reachable fraction is not the same as top-50 recovery.

## Recovered edges of the *original* per-page graph

The fraction of original observed graph edges retained by **only the 25% surviving word types**, and after adding back true hits among the top-50 nominations:

| Cohort | Surviving-only edge recall | Edge recall after correct top-50 additions |
|---|---:|---:|
| Voynich A | 5.95% | **20.03%** |
| Voynich B | 6.11% | **16.70%** |
| Austen A-sized | 5.95% | **15.08%** |
| Austen B-sized | 6.12% | **15.50%** |
| Naibbe A-sized | 5.99% | **14.42%** |
| Naibbe B-sized | 6.11% | **14.39%** |

Crucial: edge recall is computed only using predicted words that actually belong to the original page, so invented-but-unattested proposals do not count as recovered real edges. This is a useful *structural* measure of incomplete-network recovery. But these are conditional on artificially hiding attested words, not evidence that a historically missing word must exist.

## What the controls rule out and what remains

- Nearby lexical persistence is **not Voynich-exclusive**; Austen has it too.
- Dense page-level edit connectivity and high recoverability appear stronger in Voynich than in Austen and in these Naibbe samples, but a self-citation generator is specifically designed to make dense networks. Compare to multiple independently generated Timm–Schinner texts, not just Naibbe.
- Voynich B's nearby degree-histogram topology persistence is comparatively strong, but we must repeat with page-size- and frequency-matched graph rewires and block/section/quire controls.
- The source order is editorial foliation, not confirmed writing time; pages are naturally correlated.
- Long-range dependence, cross-page generation and copying direction remain separate hypotheses; page graph overlap cannot establish a copying arrow.
- The current recovery generator is one-glyph only, and uses a page-derived alphabet; future variants should fix the alphabet out-of-page, compare 2-edit and finite-state generators, and report held-out *next-page* new-form prediction to prevent overvaluing random deletions.
