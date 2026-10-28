# All-pairs Voynich edit graph and pseudo birth/death
Exploratory study, 2026-10-28. No decipherment claim.

See [full report](REPORT.md) and frozen [summary](frozen_summary.json). Run exact graph construction with the prior [all-pairs script](../all_pairs/all_pairs.py) on ZL3b-n.

## Main tests
1. One-EVA-glyph insertion/substitution/deletion graph over every word type (A/B separately).
2. Five leaf-held-out context prediction, using word shape plus graph-neighbor smoothing. Compare against **degree-preserving rewires within endpoint shape bins** (same degree for every word, same morphological endpoint signature pair counts). Set smoothing strength on training-only inner folds.
3. Folio-order pseudo birth/death: active word=appears in a nonoverlapping window of five or ten folio sides; active edge=both endpoint words appear. Compare against 250 global and 250 within-hand×section page-order shuffles.
4. Random node deletion and prediction from canonical surviving one-edit candidates; compare graph witness ranking with glyph-likelihood ranking on the same universe.

## Caveats
Folio order is *not* manuscript writing chronology. Birth/death means observational appearance/disappearance, not lexical invention. Graph edits are defined by spelling; recovering an edit edge from its two surviving endpoint strings is tautological. Do not infer meaning from connectivity.

The numerical results were run against connected GitHub source content with an exploratory JavaScript implementation. Python reproductions and tests are bundled in the downloadable study; an independent full-corpus Python rerun remains outstanding.
