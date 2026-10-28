# Voynich: computational studies of manuscript structure

Exploratory, reproducible research on **Beinecke MS 408**, including word-frequency statistics, held-out generation, cryptographic controls, next-word prediction, counterfactual substitutions, and independent-transcription comparisons.

> **No decipherment claim.** EVA transliterations are modern research artifacts. A plausible spelling, a higher model likelihood, or an apparent word class is not evidence of a translation.

## Roadmap

- **Word forms:** all-pairs glyph-distance census and held-out within-word rigidity tests.
- **Transcription ambiguity:** classify disagreements between independent ZL3b and IT2a readings; review high-confidence cases against Yale's original folio images.
- **Historical experiments:** archived, dated notebooks and code for earlier tests, each with provenance and reproduction limitations.
- **Cipher controls:** English / Naibbe / Vigenère-style and shuffle controls.

Inputs are **not** bundled: the repository's scripts download the original transcriptions with attribution; consult `docs/SOURCES.md` and the source READMEs.

## Sources

- Yale Beinecke MS 408 catalog: https://collections.library.yale.edu/catalog/2002046
- Yale IIIF manifest: https://collections.library.yale.edu/manifests/2002046
- Zandbergen–Landini ZL3b: https://www.voynich.nu/data/ZL3b-n.txt
- Takahashi IT2a: https://www.voynich.nu/data/IT2a-n.txt

## Research status

Results are **exploratory** and include negative controls. Existing historical artifacts may have frozen connected-analysis results that have not been independently recomputed with their Python implementations. This distinction is preserved in their reports.

Repository created on 2026-10-28.
