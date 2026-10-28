# Voynich: per-word stability under 500 independent 75% deletions

Exploratory experiment, October 28 2026. **Nomination robustness is not the probability a word is meaningful or real.**

The first 70% of ZL3b paragraph-bearing folio sides (separately for Currier A/B) defines the training word vocabulary. The last 30% and independently produced IT2a are *evaluation only*. Each of 500 masks retains 25% of distinct training word types. All canonical one-EVA-glyph edit candidates are proposed from the training forms only. Per mask, a candidate's support is the number of *different surviving* one-edit neighbors. Nominate the 500 most supported absent candidates (random seed breaks ties) and count per-word nomination frequency.

**Observed average composition of top 500 per mask:**

| Status | A | B |
|---|---:|---:|
| Previously attested training forms deleted by the mask | 206.86 | 226.07 |
| New in the held-out ZL folios | 25.20 | 41.94 |
| Present only in Takahashi | 13.08 | 12.25 |
| Unattested in either transcription | 254.87 | 219.73 |

Examples: A: `oor` (90% nomination, later attested), `oshy` (88%, unattested), `ochar` (58.2%, IT-only). B: `oy` (97.2%, later attested), `oey` (86%, unattested), `cheoy` (83.8%, IT-only).

**Central finding:** High edit-graph degree makes a candidate stable across deletion masks, but stability alone does not distinguish genuine later-attested word types from unattested graph constructions.

## Run

```bash
pip install numpy pytest
python experiments/robustness75/robustness.py --zl data/ZL3b-n.txt --it data/IT2a-n.txt --out results/reproduced/robustness75 --trials 500
```

Code: [robustness.py](robustness.py). A standalone tested research package (including fixture tests and frozen result summary) is linked in the associated ChatGPT research session. Source provenance: [docs/SOURCES.md](../../docs/SOURCES.md).

**Limitations:** Only one folio-order train/test split; folio order is not writing chronology; evaluation uses text presence, not verified semantic validity; IT-only may reflect transcription differences; model uses one-edit support and seeded random tie-breaking; samples from a candidate universe created with a predeclared EVA alphabet. Full-source results were computed via a connected source-analysis environment; packaged Python passed offline synthetic tests but full independent source rerun remains outstanding.
