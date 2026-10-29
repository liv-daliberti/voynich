# Stable versus reorganizing one-edit word families

A **prospective**, within-quire experiment that asks whether the degree to which
a word's edit neighborhood persists over earlier pages predicts (a) genuinely
*new-to-the-quire-prefix* word forms on the next page and (b) reuse of known
neighbor forms. Unverified folio chronology is not treated as production order.

## Define a family using only past observations

For every target page `t`, and every center word `w` present on page `t-1`,
require an earlier center occurrence within the preceding four same-quire
folio sides. Generate canonical one-EVA-glyph variants of `w` with an
alphabet computed from **pages strictly preceding `t` in that quire**.
For English controls use one ASCII letter per edit; Voynich, Naibbe and
the Timm–Schinner generator use the canonical compound EVA glyph units.

Let `A` and `B` be the observed source-only variant sets on the last
two center-containing source pages (most recent `B` is `t-1`). Require
both sets nonempty. Define stability as `Jaccard(A,B)`. Stable means >=.50;
reorganizing means <=.10, with other observations retained as intermediate.
These cutoffs are **predeclared exploratory bins**, not learned from outcomes.
A family can change category as its observed neighborhood evolves.

For the next target folio, define candidate types as all source-alphabet
one-edit forms of the center that have *never occurred in any earlier page
of that same quire*. The binary primary outcome is whether at least one
such type appears on target folio `t`. It is **not** an English dictionary
validity or semantic correctness label. A separate outcome checks whether
a one-edit neighbor present on page `t-1` reappears on `t`.

## Two independent comparisons

1. **Same-target page, frequency/degree-matched families.** Within each
   identical target folio, match a stable-family center to a reorganizing
   center, one-to-one, without seeing future forms. Predeclared calipers:
   <=2 glyph length; <=1.0 log prior center tokens; <=.9 log prior number
   of observed neighbors; <=.8 log candidate pool; <=.9 log previous-page
   neighbor count. Minimize standardized distance on additional observed
   prior-document-frequency and preceding-neighbor counts. Compute stable
   minus reorganizing difference in future novel-form presence and
   known-neighbor retention, quire-cluster 95% bootstrap intervals.
2. **Quire-held-out incremental predictive modeling.** GroupKFold (4 folds)
   holds out entire quires, never individual word events from the same quire.
   Baseline logistic regression sees prior word token count, source document
   frequency, prefix-available degree, source candidate pool, EVA word length,
   both past observed neighbor counts, relative quire position and source
   page size. Extended model adds **only the past stability score**.
   Scaling, logistic coefficients and hyperparameters are fitted exclusively
   on the training quires. Report held-out token-level log loss, ROC-AUC
   and AP; bootstrap held-out per-event log loss improvements **by quire**.
   Use separate models for Currier A and B.

## Controls

Run the same process on public-domain Jane Austen and Naibbe ciphertext
pseudo-pages matching every Voynich folio's token count. Also generate three
independent texts with the **actual released Timm/Schinner Java generator**
from pinned original 2019 commit, varying seed 19, 29 and 37 and increasing
output length from the published default 1,200 to 5,000 lines. Morphology,
source-word selection and other settings remain unchanged. These synthetic
pseudo-pages inherit manuscript quire labels as a **sampling and CV grouping
template only**, not genuine historical manuscript units.

The original full manuscript's 6,481 global ego graphs **are not used as
prospective family priors**. Global graph content includes held-out folios
and would leak future observations. This study generates word candidates
from only past same-quire pages.

## Run

```bash
python experiments/family_stability/predict.py --zl data/ZL3b-n.txt \
    --mode Voynich --lookback 4 --out results/family_stability/Voynich

python -m pytest -q tests/test_family_stability.py
```

The GitHub Actions workflow `.github/workflows/family-stability.yml`
reproduces Voynich, Austen, Naibbe and three genuine Timm outputs; it also
checks Voynich `--lookback` 3 and 5 as sensitivity runs. Source/checksum
provenance and complete event-by-event CSV plus matched pair CSV are archived.

## Limitations

- “Stable” is a property of *observed source-page neighbor occupancy*,
  not of an underlying grammatical or morphological class.
- Stable/reorganizing classes have vastly different unadjusted source
  frequencies and observed degrees in some strata. Raw outcome rates are
  confounded; matched and quire-held-out comparisons are primary.
- Even matching on prior frequency/degree cannot control all variations
  in source manuscript section, rare glyphs or actual word meanings.
- Multiple outcomes/thresholds and related experiments have been explored;
  no p-values are confirmatory or corrected for multiplicity.
- Bootstrap quire counts for separate A/B models are small (~9–10).
- “New” means newly attested in an earlier-page-prefix-specific candidate
  universe, **not** historically invented, authentic, or semantically valid.
- Target-page predicted forms absent from the observed prior glyph alphabet
  are not eligible. Report pool sizes and coverage separately.
- Current folio order is not verified historical writing chronology.
- Austen, cipher and generator “quires” are artificial segmentation.
- The Timm program is a procedural control, not established as the
  historical generating mechanism for Voynich.
