# Predicting later-attested Voynich word forms beyond graph degree

Follow-up to [robustness75](../robustness75/README.md), October 28, 2026.

**Question:** Does a candidate word form's training-only EVA glyph compatibility add information to one-edit graph degree for predicting whether the form occurs in held-out folios?

**Not a decipherment.** The binary label is later *attestation*, not word validity, meaning, or authenticity. Unattested forms may be valid but unobserved.

## Protocol

- Parse ZL3b paragraph lines with explicit, per-folio Currier A/B language labels. Evaluate each variety separately.
- Build a candidate vocabulary exclusively from canonical one-EVA-glyph edits of **training** words; restrict to new forms supported by at least two distinct training types. Training-only glyph alphabet.
- The **outer experiment** trains features on the first 70% of folios in each variety and evaluates new-form attestation in the final 30%.
- The **inner experiment** builds candidates using the first 50% and labels them using forms appearing in folios 50–70%. Fit logistic regressions on these inner observations only; then apply to the outer experiment. Compare log-degree, log-degree + length, and log-degree + length + training-token-weighted glyph bigram likelihood.
- Report ROC-AUC and average precision (important with rare positives).
- Compare glyph-bigram scores *within exact degree-and-glyph-length strata* using pair-weighted ROC-AUC among strata containing both positive and negative candidates. This is a conditional concordance, not a model fit or global ROC-AUC.
- IT2a is used only to count independently transcribed forms that were not seen in the ZL evaluation; it is never a training feature.

## Preliminary source-data results

A separate exploratory JavaScript calculation on the ZL3b transcription produced these results. **The complete Python source-data rerun is outstanding**; the provided Python script and synthetic tests are a reproducibility handoff.

| Measure | Currier A | Currier B |
|---|---:|---:|
| Outer candidate forms (degree >= 2) | 69,602 | 95,050 |
| Attested candidates in final 30% | 601 | 1,219 |
| Graph-degree raw ROC-AUC | 0.790 | 0.785 |
| Raw glyph-bigram ROC-AUC | 0.831 | 0.798 |
| Glyph bigram within exact degree+length strata, pair-weighted ROC-AUC | 0.879 | 0.885 |
| Inner-trained degree+length outer ROC-AUC | 0.804 | 0.800 |
| Inner-trained degree+length+bigram outer ROC-AUC | **0.913** | **0.906** |
| Inner-trained degree+length outer average precision | 0.058 | 0.097 |
| Inner-trained degree+length+bigram outer average precision | **0.134** | **0.181** |

Interpretation: glyph sequence compatibility may help distinguish later-attested from unattested graph-supported forms, even after controlling for degree and length. **Do not interpret the scores as calibrated word authenticity probabilities.** Exact replication, folio-block confidence intervals, feature ablations, and natural-language/cipher/rewired-graph controls remain necessary.

## Run

Inputs are **not bundled**. See [sources](../../docs/SOURCES.md) for ZL3b and IT2a.

```bash
pip install -r requirements.txt
python experiments/phonotactics/evaluate.py \
    --zl data/ZL3b-n.txt --it data/IT2a-n.txt \
    --out results/phonotactics
python -m pytest -q tests/test_phonotactics.py
```

Writes `summary.json` and `candidates_A.csv` / `candidates_B.csv`, including prediction scores for all candidates.

## Limitations

- Folio order is not verified composition chronology; later means only later in the selected order.
- Candidate prevalence depends on how many folios and manuscript words are observed. A candidate absent in both ZL3b and IT2a is not necessarily impossible.
- Neighbor connectivity and orthographic regularity can arise from language, ciphering, text copying, or procedural mechanisms. No mechanism is identified by this experiment.
- Within-degree-and-length concordance accounts for these two factors, but **does not** control for word frequency, folio topic, shape differences beyond length, writer, section, or shared transcription errors.
- Next controls: word-type frequency and substring-preserving randomized candidates, multiple folio splits with grouped uncertainty, and context compatibility measured without test-label leakage.
