# Research log and replication status

2026-10-28, exploratory research, not a decipherment.

1. **Frequency/entropy:** Compared ZL3b paragraph tokens against length-matched Austen/Darwin. Zipf-like ranks coexist with lower glyph conditional entropy.
2. **Folio-held-out generators:** Folio-topic sampling matches aggregate statistics but cannot independently invent words. Glyph-conserving word crossover adds novelty, with Currier B long-range-location mismatch.
3. **Slot grammar:** A simple 12-slot system parses >80% of cleaned tokens but independent slots overgenerate and lose glyph constraints. A conditional slot model recovers some unseen words yet lags glyph trigrams in probability.
4. **Next-word models:** Word and glyph n-grams, 8-state HMM, shuffling, English, Naibbe and Caesar/Vigenère controls. Boundary glyph predictive information exceeds position-preserving shuffles but can arise procedurally.
5. **Copy/edit mixture:** Three-component normalized model improves overall held-out likelihood but not genuinely unseen forms. Training-calibrated repetition approaches observed rates.
6. **One-word perturbations:** Morphology/frequency-matched alternatives suggest weak residual context effects, much smaller than those seen in English controls. Exploratory interchangeability clusters unstable across folds.
7. **Independent transcription:** Six frozen word-pair hypotheses from ZL3b compared with Takahashi IT2a; four distinguishable pairs share expected direction, a near-interchangeable pair remains near chance, one test poorly supported. Hands and sections are confounded.
8. **One-glyph perturbations:** About eight million replacement evaluations across two transliterations; high within-word rigidity. Model-preferred substitutions rarely corroborate independent readings. Fixed repeating-key periods 2-16 not selected; period-6 Vigenère positive control detected.
9. **All-pairs census (this iteration):** A=2,985 types / 4,453,620 pairs; B=4,637 types / 10,748,566 pairs. Exactly-one-edit counts A 12,420; B 19,609. This is a *word-form census*, not proof of meaning. Distinct types within one variety are counted once.
10. **Transcription disagreement pilot (this iteration):** Conservative aligned-word single-EVA-glyph candidate set: 84,550 glyph records, 551 disagreements. Five hashed-leaf folds; lightweight linear logistic classifier using local token-level features yields mean AUC ~0.542 versus rare-glyph-frequency baseline ~0.557. FAILED to beat baseline. This is a *restricted* alignment subset and not a reliable ambiguity adjudicator. Image review remains outstanding.

## Reproducibility warning

Much of the historic exploratory numeric analysis was conducted through a connected source-reading environment, not independently verified by running all original-data scripts end-to-end in a separate Python environment. Freeze scripts, inputs and metrics before publication; never treat a model's preferred spelling as a verified transcription correction.

The earlier research packages are preserved as downloadable session artifacts. The repository is being populated incrementally; this log describes the program of work and does not imply all archived binaries have been committed.
