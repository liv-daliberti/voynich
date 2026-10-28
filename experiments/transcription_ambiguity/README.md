# ZL3b–Takahashi disagreement prediction pilot

Goal: distinguish actual ambiguity from model rarity. Strict same-line-token-count / same-glyph-length alignments, retaining agreement sites and words differing by exactly one EVA unit. Train/test split by manuscript leaf.

Exploratory observations: 84,550 candidate glyph records (551 disagreements); a simple logistic model with rarity, word length, position, compound-glyph indicator, folio word frequency, Currier and selected hand/section group flags achieves average held-out ROC AUC 0.542. Rare-glyph-frequency baseline achieved 0.557. **Negative result:** the trained model did not improve on a simple baseline. Avoid image-ground-truth claims until manual evaluation.

Design flaws to address: selection on alignability; extreme imbalance; different word boundaries omitted; potential transcription-specific coding biases; non-independent hands and section. Next: adjudicate images using `docs/IMAGE_REVIEW_PROTOCOL.md`, add glyph-neighborhood features and visual embeddings, report AUPRC, and conduct truly folio-grouped calibration.
