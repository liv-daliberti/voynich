# Within-quire next-page prediction

Reproduced from ZL3b's 197 Currier A/B paragraph-bearing folio sides. **Within-quire** means only positions among sides assigned to the same $Q and Currier $L; it is not a verified production chronology.

## Experimental question
Does p_(t-1) predict previously unobserved word forms on p_t better than a size-matched prior source page at least TWO folio-side positions earlier **within the same quire**?

1. Build one train-only, one-EVA-glyph candidate ranker for every observed folio: each candidate must have >=2 distinct one-edit source witnesses. Candidate ranking uses source words only.
2. For each target t, choose its immediate preceding same-quire source and an earlier source >=2 positions back, with source distinct vocabulary sizes within 30% and closest token count. **No target words affect the source selection.**
3. Define fixed gold as words in target t **absent from BOTH sources**. Score recall from the top-50 nominations and subtract uniform-random expectation from each source's own eligible candidate pool. Also evaluate candidate coverage, percentile rank of forms proposed by both sources, shared vocabulary/edges, coarse graph topology.
4. Treat full pages as whole units when shuffling assignments within **(variety, quire)**, **(variety, quire, $I illustration/section)**, **(variety, quire, section, hand)**, and the latter plus type-count bins. Recompute the near-vs-earlier comparison after each permutation.
5. Use quire-block bootstrap over observed targets, rather than independent word-token bootstrap. Report eligible targets and quir es; underpowered group sizes warrant caution.
6. Currier A/B are reported separately, as well as pooled. Controls: Austen words and fresh Timm–Schinner generated text are cut into precisely the *same Voynich folio token budgets* and inherit manuscript labels solely for matching. **These are artificial strata and not true textual quir es or sections** in control corpora.

Run:
```bash
python experiments/within_quire/predict.py --zl data/ZL3b-n.txt \
  --cohort Voynich --trials 350 --out results/within_quire/voynich.json
```

GitHub Actions: `.github/workflows/within-quire-egos.yml`; full source ZIPs and checksums are archived there.

Limitations: Currier is a transcription-assigned category, not an independent dataset. Shuffling pages within a quire can disrupt actual physical recto-verso structure; preserving this is a useful stricter follow-up. Small changes in high-frequency morphological families may explain the candidate advantage even without literal copying; source-side generation model is a one-edit analogy, not the full Timm–Schinner procedural likelihood. No claim about semantic grammar or the original production sequence.

See [REPORT.md](REPORT.md) for outcomes.