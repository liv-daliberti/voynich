# Bidirectional prediction from every manuscript-line word position

October 28, 2026. Extension of [exact fixed-target lattices](../latent_grammar/REPORT.md). **No sentencehood, translation or decipherment claim.**

## Research question

Does knowing the right-hand context of a manuscript line predict a word more reliably than knowing its left-hand context, especially near the line boundaries? Are the direction and position patterns different from readable English and the published Timm–Schinner copy/edit sample?

For each retained complete held-out line/sentence, evaluate **every word position** i, not only common or eligible forms:

- Forward: `P(w_i | w_0,...,w_(i-1))`.
- Backward: `P(w_i | w_(i+1),...,w_(n-1))`.
- Bidirectional cloze: a **normalized calibrated product-of-experts conditional score** using both direction-specific predictions. This is **not** a jointly normalized autoregressive likelihood for a whole sentence.

For each fixed target position, also progressively reveal *all possible prefix lengths* from the left boundary and *all possible suffix lengths* from the right boundary. With gaps between observed words and the target, the normalized sparse first-order Markov operator exactly sums over **all intermediate word-state paths**, without beam pruning.

## Data and controls

- Strict Currier A/B paragraph lines in the ZL3b transcription. Their line ends may reflect scribal layout and should not be called grammatical sentence ends.
- Jane Austen and Darwin, independently split into heuristic English sentences, first 22,843 words each.
- Published Timm–Schinner self-citation generator **sample**, used as a procedural *output control*, not a newly parameterized generator.
- Separate training-only position-preserving word-order shuffles within each folio/pseudo-block and line-start/middle/end bins. Evaluate exactly the same held-out lines, positions, word types and targets.
- Select the first-order smoothing parameter separately for each direction from 0.5, 2, 8, 32, 128 on an inner validation subset. Tune the bidirectional fusion exponent from 0, .25, .5, .75, 1 on training-internal held-out data. Nothing is selected using final held-out labels.
- Use precisely the same training-derived vocabulary (top 700 token types plus a shared unknown category), including deterministic ties, for forward/backward. The unknown category is a coarsening, not a full open-form word model.

## Statistics and visualization data

For each direction and bidirectional cloze:

- Held-out per-token negative log likelihood (lower is better), top-5 accuracy, and full-distribution Shannon entropy.
- Stratification: first, early, middle, late, last word; exact distance from line start and line end capped at 9; five normalized line-position bins; seen vs unknown vocabulary types.
- Per-position **fixed-target reveal curves** from both ends: number of revealed words, mean target NLL, mean entropy, and log probability gain relative to revealing zero words from the same boundary. The underlying sets of targets at different reveal lengths differ; interpret conditional curves with the `n` columns, or use per-target traces for within-target comparisons.
- Paired **folio/pseudoblock cluster bootstrap** over sampled held-out lines (700 repetitions), including the backward-minus-forward NLL difference and each method's improvement over its shuffled control. Samples are exploratory and confidence intervals may be wide with few distinct groups.

```bash
python experiments/bidirectional/bidirectional.py \
  --zl data/ZL3b-n.txt --austen data/1342-0.txt \
  --darwin data/1228.txt --generated data/timm_generated.txt \
  --max-lines 50 --max-vocab 700 --out results/bidirectional
```

CI workflow `.github/workflows/bidirectional-replication.yml` downloads source corpora and saves SHA-256 checksums. `summary.json` includes tables and CIs; `*_reveal.json` contains all progression metrics and a sample of complete target traces.

## Caveats and acceptance criteria

1. Forward and backward text factorizations are both legitimate in English; **there is no premise that English intrinsically favors forward prediction**. Compare likelihood at *the same held-out positions with the same candidate space*, with genre, length and boundary distributions reported.
2. Predictability from a line end may be related to **physical margin, spacing or suffix conventions**, not meaningful backward syntax. Analyze line-initial/line-final and interior positions separately.
3. The current first-order model only remembers the nearest observed token, although its forward operator exactly marginalizes arbitrary gaps. To test longer-range word dependencies, compare bidirectional trigrams, PPM/variable-order models, Brown/anchor-HMM word classes, and normalized finite-state morphology.
4. Two-sided masked word pseudo-likelihood cannot be compared directly with an autoregressive whole-line probability. It can be compared **at the same masked positions** with held-out conditional NLL.
5. Use repeated folio-/quire-held-out splits, multiple random seeds, and corpus-size-matched comparisons. Folio order is not a verified writing chronology.
6. Repeat with all training vocabulary and separately with **new forms** generated by edit neighbors and calibrated glyph models; the current unknown bucket hides unseen spellings.
7. A surprising backward advantage is not a translation or a decipherment. Require stronger than position-shuffled and copy/edit controls, across sections/hands and independent transcriptions.

References: Currier (1976) line as structural unit; Bowern & Lindemann (2021) linguistic review, especially positional restrictions; Peters et al. (2018) forward/backward contextual language modeling; Devlin et al. (2019) masked-word prediction; Timm & Schinner (2020) self-citation generation.
