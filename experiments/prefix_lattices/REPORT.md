# First source-data results: prefix and beam lattice

October 28, 2026. Run [40783142361](https://github.com/liv-daliberti/voynich/actions/runs/40783142361) completed on source ZL3b, Austen, and Darwin. Run artifacts contain fixed-horizon branching paths and sampled prefix traces. All values below are **exploratory** from **160 held-out word positions per corpus**, one 70/30 split, no cluster uncertainty intervals. Lower NLL is better.

| Corpus | Unigram NLL | Bigram NLL | Trigram NLL | Interpolated 4gram NLL | 4gram on shuffled training |
|---|---:|---:|---:|---:|---:|
| Voynich A | **5.6089** | 5.9387 | 5.9628 | 5.9950 | 6.0560 |
| Voynich B | **6.1566** | 6.7433 | 6.7907 | 6.8435 | 6.8876 |
| Austen 11k | **5.5776** | 5.7569 | 5.8182 | 5.8234 | 6.5106 |
| Austen 23k | 5.9020 | **5.6752** | 5.7670 | 5.7918 | 6.8738 |
| Darwin 11k | 5.4972 | **5.3984** | 5.5126 | 5.6011 | 6.3726 |
| Darwin 23k | 5.7866 | 5.4849 | 5.4403 | **5.4236** | 6.6892 |

The 4gram has much *lower predictive entropy* than the unigram but *worse predictive likelihood* in both Voynich varieties: A mean entropy 6.580 -> 5.474 nats, NLL 5.609 -> 5.995; B mean entropy 6.712 -> 5.238, NLL 6.157 -> 6.843. That is **overconfident prediction**, not evidence of constrained grammar. English context benefits become clearer at 23k tokens, but even in Austen the 4gram does not beat the bigram.

The edit-prior and edit-context variants are effectively tied with the 4gram because **only 0 to 0.6% of sampled truth tokens** are in the 128 selected novel edit candidates. This experiment cannot yet test whether the edit graph improves open-vocabulary prediction. Increase candidate coverage while tuning the novelty budget on inner folds.

Beam examples have 3-word fixed horizons, diversity enforced across first-word roots, and a width of 6 paths. The first Voynich A/B and English example **did not contain the exact observed 3-word continuation**. Branches illustrate plausible-looking model paths, not genuine/meaningful sentences. The beam's summed path probability is a strict lower bound on retained sequence probability mass, not total probability or calibrated next-word entropy.

The updated source-data workflow also logs **entropy by observed-prefix length** and the effective next-word candidate count. These are distributions for *successive next-word positions*. If testing literal uncertainty reduction for one fixed future word position, keep that target fixed and marginalize unseen intermediate words across beam or sampling paths. Never assume next-word uncertainty must decrease monotonically.

## Research comparison

- Chen and Goodman (1999): modified Kneser-Ney interpolation.
- Begleiter, El-Yaniv and Yona (2004): variable-order Markov sequence prediction, PPM and suffix trees.
- Brown et al. (1992): class-based word ngrams and learned distributional equivalence.
- Mohri, Pereira and Riley (2002): weighted finite-state composition across lexicon and transition grammar.
- Vijayakumar et al. (2018): diverse beams and distinct sequence modes.
- Timm and Schinner (2020): self-citation/copy-edit procedural text generation, crucial negative mechanism control.

**Next:** add block-heldout replication and uncertainty; tune smoothing only on inner folds, implement a strong English Kneser-Ney/PPM baseline, compare against a copy/edit procedural control. Then test same-target prefix conditioning with exact/marginalized missing-slot continuation; report beam coverage versus beam width and candidate recall for novel words.