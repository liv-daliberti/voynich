# Exact fixed-target grammar lattice and cross-corpus structure transfer

Research extension, October 28 2026. **Not a decipherment or English-to-Voynich translation.**

## Questions

1. Does grammar-like regularity in a previously observed prefix improve the prediction of a **fixed later position** `T` as each word from positions 0 through `T-1` is revealed?
2. Are learned latent **word classes and class transitions** portable between English prose and Voynich paragraphs, even though no words are aligned or translated?
3. Can a published **copy/edit procedural control** produce similar predictive structure?

## Data and baselines

- Voynich: paragraph *lines* in Currier A/B from the original ZL3b transcription. A line is not assumed to be a sentence.
- English: Austen and Darwin, first 22,843 words, split at ~70% of complete training sequences. English sentence segmentation is heuristic.
- Procedural control: the **sample text published with Timm and Schinner's self-citation generator** (see their public GitHub repository). We use the actual generated file, **not a reimplementation of their algorithm or a claim that this is the only parameterization**.
- Training order null: shuffle word tokens within folio/English pseudo-block and start/middle/end position, preserving word counts and position marginals.

## Model and exact prediction

Each corpus gets a **separately learned** discrete hidden-state Markov model with six states, unsupervised Baum–Welch training, at most 450 common word types plus `<UNK>`. This is a low-data structural baseline; hidden states are *anonymous*. Both vocabulary and parameters are derived from training only. A word absent from the 450-word vocabulary is scored as `<UNK>`, not falsely assigned probability zero. We separately report seen-vocabulary scores.

For a fixed target index `T` (zero-based), after observing the first `t` words (`0 <= t <= T`):

- Filter the hidden-state distribution by the observed prefix;
- Advance this distribution with the **transition matrix for exactly the missing number of positions**;
- Emit all possible word forms by multiplying by the emission matrix.

Explicitly, for `t > 0`:

`P(W_T=w | W_0,...,W_{t-1}) = [alpha_(t-1) A^(T-t+1) B]_w`.

For `t=0`, use `pi A^T B`. This is an **exact sum across all intermediate latent state paths**. No beam pruning is involved.

At each `T in {3,5,7,9}`, evaluate all prefix depths `t=0,...,T`, on the *same* randomly sampled held-out test targets. Report mean target NLL, full-distribution entropy, Brier score, seen-vocabulary NLL, and log probability improvement relative to zero context.

An exact probabilistic **state lattice** (6 states per layer, all weighted edges retained) is saved for example targets. Unlike a beam, its total state mass is one; meaningful state class descriptions require additional validation.

## Transfer from English structure (without lexical translation)

Fit models separately to English and Voynich; align anonymous latent states with **Hungarian matching on per-state descriptors learned exclusively from training**: start occupancy, stationary occupancy, self-transition, emission entropy normalized by vocabulary size, and outgoing transition entropy. The donor state's **transition probabilities only** are transplanted into a Voynich model. Voynich emission probabilities and vocabulary remain Voynich; there is no bilingual mapping.

Predict held-out Voynich targets using (a) its native transition model, (b) Austen transitions, (c) Darwin transitions, (d) Timm/Schinner generated sample transitions, and (e) Voynich trained on position-preserving shuffled sequences. **Successful transfer would mean shared statistical roles, not that a Voynich word translates to any English word.** The Hungarian descriptor approach itself is a weak heuristic and can produce arbitrary mappings for nonidentifiable states.

## Run

```bash
python experiments/latent_grammar/fixed_target.py \
  --zl data/ZL3b-n.txt --austen data/1342-0.txt --darwin data/1228.txt \
  --generated data/timm_generated.txt --out results/latent_grammar_lattice \
  --states 6 --em-iterations 5 --max-targets 130
```

CI downloads original sources and reports SHA-256 fingerprints and outputs. Detailed per-example traces and source hashes are archived, not committed transcriptions.

## Interpretation constraints

- The 450-type vocabulary is **not** a true generative open-vocabulary model. The `<UNK>` channel collapses many rare/unobserved forms. Do not interpret cross-corpus raw NLLs without matching token coverage.
- Hidden-state class correspondences are not identifiable without semantic anchors. Different initial seeds can permute or mix states. **No decipherment claim**.
- Natural English sentence boundaries and Voynich line boundaries are not equivalent.
- A held-out result for one folio order is not chronological language change. Repeat by blocked folio/section, many seeds, and multiple text windows.
- The procedural control's source text is a sample with fixed configuration; compare more Timm/Schinner generator configurations before treating it as a mechanistic falsification.
- English vs Voynich should ultimately be tested against the same alternative model families and matched token budgets, while controlling for differences in alphabet, word length and sampling.
- Cross-language structural alignment needs a positive control (Austen to Darwin, and preferably an artificially enciphered English corpus) and a shuffled/random alignment control before any interpretation.
- Exact state-path marginalization is possible because the hidden-state space is small and Markov. A full word-level beam/lattice with spelling invention would require a **separate normalized emission model** and pruning-mass accounting.

## References

- Rabiner, 1989: HMM forward-backward methodology.
- Bowern and Lindemann, 2021: *The Linguistics of the Voynich Manuscript*, Annual Review of Linguistics.
- Timm and Schinner, 2020: *A possible generating algorithm of the Voynich manuscript*, Cryptologia. Published generator sample: https://github.com/TorstenTimm/SelfCitationTextgenerator
- Conneau et al., 2018: *Word Translation Without Parallel Data* (ICLR).
- Søgaard, Ruder and Vulić, 2018: *On the Limitations of Unsupervised Bilingual Dictionary Induction*, ACL.
- Marchisio, Duh and Koehn, 2020: *When Does Unsupervised Machine Translation Work?*, WMT.
