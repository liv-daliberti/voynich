# Prefix-conditioned prediction and branching lattices

The model scores a **normalized** next-word distribution over a fixed train-only candidate vocabulary. For Voynich, each sequence is a transcribed paragraph **line**, not a known sentence. Austen and Darwin provide readable English controls.

Models: unigram, prefix-position, bigram transition graph, trigram, interpolated 4-gram, 4-gram with edit-graph new-form prior, and 4-gram with edit-neighbor contextual transfer. Known words, 128 training-only edit candidates, and `<UNK>` share a fixed prediction space. Novelty (3%) and `<UNK>` (1%) mass are predeclared, not calibrated.

The decoder retains several distinct first-word branches and expands them for three words using a fixed-horizon diversity-constrained beam. The beam is a search result, **not** a normalized probability distribution. Entropy is always computed from full one-step probabilities, not beam path counts. Evaluation includes held-out log-loss, top-1/top-5, entropy, prefix-depth analyses, OOV/candidate coverage, and position-preserving shuffled word-order controls.

Run from repository root:

```bash
python experiments/prefix_lattices/prefix_beam.py --zl data/ZL3b-n.txt --austen data/1342-0.txt --darwin data/1228.txt --out results/prefix_lattices --max-events 200 --max-novel 128
```

Next: evaluate blocked folio splits with confidence intervals; test Kneser-Ney, probabilistic suffix trees, finite-state morphological templates, learned word classes, and copy-edit procedural controls. Any graph model must beat a transition-only model on the same candidate set and held-out positions. English and Voynich boundary definitions differ and must not be interpreted as directly comparable syntax.

Related methods: Chen and Goodman (1999) on n-gram smoothing; Begleiter, El-Yaniv and Yona (2004) on variable-order Markov models; Brown et al. (1992) on word classes; Mohri, Pereira and Riley (2002) on weighted finite-state systems; Vijayakumar et al. (2018) on diverse beam search; Bowern and Lindemann (2021) on Voynich linguistics; Timm and Schinner (2020) on self-citation procedural generation.
