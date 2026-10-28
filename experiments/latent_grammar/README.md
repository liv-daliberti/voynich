# Fixed future words, exact probability lattices, grammar transfer and graph matching

This folder contains three **distinct** experiments on the same ZL3b, Austen and Darwin sources, with a fourth *procedural null*: the openly published Timm–Schinner self-citation generated text sample.

- `fixed_target.py`: a six-state unsupervised HMM. Exact forward marginal `P(W_T | W_<t)` over **all latent paths**. For Voynich, the first run found no useful improvement over position-preserving shuffling, **including failure to find a strong positive control in English**. The model's learned latent classes are **not** human-identified parts of speech. A heuristic Hungarian matching of state descriptors transplants English transitions into a Voynich model with Voynich emissions. Even if NLL improves by a small amount, **without an English positive control and repeated seeds it is not evidence of translation or transferable grammar**.
- `word_lattice.py`: more robust exact word-transition graph with sparse, normalized next-word transitions. Smoothing is selected by a **training-only inner validation split** from candidates 0.5, 2, 8, 32, 128. For one held-out target position, the forward operator sums probability over every possible intervening word state, including an `<UNK>` bucket. Compare no prefix, partial prefix, and immediate predecessor against a word-order shuffle that preserves positions and token frequency. A bigram only remembers the immediately preceding observed word and has short predictive memory, by design.
- `structural_alignment.py`: a separate **graph matching** hypothesis. Construct directed word-context graphs using *word frequency, line-start/end propensity, predecessor/successor diversity and entropy*, not token spellings. First try to recover a **known random renaming of English words** across two disjoint English subsets. Then fit English-to-Voynich **structural analogies**, without semantic labels. Compare similarity against position-preserving shuffled target word order. Descriptors are approximate; matching can be arbitrary and should not be read as an English-Voynich dictionary.

**Do not conflate**: A self-citation sample vs a full replication of its generator; anonymous latent roles vs human grammar; graph similarity vs semantic alignment; unseen words collapsed into `<UNK>` vs explicit prediction of novel EVA strings; corpus folio order vs genuine time.

## Reproduction

GitHub Actions workflow `.github/workflows/latent-grammar-replication.yml` downloads the original four source texts, saves SHA256 hashes, runs all three experiments and archives their summary JSON and prefix/lattice traces. Unit tests check exact forward probabilities against matrix powers and brute-force enumeration; test that renaming destroys spellings without destroying graph signal; and verify that hidden-path sums remain normalized.

```bash
python experiments/latent_grammar/fixed_target.py --zl data/ZL3b-n.txt --austen data/1342-0.txt --darwin data/1228.txt --generated data/timm_generated.txt --out results/latent_grammar_lattice
python experiments/latent_grammar/word_lattice.py --zl data/ZL3b-n.txt --austen data/1342-0.txt --darwin data/1228.txt --generated data/timm_generated.txt --out results/word_lattice
python experiments/latent_grammar/structural_alignment.py --zl data/ZL3b-n.txt --austen data/1342-0.txt --darwin data/1228.txt --out results/structural_alignment
```

## Literature and next rigorous tests

- Rabiner (1989), HMMs and exact forward/backward lattice marginalization.
- Timm & Schinner (2020), *A possible generating algorithm of the Voynich manuscript*, Cryptologia; original sample generated with their published code at https://github.com/TorstenTimm/SelfCitationTextgenerator.
- Bowern & Lindemann (2021), *The Linguistics of the Voynich Manuscript*, Annual Review of Linguistics.
- Conneau et al. (2018), *Word Translation Without Parallel Data*, and Lample et al. (2018), *Unsupervised Machine Translation Using Monolingual Corpora Only*.
- Søgaard, Ruder & Vulić (2018), *On the Limitations of Unsupervised Bilingual Dictionary Induction*. Cross-domain / low-resource failure modes are especially relevant.
- Marchisio, Duh & Koehn (2020), *When Does Unsupervised Machine Translation Work?*
- Graça et al. (2011), *Controlling Complexity in Part-of-Speech Induction*. Vanilla maximum likelihood HMM class induction has well-known limitations; anchor-HMM or regularized class models are stronger comparisons.
- Stratos, Collins & Hsu (2016), *Unsupervised Part-of-Speech Tagging with Anchor Hidden Markov Models*.
- Brown et al. (1992), *Class-Based n-gram Models of Natural Language*.

Before making a Voynich-to-English semantic proposal: show known-renaming accuracy above frequency-rank matching on **independent English splits** and ideally on artificial English substitution ciphers with independent text; compare random assignment and Timm sample; use several folio blocks/seeds and uncertainty; document which aspect of morphology/word order anchors the two corpora. A high-scoring graph correspondence without independent bilingual evidence is **not a decipherment**.
