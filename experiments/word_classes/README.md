# Interpretable structural classes, not Voynich POS tags

Research implementation companion to
`experiments/persistence_innovation/README.md`.

**Goal:** Before calling a Voynich word "noun-like" or "verb-like",
test whether an analogous method recognizes real English POS categories
based *only* on its observable position and word-order relations.

## Positive control and independent genre transfer

Use the first 22,843 whole words from Austen's *Pride and Prejudice* and
Darwin's *Origin of Species*, segmented into English heuristic sentences.
NLTK's `averaged_perceptron_tagger_eng` assigns English Penn Treebank POS
labels, which are **noisy automatic labels**, not human-verified ground truth.

Aggregate each word type's sentence occurrence labels, retain those with
>=2 noun/verb instances, at least 70% dominance of one over the other,
and at least 60% of all observed uses as noun or verb. Only types occurring
at least three times are eligible for structural feature construction.

Extract 21 dimensionless distributional features: normalized word and
line frequency, start/end and second/penultimate rates, left/right
neighbor diversity, entropy, common-context word-frequency-rank classes,
mean neighbor frequencies and adjacent exact repetition. Neither
word spelling, EVA glyph structure, suffixes nor POS labels are inputs.

Fit a class-weight-balanced logistic regression trained on 70% of
**Austen types**. Test on 30% held-out *word types* using a fixed seed;
then train on all Austen types and test on an independent Darwin book.
Never test on tokens of the same type distributed across both Austen
training and held-out subsets in the within-book split.

Results from
[GitHub Actions](https://github.com/liv-daliberti/voynich/actions/runs/40811785763):

| Positive control | Distinct held-out types | ROC-AUC | Average precision | Balanced accuracy |
|---|---:|---:|---:|---:|
| Austen type holdout | 192 | .83617 | .87896 | .78512 |
| Darwin, independent book | 608 | .69383 | .76831 | .64362 |

**Important:** the loss of accuracy across English genres shows that
these observable features are not universal or invariant syntactic
roles, let alone universally transferable from English to Voynich.

## Transfer to Voynich only as an anonymous functional analogy

Freeze the full-Austen model and label all sufficiently frequent types
in each corpus with a **score** indicating similarity to English
noun-vs-verb distributional features. This is NOT a Voynich POS tag.
Compare results with original Timm/Schinner procedural text and
Naibbe ciphertext. A score near 1 or 0 is NOT a calibrated probability
that a Voynich word is truly a noun or verb.

| Cohort | Frequent word types scored | >English reference 95%-distance feature threshold |
|---|---:|---:|
| Voynich A | 605 | 18.0% |
| Voynich B | 996 | 15.0% |
| Austen | 1017 | 5.0% |
| Darwin | 1068 | 4.0% |
| Naibbe ciphertext | 1558 | 5.8% |
| Timm published text | 664 | 12.5% |

Top extrapolated "noun-like" Voynich types often have high start/end-of-line
rates. For example `sor` and `tchedy` in B are line-initial in
approximately 74% of their observations, and `qokam` is line-final in
86%. These boundary rates may reflect manuscript layout conventions
rather than grammatical status, **especially because Voynich line
breaks are NOT verified sentence boundaries**.

Independently apply five-cluster KMeans to structural word-type features
within each corpus; clusters have **anonymous numbers**, not noun/verb
names. Report clusters, frequent exemplar words and start/end
propensities as a hypothesis-generation resource.

Next decisive test: a *known* artificial substitution cipher of English
(with sentence boundaries retained), plus cross-language / cross-genre
transfer, and proper class-based probabilistic LM predictive evaluation.
Hypothesized nouns/verbs require cross-context substitution patterns,
agreement, morphology, and external/semantic anchors. Unsupervised
clusters alone do not establish them.

## Run

```bash
python -m pip install nltk==3.9.1
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
python experiments/word_classes/english_pos_probe.py \
  --zl data/ZL3b-n.txt --austen data/1342-0.txt \
  --darwin data/1228.txt --naibbe data/naibbe.txt \
  --generated data/timm.txt --out results/word_classes
```
