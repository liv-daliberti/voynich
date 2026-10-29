# Persistence vs innovation graphs, and noun-like vs verb-like structure

October 28–29 2026, stacked on the prior `stable-reorganizing-families` branch.

## Different graphs answer different questions

Given actual ZL3b ordered folio sides within one Currier/quire,
construct a **directed cross-page event graph** from every distinct word
center on page `t-1` to every canonical one-EVA-glyph variant.

**Persistence graph, P:** include `w -> v` only when `v` was present on
source page `t-1` AND appears again on target page `t`.
These edges measure repeated already-observed morphological relations.

**Innovation graph, I:** include `w -> v` only when `v` was not present
anywhere in earlier observed pages of the *same quire*, but first appears on
target page `t`. This is **new attestation in a prefix**; not creation
of a true new linguistic word. Only prior-source EVA glyph alphabets are
used to generate permissible edit variants. Neither graph determines an
actual copying source or historical production chronology.

Each network stores: weighted edge list (source center, target form, number
of cross-page transition occurrences), individual events, top recurrent
source centers, token/source-type normalized rates, fraction of previously
unseen target vocabulary covered by proposed innovations, and how often a
new form has multiple possible prior parent words.

A null performs 80 page-order permutations **within Currier variety, quire,
illustration section and hand**, preserving entire folio contents.
Recomputing innovation/persistence in each permutation avoids leaking
target-page labels into prior source models. This is a test of observational
sequence organization conditional on section/quire, not historical causation.

Outputs: `persistence_graph_edges.csv`,
`innovation_graph_edges.csv`, `edge_occurrences.csv`,
`page_transition_counts.csv`, and top-85-edge Graphviz `.dot` files.

Controls: English Austen (one-character edit units), Naibbe ciphertext
and the original public Timm/Schinner generated sample (EVA compound units).
These are matched-length pseudo-folios with artificial inherited Currier
and quire labels, not naturally sectioned documents.

## Testing a noun-versus-verb hypothesis

Unknown Voynich forms cannot be honestly labeled `noun` or `verb`
without decipherment or independent anchors. Instead test whether two
observable *distributional roles* appear and whether an English-trained
diagnostic can distinguish English nouns and verbs.

- Form separate English sentence corpora from first 22,843 tokens of
  Austen and Darwin; create English word-type labels using NLTK's
  `averaged_perceptron_tagger_eng` averaged-perceptron Penn tags.
  Restrict to frequent, predominantly nominal or verbal word types;
  type labels are **automatic noisy estimates**, not hand-annotated gold.
- Build **21 dimensionless word-context features** for each observed type:
  sentence/line-start/end and second/penultimate position rates,
  preceding/following type diversity and entropy, common vs uncommon
  context rank distributions, prior/following mean frequencies, repetition
  and overall word/sentence frequencies. Do NOT use word identities, edit
  distance, literal spellings or suffix/prefix cues in this probe.
- Test whether English POS classes can be distinguished in a 30% heldout
  Austen *word-type* split, plus **Darwin as an independent full book**.
- Freeze the model trained on Austen structural features; score Voynich
  A/B types and Naibbe/Timm controls as **noun-like vs verb-like only in
  the sense of similarity to this particular English statistical model**.
  Do not call these POS tags, translations or verified syntactic categories.
- Report how often the target corpus's feature vectors lie outside the
  reference Austen 95th-percentile Euclidean distance in standardized
  feature space. If most Voynich forms are out-of-domain, scores are
  extrapolations and should not be interpreted.
- Independently cluster each corpus into five **anonymous distributional
  word classes**. Report positional patterns, frequent words, counts;
  do not name a class "nouns" just because it is cluster 2.

This is a diagnostic, **not unsupervised machine translation**. A higher
English POS AUC does not prove that the same feature-role correspondence
holds in Voynich. Compare to cipher and procedural samples to detect
spurious language-like structure. English tagger is called ONLY on English
controls; it is never applied to Voynich as if Voynich were English.

## Reproduce

```bash
python experiments/persistence_innovation/graphs.py --zl data/ZL3b-n.txt \
  --out results/persistence_innovation/Voynich --mode Voynich \
  --permutations 80

python -m pip install nltk==3.9.1
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
python experiments/word_classes/english_pos_probe.py \
  --zl data/ZL3b-n.txt --austen data/1342-0.txt \
  --darwin data/1228.txt --naibbe data/naibbe.txt \
  --generated data/timm.txt --out results/word_classes
```

CI: `.github/workflows/persistence-innovation-pos.yml`, five
independent jobs with source text SHA256 fingerprints and archive
artifacts. Test: `tests/test_persistence_innovation_pos.py`.

## Next scientific tests

1. Split each quire into independent train/validation/test segments for
   *predictive* innovation modeling; current observed edge graph descriptions
   use future labels only to assign edge type, and its null tests are
   order-conditioned descriptive rather than prospective scores.
2. Reconstruct the actual graph output across quire boundaries when metadata
   allows, and test different putative writing orders.
3. Compare POS-probe calibration on completely different English genres and
   an artificially enciphered English corpus. A robust noun/verb signal on
   English alone cannot assign Voynich grammatical categories.
4. Learn class-based HMMs or nonparametric suffix-tree classes on Voynich,
   conditioned on line boundaries, and test whether those classes improve
   held-out cloze/log loss over morphology and positional nulls.
