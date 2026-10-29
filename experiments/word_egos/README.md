# One center word = one edit graph

This experiment creates a separate **ego graph centered at every observed word form**, then records the observed neighbors of each center on every folio where the center appears.

- Radius 1: all corpus-attested forms exactly one canonical EVA glyph insertion/deletion/substitution from the center. Center is connected to every observed one-edit neighbor; edges among neighbors are included.
- Radius 2: unique observed forms within exactly two one-edit steps, but not themselves radius 1. Stored separately as a ring list, not the same as a dense radius-1 induced graph.
- Page ego: intersection of the global radius-1 neighborhood and a folio's observed word inventory. Store active nodes, present edit edges, active second ring, local triangles, local degree and token frequency.
- Global graph is descriptive and uses **the whole transcription**, including later folios, so it must NOT be mistaken for a prospective prediction or possible historical scribe source inventory.
- For every word type: document frequency, token frequency, global degree, two-hop count, mean/SD of active local degree and number of folios supporting it as a missing, one-edit-neighbor word.

The full output includes `all_word_egos.jsonl`, `all_word_folio_egos.jsonl`, `word_summary.csv`, `word_ego_per_folio.csv`, `selected_word_histories.json`, and 25 examples as editable `selected_graphviz/*.dot` files.

## Within-quire persistence hypothesis

For the 180 frequent centers with document frequency >=4 and global degree >=3, compare the Jaccard overlap of observed radius-1 neighbor forms in folio pairs separated by <=2 versus >=3 positions within the SAME quire and Currier variety. A center contributes one near-minus-far vote if it has both types of pairs. Do not count empty–empty neighborhoods as proof of stability.

Controls permute whole pages within variety+quire, and then preserve section and hand. Record how many pages move, matched centers, and nearby/distant pair counts. This is exploratory: high document-frequency centers are selected, graph topology is based on all attested forms, folios differ in size, and a long word might produce a different candidate distribution than a short word.

Reproduce:
```bash
python experiments/word_egos/evaluate.py \
  --zl data/ZL3b-n.txt --out results/word_egos --trials 350
```

See [REPORT.md](REPORT.md) and the archived GitHub Actions artifacts for complete editable graphs. Word nodes are EVA transcription strings, *not decoded meanings*.