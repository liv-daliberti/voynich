# Exhaustive word-form comparison

Enumerate each distinct word type (within Currier A or B) and every unordered pair. Compare EVA-unit length, first+last EVA units, and exact Levenshtein distance 1. Outputs include full pair counts, within-boundary counts and the near-neighbor graph.

Run: `python experiments/all_pairs/all_pairs.py --zl data/ZL3b-n.txt --out results/all_pairs`.

A: 2,985 types, 4,453,620 unordered pairs, 12,420 pairs at distance <=1 (excluding identical types), 5,206 at distance 1 with the same boundary units.

B: 4,637 types, 10,748,566 unordered pairs, 19,609 pairs at distance <=1, 9,749 at distance 1 with matching boundaries.

**Scope:** only paragraph tokens matched by the script. All-pairs lexical similarity is not contextual substitutability and is not meaning.
