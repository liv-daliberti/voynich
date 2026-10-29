# Per-word edit ego graphs: scope and first results

Full [reproducible run](https://github.com/liv-daliberti/voynich/actions/runs/40808005107) completed. ZL3b Currier A+B, 197 eligible folio sides, all observed word types included.

- **6,481** distinct word centers, one descriptive global graph each.
- **22,875** center–folio instances, each with observed one-edit neighbors and induced present edges.
- **180** frequently attested words selected for the statistical near/far neighborhood-overlap test, matched within actual quire+variety strata.
- **7,598** nearby within-quire center-word page pairs (gap 1–2) and **25,282** more distant pairs (gap >=3), after excluding empty–empty neighborhoods.
- Global degree illustration: `daiin` has 47 corpus-observed one-edit neighbors across **186** folios, but on `f1r` only 8 are present, on `f1v` 0, and on `f2r` 5; `chedy` has 65 one-edit neighbors across 77 folios, with per-page activation varying between 0 and 18.

## Neighborhood persistence within quires

Mean of center-level [Jaccard(near page observed neighbor set)-Jaccard(far page observed neighbor set)] is **−.004096**, averaged one vote per eligible center. Negative does not mean there is no structure: the corresponding null after quire-only page-order permutation is **−.012404**. Thus the observed difference is *less negative*, representing an excess +.008308 over the null, with one-sided permutation p=.034188 across 350 shuffles.

| Page-order null | Expected delta | Observed minus null | Exploratory one-sided p |
|---|---:|---:|---:|
| Within variety and quire | −.012404 | +.008308 | .034188 |
| Within variety, quire, section | −.012210 | +.008113 | .045584 |
| Within variety, quire, section and hand | −.012212 | +.008115 | .037037 |

This is **weak exploratory evidence** for word-centered neighborhood organization within a quire. The raw absolute near-minus-far value is negative, and these p-values are uncorrected for the multiple center/graph and stratum hypotheses investigated. Coarse page-size and repeated-common-word effects still need stronger matching; no historical directionality or grammatical meanings are established.

Editable JSONL records allow opening any center, inspecting its global radius-1 induced graph and its *per-folio* activated subgraph. See [README.md](README.md).