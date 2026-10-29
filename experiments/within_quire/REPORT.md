# Findings: within-quire source predictability

October 28–29, 2026. [Successful independent workflow](https://github.com/liv-daliberti/voynich/actions/runs/40808005107) with 350 Voynich and 200 Austen/Timm position-preserving permutations. Experimental and exploratory results, not decipherment.

## Same-target chance-adjusted top-50 recall (near minus earlier source)

| Cohort | Eligible target folios | Q uires (matched) | Point delta | Quire bootstrap 95% |
|---|---:|---:|---:|---:|
| Currier A | 82 | 10 | +.011797 | [.005999,.018495] |
| Currier B | 60 | 8 | +.003102 | [-.005248,.008641] |
| Combined A/B | 142 | 18 | +.008123 | [.003582,.013454] |
| Austen pseudo-pages | 139 | 18 | +.000015 | [-.002800,.002729] |
| Timm generated pseudo-pages, seed 29 | 139 | 17 | +.008156 | [.002691,.014811] |

Positive means the immediately preceding source ranks a fixed target's genuinely source-novel forms better than a size-matched earlier source, *net of* uniform random ranking within each source candidate pool.

### Strictest quire + section + hand + vocabulary-size order null

| Cohort | Observed delta | Expected under null | Excess | One-sided Monte Carlo p |
|---|---:|---:|---:|---:|
| Currier A | .011797 | .000523 | **+.011274** | .008547 |
| Currier B | .003102 | -.000862 | +.003964 | .113960 |
| Austen | .000015 | -.000185 | +.000200 | .452736 |
| Timm seed 29 | .008156 | .000439 | **+.007717** | .014925 |
| Voynich pooled | .008123 | -.000062 | **+.008185** | .008547 |

The naive ranked-percentile improvement restricted to forms present in BOTH candidate spaces is less robust (combined +.017417, strict null p=.128205). Therefore **stronger near-source candidate coverage and true hit counts**, not necessarily fine ranking among jointly proposed forms, may drive the effect.

Also, within the same matching, the *raw* near-over-earlier word Jaccard advantage pooled is +.011661, with a strict-null excess +.010334 (p=.002849), i.e. topic/word-family continuation remains an explanation.

**Interpretation:** Under these exact conditions Currier A shows a small, reproducible within-quire prospective-form prediction signal, while B is inconclusive. A single Timm procedural sample can reproduce a signal of similar magnitude. Austen's corresponding matched pseudo-pages do NOT. This cannot distinguish literal self-citation from locally persistent morphology, genre or topic, or prove historical folio creation order.

The artificial Timm/Austen "quires" are inherited from real Voynich positional templates, and the generator was run only under one seed and length here. Five seeds were used in the earlier cross-page study, not the same experiment. Additional seeds with genuinely native generator quire/page boundaries and bootstrap uncertainty are needed before mechanism comparison. All p-values are exploratory and not multiplicity adjusted.