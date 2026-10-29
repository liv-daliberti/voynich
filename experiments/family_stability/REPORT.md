# Family-stability versus reorganization: preliminary findings

October 28–29, 2026. [Full source-data run 40809993914](https://github.com/liv-daliberti/voynich/actions/runs/40809993914) completed successfully on ZL3b Voynich A/B, matched Austen, Naibbe, and Timm/Schinner's **original Java generator** run with three new random seeds 19, 29, 37. Synthetic tests passed. All candidate proposals and family classifications are **strictly source-only**.

## Predeclared binary states and prospective outcome

For a center observed on the previous page and on another earlier page within a four-page lookback, calculate Jaccard of the two latest observed nonempty one-edit neighbor sets. Stable >=0.50, reorganizing <=0.10, otherwise intermediate. Target outcome: whether the next page contains any source-alphabet one-edit variant of this center not seen on any earlier page of that quire. All data for candidate list, class and rank precede the target page.

## Same-target page, source-frequency/degree matched comparison

| Cohort | Eligible events | Strict same-page matched pairs | Stable minus reorganizing novel-form probability | 95% quire bootstrap interval |
|---|---:|---:|---:|---:|
| Voynich A | 1737 | 171 | +.011696 | [-.0741,+.0861] |
| Voynich B | 4748 | 350 | +.022857 | [-.0350,+.0957] |
| Voynich pooled | 6485 | 521 | +.019194 | [-.0272,+.0605] |
| Austen pooled | 3006 | 255 | -.015686 | [-.0780,+.0444] |
| Naibbe pooled | 5767 | 656 | +.059451 | [+.0194,+.1018] |
| Timm19 pooled | 5635 | 497 | -.012072 | [-.0680,+.0645] |
| Timm29 pooled | 5338 | 470 | +.010638 | [-.0509,+.1267] |
| Timm37 pooled | 5703 | 529 | -.007561 | [-.0545,+.0355] |

**No decisive stable-vs-reorganizing advantage in Voynich after matching.** Naibbe ciphertext shows a stronger apparent advantage, underlining that this metric is not linguistic evidence.

## Held-out quire-level prediction (baseline vs +source neighborhood stability)

Four GroupKFold splits by *entire quire*. Positive log-loss gain is useful incremental information.

| Cohort | Baseline NLL | Plus stability NLL | Improvement | 95% quire-bootstrap CI for gain |
|---|---:|---:|---:|---:|
| Voynich A | **.684822** | .685683 | -.000861 | [-.002452,+.000734] |
| Voynich B | .661868 | **.660121** | +.001746 | [+.000816,+.006142] |
| Austen pooled | **.316823** | .316995 | -.000172 | [-.001718,+.001008] |
| Naibbe pooled | **.631664** | .631989 | -.000325 | [-.001050,-.000031] |
| Timm19 pooled | **.639980** | .640118 | -.000138 | [-.000670,+.000263] |
| Timm29 pooled | .626872 | **.626009** | +.000863 | [-.001098,+.002229] |
| Timm37 pooled | **.649066** | .649561 | -.000496 | [-.001450,-.000080] |

**Currier B has a small improvement** in held-out log loss when source-only
stability is added, but it corresponds to a tiny AUC increase (.6331→.6365)
and AP increase (.5720→.5767), with multiple related tests explored and
few independent quires. Treat as a reproducible exploratory lead.

The secondary outcome **reappearance of already-observed last-page
neighbors** shows larger held-out improvement from stability, Voynich A
+.013979 [.007475,.021059] and B +.014423 [.001438,.018039] nats.
This is unsurprising: the stability score itself measures repeated
occurrence of known forms, so predicting further reoccurrence is partly
a positive-control tautology, not evidence of lexical innovation.

## Examples from past-only events

These are **illustrative families**, summarized across eligible events,
not held-out ground-truth labels for a fixed grammatical category.
Among A centers with >=6 eligible events:
- More often stable: `saiin` (mean prior neighborhood Jaccard ~.634,
  n=11), `dain` (~.469, n=34), `qoky` (~.452, n=9).
- More frequently reorganizing: `chody` (~.056, n=12),
  `okaiin` (~.080, n=11), `dar` (~.152, n=28).
Among B centers with >=6 eligible events:
- More often stable: `checkhy` (~.593, n=20),
  `qotain` (~.513, n=21), `qotedy` (~.472, n=32).
- More frequently reorganizing: `chedar` (~.096, n=14),
  `opchedy` (~.182, n=24), `oky` (~.194, n=20).

These forms may be stable in some folios and reorganizing in others.
Their raw novelty rates differ and confound frequency and candidate-pool
size; do not infer causation or grammatical roles from the examples.

## Why this is important

Stable families are more often associated with future occurrence of
related forms in some **raw** comparisons, but those families also tend to
be more frequent and better connected. Matching those features removes
most of the apparent advantage. Currier B retains only a small
out-of-quire incremental gain.

This result argues against claiming that *a visibly stable word graph
necessarily generates more genuinely novel forms*. Stability seems more
useful for predicting re-use of existing forms than novel-form birth,
and the difference between A and B is currently too small and
underpowered for strong mechanistic interpretation.

The broader observational study also found that cross-page and graph
similarity are strongly shaped by quire/section organization, and
comparison to the original Timm generated mechanism is necessary.
Further sensitivity: use three- and five-page source histories,
additional source/model controls, stable/reorganizing cutoffs,
and truly independent heldout manuscripts/transcriptions.
