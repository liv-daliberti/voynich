# Learned masked VQ visual generation on quire-heldout Voynich handwriting

Source-data study October 28–29, 2026, on the actual pixel scans.
**Primary conclusion: a successfully trained image model, but unsuccessful
generation of recognizable Voynich glyphs or EVA text.** No decipherment
claim.

## Data / quire split

11 Beinecke-derived publicly mirrored manuscript scan files (names and
SHA256 recorded). 149 candidate handwritten line fragments in **8 source
training folios**, and 66 line fragments in **three held-out folio images
from quires M and O**. The split is by whole quire, not random image patches.
Original grayscale crops and locally thresholded script crops are both
archived. The line detector uses small connected-component peaks and
conservative cropping; **human alignment has not been established**.

Original gray-crop pilot first identified only 17/7 lines because its
row-run segmentor conflated manuscripts' drawings with script. We
inspected the actual crop montage and replaced it with a small-ink
component detector, obtaining the 149/66 candidate line dataset.
This correction is itself documented, and source-image IDs are frozen.

## Model actually trained

- Continuous convolutional autoencoder stroke-reconstruction warmup:
  140 optimization steps, weighted pixel BCE, horizontal stroke L1, and
  NT-Xent contrastive loss over augmented views of same line.
- From encoder's *real training line* vectors, initialize 96-code
  quantization vocabulary using MiniBatchKMeans; then train a
  straight-through VQ encoder with learned transposed-convolution decoder
  for 280 additional optimization steps.
- Train a 2-layer, 4-head masked visual-token Transformer for 160 steps,
  with 25-87% random code masking and cross-entropy to original
  *visual code indices*, not EVA glyphs.
- Iterative confidence-based token sampling from ALL-MASK produces
  17 entirely synthetic line images. Assemble these into a watermarked
  synthetic page with blank margins. IID visual-code sampling through
  the SAME decoder is the baseline.
- Codebook no longer collapses: final minibatch uses **82 of 96** visual
  codewords after the continuous-decoder/KMeans initialization.
  Masked-code CE during training falls **4.576→1.617**. This is
  training improvement, not visual/language generalization.

## Primary honest heldout results

Real heldout reference comprises 66 candidate lines from different
quires. Comparisons use encoded real heldout lines, not training images.

| Metric | Masked visual generator | IID token baseline | Preferred |
|---|---:|---:|---|
| JS visual-code unigram divergence to holdout | .29963 | .01808 | IID |
| JS horizontal-code transition divergence to holdout | .41778 | .53555 | Masked |
| Pixel fraction ink-like (same threshold) | .09994 | .17406 | Reference .05502 |
| Reference holdout VQ reconstruction MAE | .06305 | N/A | Low is good |
| Holdout 50%-masked code prediction accuracy | .6095 | N/A | No frequency baseline calibrated |

Visual output: the spatial masked-token generator produces **coherent
horizontal strokes rather than recognizable Voynich glyphs**.
The unstructured baseline yields isolated blocky fragments.
The generator improves one spatial visual-token statistic (horizontal
transitions), yet badly distorts the image-token frequency distribution;
neither sample looks like genuine handwriting.

This is a negative **word-writing** result with a very limited image
training set. It does NOT rule out a much larger MAGE/VQ-based system.

## Recognition / true language-distribution test is BLOCKED

`htr_alignment_gate.json` compares automatically detected candidate line
counts with ZL3b known paragraph line counts. Only 4/11 tested folios
have line counts within 20%; **none** have manually verified crop/text
alignment, and several contain illustrations, marginal script or layout
difficulties. There is **no** valid Voynich HTR system, no calibrated CER
or WER, and therefore NO generated EVA transcription or comparison of
synthetic vs real Voynich word-length, unigram, positional, graph,
persistence or innovation distributions.

A separate `visual_symbol_audit.py` may compare unsupervised connected
component shapes and pseudo-word group gaps using a shape vocabulary
fitted on real training line crops. Such identifiers are **NOT EVA glyphs**.

## Reproduction and figure assets

Pipeline:
- `experiments/visual_gen/line_dataset.py`
- `experiments/visual_gen/train_line_vq.py`
- `experiments/visual_gen/htr_alignment_gate.py`
- `experiments/visual_gen/visual_symbol_audit.py`
- `.github/workflows/voynich-masked-line-gen.yml`

The real successful quire-heldout line VQ run is
[GitHub Actions 40815876078](https://github.com/liv-daliberti/voynich/actions/runs/40815876078)
with source hashes, binarized and original line crops, the synthesized
page, IID baseline, line montages and reported quantitative metrics.
The subsequent combined workflow also runs shape/gap auditing.

## What would count as eventual success

Use the authoritative Yale MS408 full IIIF corpus rather than
11 public image mirror examples. Hand-verify text regions, line bounding
boxes, and proper transcription ID alignments. Train the recognizer
and image generator on disjoint quires with a calibrated heldout CER/WER
benchmark. For the generative model, substantially increase script-line
diversity, VQ decoder spatial detail and masked context range, and
explicitly compare generated glyph identity and word-level statistics
against actual heldout scans only after HTR passes.

Also measure nearest-training-patch memorization, line-layout/copy-paste
controls and self-citation-generated text rendered in a separately
learned stroke style. The null controls must be matched in image
resolution, handwriting density and quire/hand composition.

A visually impressive fake page is not sufficient; the target is to
reproduce known manuscript *language-form statistics* while generating
distinct forms.
