# Quire-heldout script-line reconstruction and generation

A second visual-only baseline, addressing the failures of the earlier
[whole-page patch prototype](../visual_gen/REPORT.md).

## Scientific target

Does a learned, **stroke-preserving** visual tokenizer + contrastive and
masked-code prediction learn to generate novel handwriting lines with the
geometric and statistical properties of *held-out real manuscript script*?

The output **cannot be treated as Voynich EVA words** unless a separately
validated, real-image-to-EVA recognizer passes a quire-heldout handwriting
recognition test. A model can generate handwriting-looking texture without
learning the manuscript's language.

## Sources and split

The workflow downloads 11 high-resolution, publicly mirrored Beinecke-derived
MS408 folio scans, with source URLs and SHA256. The source mirror is
`https://github.com/d2kx/voynich-research`; the library's authoritative
full-manuscript source is Yale Beinecke
`https://collections.library.yale.edu/manifests/2002046`.

For the current small CPU experiment:

- Hold out all images in **quires M and O** (f76r, f78r, f88r).
- Training images are other quires (A, B, C, E, G, H) and no patch
  from the holdout is ever used to train the encoder/codebook.
- The 11 images are **not a sufficient or representative training corpus**.
  Some originals contain illustrations, contrast-enhanced scans and text
  in several page regions. Crop quality is a major confound.
- Transcription labels from original ZL3b are used only for the **quire
  assignment** and after-training recognition-readiness audit, not to guide
  image generation.

## Image-only preprocessing

`line_dataset.py` uses page-local grayscale contrast,
small connected-component shapes and 1D handwriting-row peak detection.
A line crop must have sufficient horizontal component spread and pass
dark-ink coverage checks. This **heuristic is not a verified segmentor**;
human review of the crop montages is mandatory. Keep original folio
ID, quire, pixel bounding box and source file hash per crop.

Crop normalization pads to 64×256 without stretching text glyph widths,
but can truncate long lines. Treat these as *line fragments*, not certified
full manuscript lines. The generator produces fragments and arranges them
into synthetic page layouts, rather than attempting to reproduce
illustrations.

## Learned visual architecture

`train_line_vq.py` trains a small convolutional VQ-like image
autoencoder on black-ink line fragments, with 8×32 latent tokens from
64×256 input pixels (downsampling ×8).

1. Encoder: conv layers to 64-dimensional 2D continuous codes.
2. Learned codebook: 96 vector-quantization embeddings, straight-through
   estimator and standard codebook/commitment loss.
3. **Learned decoder**: transposed convolutions back to pixel ink density,
   trained with ink-weighted BCE and horizontal stroke-gradient L1.
   This is not the previous model's centroid-patch averaging.
4. Contrastive NT-Xent loss between augmented versions of the same line
   gives an additional image-space self-supervised objective.
5. A 2-layer, 4-head masked visual-code Transformer learns to predict
   randomly hidden tokens, conditioned on all unmasked code positions.
   Mask rates vary between 25% and 87%.
6. Iterative confidence-based masked-code filling samples entirely new
   8×32 token grids from a fully masked input. Decode 17 synthetic line
   images and assemble a **clearly labeled synthetic page**.

Controls: train-frequency IID visual-code sampling, the exact same learned
VQ decoder and synthetic page layout, and heldout real-line reconstruction.
Use visual-token unigram/transition JS divergence to **heldout real
scan codes**, ink-density differences, and max token-position agreement
with any training line as a memorization warning.

The VQ/token mask loss is *not* a supervised EVA token loss, and masked
code accuracy does not imply glyph recognition.

## EVA recognition readiness gate

`htr_alignment_gate.py` compares crop counts to the correct ZL3b
paragraph line counts for each folio and records potential source-line
alignment compatibility. It **always blocks EVA inference** without
manually verified image crop to transcription line matching, true
heldout-quire CER/WER, confidence calibration, and review of ambiguous
glyphs. Naively pairing the nth detected image stripe to the nth ZL
line would introduce label noise and potentially circular evaluations.

Thus this branch satisfies the *image generation and visual comparison*
part of the user's request. It does NOT pretend that this small experiment
has recovered a validated language distribution from synthetic images.

## Reproduction

`.github/workflows/voynich-masked-line-gen.yml` installs CPU PyTorch
and OpenCV, downloads/sha256-checks all sources, extracts script line
candidates, checks quire split, audits EVA readiness, trains the
autoencoder/masked Transformer, samples two page images and archives
result JSON and real/heldout/synthetic line montages.

```bash
python experiments/visual_gen/line_dataset.py \
  --zl data/visual_line/ZL3b-n.txt \
  --images data/visual_line/*.png --heldout-quires M O \
  --out results/visual_line

python experiments/visual_gen/htr_alignment_gate.py \
  --line-manifest results/visual_line/line_manifest.json \
  --zl data/visual_line/ZL3b-n.txt \
  --out results/visual_line/htr_alignment_gate.json

python experiments/visual_gen/train_line_vq.py \
  --dataset results/visual_line/line_images.npz \
  --out results/visual_line --vq-steps 280 \
  --mask-steps 160 --codes 96 --nlines 17
```

### Next stage if this succeeds

Expand to **all available Yale IIIF page images** with source/quire and
script-region QA, at full original resolution. Sample from multiple image
styles. Use an independently annotated script-region and line-box
dataset, train a quire-validated handwriting recognition system on
paired line images + EVA glyph sequences, then compare generated text
with the real manuscript's word lengths, Zipf law, edit connectivity,
Currier A/B behavior and persistence/innovation graphs.

A *negative* result from this small model does **not** establish that
the Voynich writing system cannot be learned or generated by a stronger
architecture.
