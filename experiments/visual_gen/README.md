# Image-space generative Voynich pilot: representation vs generation

**Status:** runnable CPU research pilot on 11 publicly accessible
Beinecke-derived folio images. **Not** manuscript decipherment or a
validated visual-to-EVA generator. Generated folio-like textures and proxy
component distributions must never be described as Voynich words.

## Why image-based generation is scientifically useful

Our prior experiments constrained word types to human-designed EVA units.
The present experiment asks whether a model given **only raster images**
can learn visual statistics independently. It can potentially find
handwriting regularities that EVA transcription erases or normalizes.
In addition to manuscript style, page structure, drawings, pigment and
paper noise can confound these results; avoid equating visual realism with
learned language.

MAGE (Li et al., CVPR 2023) is a stronger principled model family,
combining contrastive representation learning with masked discrete image
token prediction and a visual decoder. The pilot implements only a
computationally inexpensive approximation to test the input and
evaluation pipeline.

## Implemented pilot

1. Source images: 11 folio scans publicly mirrored at
   `d2kx/voynich-research/pages` with Beinecke attribution.
   9 train and **2 whole-image heldout scans**. Each scan is resized to
   384×512 grayscale. The source filenames and SHA256 fingerprints are
   exported, and specific training/test filenames are fixed in CI.
   **IMPORTANT:** the 11 images span different quires and some enhanced
   versions, are not a representative manuscript sample, and may mix
   drawings and script. The source image mirror is not authoritative;
   full work should use Yale IIIF MS 408 at
   `https://collections.library.yale.edu/manifests/2002046`.
2. Break each train page into non-overlapping 32×32 image patches. Apply
   small random translations, contrast and noise augmentations; train a
   tiny CNN encoder by **NT-Xent contrastive loss** without EVA labels.
3. Quantize patch embeddings to 40 visual codewords by MiniBatchKMeans.
   Decode by mean training pixel patch of each codeword. This is a
   crude **centroid-based decoder**, not VQGAN or a generative image
   autoencoder; the output will look blocky or blurred.
4. Fit a normalized spatial discrete-token generator with smoothed
   token probabilities conditioned on row bin, immediately left code,
   immediately above code, and their joint pair. Sample one new 384×512
   "page" code grid, render using the codebook; also render an IID patch
   sampling null.
5. Compare generated and heldout representation distributions:
   code-unigram and horizontal code-transition Jensen–Shannon
   divergences, a detector for trivially repeated visual patches, and
   image-space pseudo-script tokenization based on connected component
   positions and shape signatures.
6. **The pseudo-script extractor is NOT a Voynich handwriting recognizer.**
   Component clusters and whitespace gaps are weak visual surrogates.
   Do not compare its symbols to EVA word types or claim we have
   recovered Voynich linguistic word distributions.

Reproduce:
```bash
python experiments/visual_gen/train_pilot.py \
  --images data/visual_gen/f1r_full.png \
    data/visual_gen/f1v_full.png \
    data/visual_gen/f16v_full_enh.png \
    data/visual_gen/f17r_full.png \
    data/visual_gen/f34r_full_enh.png \
    data/visual_gen/f49v_full.png \
    data/visual_gen/f57v_full_enh.png \
    data/visual_gen/f58r_full.png \
    data/visual_gen/f76r_full.png \
    data/visual_gen/f78r_full.png \
    data/visual_gen/f88r_full_enh.png \
  --steps 160 --codes 40 --out results/visual_gen
```

The GitHub Actions workflow downloads all 11 public images,
installs the optional PyTorch/OpenCV dependencies, trains the model,
and archives the *actual* generated image, IID patch baseline,
heldout image samples, loss log and quantitative summary. Base tests
skip gracefully when image-specific dependencies are unavailable.

## Required next-stage scientific experiment

A credible model must **first separate script from illustrations/layout**,
with line and word bounding boxes. Train handwriting encoder on multiple
whole heldout quires, not just individual pages or cropped patches. Replace
centroid patches with a learned VQ autoencoder + masked latent generative
model (MAGE-style, or a VQ latent diffusion model), condition hierarchical
state on page, line and glyph geometry, and train on much more image data.
Strong negatives: uniform image patches, text patches shuffled within page,
predecessor-line shuffle, an artificial pseudoglyph corpus, genuine
self-citation generator rendering, and nearest-neighbor collage.

Most importantly, develop and validate a real *visual-to-EVA recognizer*:
pair line crops with ZL3b/IT2a text, allow known transcription
disagreement, keep full heldout quires, report CER/WER on real scans,
confidence/abstention and segmentation precision before interpreting
generated images' putative text. Only generated *high-confidence
transcriptions* may be compared with the real manuscript's
word-type distribution, bigrams, edit neighbors, conditional entropy,
forward/backward asymmetry and local family innovation.

Use the **same** recognizer on real heldout images and generated images
so recognizer biases affect both sides; include a human-verified
subset and a known-language handwritten positive control.

All generated outputs must remain labeled synthetic, to avoid
confusion with genuine folio images.
