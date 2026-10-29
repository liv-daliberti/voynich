# Real image-space Voynich generation pilot: negative result

Completed October 28–29, 2026. Real fully trained CPU prototype, source-tracked
Beinecke-derived manuscript scans, NOT an EVA-producing generator.

## Model actually trained

- 11 page scans from `d2kx/voynich-research/pages`, 9 complete train images,
  2 complete held-out images. Some have enhanced contrast; this is a small
  mixed-quire/mixed-content subset, not a representative image dataset.
  All source SHA-256s are archived.
- Resize to 384×512 grayscale and segment into 32×32 patches.
- A small convolutional encoder trained from scratch with actual NT-Xent
  contrastive loss for 160 optimizer steps, augmentation: +/-2 pixels translation,
  contrast jitter, Gaussian noise.
- MiniBatchKMeans on learned embeddings, 40 visual patch tokens.
- Mean original patch per cluster serves as a nonlearned *decoder*. This is
  a major blurring bottleneck and NOT a VQGAN decoder.
- Autoregressive 2D spatial patch-token model (row bucket, left token,
  above token, left/top pair with smoothing), generates one new page by sampling
  token by token, independently of the original EVA transcription.
- Control: IID token sampling from the same learned codebook, so both systems
  use exactly the same visual patch vocabulary.

Completed [successful pilot workflow](https://github.com/liv-daliberti/voynich/actions/runs/40813461191).
Earlier success [40813367111](https://github.com/liv-daliberti/voynich/actions/runs/40813367111)
uses the same training with initial *train*-reference statistics; the main run
adds honest held-out distribution comparisons.

## Primary results

Contrastive loss on sampled training patches at steps 0 → 159:
**4.8287 → 2.3793**. This proves the SSL optimization runs, not that the model
learned glyph semantics or a reusable writing representation.

| Metric | Spatial model | IID patch model | Better |
|---|---:|---:|---|
| JS visual-token frequency divergence from **held-out real images** | .31546 | .12378 | IID |
| JS horizontal-token transition divergence from **held-out real images** | .17691 | .16316 | IID |
| JS visual-token frequency divergence from train images | .12431 | .01528 | IID |
| JS horizontal-token transition divergence from train images | .20276 | .27299 | Spatial |
| Automatically extracted pseudo-glyph foreground components | 8 | 17 | Neither |

Two real held-out images have 116 and 102 components under the same
rough threshold/connected-component extractor. The generated images show
large abstract rectangular strokes and blank page textures, **NOT**
recognizable Voynich words or glyphs.

This model does NOT pass a sensible generative visual Turing test. Its
seeming lower train spatial-transition divergence is overfitting or a
weak spatial-correlation effect and **does not transfer to heldout pages**.
Even the IID patch model does better for both held-out visual metrics.

## Strict interpretation

No valid automated EVA transcript can be recovered from the generated page.
The included pseudo-script extractor measures crude connected-components
and gaps, NOT a Voynich HTR, and its “pseudo-word types” are NOT real words.
Do not calculate previous research's Voynich word-length, digraph, PMD,
one-edit graph, or quire innovation claims from those pseudo-code IDs.

This negative result is unsurprising given extreme limitations:
- 11 scans, including multiple illustration-rich pages.
- 9 training scans, much too little diversity.
- No text localization, script-only masking or line boxes.
- 32×32 *nonoverlapping* patch tokens have harsh seams.
- Centroid-patch decoder erases fine glyph strokes.
- 40 visual codes and one-step 2D Markov dependence cannot model long glyph
  sequences, words, lines, or page geometry.
- Source images have mixed contrast enhancement and page styles.

## Research roadmap for a real test

1. Source 200+ Yale Beinecke MS408 IIIF page images and document SHA and
   folio/quire/hand labels. Split **entire quires** into held-out test images,
   with separate validation. Never shuffle individual patches across splits.
2. Detect **script regions/lines** and exclude illustration/paint/body text
   entirely from the first writing-model training. Train positive-control
   word/line segmentor on hand-reviewed genuine scans.
3. Learn script patches by contrastive + masked latent image modeling
   (MAGE-style) or modern VQ-MAE, with a learned stroke-preserving decoder.
   Mask random patches plus contiguous strips and whole line spans, to force
   long-range visual context rather than local copying.
4. Learn hierarchical page layout, line flow and glyph-level generation;
   then produce hundreds of synthetic pages and compare to heldout
   *genuine image* structure and nearest-training-patch memorization.
5. Train/test a **real Voynich image-to-EVA recognizer** with line crops and
   aligned scholarly transcription, held-out quire CER/WER and abstention
   thresholds; use two transcriptions to bound uncertainty. Only if
   the recognizer performs adequately on real pages can it measure
   generated word distributions. Do not assume that standard English OCR
   is suitable for unidentified Voynich glyphs.
6. Compare real, generative, IID patch null, shuffled line patch null and
   Timm–Schinner synthetic text rendered in a separately learned stroke
   style using exactly the same recognizer/geometry pipeline. Test source
   support, novel forms, positional entropy, page persistence and
   word-family innovation.

Strong science needs distinct validation of visual style, transcribability,
generative novelty, and linguistic-level distributional match. **Current
pilot fails the first two.**

References: MAGE (Li et al., CVPR 2023), Yale Beinecke digital MS408;
learned-representation-guided diffusion for large image generation (2024).
