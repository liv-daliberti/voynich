# Manuscript-image adjudication protocol

**Status: awaiting original-image annotation.** The model has not verified individual manuscript glyphs from the page images. Do not call a model-selected alternate the true reading.

Source: Yale Beinecke MS 408, https://collections.library.yale.edu/catalog/2002046 and IIIF manifest https://collections.library.yale.edu/manifests/2002046 . Match folio IDs carefully to IIIF canvases; verify folio recto/verso and orientation manually.

For each candidate, record: folio, line identifier, EVA token index, aligned glyph index, ZL spelling, IT spelling, crop/canvas URL, whether glyph segmentation agrees, ZL reading plausible (yes/no/uncertain), IT reading plausible (yes/no/uncertain), third reading if needed, image resolution, adjudicator ID, blind/revealed status, date, and notes. Require two blind readers and disagreements adjudicated by a third.

Select **two separate evaluation sets** by whole folios: (a) disagreement-enriched cases; (b) random agreed locations (negative controls). Do not estimate real-world accuracy from disagreement-enriched cases alone. Blind annotators to the model's score and the preferred transcription. Evaluate confidence calibration, per-glyph confusion, AUPRC against prevalence, grouped AUC and bootstrapped 95% confidence intervals by leaf. Release crop coordinates and source links, not unsupported ground-truth labels.
