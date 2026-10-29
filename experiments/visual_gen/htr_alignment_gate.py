#!/usr/bin/env python3
"""Fail-closed readiness audit for Voynich image-to-EVA recognition.

A manuscript line crop must be paired with its *correct* ZL transcription
line and reviewed for geometric alignment. The model does NOT assume that
the nth detected image stripe is the nth transcribed line without review.
This audit reports potential automatically matched folios but refuses
to call generated connected components EVA word forms.

A future HTR training script can consume manually verified line pairs
via verified_alignment.json; it must report heldout quire CER/WER before
synthetic text is analyzed linguistically.
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path
from experiments.context.slot_filling import voynich_lines


def eva_glyphs(word):
    from experiments.robustness75.robustness import glyphs
    return glyphs(word)


def audit(manifest,zl,source_line_margin=.2):
    data=json.loads(Path(manifest).read_text())
    text=Path(zl).read_text(encoding="utf-8-sig")
    trans=defaultdict(list)
    for lang in ("A","B"):
        for folio,words in voynich_lines(text,lang):
            trans[folio].append(words)
    boxes=defaultdict(list)
    for row in data["rows"]:
        boxes[row["folio"]].append(row)
    rows=[]
    for folio,detected in sorted(boxes.items()):
        target=trans.get(folio,[])
        actual_n=len(detected);trans_n=len(target)
        ratio=abs(actual_n-trans_n)/max(actual_n,trans_n,1)
        rows.append({
            "folio":folio,"quire":detected[0]["quire"],
            "split":detected[0]["split"],
            "detected_script_line_crops":actual_n,
            "paragraph_lines_in_ZL":trans_n,
            "relative_count_error":round(ratio,4),
            "maybe_count_compatible":ratio<=source_line_margin,
            "geom_alignment_verified":False,
            "ocr_training_admissible":False,
            "reason":"Folio line counts alone cannot verify handwritten crop alignment"})
    candidates=sum(r["maybe_count_compatible"] for r in rows)
    return {"real_image_to_eva_status":"BLOCKED: no manually verified crop-to-EVA alignments",
            "number_folios":len(rows),
            "compatible_count_only":candidates,
            "alignment_rows":rows,
            "extraction_cer":None,
            "extraction_wer":None,
            "generated_eva_text":None,
            "generated_word_distribution_comparison":None,
            "required_to_unlock":[
                "Independent annotation of line bounding boxes and matching ZL IDs",
                "Train HTR on train quires only",
                "Evaluate real heldout quire CER and WER, confidence calibration",
                "Report corpus-level transcription uncertainty and abort on poor quality",
                "Only then transcribe generated script and compare with real word distributions"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--line-manifest",required=True)
    p.add_argument("--zl",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    result=audit(a.line_manifest,a.zl)
    out=Path(a.out);out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({
        "status":result["real_image_to_eva_status"],
        "folios":result["number_folios"],
        "count_compatible":result["compatible_count_only"]}))

if __name__=="__main__":
    main()
