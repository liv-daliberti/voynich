#!/usr/bin/env python3
"""Re-evaluate actual fresh Timm/Schinner generator output with identical assays.

Uses the released Java generator, **not a hand-made reimplementation**.
Source forms are parsed as EVA, and synthetic pages are reconstructed using
the published lines-per-page configuration. Five seeds vary randomness only.

All measurement routines are imported from the existing research experiments.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.higher_order_direction.evaluate import one_corpus
from experiments.local_copy.evaluate import run_corpus
from experiments.phonotactics.evaluate import make_rows, fit_predict, degree_length_matched_auc, metrics


def parse_generated(text: str, lines_per_page=29):
    """Skip recorded parameters and statistics, preserve generator text lines."""
    paragraphs=[]
    for raw in text.replace("\r\n","\n").splitlines():
        raw=raw.strip()
        if not raw or raw.startswith("#"):
            continue
        words=re.findall(r"[a-z]+",raw.lower())
        if words:
            paragraphs.append(words)
    if not paragraphs:
        raise ValueError("No generated text lines")
    pages=[]
    for i,line in enumerate(paragraphs):
        page_id=f"gen_{i//lines_per_page:04d}"
        if not pages or pages[-1]["folio"]!=page_id:
            pages.append({"folio":page_id,"language":"A","words":[]})
        pages[-1]["words"].extend(line)
    sequences=[(f"gen_{i//lines_per_page:04d}",line)
               for i,line in enumerate(paragraphs)]
    return pages,sequences


def prospective(pages):
    n=len(pages)
    early=int(.5*n);cut=int(.7*n)
    inner,inner_meta=make_rows(pages[:early],pages[early:cut])
    outer,outer_meta=make_rows(pages[:cut],pages[cut:])
    y=[row["attested_later"] for row in outer]
    summaries={"inner":inner_meta,"outer":outer_meta,
               "degree_length_matched_bigram":degree_length_matched_auc(outer)}
    for name,cols in (
        ("degree_length",["log_degree","glyph_length"]),
        ("degree_length_bigram",["log_degree","glyph_length","bigram"])):
        scores=fit_predict(inner,outer,cols)
        summaries[name]=metrics(y,scores)
    summaries["raw_degree"]=metrics(y,[r["degree"] for r in outer])
    summaries["sample_future_new"]=[x["word"] for x in outer
                                   if x["attested_later"]][:20]
    return summaries


def run_text(generated_file,output,seed=None,
             local_permutations=80,strong_permutations=50,
             run_phonotactics=True,run_higher_order=True):
    payload=Path(generated_file).read_bytes()
    pages,seq=parse_generated(payload.decode("utf-8-sig"))
    report={"seed":seed,
            "generated_text_sha256":hashlib.sha256(payload).hexdigest(),
            "synthetic_pages":len(pages),"text_lines":len(seq),
            "tokens":sum(len(w) for _,w in seq),
            "distinct_word_types":len({w for _,line in seq for w in line}),
            "source":"Timm-Schinner unmodified published Java generator; seed varied"}
    if run_phonotactics:
        report["prospective_attestation"]=prospective(pages)
    if run_higher_order:
        model=one_corpus(seq,folds=4,max_vocab=1200,seed=20261028)
        report["higher_order_four_folds"]=model["aggregate"]
        report["higher_order_smoothing_by_fold"]=[
            {"fold":x["fold"],"direction":direction,
             "alpha":d["alpha"],"position_alpha":d["position_alpha"]}
            for x in model["folds"]
            for direction,d in x["directions"].items()]
    if local_permutations:
        local=run_corpus(seq,True,permutations=local_permutations,
                         seed=20261028,null_mode="token")
        report["local_copy_token_shuffle"]=local
    if strong_permutations:
        strong=run_corpus(seq,True,permutations=strong_permutations,
                          seed=20261028,null_mode="line")
        report["local_copy_whole_line_shuffle"]=strong
    output=Path(output);output.parent.mkdir(exist_ok=True,parents=True)
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({
        "seed":seed,"lines":report["text_lines"],"tokens":report["tokens"],
        "prospective_auc":report.get("prospective_attestation",{}).get(
            "degree_length_bigram",{}).get("roc_auc"),
        "higher_order_reverse_specific_gain":report.get(
            "higher_order_four_folds",{}).get("fourgram",{}).get(
            "reverse_specific_gain",{}).get("mean"),
        "local_before_edit_excess":report.get("local_copy_token_shuffle",{}).
            get("metrics",{}).get("all",{}).
            get("previous_line_same_column",{}).get("one_edit",{}).
            get("excess_pp"),
        "local_before_after_excess":report.get("local_copy_token_shuffle",{}).
            get("directional_contrasts",{}).get(
            "all:vertical_one_line:one_edit",{}).get("excess_directional_pp")
    }),flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--generated",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--seed",type=int)
    p.add_argument("--local-permutations",type=int,default=80)
    p.add_argument("--strong-permutations",type=int,default=50)
    p.add_argument("--skip-phonotactics",action="store_true")
    p.add_argument("--skip-higher-order",action="store_true")
    a=p.parse_args()
    run_text(a.generated,a.out,seed=a.seed,
             local_permutations=a.local_permutations,
             strong_permutations=a.strong_permutations,
             run_phonotactics=not a.skip_phonotactics,
             run_higher_order=not a.skip_higher_order)
