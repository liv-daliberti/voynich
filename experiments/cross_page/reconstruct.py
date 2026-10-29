#!/usr/bin/env python3
"""Prospective cross-page one-EVA-glyph reconstruction and distance decay.

Each source page nominates absent forms using ONLY its own word types.
Every later page is an independent evaluation target. Nulls permute pages
within Currier strata while keeping their types, candidate lists and sizes.
A matched distant-source comparison targets the very SAME evaluation page.
No inference of historical writing order or linguistic meaning is made.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.robustness75.robustness import glyphs, one_edit_forms
from experiments.english_controls.benchmark import character_edits, clean_gutenberg
from experiments.page_graphs.compare import parse_voynich
from experiments.page_graphs.reconstruct_controls import matched_pages

LAGS = (1, 2, 3, 5, 10, 20, 40)


def generated_words(text):
    """Published Java generator format: comments start #, text is ASCII EVA."""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        tokens = re.findall(r"[a-z]+", line.lower())
        if tokens:
            lines.extend(tokens)
    return lines


def native_generated_pages(text, lines_per_page=29):
    """Keep original Java generator line breaks and published page size."""
    lines=[]
    for raw in text.splitlines():
        raw=raw.strip()
        if not raw or raw.startswith("#"):continue
        words=re.findall(r"[a-z]+",raw.lower())
        if words:lines.append(words)
    pages=[]
    for first in range(0,len(lines),lines_per_page):
        content=[w for line in lines[first:first+lines_per_page] for w in line]
        pages.append({"folio":f"native_{first//lines_per_page:04d}",
                      "language":"G","words":content})
    return pages


def source_model(page, eva=True, top_k=50, min_support=2):
    """No target-page words, alphabet, or outcomes are used."""
    forms = set(page["words"])
    unitize = glyphs if eva else tuple
    alphabet = sorted({unit for word in forms for unit in unitize(word)})
    degree = Counter()
    for word in forms:
        edits = (one_edit_forms(unitize(word), alphabet) if eva
                 else character_edits(word, tuple(alphabet)))
        for proposed in edits:
            if proposed not in forms:
                degree[proposed] += 1
    eligible = {w for w,n in degree.items() if n >= min_support}
    # Fixed, deterministic support-first ranking, no target-dependent ties.
    ranked = sorted(eligible, key=lambda w:(-degree[w],w))
    return {"vocab":forms, "candidate":eligible,
            "top":set(ranked[:top_k]),"top_order":ranked[:top_k],
            "candidate_count":len(eligible),
            "top_k_actual":min(top_k,len(eligible)),
            "training_tokens":len(page["words"]),
            "training_types":len(forms)}


def score_pair(source, target_set):
    """Target types absent from SOURCE, not globally unseen or necessarily new."""
    novel = target_set - source["vocab"]
    candidate = source["candidate"]
    top = source["top"]
    hit = len(top & novel)
    reachable = len(candidate & novel)
    k=source["top_k_actual"]
    expected = (k*reachable/source["candidate_count"]
                if source["candidate_count"] else 0.)
    return {
        "hits":hit, "novel":len(novel), "reachable":reachable,
        "topk":k, "precision":hit/k if k else 0.,
        "novel_recall":hit/len(novel) if novel else 0.,
        "candidate_coverage":reachable/len(novel) if novel else 0.,
        "random_hits":expected,
        "word_jaccard":len(source["vocab"] & target_set) /
            max(1,len(source["vocab"] | target_set))}


def score_matrices(pages,eva=True,top_k=50,min_support=2):
    models=[source_model(p,eva,top_k,min_support) for p in pages]
    targets=[set(p["words"]) for p in pages]
    metrics=("hits","novel","reachable","topk","precision",
             "novel_recall","candidate_coverage","random_hits","word_jaccard")
    arrays={metric:np.zeros((len(pages),len(pages))) for metric in metrics}
    for i,model in enumerate(models):
        for j,forms in enumerate(targets):
            if i==j: continue
            measure=score_pair(model,forms)
            for name in metrics:arrays[name][i,j]=measure[name]
    return models,arrays


def group_indices(pages):
    groups={"physical_all":list(range(len(pages)))}
    for l in sorted({p["language"] for p in pages}):
        ids=[i for i,p in enumerate(pages) if p["language"]==l]
        if ids:groups[l]=ids
    return groups


def summarize_lag(arrays, ids, lag, backwards=False):
    n=len(ids)
    if lag>=n:return None
    if backwards:
        src=np.asarray(ids[lag:]);tar=np.asarray(ids[:-lag])
    else:
        src=np.asarray(ids[:-lag]);tar=np.asarray(ids[lag:])
    denom=len(src)
    out={"n_pairs":denom}
    for key,a in arrays.items():
        out["mean_"+key]=round(float(a[src,tar].mean()),6)
    return out


def ordered_diagnostics(arrays,ids,lags=LAGS,randomizations=300,seed=20261028,
                        fixed_language_permutation=False,all_pages=None):
    results={}
    if len(ids)<25:return results
    rng=np.random.default_rng(seed)
    for d in lags:
        if d>=len(ids):continue
        forward=summarize_lag(arrays,ids,d)
        reverse=summarize_lag(arrays,ids,d,True)
        # For the physical_all series, shuffle pages WITHIN A/B slots only.
        outcomes={kind:[] for kind in ("precision","hits","novel_recall")}
        if randomizations:
            ii=np.asarray(ids)
            for _ in range(randomizations):
                if fixed_language_permutation:
                    shuffled=ii.copy()
                    labels=[all_pages[k]["language"] for k in ii]
                    for lang in set(labels):
                        slots=np.asarray([z for z,x in enumerate(labels)
                                          if x==lang],dtype=int)
                        shuffled[slots]=rng.permutation(shuffled[slots])
                else:
                    shuffled=rng.permutation(ii)
                for field in outcomes:
                    a=arrays[field]
                    outcomes[field].append(float(
                        a[shuffled[:-d],shuffled[d:]].mean()))
        null={}
        for field,draw in outcomes.items():
            if not draw:continue
            sample=np.asarray(draw)
            actual=forward["mean_"+field]
            null[field]={"mean":round(float(sample.mean()),6),
                "null_ci95":[round(float(v),6) for v in
                             np.quantile(sample,(.025,.975))],
                "excess":round(actual-float(sample.mean()),6),
                "one_sided_p":round(float(
                    (1+np.count_nonzero(sample>=actual))/(len(sample)+1)),6)}
        results[str(d)]={"forward":forward,"reverse_placebo":reverse,
                         "page_order_null":null}
    return results


def match_distant_sources(pages,models,arrays,ids,far_distance=20):
    """Compare source one page earlier vs closest SIZE-MATCHED source >=20
    earlier; exact same target, no candidate-label-informed selection."""
    comparisons=[]
    for j in range(far_distance,len(ids)):
        close=ids[j-1];target=ids[j]
        options=ids[:j-far_distance+1]
        far=min(options,key=lambda src:(
            abs(models[src]["training_types"]-models[close]["training_types"]),
            abs(models[src]["training_tokens"]-models[close]["training_tokens"]),
            src))
        comparisons.append({
            "target":pages[target]["folio"],"close":pages[close]["folio"],
            "far":pages[far]["folio"],"target_size":len(set(pages[target]["words"])),
            "source_size_close":models[close]["training_types"],
            "source_size_far":models[far]["training_types"],
            "precision_close":arrays["precision"][close,target],
            "precision_far":arrays["precision"][far,target],
            "hits_close":arrays["hits"][close,target],
            "hits_far":arrays["hits"][far,target],
            "random_excess_close":arrays["hits"][close,target]-
                                  arrays["random_hits"][close,target],
            "random_excess_far":arrays["hits"][far,target]-
                                arrays["random_hits"][far,target]})
    if not comparisons:return {"targets":0}
    summary={"targets":len(comparisons),"max_far_offset":len(ids)}
    for key in ("precision","hits","random_excess"):
        diffs=np.asarray([x[f"{key}_close"]-x[f"{key}_far"]
                          for x in comparisons])
        summary[f"{key}_close_minus_far"]=round(float(diffs.mean()),6)
        rng=np.random.default_rng(20261028)
        sample=[float(np.mean(rng.choice(diffs,size=len(diffs),replace=True)))
                for _ in range(1500)]
        summary[f"{key}_ci95"]=[round(float(x),6)
                                for x in np.quantile(sample,(.025,.975))]
    summary["mean_source_type_size_difference"]=round(float(
        np.mean([abs(x["source_size_close"]-x["source_size_far"])
                 for x in comparisons])),2)
    summary["examples"]=comparisons[:8]
    return summary


def run_pages(pages,eva=True,top_k=50,min_support=2,
              lags=LAGS,shuffles=300,name="Voynich"):
    models,matrices=score_matrices(pages,eva,top_k,min_support)
    groups=group_indices(pages)
    result={"name":name,"pages":len(pages),
            "tokens":sum(len(p["words"]) for p in pages),
            "types":len(set(w for p in pages for w in p["words"])),
            "candidate_ranker":"unique one-edit witnesses, deterministic ties",
            "min_support":min_support,"top_k":top_k,
            "within_variety_lags":"subsequence of that Currier variety",
            "physical_all_lags":"original observed folio order including mixed A/B",
            "groups":{}}
    for group,ids in groups.items():
        result["groups"][group]={
            "pages":len(ids),
            "lag_decay":ordered_diagnostics(
                matrices,ids,lags,shuffles,seed=20261028,
                fixed_language_permutation=(group=="physical_all"),
                all_pages=pages),
            "paired_near_vs_size_matched_far":
                match_distant_sources(pages,models,matrices,ids)}
    return result


def load_cohort(zl,austen=None,naibbe=None,generated=None,mode="voynich"):
    pages=parse_voynich(Path(zl).read_text(encoding="utf-8-sig"))
    if mode=="voynich":return pages,True
    if mode=="austen":
        words=clean_gutenberg(Path(austen).read_text(encoding="utf-8-sig"),"austen")
    elif mode=="naibbe":
        words=re.findall(r"[a-z]+",Path(naibbe).read_text(
            encoding="utf-8-sig").lower())
    elif mode=="timm":
        words=generated_words(Path(generated).read_text(encoding="utf-8-sig"))
    else:raise ValueError(mode)
    if mode=="timm":
        # Protect direct mechanism's native page structure as a secondary
        # sensitivity test by constructing 29 physical lines per page.
        # This primary cohort uses identical token counts to real folios.
        pass
    return matched_pages(pages,words,mode),mode!="austen"


def run(zl,out,mode="voynich",austen=None,naibbe=None,
        generated=None,seed=None,shuffles=300,top_k=50):
    pages,eva=load_cohort(zl,austen,naibbe,generated,mode)
    summary=run_pages(pages,eva,top_k=top_k,shuffles=shuffles,name=mode)
    summary["seed"]=seed
    if mode=="timm":
        raw=Path(generated).read_text(encoding="utf-8-sig")
        native=native_generated_pages(raw,lines_per_page=29)
        summary["native_generator_page_sensitivity"]=run_pages(
            native,True,top_k=top_k,shuffles=shuffles,
            name="timm_native_lines_per_page_29")
        print("timm native",seed,"pages",len(native),
              "lag1",summary["native_generator_page_sensitivity"]["groups"]["G"]
                    ["lag_decay"]["1"]["forward"]["mean_novel_recall"],flush=True)
    summary["source_sha256"]={
       "zl":hashlib.sha256(Path(zl).read_bytes()).hexdigest()}
    for label,file in (("austen",austen),("naibbe",naibbe),
                       ("generated",generated)):
        if mode==label or (mode=="timm" and label=="generated"):
            summary["source_sha256"][label]=hashlib.sha256(
                Path(file).read_bytes()).hexdigest()
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(summary,indent=2)+"\n")
    for key in ("A","B","physical_all"):
        if key not in summary["groups"]:continue
        g=summary["groups"][key]
        lag1=g["lag_decay"].get("1",{}).get("forward",{})
        lag20=g["lag_decay"].get("20",{}).get("forward",{})
        print(mode,"seed",seed,key,"pages",g["pages"],
              "hit_precision_lag1",lag1.get("mean_precision"),
              "lag20",lag20.get("mean_precision"),
              "matched_distant",g["paired_near_vs_size_matched_far"].
                                    get("precision_close_minus_far"),
              flush=True)
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--zl",required=True)
    p.add_argument("--austen")
    p.add_argument("--naibbe")
    p.add_argument("--generated")
    p.add_argument("--mode",choices=("voynich","austen","naibbe","timm"),
                   required=True)
    p.add_argument("--seed",type=int)
    p.add_argument("--top-k",type=int,default=50)
    p.add_argument("--shuffles",type=int,default=300)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    run(a.zl,a.out,a.mode,a.austen,a.naibbe,a.generated,a.seed,
        a.shuffles,a.top_k)
