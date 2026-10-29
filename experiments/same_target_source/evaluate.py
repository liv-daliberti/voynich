#!/usr/bin/env python3
"""Paired, same-target source-selection test with source metadata matching.

For target t compare page t-1 to a source >=20 *same-variety* pages earlier.
Fixed gold = target word types absent from BOTH sources. Rank each gold form
using only that source's observed vocabulary and its one-edit proposal graph.
No target labels affect source choice or candidate formation.

Separate strict (same section AND hand; adjacent target same stratum),
section-only, and relaxed comparisons. Report coverage, rank, conditional
rank among jointly supported gold, and chance-adjusted top-50 recall.
No claim of writing chronology or decipherment.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.robustness75.robustness import glyphs, one_edit_forms
from experiments.english_controls.benchmark import clean_gutenberg, character_edits
from experiments.page_graphs.compare import parse_voynich
from experiments.page_graphs.reconstruct_controls import matched_pages
from experiments.cross_page.reconstruct import (
    generated_words, native_generated_pages,
)

SCHEMES = ("strict_section_hand", "same_section", "same_variety")
TOP_K = 50


def precompute_source(page, eva=True, top_k=TOP_K):
    """Candidate ranks are pure functions of SOURCE words and glyph alphabet."""
    forms=set(page["words"])
    units=glyphs if eva else tuple
    abc=sorted({u for w in forms for u in units(w)})
    degree=Counter()
    for w in forms:
        variants=(one_edit_forms(units(w),abc) if eva else
                  character_edits(w,tuple(abc)))
        degree.update(x for x in variants if x not in forms)
    ranking=sorted((w for w,d in degree.items() if d>=2),
                   key=lambda w:(-degree[w],w))
    return {"vocab":forms,"rank":{w:i+1 for i,w in enumerate(ranking)},
            "n_candidates":len(ranking),"top":set(ranking[:top_k]),
            "types":len(forms),"tokens":len(page["words"]),
            "topk":min(top_k,len(ranking))}


def matched_far_sources(pages,models,ids,j,min_lag=20,scheme="strict_section_hand",
                        type_tolerance=.25,max_matches=8):
    """Match nearest source t-1 with earlier source without seeing targets.

    Strict scheme requires previous and target within section/hand, with the
    distant source also matching both. Always require same variety.
    """
    if j<min_lag or j<1:return []
    near_idx=ids[j-1]
    tar_idx=ids[j]
    near=pages[near_idx]
    target=pages[tar_idx]
    section=near.get("section_code","?")
    hand=near.get("hand_code","?")
    if scheme=="strict_section_hand":
        if (target.get("section_code","?")!=section or
            target.get("hand_code","?")!=hand):return []
    elif scheme=="same_section":
        if target.get("section_code","?")!=section:return []
    selected=[]
    for k in range(0,j-min_lag+1):
        idx=ids[k]
        p=pages[idx]
        if scheme in ("strict_section_hand","same_section") and p.get(
            "section_code","?")!=section:continue
        if scheme=="strict_section_hand" and p.get("hand_code","?")!=hand:
            continue
        source_size=models[idx]["types"]
        nearby_size=models[near_idx]["types"]
        frac=abs(source_size-nearby_size)/max(nearby_size,1)
        if frac>type_tolerance:continue
        selected.append((frac,
            abs(models[idx]["tokens"]-models[near_idx]["tokens"])/
            max(models[near_idx]["tokens"],1),
            -k,idx))
    selected.sort()
    return [x[-1] for x in selected[:max_matches]]


def score_identical_gold(near,far,target):
    """Exactly the SAME novel gold set, same target, each source's own rank."""
    gold=(target-near["vocab"])-far["vocab"]
    if not gold:return None
    K=TOP_K
    def calc(model):
        supported=gold&model["rank"].keys()
        hits=len(gold&model["top"])
        expected=(model["topk"]*len(supported)/model["n_candidates"]
                  if model["n_candidates"] else 0.)
        # Raw rank is not comparable across different candidate-set sizes.
        return {"hits":hits,"coverage":len(supported)/len(gold),
                "precision":hits/max(model["topk"],1),
                "recall":hits/len(gold),
                "chance_adjusted_recall":(hits-expected)/len(gold),
                "expected_hits":expected}
    n=calc(near);f=calc(far)
    common=gold & near["rank"].keys() & far["rank"].keys()
    def rank_quality(model,w):
        N=model["n_candidates"];r=model["rank"][w]
        # 1 for best-ranked proposal, 0 for worst; normalized by pool size.
        return 1. if N<2 else 1-(r-1)/(N-1)
    rank_difference=(sum(rank_quality(near,w)-rank_quality(far,w)
                         for w in common)/len(common) if common else None)
    out={"n_gold":len(gold),"n_common_proposed":len(common),
         "near":n,"far":f,"mean_common_rank_percentile_delta":rank_difference}
    for key in ("recall","coverage","precision","chance_adjusted_recall"):
        out[f"delta_{key}"]=n[key]-f[key]
    return out


def choose_source_pairs(pages,models,ids,scheme,min_lag=20,
                        size_tolerance=.25,max_matches=8):
    pairs=[]
    for j in range(min_lag,len(ids)):
        tar=ids[j];near=ids[j-1]
        for far in matched_far_sources(pages,models,ids,j,min_lag,scheme,
                                       size_tolerance,max_matches):
            pairs.append((tar,near,far))
    return pairs


def evaluate_pairs(pages,models,pairs):
    """Use one best size-matched far per target as primary paired estimate.

    Repeat all alternative far choices as secondary sensitivity, with the
    exact target novel gold fixed within each near/far comparison.
    """
    bytarget=defaultdict(list)
    for tar,near,far in pairs:
        bytarget[tar].append((near,far))
    rows=[]
    for tar,choices in bytarget.items():
        # Primary far chosen from predeclared source size ranking only.
        near,far=choices[0]
        rec=score_identical_gold(models[near],models[far],set(pages[tar]["words"]))
        if rec is None:continue
        rec.update({"folio":pages[tar]["folio"],"target_idx":tar,
                    "near_idx":near,"far_idx":far,
                    "near_folio":pages[near]["folio"],
                    "far_folio":pages[far]["folio"],
                    "near_types":models[near]["types"],
                    "far_types":models[far]["types"],
                    "size_diff":abs(models[near]["types"]-models[far]["types"]),
                    "target_types":len(set(pages[tar]["words"])),
                    "section_code":pages[tar].get("section_code","?"),
                    "hand_code":pages[tar].get("hand_code","?"),
                    "n_alternate_far":len(choices)})
        rows.append(rec)
    return rows


def weighted_result(rows,field,weight="gold"):
    valid=[row for row in rows if row[field] is not None]
    if not valid:return None
    w=np.asarray([x["n_gold"] if weight=="gold"
                  else (x["n_common_proposed"] if weight=="common"
                        else 1) for x in valid],float)
    if w.sum()<=0:return None
    y=np.asarray([x[field] for x in valid])
    return float(np.average(y,weights=w))


def confidence(rows,field,repetitions=1500,seed=20261028,weight="gold"):
    """Cluster by target folio; no pairwise candidate observations as iid."""
    valid=[r for r in rows if r[field] is not None]
    if not valid:return {"mean":None,"ci95":None,"targets":0}
    point=weighted_result(valid,field,weight)
    rng=np.random.default_rng(seed)
    sample=[]
    for _ in range(repetitions):
        indices=rng.integers(0,len(valid),len(valid))
        sample.append(weighted_result([valid[int(i)] for i in indices],
                                      field,weight))
    return {"mean":round(point,6),
            "ci95":[round(float(v),6) for v in np.quantile(sample,[.025,.975])],
            "targets":len(valid)}


def test_conditions(pages,eva,scheme="strict_section_hand",
                    min_lag=20,size_tolerance=.25,repetitions=1000):
    models=[precompute_source(p,eva) for p in pages]
    ids_by_group=defaultdict(list)
    for i,p in enumerate(pages):ids_by_group[p["language"]].append(i)
    aggregate={}
    primary_rows={}
    for lang,ids in ids_by_group.items():
        pairs=choose_source_pairs(pages,models,ids,scheme,min_lag,
                                  size_tolerance)
        rows=evaluate_pairs(pages,models,pairs)
        primary_rows[lang]=rows
        metrics={k:confidence(rows,k,repetitions,weight=(
            "common" if k=="mean_common_rank_percentile_delta" else "gold"))
            for k in ("delta_recall","delta_chance_adjusted_recall",
                      "delta_coverage","delta_precision",
                      "mean_common_rank_percentile_delta")}
        aggregate[lang]={"eligible_targets":len(rows),
                         "preselected_pairs":len(pairs),
                         "distinct_source_target_folios":len(ids),
                         "mean_size_difference":round(np.mean(
                             [r["size_diff"] for r in rows]),3)
                             if rows else None,
                         "mean_fixed_novel_forms":round(np.mean(
                             [r["n_gold"] for r in rows]),3)
                             if rows else None,
                         "matched_results":metrics}
    return aggregate,primary_rows


def permutation_null(pages,eva,scheme="strict_section_hand",min_lag=20,
                     size_tolerance=.25,trials=120,seed=20261028):
    """Permute page assignments within same language, SECTION AND HAND,
    and source-size tertile; recompute source selection and same-target gold.

    Ordering is arbitrary within metadata-size strata under this null.
    This tests whether *original* near-source advantage exceeds such
    constrained orderings; it cannot establish historical production order.
    """
    rng=np.random.default_rng(seed)
    groups=defaultdict(list)
    models=[precompute_source(p,eva) for p in pages]
    by_key=defaultdict(list)
    for idx,p in enumerate(pages):
        by_key[(p["language"],p.get("section_code","?"),
                p.get("hand_code","?"))].append(idx)
    for key,ids in by_key.items():
        ordered=sorted(ids,key=lambda i:models[i]["types"])
        for rank,idx in enumerate(ordered):
            # Half-bin with a minimum of 2 pages per source-size band
            size_bin=min(2,(3*rank)//max(1,len(ordered)))
            groups[key+(size_bin,)].append(idx)
    observed,_=test_conditions(pages,eva,scheme,min_lag,
                                size_tolerance,250)
    null=defaultdict(list)
    for trial in range(trials):
        order=list(range(len(pages)))
        for ids in groups.values():
            perm=rng.permutation(ids)
            for slot,original in zip(ids,perm):
                order[slot]=int(original)
        shuffled=[pages[i] for i in order]
        # Avoid refitting candidate source ranking when only page order changes.
        shuffled_models=[models[i] for i in order]
        grouped=defaultdict(list)
        for i,p in enumerate(shuffled):grouped[p["language"]].append(i)
        for lang,ids in grouped.items():
            pairs=choose_source_pairs(shuffled,shuffled_models,ids,scheme,
                                     min_lag,size_tolerance)
            rows=evaluate_pairs(shuffled,shuffled_models,pairs)
            value=weighted_result(rows,"delta_chance_adjusted_recall")
            if value is not None:
                null[lang].append(value)
    summary={}
    for lang,result in observed.items():
        score=result["matched_results"]["delta_chance_adjusted_recall"]["mean"]
        samples=np.asarray(null.get(lang,[]),float)
        if score is None or not len(samples):
            summary[lang]={"available":False,"trial_count":len(samples)}
            continue
        summary[lang]={
            "available":True,
            "observed_delta":score,
            "null_mean":round(float(samples.mean()),6),
            "null_ci95":[round(float(q),6)
                         for q in np.quantile(samples,[.025,.975])],
            "excess_over_null":round(score-float(samples.mean()),6),
            "one_sided_p":round(float((1+(samples>=score).sum())/
                                       (len(samples)+1)),6),
            "trial_count":len(samples)}
    return summary


def run(pages,eva,out,name="Voynich",shuffles=100):
    result={"cohort":name,"pages":len(pages),"tokens":sum(
        len(p["words"]) for p in pages),
        "method":"same target gold novel to BOTH near/far source pages"}
    all_rows={}
    for scheme in SCHEMES:
        groups,rows=test_conditions(pages,eva,scheme)
        result[scheme]={"observed":groups}
        all_rows[scheme]=rows
        # Shuffled order within section/hand, not a target-word shuffle.
        result[scheme]["page_order_null"]=permutation_null(
            pages,eva,scheme,trials=shuffles)
        print(name,scheme,json.dumps(result[scheme]["observed"]),flush=True)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    for scheme,groups in all_rows.items():
        p=out.with_name(out.stem+"_"+scheme+"_paired_targets.json")
        p.write_text(json.dumps(groups,indent=2)+"\n")
    return result


def get_pages(zl,mode,austen=None,naibbe=None,generated=None,
              native=False):
    p=parse_voynich(Path(zl).read_text(encoding="utf-8-sig"))
    if mode=="voynich":return p,True
    if mode=="austen":
        words=clean_gutenberg(Path(austen).read_text(encoding="utf-8-sig"),
                              "austen")
    elif mode=="naibbe":
        words=re.findall(r"[a-z]+",Path(naibbe).read_text(
            encoding="utf-8-sig").lower())
    elif mode=="timm":
        text=Path(generated).read_text(encoding="utf-8-sig")
        if native:
            q=native_generated_pages(text)
            for x in q:
                x["section_code"]="native"
                x["hand_code"]="?"
            return q,True
        words=generated_words(text)
    else:raise ValueError(mode)
    return matched_pages(p,words,mode),mode!="austen"


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--zl",required=True)
    ap.add_argument("--austen")
    ap.add_argument("--naibbe")
    ap.add_argument("--generated")
    ap.add_argument("--mode",choices=["voynich","austen","naibbe","timm"],
                    required=True)
    ap.add_argument("--native",action="store_true")
    ap.add_argument("--shuffles",type=int,default=100)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    p,eva=get_pages(args.zl,args.mode,args.austen,args.naibbe,
                    args.generated,args.native)
    run(p,eva,args.out,
        name=args.mode+("_native" if args.native else ""),
        shuffles=args.shuffles)


if __name__=="__main__":
    main()
