#!/usr/bin/env python3
"""Within-quire page-order predictability, conditional on same target.

Controls keep entire folios fixed and rearrange their assignment to positions
within each quire, optionally within the same illustration category and hand.
A source page NEVER sees its target-page word forms at ranking time.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.same_target_source.evaluate import precompute_source, score_identical_gold, get_pages
from experiments.page_graphs.compare import build_adjacency, graph_stats, shape_cos
from experiments.english_controls.benchmark import clean_gutenberg
from experiments.page_graphs.reconstruct_controls import matched_pages
from experiments.cross_page.reconstruct import generated_words

MODES={
    "within_quire":("language","quire_code"),
    "within_quire_section":("language","quire_code","section_code"),
    "within_quire_section_hand":("language","quire_code","section_code","hand_code"),
    "within_quire_section_hand_size":("language","quire_code","section_code","hand_code","size_bin"),
}
OUTCOMES=("target_recall","chance_adjusted_recall","candidate_coverage",
          "rank_percentile","word_jaccard","edge_jaccard","degree_cosine")


def get_controls(zl,austen=None,generated=None,cohort="Voynich"):
    pages,eva=get_pages(zl,"voynich")
    if cohort=="Voynich":
        return pages,True
    if cohort=="Austen":
        tokens=clean_gutenberg(Path(austen).read_text(encoding="utf-8-sig"),"austen")
        eva=False
    elif cohort=="Timm":
        tokens=generated_words(Path(generated).read_text(encoding="utf-8-sig"))
        eva=True
    else:raise ValueError(cohort)
    mapped=matched_pages(pages,tokens,cohort.lower())
    for a,b in zip(mapped,pages):
        for key in ("section_code","quire_code","hand_code"):
            a[key]=b[key]
    return mapped,eva


def strata(pages,keys):
    groups=defaultdict(list)
    meta=[k for k in keys if k!="size_bin"]
    for idx,p in enumerate(pages):
        groups[tuple(p.get(k,"?") for k in meta)].append(idx)
    if "size_bin" not in keys:
        return groups
    matched=defaultdict(list)
    for category,ids in groups.items():
        if len(ids)<6:
            matched[category+("all",)].extend(ids)
        else:
            ranked=sorted(ids,key=lambda i:(len(set(pages[i]["words"])),i))
            for j,idx in enumerate(ranked):
                matched[category+(min(2,(3*j)//len(ranked)),)].append(idx)
    return matched


def permute(pages,keys,rng):
    order=np.arange(len(pages),dtype=int)
    groups=strata(pages,keys)
    for ids in groups.values():
        if len(ids)>1:
            order[ids]=rng.permutation(ids)
    return order


def graph_pairs(pages):
    adj=build_adjacency(pages)
    graphs=[graph_stats(p,adj) for p in pages]
    nodes=[x[1] for x in graphs]
    edges=[x[2] for x in graphs]
    shapes=[x[0]["degree_histogram"] for x in graphs]
    return nodes,edges,shapes


def jaccard(a,b):
    return len(a&b)/len(a|b) if a or b else 1.


def source_folio_metrics(near,far,target,pages,models,graphs):
    ni,fi,ti=near,far,target
    score=score_identical_gold(models[ni],models[fi],set(pages[ti]["words"]))
    if score is None:
        return None
    nodes,edges,shapes=graphs
    metrics={
        "target_recall":score["delta_recall"],
        "chance_adjusted_recall":score["delta_chance_adjusted_recall"],
        "candidate_coverage":score["delta_coverage"],
        "rank_percentile":score["mean_common_rank_percentile_delta"],
        "word_jaccard":jaccard(nodes[ni],nodes[ti])-jaccard(nodes[fi],nodes[ti]),
        "edge_jaccard":jaccard(edges[ni],edges[ti])-jaccard(edges[fi],edges[ti]),
        "degree_cosine":shape_cos(shapes[ni],shapes[ti])-
                        shape_cos(shapes[fi],shapes[ti])}
    return {"folio":pages[ti]["folio"],"near_folio":pages[ni]["folio"],
            "far_folio":pages[fi]["folio"],"quire_code":pages[ti]["quire_code"],
            "section_code":pages[ti]["section_code"],"language":pages[ti]["language"],
            "n_gold":score["n_gold"],
            "n_common_proposed":score["n_common_proposed"],
            "size_difference":abs(models[ni]["types"]-models[fi]["types"]),
            "values":metrics}


def select_pairs(pages,models,order,require_section_hand=False,
                 far_min_quire_lag=2,size_tolerance=.30):
    """For each target in a quire, choose its directly preceding folio side
    and one earlier source 2+ quire positions behind, matched on source
    vocabulary size; no target words enter that choice.
    """
    byquire=defaultdict(list)
    for idx in order:
        p=pages[int(idx)]
        byquire[(p["language"],p["quire_code"])].append(int(idx))
    pairs=[]
    for (lang,q),ids in byquire.items():
        for t in range(far_min_quire_lag,len(ids)):
            near=ids[t-1];tar=ids[t]
            if require_section_hand and (
                pages[tar]["section_code"]!=pages[near]["section_code"] or
                pages[tar]["hand_code"]!=pages[near]["hand_code"]):
                continue
            choices=[]
            for r in range(t-far_min_quire_lag+1):
                far=ids[r]
                if require_section_hand and (
                    pages[far]["section_code"]!=pages[near]["section_code"] or
                    pages[far]["hand_code"]!=pages[near]["hand_code"]):
                    continue
                difference=abs(models[far]["types"]-models[near]["types"])
                rel=difference/max(models[near]["types"],1)
                if rel>size_tolerance:continue
                choices.append((rel,
                    abs(models[far]["tokens"]-models[near]["tokens"]),
                    -r,far))
            if choices:
                choices.sort()
                pairs.append((near,choices[0][-1],tar))
    return pairs


def evaluate_order(pages,models,graph,order,strict=False):
    pairs=select_pairs(pages,models,order,require_section_hand=strict)
    values=[]
    for near,far,t in pairs:
        row=source_folio_metrics(near,far,t,pages,models,graph)
        if row:
            values.append(row)
    return values


def group_stats(rows,field,weighted=False):
    selected=[r for r in rows if r["values"][field] is not None]
    if not selected:return None
    v=np.asarray([r["values"][field] for r in selected],float)
    if weighted:
        w=np.asarray([r["n_common_proposed"] if field=="rank_percentile"
                      else r["n_gold"] for r in selected],float)
        if w.sum()==0:return None
        return float(np.average(v,weights=w))
    return float(v.mean())


def block_bootstrap(rows,field,seed=20261028,reps=1200):
    eligible=[r for r in rows if r["values"][field] is not None]
    if not eligible:return {"value":None,"ci95":None,"n_targets":0,"n_quires":0}
    blocks=defaultdict(list)
    for row in eligible:
        blocks[(row["language"],row["quire_code"])].append(row)
    keys=sorted(blocks)
    rng=np.random.default_rng(seed)
    samples=[]
    for _ in range(reps):
        draw=rng.choice(len(keys),len(keys),replace=True)
        group=[row for j in draw for row in blocks[keys[int(j)]]]
        samples.append(group_stats(group,field))
    return {"value":round(group_stats(eligible,field),6),
            "ci95":[round(float(v),6) for v in np.quantile(samples,[.025,.975])],
            "n_targets":len(eligible),"n_quires":len(keys)}


def analyze(pages,eva,trials=300,seed=20261028):
    models=[precompute_source(p,eva) for p in pages]
    graph=graph_pairs(pages)
    original=np.arange(len(pages),dtype=int)
    result={"pages":len(pages),
            "quires":len(strata(pages,("language","quire_code"))),
            "source_match":"within same quire, nearest prior vs earlier 2+ steps, <=30% source-size mismatch",
            "nulls":{}}
    rng=np.random.default_rng(seed)
    for strict in (False,True):
        label="all_within_quire" if not strict else "same_section_hand"
        observed=evaluate_order(pages,models,graph,original,strict)
        report={"observed":{metric:block_bootstrap(observed,metric)
                             for metric in OUTCOMES},
                "by_currier_variety":{
                    lang:{"matched_targets":sum(
                        r["language"]==lang for r in observed),
                          "metrics":{metric:block_bootstrap([
                              r for r in observed if r["language"]==lang],
                              metric) for metric in OUTCOMES}}
                    for lang in ("A","B")},
                "matched_targets":len(observed),
                "per_target_rows":observed[:30],
                "nulls":{}}
        for strat_name,keys in MODES.items():
            draws=defaultdict(list)
            moved=[]
            targets=[]
            for rep in range(trials):
                order=permute(pages,keys,rng)
                moved.append(int(np.sum(order!=original)))
                trialrows=evaluate_order(pages,models,graph,order,strict)
                targets.append(len(trialrows))
                for field in OUTCOMES:
                    draws[field].append(group_stats(trialrows,field))
                    for lang in ("A","B"):
                        draws[lang,field].append(group_stats([
                            r for r in trialrows if r["language"]==lang],
                            field))
            x={"strata":list(keys),
               "n_strata":len(strata(pages,keys)),
               "mean_moved_pages":round(float(np.mean(moved)),2),
               "mean_matched_targets":round(float(np.mean(targets)),2),
               "observations":{}}
            for field in OUTCOMES:
                actual=report["observed"][field]["value"]
                v=np.asarray([z for z in draws[field] if z is not None],float)
                if actual is None or not len(v):
                    x["observations"][field]={"available":False}
                    continue
                x["observations"][field]={
                    "observed":actual,
                    "null_mean":round(float(v.mean()),6),
                    "excess":round(float(actual-v.mean()),6),
                    "null_ci95":[round(float(q),6)
                                 for q in np.quantile(v,[.025,.975])],
                    "p_one_sided":round(float(
                        (1+np.sum(v>=actual))/(len(v)+1)),6),
                    "valid_permutations":len(v)}
            x["by_currier_variety"]={}
            for lang in ("A","B"):
                lx={}
                for field in ("chance_adjusted_recall","word_jaccard",
                              "rank_percentile"):
                    actual=report["by_currier_variety"][lang]["metrics"][field]["value"]
                    values=np.asarray([v for v in draws[lang,field]
                                       if v is not None],float)
                    if actual is None or not len(values):
                        lx[field]={"available":False}
                    else:
                        lx[field]={
                            "observed":actual,
                            "null_mean":round(float(values.mean()),6),
                            "excess":round(float(actual-values.mean()),6),
                            "p_one_sided":round(float(
                                (1+np.sum(values>=actual))/(len(values)+1)),6),
                            "permutations":len(values)}
                x["by_currier_variety"][lang]=lx
            report["nulls"][strat_name]=x
        result[label]=report
        for name,r in report["observed"].items():
            if name in ("chance_adjusted_recall","word_jaccard","rank_percentile"):
                print(label,name,r,flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--zl",required=True)
    p.add_argument("--austen")
    p.add_argument("--generated")
    p.add_argument("--cohort",choices=["Voynich","Austen","Timm"],default="Voynich")
    p.add_argument("--trials",type=int,default=300)
    p.add_argument("--out",required=True)
    args=p.parse_args()
    pages,eva=get_controls(args.zl,args.austen,args.generated,args.cohort)
    result=analyze(pages,eva,args.trials)
    result["cohort"]=args.cohort
    out=Path(args.out);out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(result,indent=2)+"\n")

if __name__=="__main__":
    main()
