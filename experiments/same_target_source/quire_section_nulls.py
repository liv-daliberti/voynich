#!/usr/bin/env python3
"""Quire- and section-aware page-order nulls for local graph continuity.

ZL metadata: $L = Currier variety; $I = illustration/section;
$Q = quire; $H = hand; $F = position within quire (NOT section).

Permute WHOLE folio sides, not individual tokens. Preserve any specified
metadata combinations and optionally match source-size tertile. Compare
against unshuffled (i) local graph similarity and (ii) source-only,
same-target novel-form ranking over distant source controls.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.page_graphs.compare import build_adjacency,graph_stats,shape_cos
from experiments.same_target_source.evaluate import (
    get_pages,precompute_source,choose_source_pairs,evaluate_pairs,
    weighted_result,
)

STRATA={
    "currier":("language",),
    "currier_section":("language","section_code"),
    "currier_quire":("language","quire_code"),
    "currier_section_quire":("language","section_code","quire_code"),
    "currier_section_quire_hand":(
        "language","section_code","quire_code","hand_code"),
    "currier_section_quire_hand_size":(
        "language","section_code","quire_code","hand_code","size_bin"),
}
FIELDS=("word_jaccard","edge_jaccard","topology_cosine")
SCHEMES=("strict_section_hand","same_section","same_variety")


def cells(pages,models,fields):
    grouped=defaultdict(list)
    for i,p in enumerate(pages):
        key=tuple(p.get(x,"?") for x in fields if x!="size_bin")
        grouped[key].append(i)
    if "size_bin" not in fields:
        return grouped
    out=defaultdict(list)
    for key,ids in grouped.items():
        # Avoid artificial singleton strata in groups of only a few pages.
        if len(ids)<6:
            out[key+("all",)].extend(ids)
        else:
            ordered=sorted(ids,key=lambda i:(models[i]["types"],i))
            for rank,idx in enumerate(ordered):
                out[key+(min(2,(3*rank)//len(ordered)),)].append(idx)
    return out


def sample_order(groups,n,rng):
    # Group assignment preserves the original metadata at each slot.
    order=np.arange(n,dtype=int)
    for ids in groups.values():
        if len(ids)>=2:
            order[ids]=rng.permutation(ids)
    return order


def movable_summary(groups,n):
    move=[x for x in groups.values() if len(x)>=2]
    return {"strata":len(groups),
            "movable_groups":len(move),
            "movable_pages":sum(len(x) for x in move),
            "unmovable_pages":n-sum(len(x) for x in move),
            "largest_group":max((len(x) for x in groups.values()),default=0),
            "small_groups_1_or_2":sum(len(x)<=2 for x in groups.values())}


def graph_matrices(pages):
    adjacency=build_adjacency(pages)
    G=[graph_stats(p,adjacency) for p in pages]
    n=len(pages)
    word=np.zeros((n,n),dtype=float)
    edge=np.zeros((n,n),dtype=float)
    topo=np.zeros((n,n),dtype=float)
    for i in range(n):
        for j in range(i+1,n):
            x,y=G[i],G[j]
            a,b=x[1],y[1]
            u=len(a|b)
            word[i,j]=word[j,i]=len(a&b)/u if u else 0
            a,b=x[2],y[2]
            u=len(a|b)
            edge[i,j]=edge[j,i]=len(a&b)/u if u else 0
            topo[i,j]=topo[j,i]=shape_cos(
                x[0]["degree_histogram"],y[0]["degree_histogram"])
    return {"word_jaccard":word,"edge_jaccard":edge,
            "topology_cosine":topo}


def graph_stats_in_order(pages,order,matrices):
    out={}
    langs=sorted(set(p["language"] for p in pages))
    # Eligible near/far pairs are exactly the same positional slots across
    # permutations, preserving the A/B sequence and counts.
    for lang in langs:
        idx=np.asarray([k for k,i in enumerate(order)
                        if pages[int(i)]["language"]==lang],dtype=int)
        ids=order[idx]
        if len(ids)<25:continue
        a,b=np.triu_indices(len(ids),1)
        nearby=b-a<=3
        distant=b-a>=20
        out[lang]={}
        for field,mat in matrices.items():
            sub=mat[np.ix_(ids,ids)]
            near=float(sub[a[nearby],b[nearby]].mean())
            far=float(sub[a[distant],b[distant]].mean())
            out[lang][field]={"near":near,"far":far,"delta":near-far}
    return out


def same_target_stat(pages,models,order):
    perm_pages=[pages[i] for i in order]
    perm_models=[models[i] for i in order]
    lang_indices=defaultdict(list)
    for i,p in enumerate(perm_pages):
        lang_indices[p["language"]].append(i)
    out={}
    for scheme in SCHEMES:
        out[scheme]={}
        for lang,ids in lang_indices.items():
            choices=choose_source_pairs(perm_pages,perm_models,ids,scheme,
                                        min_lag=20,size_tolerance=.25)
            target_rows=evaluate_pairs(perm_pages,perm_models,choices)
            out[scheme][lang]={
                "target_count":len(target_rows),
                "delta_chance_adjusted_recall":weighted_result(
                    target_rows,"delta_chance_adjusted_recall"),
                "delta_recall":weighted_result(target_rows,"delta_recall"),
                "delta_coverage":weighted_result(target_rows,"delta_coverage"),
                "mean_matched_source_size_gap":float(np.mean([
                    row["size_diff"] for row in target_rows]))
                    if target_rows else None}
    return out


def distribution(observed,draws):
    samples=np.asarray([v for v in draws if v is not None],dtype=float)
    if observed is None or not len(samples):
        return {"n_valid_permutations":len(samples),
                "observed":observed,"p_one_sided":None}
    return {"observed":round(float(observed),6),
        "null_mean":round(float(samples.mean()),6),
        "excess_vs_null":round(float(observed-samples.mean()),6),
        "null_95_interval":[round(float(x),6)
                            for x in np.quantile(samples,[.025,.975])],
        "p_one_sided":round(float(
            (1+np.count_nonzero(samples>=observed))/(len(samples)+1)),6),
        "n_valid_permutations":len(samples)}


def evaluate(pages,eva,permutations=150,seed=20261028,
             null_labels=None,graph_only=False):
    models=([{"types":len(set(p["words"]))} for p in pages]
            if graph_only else [precompute_source(p,eva) for p in pages])
    graph=graph_matrices(pages)
    original=np.arange(len(pages),dtype=int)
    ref_graph=graph_stats_in_order(pages,original,graph)
    ref_source={} if graph_only else same_target_stat(pages,models,original)
    results={
        "metadata_interpretation":{
            "$L":"Currier A/B","$I":"illustration/section",
            "$Q":"quire","$H":"hand","$F":"within-quire folio position, NOT section"},
        "n_pages":len(pages),
        "permutations_per_null":permutations,
        "observed_graph":ref_graph,"observed_source":ref_source,
        "nulls":{}}
    rng=np.random.default_rng(seed)
    for label,fields in STRATA.items():
        if null_labels is not None and label not in null_labels:continue
        groups=cells(pages,models,fields)
        graph_draws=defaultdict(list)
        source_draws=defaultdict(list)
        moved=[]
        eligible=defaultdict(list)
        for _ in range(permutations):
            order=sample_order(groups,len(pages),rng)
            moved.append(int(np.count_nonzero(order!=original)))
            gs=graph_stats_in_order(pages,order,graph)
            for lang,by_field in gs.items():
                for field,x in by_field.items():
                    graph_draws[lang,field].append(x["delta"])
            if not graph_only:
                ss=same_target_stat(pages,models,order)
                for scheme,by_lang in ss.items():
                    for lang,x in by_lang.items():
                        eligible[scheme,lang].append(x["target_count"])
                        for m in ("delta_chance_adjusted_recall",
                                  "delta_recall","delta_coverage"):
                            source_draws[scheme,lang,m].append(x[m])
        result={"strata_keys":fields,
                "permutation_mobility":movable_summary(groups,len(pages)),
                "mean_changed_pages_per_permutation":round(
                    float(np.mean(moved)),3),
                "min_changed_pages_per_permutation":min(moved),
                "max_changed_pages_per_permutation":max(moved),
                "graph":{},"matched_source":{}}
        for lang,fields2 in ref_graph.items():
            result["graph"][lang]={
                field:distribution(x["delta"],graph_draws[lang,field])
                for field,x in fields2.items()}
        for scheme,by_lang in ref_source.items():
            result["matched_source"][scheme]={}
            for lang,x in by_lang.items():
                record={
                    key:distribution(x[key],source_draws[scheme,lang,key])
                    for key in ("delta_chance_adjusted_recall",
                                "delta_recall","delta_coverage")}
                t=np.asarray(eligible[scheme,lang])
                record["eligible_target_counts"]={
                    "observed":x["target_count"],
                    "null_mean":round(float(t.mean()),3),
                    "range":[int(t.min()),int(t.max())]}
                result["matched_source"][scheme][lang]=record
        results["nulls"][label]=result
        for lang,records in result["graph"].items():
            print(label,lang,"mobility",result["mean_changed_pages_per_permutation"],
                  "Jaccard observed",records["word_jaccard"]["observed"],
                  "null",records["word_jaccard"].get("null_mean"),
                  "p",records["word_jaccard"]["p_one_sided"],flush=True)
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--zl",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--permutations",type=int,default=150)
    p.add_argument("--nulls",nargs="*",choices=tuple(STRATA))
    p.add_argument("--graph-only",action="store_true")
    a=p.parse_args()
    pages,eva=get_pages(a.zl,"voynich")
    result=evaluate(pages,eva,a.permutations,
                    null_labels=a.nulls,graph_only=a.graph_only)
    target=Path(a.out);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
