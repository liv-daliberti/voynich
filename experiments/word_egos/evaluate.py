#!/usr/bin/env python3
"""Center one edit-graph ego around EVERY observed Voynich word type.

Build global radius-1 and radius-2 neighborhoods (descriptive only).
Record how each center's induced local ego graph changes by folio.
For stable high-support centers, test within-quire local-neighborhood
persistence against metadata-preserving page-order shuffles.

Global neighbors use the full transcription, so these plots are NOT a
prospective vocabulary prediction study. Per-page observed neighbors
are restricted to words actually present on that folio.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.page_graphs.compare import build_adjacency
from experiments.same_target_source.evaluate import get_pages
from experiments.within_quire.predict import permute


def two_hops(word,neighbors):
    one=neighbors[word]
    second=set().union(*(neighbors[n] for n in one)) if one else set()
    return second-one-{word}


def jaccard(a,b):
    if not (a or b):return 1.
    return len(a&b)/len(a|b)


def ego_profiles(pages):
    adjacency=build_adjacency(pages)
    page_types=[set(p["words"]) for p in pages]
    counts=Counter(w for p in pages for w in set(p["words"]))
    token_counts=Counter(w for p in pages for w in p["words"])
    two={w:two_hops(w,adjacency) for w in adjacency}
    records=[]
    centers=defaultdict(list)
    active_ghost=Counter()
    for j,p in enumerate(pages):
        V=page_types[j]
        # Invert observed node neighbors to find hypothetical absent
        # centers supported by 2+ page words, without outputting fake
        # form *predictions* as facts.
        inverted=defaultdict(set)
        for w in V:
            for center in adjacency[w]:
                if center not in V:
                    inverted[center].add(w)
        for absent,nbr in inverted.items():
            if len(nbr)>=2:active_ghost[absent]+=1
        for word in V:
            one=adjacency[word]
            active=one & V
            second=two[word]&V
            local_edges=sum(len(adjacency[n]&active) for n in active)//2
            row={
                "word":word,"folio":p["folio"],"page_idx":j,
                "language":p["language"],"quire":p.get("quire_code","?"),
                "section":p.get("section_code","?"),
                "hand":p.get("hand_code","?"),
                "center_token_count":p["words"].count(word),
                "global_degree":len(one),
                "global_second_ring":len(two[word]),
                "active_one_hop":len(active),
                "active_two_hop":len(second),
                "ego_edges":len(active)+local_edges,
                "ego_triangles":local_edges,
                "local_neighborhood_activation":(
                    len(active)/len(one) if one else 0.)}
            records.append(row)
            centers[word].append((j,frozenset(active)))
    summary=[]
    for word in adjacency:
        rows=centers[word]
        act=np.asarray([len(a) for _,a in rows],float)
        summary.append({
            "word":word,"language_occurrences":",".join(sorted({
                pages[i]["language"] for i,_ in rows})),
            "document_frequency":counts[word],
            "token_frequency":token_counts[word],
            "global_degree":len(adjacency[word]),
            "global_two_hop_count":len(two[word]),
            "mean_active_one_hop":round(float(act.mean()),5) if len(act) else 0.,
            "std_active_one_hop":round(float(act.std()),5) if len(act) else 0.,
            "max_active_one_hop":int(act.max()) if len(act) else 0,
            "ghost_supported_pages":active_ghost[word]})
    return adjacency,centers,summary,records


def pair_tables(pages,centers,summary,limit_centers=180,
                min_document_frequency=4,min_global_degree=3):
    eligible=[r for r in summary if
              r["document_frequency"]>=min_document_frequency and
              r["global_degree"]>=min_global_degree]
    eligible.sort(key=lambda x:(-x["document_frequency"],
                                 -x["global_degree"],x["word"]))
    eligible=eligible[:limit_centers]
    # Fix selected centers across permutations. A pair is included
    # only if the same center occurs on both source pages.
    groups=defaultdict(list)
    for i,p in enumerate(pages):
        groups[(p["language"],p.get("quire_code","?"))].append(i)
    local_pos={v:i for ids in groups.values() for i,v in enumerate(ids)}
    word_i=[];page_i=[];page_j=[];sim=[];lag=[]
    for wi,row in enumerate(eligible):
        items=centers[row["word"]]
        for i in range(len(items)):
            left,ln=items[i]
            for j in range(i+1,len(items)):
                right,rn=items[j]
                if (pages[left]["language"],pages[left].get("quire_code"))!=(
                    pages[right]["language"],pages[right].get("quire_code")):
                    continue
                gap=abs(local_pos[left]-local_pos[right])
                if not gap:continue
                # Empty observed ring on both pages is not evidence of
                # identical morphology in those pages.
                if not (ln or rn):continue
                word_i.append(wi);page_i.append(left);page_j.append(right)
                sim.append(jaccard(ln,rn));lag.append(gap)
    return eligible,{"center_idx":np.asarray(word_i,dtype=int),
                     "p":np.asarray(page_i,dtype=int),
                     "q":np.asarray(page_j,dtype=int),
                     "similarity":np.asarray(sim,dtype=float),
                     "lag":np.asarray(lag,dtype=int)}


def ego_order_stat(pages,pairs,order,min_far_lag=3):
    """Average near-versus-far ego overlap, giving each comparable
    center word one vote instead of allowing ubiquitous words to dominate."""
    n=len(pages)
    inv=np.empty(n,dtype=int)
    inv[order]=np.arange(n)
    byquire=defaultdict(list)
    for slot,src in enumerate(order):
        page=pages[int(src)]
        byquire[(page["language"],page.get("quire_code","?"))].append(slot)
    withinpos=np.zeros(n,dtype=int)
    for slots in byquire.values():
        withinpos[slots]=np.arange(len(slots))
    gap=np.abs(withinpos[inv[pairs["p"]]]-
               withinpos[inv[pairs["q"]]])
    near=gap<=2
    far=gap>=min_far_lag
    sim=pairs["similarity"]
    ci=pairs["center_idx"]
    diffs=[]
    observations=[]
    for c in np.unique(ci):
        yes=(ci==c)&near;no=(ci==c)&far
        if yes.any() and no.any():
            diffs.append(float(sim[yes].mean()-sim[no].mean()))
            observations.append((int(yes.sum()),int(no.sum())))
    return {"mean_delta":float(np.mean(diffs)) if diffs else None,
            "matched_centers":len(diffs),
            "near_word_pairs":int(sum(x[0] for x in observations)),
            "far_word_pairs":int(sum(x[1] for x in observations))}


def evaluate(pages,trials=400,seed=20261028,limit_centers=180):
    adjacency,centers,word_rows,page_rows=ego_profiles(pages)
    selected,pairs=pair_tables(pages,centers,word_rows,limit_centers)
    order=np.arange(len(pages),dtype=int)
    actual=ego_order_stat(pages,pairs,order)
    results={"n_pages":len(pages),"n_global_word_types":len(adjacency),
             "page_ego_rows":len(page_rows),
             "n_selected_center_words":len(selected),
             "center_selection":"top page-document-frequency, df>=4, degree>=3",
             "observed":actual,"nulls":{}}
    rng=np.random.default_rng(seed)
    for name,fields in (
        ("quire",("language","quire_code")),
        ("quire_section",("language","quire_code","section_code")),
        ("quire_section_hand",("language","quire_code","section_code","hand_code"))
    ):
        values=[]
        eligible=[]
        mobility=[]
        for _ in range(trials):
            shuffled=permute(pages,fields,rng)
            mobility.append(int(np.count_nonzero(shuffled!=order)))
            x=ego_order_stat(pages,pairs,shuffled)
            if x["mean_delta"] is not None:
                values.append(x["mean_delta"])
                eligible.append(x["matched_centers"])
        if values and actual["mean_delta"] is not None:
            v=np.asarray(values)
            results["nulls"][name]={
                "observed_delta":round(actual["mean_delta"],6),
                "null_mean":round(float(v.mean()),6),
                "excess":round(actual["mean_delta"]-float(v.mean()),6),
                "null_ci95":[round(float(x),6) for x in np.quantile(v,[.025,.975])],
                "p_one_sided":round(float((1+np.sum(v>=actual["mean_delta"]))/
                                             (len(v)+1)),6),
                "n_permutations":len(v),
                "mean_eligible_centers":round(float(np.mean(eligible)),2),
                "mean_moved_pages":round(float(np.mean(mobility)),2)}
        else:
            results["nulls"][name]={"available":False}
    # Representative local graph histories for each sampled center word
    selected_names={x["word"] for x in selected}
    histories={}
    for word in selected_names:
        present=[r for r in page_rows if r["word"]==word]
        histories[word]=sorted([
            {"folio":r["folio"],"quire":r["quire"],"language":r["language"],
             "neighbors":r["active_one_hop"],"two_hop":r["active_two_hop"],
             "triangles":r["ego_triangles"],"local_degree":r["active_one_hop"]}
            for r in present],key=lambda x:x["folio"])
    return results,word_rows,page_rows,histories


def run(zl,out,trials=400):
    pages,_=__import__(
        "experiments.same_target_source.evaluate",
        fromlist=["get_pages"]).get_pages(zl,"voynich")
    result,words,ego_pages,histories=evaluate(pages,trials)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for filename,rows in (("word_summary.csv",words),
                          ("word_ego_per_folio.csv",ego_pages)):
        with (out/filename).open("w",newline="") as fp:
            writer=csv.DictWriter(fp,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    (out/"selected_word_histories.json").write_text(
        json.dumps(histories,indent=2)+"\n")
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--zl",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--trials",type=int,default=400)
    a=p.parse_args()
    run(a.zl,a.out,a.trials)
