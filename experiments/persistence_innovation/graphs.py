#!/usr/bin/env python3
"""Two directed cross-page word networks: persistence versus innovation.

SOURCE-ONLY one-glyph edit families are formed from the immediately
previous page and the alphabet observed earlier in the SAME quire.
Persistence: w -> v if both w and v were on the preceding page, v occurs
again on the next page, and v is a one-glyph variant of w.
Innovation: w -> v if w was on the preceding page, v is a one-glyph
candidate, is entirely new to the preceding quire prefix, and appears
on the next page.

These are observational edge types, not identified lexical derivatives,
temporal causation, parts of speech, or verified historical production order.
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path
from collections import defaultdict, Counter
from functools import lru_cache
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.same_target_source.evaluate import get_pages
from experiments.cross_page.reconstruct import generated_words
from experiments.english_controls.benchmark import clean_gutenberg,character_edits
from experiments.page_graphs.reconstruct_controls import matched_pages
from experiments.robustness75.robustness import glyphs,one_edit_forms
from experiments.page_graphs.compare import build_adjacency


def get_cohort(zl,cohort="Voynich",austen=None,naibbe=None,generated=None):
    pages,_=get_pages(zl,"voynich")
    if cohort=="Voynich":return pages,True
    if cohort=="Austen":
        text=Path(austen).read_text(encoding="utf-8-sig")
        words=clean_gutenberg(text,"austen");eva=False
    elif cohort=="Naibbe":
        import re
        words=re.findall(r"[a-z]+",Path(naibbe).read_text(
            encoding="utf-8-sig").lower());eva=True
    elif cohort=="Timm":
        words=generated_words(Path(generated).read_text(
            encoding="utf-8-sig"));eva=True
    else:raise ValueError(cohort)
    synthetic=matched_pages(pages,words,cohort.lower())
    for p,source in zip(synthetic,pages):
        for key in ("section_code","quire_code","hand_code"):
            p[key]=source.get(key,"?")
    return synthetic,eva


def by_quire(pages):
    g=defaultdict(list)
    for idx,p in enumerate(pages):
        g[(p["language"],p.get("quire_code","?"))].append(idx)
    return g


@lru_cache(maxsize=8192)
def propose(word,alphabet,eva):
    if not alphabet:return set()
    return frozenset((one_edit_forms(glyphs(word),alphabet) if eva else
            character_edits(word,tuple(alphabet)))-{word})


def tally(pages,eva,order=None,record_edges=True,indexed_neighbors=None):
    """All target outcome observations excluded from source candidate choice.

    indexed_neighbors is an optional GLOBAL index of *observed* one-edit
    matches, used only to speed intersection with actual target words.
    Any newly introduced target glyph still must belong to the source-only
    prefix alphabet. This indexing cannot invent or make an edit eligible.
    Candidate-pool size for original ordered pages is still calculated
    from entirely source-only proposals and is ignored in shuffled nulls.
    """
    if order is None:order=np.arange(len(pages),dtype=int)
    order_groups=by_quire([pages[int(i)] for i in order])
    counters=defaultdict(Counter)
    edges={"persistence":Counter(),"innovation":Counter()}
    edge_instances=[]
    page_reports=[]
    for (language,quire),slots in order_groups.items():
        seen=set()
        for j,slot in enumerate(slots):
            current_idx=int(order[slot])
            current=set(pages[current_idx]["words"])
            if j==0:
                seen.update(current)
                continue
            prior_idx=int(order[slots[j-1]])
            previous=set(pages[prior_idx]["words"])
            units=glyphs if eva else tuple
            alphabet=sorted({x for w in seen for x in units(w)})
            possible=0
            retained=0
            novel=0
            distinct_new_sources=defaultdict(set)
            actual_innovations=set()
            cache={}
            novel_on_target=current-seen
            stable_on_target=previous & current
            allowed=set(alphabet)
            for center in previous:
                if indexed_neighbors is None:
                    if center not in cache:
                        cache[center]=propose(center,tuple(alphabet),eva)
                    family=cache[center]
                    persistence=family & stable_on_target
                    innovations=family & novel_on_target
                    if record_edges:
                        possible+=len(family-seen)
                else:
                    # Every target comparison uses an exact one-glyph edge
                    # in the corpus-wide INDEX, but source-only alphabet is
                    # checked before an innovation qualifies.
                    matched=indexed_neighbors[center]
                    persistence=matched & stable_on_target
                    innovations={word for word in (matched & novel_on_target)
                                 if all(g in allowed for g in
                                        (glyphs(word) if eva else tuple(word)))}
                    if record_edges:
                        possible+=len(propose(center,tuple(alphabet),eva)-seen)
                retained+=len(persistence)
                novel+=len(innovations)
                for word in persistence:
                    if record_edges:
                        edges["persistence"][center,word]+=1
                        edge_instances.append({
                            "group":f"{language}:{quire}","folio":pages[current_idx]["folio"],
                            "source_folio":pages[prior_idx]["folio"],
                            "center":center,"target":word,
                            "edge_type":"persistence"})
                for word in innovations:
                    actual_innovations.add(word)
                    distinct_new_sources[word].add(center)
                    if record_edges:
                        edges["innovation"][center,word]+=1
                        edge_instances.append({
                            "group":f"{language}:{quire}","folio":pages[current_idx]["folio"],
                            "source_folio":pages[prior_idx]["folio"],
                            "center":center,"target":word,
                            "edge_type":"innovation"})
            t=language
            stats=counters[t]
            stats["eligible_steps"]+=1
            stats["source_types"]+=len(previous)
            stats["target_types"]+=len(current)
            stats["retained_edges"]+=retained
            stats["innovative_edges"]+=novel
            stats["candidate_pool_total"]+=possible
            stats["novel_target_types"]+=len(current-seen)
            stats["innovation_hit_types"]+=len(actual_innovations)
            stats["innovation_multi_parent"]+=sum(len(v)>1 for v in
                                                 distinct_new_sources.values())
            if record_edges:
                page_reports.append({
                    "folio":pages[current_idx]["folio"],
                    "source_folio":pages[prior_idx]["folio"],
                    "language":language,"quire":quire,"source_types":len(previous),
                    "target_types":len(current),
                    "previously_unseen_types":len(current-seen),
                    "persistence_edges":retained,
                    "innovation_edges":novel,
                    "innovation_distinct_types":len(actual_innovations),
                    "innovation_multiple_parents":sum(len(v)>1 for v in
                                                       distinct_new_sources.values()),
                    "source_candidate_count":possible})
            seen.update(current)
    result={}
    for lang,counts in counters.items():
        steps=counts["eligible_steps"]
        result[lang]={
            "steps":steps,
            "mean_persistence_edges_per_source_type":round(
                counts["retained_edges"]/max(1,counts["source_types"]),6),
            "mean_innovation_edges_per_source_type":round(
                counts["innovative_edges"]/max(1,counts["source_types"]),6),
            "new_target_word_coverage":round(
                counts["innovation_hit_types"]/max(1,counts["novel_target_types"]),6),
            "multi_parent_share_of_innovation_forms":round(
                counts["innovation_multi_parent"]/
                max(1,counts["innovation_hit_types"]),6),
            "mean_innovation_edges_per_step":round(
                counts["innovative_edges"]/steps,5) if steps else None,
            "mean_persistence_edges_per_step":round(
                counts["retained_edges"]/steps,5) if steps else None,
            "new_target_types":counts["novel_target_types"],
            "innovative_edge_instances":counts["innovative_edges"],
            "persistent_edge_instances":counts["retained_edges"]}
    return result,edges,edge_instances,page_reports


def null_order(pages,rng):
    order=np.arange(len(pages),dtype=int)
    groups=defaultdict(list)
    for i,p in enumerate(pages):
        groups[(p["language"],p.get("quire_code","?"),
                p.get("section_code","?"),p.get("hand_code","?"))].append(i)
    for idx in groups.values():
        if len(idx)>1:
            order[idx]=rng.permutation(idx)
    return order


def evaluate(pages,eva,permutations=50,seed=20261028):
    # Full observed vocabulary is ONLY an intersection index; possible
    # innovations are still filtered by prior-quire alphabet. This is
    # equivalent to source-only enumeration for observed edge outcomes.
    if eva:
        indexed=build_adjacency(pages)
    else:
        all_types={w for p in pages for w in p["words"]}
        alphabet=tuple(sorted({c for w in all_types for c in w}))
        indexed={w:character_edits(w,alphabet)&all_types
                 for w in all_types}
    observed,edges,instances,steps=tally(
        pages,eva,indexed_neighbors=indexed)
    rng=np.random.default_rng(seed)
    measurements=defaultdict(list)
    for _ in range(permutations):
        order=null_order(pages,rng)
        trial,_,_,_=tally(pages,eva,order,record_edges=False,
                          indexed_neighbors=indexed)
        for lang,stats in trial.items():
            for m in ("mean_persistence_edges_per_source_type",
                      "mean_innovation_edges_per_source_type",
                      "new_target_word_coverage"):
                measurements[lang,m].append(stats[m])
    summary={"observed":observed,"page_count":len(pages),
             "null_permutations":permutations,
             "metadata_preserved":"Currier + quire + illustration/section + hand",
             "graph_edges":{},"page_sequence":steps,"null":{}}
    for lang,stats in observed.items():
        summary["null"][lang]={}
        for m in ("mean_persistence_edges_per_source_type",
                  "mean_innovation_edges_per_source_type",
                  "new_target_word_coverage"):
            vals=np.asarray(measurements[lang,m])
            actual=stats[m]
            summary["null"][lang][m]={
                "observed":actual,
                "expected":round(float(vals.mean()),6),
                "excess":round(float(actual-vals.mean()),6),
                "null_95":[round(float(v),6) for v in np.quantile(vals,[.025,.975])],
                "p_excess":round(float((1+sum(vals>=actual))/(1+len(vals))),6)}
    for kind,cnt in edges.items():
        by_center=Counter()
        for (src,dst),n in cnt.items():by_center[src]+=n
        summary["graph_edges"][kind]={
            "distinct_edges":len(cnt),
            "distinct_source_centers":len(by_center),
            "total_page_transition_edge_instances":sum(cnt.values()),
            "top_source_centers":by_center.most_common(15)}
    return summary,edges,instances


def write_outputs(out,summary,edges,instances):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    result={k:v for k,v in summary.items() if k!="page_sequence"}
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    with (out/"page_transition_counts.csv").open("w",newline="") as f:
        rows=summary["page_sequence"]
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader();w.writerows(rows)
    with (out/"edge_occurrences.csv").open("w",newline="") as f:
        if instances:
            w=csv.DictWriter(f,fieldnames=list(instances[0]))
            w.writeheader();w.writerows(instances)
    for kind,counter in edges.items():
        items=[{"source_center":w,"target_neighbor":v,"weight":n}
               for (w,v),n in counter.most_common()]
        with (out/f"{kind}_graph_edges.csv").open("w",newline="") as f:
            if items:
                w=csv.DictWriter(f,fieldnames=list(items[0]))
                w.writeheader();w.writerows(items)
        # Machine-editable Graphviz excerpt (full graph in edge CSV).
        popular=counter.most_common(85)
        output=["digraph G {","  rankdir=LR;"]
        for (a,b),n in popular:
            output.append(f'  "{a}" -> "{b}" [label="{n}"];')
        output.append("}")
        (out/f"{kind}_top85.dot").write_text("\n".join(output)+"\n")


def run(zl,out,mode="Voynich",austen=None,naibbe=None,
        generated=None,permutations=50):
    pages,eva=get_cohort(zl,mode,austen,naibbe,generated)
    summary,edges,instances=evaluate(pages,eva,permutations)
    summary["cohort"]=mode
    write_outputs(out,summary,edges,instances)
    print(mode,json.dumps(summary["observed"]),flush=True)
    return summary


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--zl",required=True)
    ap.add_argument("--austen")
    ap.add_argument("--naibbe")
    ap.add_argument("--generated")
    ap.add_argument("--mode",choices=("Voynich","Austen","Naibbe","Timm"),
                    default="Voynich")
    ap.add_argument("--out",required=True)
    ap.add_argument("--permutations",type=int,default=50)
    args=ap.parse_args()
    run(args.zl,args.out,args.mode,args.austen,args.naibbe,args.generated,
        args.permutations)
