#!/usr/bin/env python3
"""Do stable one-edit word families predict future within-quire novelty?

All family stability, candidate dictionaries and matching covariates are built
from *preceding pages of the same quire only*. Never use the held-out page
to form an ego graph, choose a family, set cutpoints, or match controls.

Gold = previously unobserved, source-generated one-EVA-glyph variants on the
NEXT page. Secondary positive control = last-page neighbors reappearing.
Observational folio sequence is not proved to be writing chronology.
"""
from __future__ import annotations
import argparse, csv, json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.robustness75.robustness import glyphs, one_edit_forms
from experiments.english_controls.benchmark import character_edits, clean_gutenberg
from experiments.same_target_source.evaluate import get_pages
from experiments.page_graphs.reconstruct_controls import matched_pages
from experiments.cross_page.reconstruct import generated_words

F = ("log_prior_tokens","log_prior_document_pages","log_prefix_neighbors",
     "log_candidate_pool","glyph_length","log_previous_neighbor_count",
     "log_earlier_neighbor_count","quire_position","log_previous_page_types")
MATCH_F = ("log_prior_tokens","log_prior_document_pages","log_prefix_neighbors",
           "log_candidate_pool","glyph_length","log_previous_neighbor_count",
           "log_earlier_neighbor_count")
KEYS = ("stable","reorganizing","intermediate")


def variants(word,alphabet,eva=True):
    return (one_edit_forms(glyphs(word),alphabet) if eva else
            character_edits(word,tuple(alphabet))) - {word}


def ordered_quire_pages(pages):
    grouped=defaultdict(list)
    for page in pages:
        grouped[(page["language"],page.get("quire_code","?"))].append(page)
    return grouped


def page_set(page):
    return set(page["words"])


def extract_events(pages,eva=True,lookback=4,
                   stability_stable=.5,stability_reorg=.1):
    """Use at most four preceding pages to calculate *past* stability.

    Center must occur on previous page and >=1 other page among the
    immediately preceding lookback window. Require nonempty neighbor
    sets on both observed pages to distinguish turnover from absence.
    """
    out=[]
    for (language,quire),folios in ordered_quire_pages(pages).items():
        seen=set()
        prior_counts=Counter()
        prior_page_df=Counter()
        page_sets=[page_set(page) for page in folios]
        for t,page in enumerate(folios):
            past=folios[:t]
            current=page_sets[t]
            if t>=2:
                alphabet=sorted({g for w in seen for g in
                                  (glyphs(w) if eva else tuple(w))})
                # Cache candidate proposals inside this quire/prefix only.
                cache={}
                def get_candidates(word):
                    if word not in cache:
                        cache[word]=variants(word,alphabet,eva)
                    return cache[word]

                previous=past[-1]
                previous_set=page_sets[t-1]
                for center in previous_set:
                    earlier=[i for i in range(max(0,t-lookback),t-1)
                             if center in page_sets[i]]
                    if not earlier:continue
                    earlier_idx=earlier[-1]
                    pool=get_candidates(center)
                    # Observed neighbors are REAL past tokens, but the
                    # *permitted* candidate universe is source-only.
                    a=pool & page_sets[earlier_idx]
                    b=pool & previous_set
                    if not a or not b:continue
                    stability=len(a&b)/len(a|b)
                    family_seen=pool & seen
                    family_novel=pool-seen
                    new_hits=family_novel & current
                    retained=b & current
                    row={
                        "word":center,"quire":str(quire),"language":language,
                        "group":str(language)+":"+str(quire),
                        "folio":page["folio"],"target_index_in_quire":t,
                        "prior_folio":previous["folio"],
                        "earlier_folio":folios[earlier_idx]["folio"],
                        "stability":stability,
                        "class":("stable" if stability>=stability_stable
                                 else "reorganizing" if stability<=stability_reorg
                                 else "intermediate"),
                        "new_neighbor_count":len(new_hits),
                        "new_neighbor_any":int(bool(new_hits)),
                        "new_neighbor_forms":";".join(sorted(new_hits)),
                        "retained_neighbor_any":int(bool(retained)),
                        "retained_neighbor_count":len(retained),
                        "candidate_pool_size":len(family_novel),
                        "prefix_family_degree":len(family_seen),
                        "prior_tokens":prior_counts[center],
                        "prior_document_pages":prior_page_df[center],
                        "previous_neighbor_count":len(b),
                        "earlier_neighbor_count":len(a),
                        "glyph_length":len(glyphs(center) if eva else center),
                        "previous_page_types":len(previous_set),
                        "log_prior_tokens":math.log1p(prior_counts[center]),
                        "log_prior_document_pages":math.log1p(prior_page_df[center]),
                        "log_prefix_neighbors":math.log1p(len(family_seen)),
                        "log_candidate_pool":math.log1p(len(family_novel)),
                        "log_previous_neighbor_count":math.log1p(len(b)),
                        "log_earlier_neighbor_count":math.log1p(len(a)),
                        "quire_position":t/max(1,len(folios)-1),
                        "log_previous_page_types":math.log1p(len(previous_set))
                    }
                    out.append(row)
            seen.update(current)
            prior_counts.update(page["words"])
            prior_page_df.update(current)
    return out


def match_stable_reorganizing(events,seed=20261028):
    """One-to-one nearest-neighbor matching WITHIN SAME TARGET PAGE.

    This controls target page length, quire, illustration, topic and hand
    automatically. Source-page covariates and a predeclared caliper prevent
    trivially matching ubiquitous high-degree forms to rare low-degree ones.
    No gold labels are examined while constructing the pairs.
    """
    group=defaultdict(lambda:{"stable":[],"reorganizing":[]})
    for e in events:
        if e["class"] in ("stable","reorganizing"):
            group[(e["group"],e["folio"])][e["class"]].append(e)
    matched=[]
    for (block,folio),g in group.items():
        stable=sorted(g["stable"],key=lambda x:(x["word"]))
        turnover=sorted(g["reorganizing"],key=lambda x:(x["word"]))
        remaining=set(range(len(turnover)))
        for a in stable:
            scores=[]
            for j in remaining:
                b=turnover[j]
                if abs(a["glyph_length"]-b["glyph_length"])>2:continue
                if abs(a["log_prior_tokens"]-b["log_prior_tokens"])>1.0:continue
                if abs(a["log_prefix_neighbors"]-b["log_prefix_neighbors"])>.9:continue
                if abs(a["log_candidate_pool"]-b["log_candidate_pool"])>.8:continue
                if abs(a["log_previous_neighbor_count"]-
                       b["log_previous_neighbor_count"])>.9:continue
                difference=sum(((a[k]-b[k])/sc)**2 for k,sc in zip(
                    MATCH_F,(1.,.6,.9,.8,2.,.9,.9)))
                scores.append((difference,b["word"],j))
            if not scores:continue
            _,_,j=min(scores)
            remaining.remove(j)
            b=turnover[j]
            matched.append({
                "group":block,"folio":folio,
                "stable_word":a["word"],"reorganizing_word":b["word"],
                "stable_new":a["new_neighbor_any"],
                "reorg_new":b["new_neighbor_any"],
                "stable_retain":a["retained_neighbor_any"],
                "reorg_retain":b["retained_neighbor_any"],
                "stable_new_count":a["new_neighbor_count"],
                "reorg_new_count":b["new_neighbor_count"],
                "stable_pool":a["candidate_pool_size"],
                "reorg_pool":b["candidate_pool_size"],
                "stable_stability":a["stability"],
                "reorg_stability":b["stability"],
                "freq_log_difference":a["log_prior_tokens"]-
                                      b["log_prior_tokens"],
                "degree_log_difference":a["log_prefix_neighbors"]-
                                        b["log_prefix_neighbors"]})
    return matched


def bootstrap_by_quire(rows,delta_fn,seed=20261028,reps=1800):
    groups=defaultdict(list)
    for row in rows:groups[row["group"]].append(row)
    labels=sorted(groups)
    if not labels:return {"n_quires":0,"point":None,"ci95":None}
    def calc(sample):
        values=[delta_fn(r) for label in sample for r in groups[label]]
        return float(np.mean(values))
    rng=np.random.default_rng(seed)
    values=[calc(rng.choice(labels,len(labels),replace=True))
            for _ in range(reps)]
    return {"n_quires":len(labels),"point":round(calc(labels),6),
            "ci95":[round(float(x),6) for x in np.quantile(values,(.025,.975))]}


def match_report(matched):
    if not matched:return {"matched_pairs":0}
    stable=sum(x["stable_new"] for x in matched)/len(matched)
    reorg=sum(x["reorg_new"] for x in matched)/len(matched)
    return {"matched_pairs":len(matched),
            "stable_novel_any_rate":round(stable,6),
            "reorganizing_novel_any_rate":round(reorg,6),
            "new_form_pair_advantage":bootstrap_by_quire(
                matched,lambda x:x["stable_new"]-x["reorg_new"]),
            "retention_pair_advantage":bootstrap_by_quire(
                matched,lambda x:x["stable_retain"]-x["reorg_retain"]),
            "mean_matched_log_prior_frequency_gap":round(float(
                np.mean([x["freq_log_difference"] for x in matched])),5),
            "mean_matched_log_degree_gap":round(float(
                np.mean([x["degree_log_difference"] for x in matched])),5)}


def cross_validate(events,label="new_neighbor_any",nfolds=4):
    """Quire-grouped out-of-sample incremental prediction.

    Fit a strictly source-feature-only logistic model with or without the
    preceding-neighborhood Jaccard. Report held-out log-loss improvement,
    not in-sample class separation.
    """
    groups=np.asarray([e["group"] for e in events])
    y=np.asarray([e[label] for e in events],dtype=int)
    cols=list(F)
    x0=np.asarray([[e[k] for k in cols] for e in events],dtype=float)
    x1=np.column_stack((x0,np.asarray([e["stability"] for e in events])))
    folds=min(nfolds,len(set(groups)))
    if folds<2 or len(set(y))<2:return {"available":False}
    pred0=np.zeros(len(y));pred1=np.zeros(len(y))
    splits=GroupKFold(folds)
    for train,test in splits.split(x0,y,groups):
        if len(set(y[train]))<2:
            return {"available":False,"reason":"single training class"}
        for matrix,dst in ((x0,pred0),(x1,pred1)):
            model=make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=1500,C=1.0))
            model.fit(matrix[train],y[train])
            dst[test]=np.clip(model.predict_proba(matrix[test])[:,1],
                              1e-7,1-1e-7)
    ll0=-(y*np.log(pred0)+(1-y)*np.log(1-pred0))
    ll1=-(y*np.log(pred1)+(1-y)*np.log(1-pred1))
    per_event=[{"group":str(group),"delta":float(a-b)}
               for group,a,b in zip(groups,ll0,ll1)]
    report={"events":len(y),"positive_rate":round(float(y.mean()),6),
            "heldout_log_loss_baseline":round(float(ll0.mean()),6),
            "heldout_log_loss_with_stability":round(float(ll1.mean()),6),
            "log_loss_gain_from_stability":bootstrap_by_quire(
                per_event,lambda x:x["delta"]),
            "roc_auc_baseline":round(float(roc_auc_score(y,pred0)),6),
            "roc_auc_with_stability":round(float(roc_auc_score(y,pred1)),6),
            "average_precision_baseline":round(float(
                average_precision_score(y,pred0)),6),
            "average_precision_with_stability":round(float(
                average_precision_score(y,pred1)),6),
            "folds":folds}
    return report


def summarize(events,lookback=4):
    summary={"events":len(events),"quires":len(set(e["group"] for e in events)),
             "lookback_preceding_folios":lookback}
    for variant in KEYS:
        rows=[e for e in events if e["class"]==variant]
        summary[variant]={"n":len(rows),
            "new_form_rate":round(float(np.mean(
                [r["new_neighbor_any"] for r in rows])),6) if rows else None,
            "retention_rate":round(float(np.mean(
                [r["retained_neighbor_any"] for r in rows])),6) if rows else None,
            "mean_prior_tokens":round(float(np.mean(
                [r["prior_tokens"] for r in rows])),2) if rows else None,
            "mean_prefix_degree":round(float(np.mean(
                [r["prefix_family_degree"] for r in rows])),2) if rows else None}
    paired=match_stable_reorganizing(events)
    summary["within_same_page_matched"]=match_report(paired)
    summary["quire_blocked_cv_new_forms"]=cross_validate(events)
    summary["quire_blocked_cv_neighbor_retention"]=cross_validate(
        events,label="retained_neighbor_any")
    return summary,paired


def run(zl,out,austen=None,naibbe=None,generated=None,mode="Voynich",
        lookback=4):
    if mode=="Voynich":
        pages,eva=get_pages(zl,"voynich")
    else:
        real,_=get_pages(zl,"voynich")
        if mode=="Austen":
            words=clean_gutenberg(Path(austen).read_text(
                encoding="utf-8-sig"),"austen")
            eva=False
        elif mode=="Naibbe":
            words=re.findall(r"[a-z]+",Path(naibbe).read_text(
                encoding="utf-8-sig").lower())
            eva=True
        elif mode=="Timm":
            words=generated_words(Path(generated).read_text(
                encoding="utf-8-sig"))
            eva=True
        else:raise ValueError(mode)
        pages=matched_pages(real,words,mode.lower())
        # Synthetic pages inherit metadata solely as sample/length strata,
        # not because generated text has historical quires or sections.
        for p,src in zip(pages,real):
            for k in ("quire_code","section_code","hand_code"):
                p[k]=src.get(k,"?")
    events=extract_events(pages,eva,lookback)
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    with (out/"events.csv").open("w",newline="") as file:
        if events:
            writer=csv.DictWriter(file,fieldnames=list(events[0]))
            writer.writeheader();writer.writerows(events)
    totals={}
    for variety in sorted(set(e["language"] for e in events)):
        subset=[e for e in events if e["language"]==variety]
        totals[variety],matched=summarize(subset,lookback)
        with (out/f"matched_pairs_{variety}.csv").open(
            "w",newline="") as file:
            if matched:
                writer=csv.DictWriter(file,fieldnames=list(matched[0]))
                writer.writeheader();writer.writerows(matched)
    totals["ALL"],matched=summarize(events,lookback)
    with (out/"matched_pairs_ALL.csv").open("w",newline="") as file:
        if matched:
            writer=csv.DictWriter(file,fieldnames=list(matched[0]))
            writer.writeheader();writer.writerows(matched)
    totals["_design"]={"cohort":mode,"page_count":len(pages),
        "source_only_prefix":True,
        "target_gold":"one-edit candidate absent from all preceding quire pages",
        "stable_threshold":.5,"reorganizing_threshold":.1,
        "requires_active_neighbors_in_both_history_pages":True,
        "matched_target_page_same_for_both_classes":True,
        "grouped_cv_leave_out_quire_folds":4,
        "historical_order_not_verified":True}
    (out/"summary.json").write_text(json.dumps(totals,indent=2)+"\n")
    print(mode,json.dumps({l:{"events":v["events"],
                 "matched":v["within_same_page_matched"].get("matched_pairs"),
                 "stable_vs_reorg":v["within_same_page_matched"].get(
                        "new_form_pair_advantage"),
                 "CV_gain":v["quire_blocked_cv_new_forms"].get(
                        "log_loss_gain_from_stability")}
                 for l,v in totals.items() if l!="_design"}),flush=True)
    return totals


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--zl",required=True)
    p.add_argument("--austen")
    p.add_argument("--naibbe")
    p.add_argument("--generated")
    p.add_argument("--mode",choices=("Voynich","Austen","Naibbe","Timm"),
                   default="Voynich")
    p.add_argument("--lookback",type=int,default=4)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    run(a.zl,a.out,a.austen,a.naibbe,a.generated,a.mode,a.lookback)
