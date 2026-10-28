#!/usr/bin/env python3
"""Conservative ZL/IT disagreement classifier, grouped by physical leaf.

Requires ZL3b-n.txt and IT2a-n.txt. No image labels are inferred.
Run: python experiments/transcription_ambiguity/pilot.py --zl data/ZL3b-n.txt --it data/IT2a-n.txt
Uses scikit-learn for reproducible 5-fold logistic baseline and AUC.
"""
from __future__ import annotations
import argparse, collections, csv, json, math, re
from pathlib import Path
from functools import lru_cache

COMPOUND=("ckh","cth","cph","cfh","ch","sh")

@lru_cache(maxsize=None)
def glyphs(w):
    out=[];i=0
    while i<len(w):
        g=next((t for t in COMPOUND if w.startswith(t,i)),None)
        if g:out.append(g);i+=len(g)
        else:out.append(w[i]);i+=1
    return tuple(out)

def read_ivtff(path):
    pages={};lines={}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        h=re.match(r"^<([^.,>]+)>\s*<!\s*(.*?)>",line)
        if h:
            pages[h[1]]=dict(re.findall(r"\$(\w)=([^\s>]+)",h[2]));continue
        m=re.match(r"^<([^>,]+),([^>]+)>\s*(.*)$",line)
        if not m or not re.search(r"[+@*=]P",m[2]):continue
        folio=m[1].split(".")[0]
        raw=re.sub(r"<![^>]*>|<%>|<\$>|\{[^}]*\}",".",m[3]).replace("<->",".")
        words=[w for w in re.split(r"[.,\s]+",raw) if re.fullmatch(r"[a-z]+",w)]
        if words:lines[m[1]]=(folio,pages.get(folio,{}),words)
    return lines

def aligned(zl,it):
    rows=[]
    # Any differing word count is discarded; no false precision from shifted indices.
    for id,(folio,meta,ws) in zl.items():
        if id not in it:continue
        _,_,other=it[id]
        if len(ws)!=len(other) or meta.get("L") not in ("A","B"):continue
        for w,w2 in zip(ws,other):
            a,b=glyphs(w),glyphs(w2)
            if len(a)!=len(b):continue
            if sum(x!=y for x,y in zip(a,b))>1:continue
            for j,(g,h) in enumerate(zip(a,b)):
                rows.append({"leaf":re.sub(r"[rv].*$","",folio),"folio":folio,
                    "language":meta["L"],"hand":meta.get("H","?"),
                    "section":meta.get("I","?"),"glyph":g,"alternative":h,
                    "word":w,"n_glyphs":len(a),
                    "position":0 if j==0 else 2 if j==len(a)-1 else 1,
                    "is_compound":int(len(g)>1),"disagreement":int(g!=h)})
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--zl",required=True);p.add_argument("--it",required=True)
    p.add_argument("--out",default="results/ambiguity_pilot")
    args=p.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    import numpy as np
    from sklearn.model_selection import GroupKFold
    rows=aligned(read_ivtff(args.zl),read_ivtff(args.it))
    glyph_freq=collections.Counter(r["glyph"] for r in rows)
    X=np.asarray([[math.log1p(glyph_freq[r["glyph"]]),r["n_glyphs"],
       int(r["position"]==0),int(r["position"]==2),r["is_compound"],
       int(r["language"]=="B"),int(r["hand"]=="3"),int(r["section"]=="S")]
       for r in rows],dtype=float)
    y=np.array([r["disagreement"] for r in rows])
    groups=[r["leaf"] for r in rows]
    output=[]
    for i,(a,b) in enumerate(GroupKFold(n_splits=5).split(X,y,groups)):
        mean=X[a].mean(0);std=X[a].std(0)+1e-8
        model=LogisticRegression(max_iter=1000,class_weight=None)
        model.fit((X[a]-mean)/std,y[a])
        preds=model.predict_proba((X[b]-mean)/std)[:,1]
        base=-X[b,0]
        output.append({"fold":i,"n":len(b),"disagreements":int(y[b].sum()),
                       "auc":float(roc_auc_score(y[b],preds)),
                       "rarity_auc":float(roc_auc_score(y[b],base)),
                       "average_precision":float(average_precision_score(y[b],preds)),
                       "prevalence":float(y[b].mean()),
                       "brier":float(brier_score_loss(y[b],preds))})
    dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    (dest/"summary.json").write_text(json.dumps({"n":len(rows),"disagreements":int(y.sum()),
      "folds":output},indent=2)+"\n")
    with (dest/"candidates.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(r for r in rows if r["disagreement"])
    print(json.dumps(output,indent=2))

if __name__=="__main__":main()
