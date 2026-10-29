#!/usr/bin/env python3
"""Structural word-class probes with an English noun/verb positive control.

No Voynich type is labeled a noun or verb. A supervised English POS probe
must generalize from Austen to *independent* Darwin, using only distributional
word-order/position features. Apply the frozen probe to Voynich, Naibbe and
original Timm-generated text as OUT-OF-DOMAIN structural analogies.

The English gold labels come from an off-the-shelf POS tagger, not manual
annotation; ambiguous/low-frequency types are filtered. No edit spelling,
word identity, or lexical translation enters the classifier.
"""
from __future__ import annotations
import argparse, csv, json, math, re, sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.context.slot_filling import english_sentences, voynich_lines, prefix_budget
from experiments.cross_page.reconstruct import generated_words


FEATURES=(
    "log_type_freq","log_sentence_coverage","relative_start","relative_end",
    "relative_second","relative_penultimate","left_distinct_ratio",
    "right_distinct_ratio","left_entropy","right_entropy",
    "preceding_top10","following_top10","preceding_top50","following_top50",
    "preceding_rare","following_rare","preceding_freq_mass",
    "following_freq_mass","self_repeat_before","self_repeat_after",
    "left_right_diversity_ratio"
)


def flatten_groups(sequences):
    return [(group,list(words)) for group,words in sequences]


def count_features(sequences,min_count=3):
    freq=Counter(w for _,seq in sequences for w in seq)
    linefreq=Counter(w for _,seq in sequences for w in set(seq))
    vocabulary=sorted(freq,key=lambda w:(-freq[w],w))
    rank={w:i for i,w in enumerate(vocabulary)}
    pos=defaultdict(Counter)
    left,right=defaultdict(Counter),defaultdict(Counter)
    left_common,right_common=defaultdict(Counter),defaultdict(Counter)
    n_docs=len(sequences);n_tokens=sum(freq.values())
    for _,tokens in sequences:
        for i,w in enumerate(tokens):
            if i==0:pos[w]["start"]+=1
            if i==len(tokens)-1:pos[w]["end"]+=1
            if i==1:pos[w]["second"]+=1
            if i==len(tokens)-2:pos[w]["penultimate"]+=1
            if i:
                v=tokens[i-1];left[w][v]+=1
                r=rank[v]
                if r<10:left_common[w]["top10"]+=1
                if r<50:left_common[w]["top50"]+=1
                if r>=100:left_common[w]["rare"]+=1
                left_common[w]["freq_mass"]+=freq[v]
            if i+1<len(tokens):
                v=tokens[i+1];right[w][v]+=1
                r=rank[v]
                if r<10:right_common[w]["top10"]+=1
                if r<50:right_common[w]["top50"]+=1
                if r>=100:right_common[w]["rare"]+=1
                right_common[w]["freq_mass"]+=freq[v]
    def ent(c):
        n=sum(c.values())
        if not n:return 0.
        return -sum(v/n*math.log(v/n) for v in c.values())/math.log(max(2,len(c)))
    rows=[]
    for w in vocabulary:
        f=freq[w]
        if f<min_count:continue
        L=left[w];R=right[w]
        countL=sum(L.values());countR=sum(R.values())
        z={
            "word":w,"count":f,"doc_freq":linefreq[w],
            "log_type_freq":math.log1p(f)/math.log1p(n_tokens),
            "log_sentence_coverage":math.log1p(linefreq[w])/math.log1p(n_docs),
            "relative_start":pos[w]["start"]/f,
            "relative_end":pos[w]["end"]/f,
            "relative_second":pos[w]["second"]/f,
            "relative_penultimate":pos[w]["penultimate"]/f,
            "left_distinct_ratio":len(L)/max(1,countL),
            "right_distinct_ratio":len(R)/max(1,countR),
            "left_entropy":ent(L),"right_entropy":ent(R),
            "preceding_top10":left_common[w]["top10"]/max(1,countL),
            "following_top10":right_common[w]["top10"]/max(1,countR),
            "preceding_top50":left_common[w]["top50"]/max(1,countL),
            "following_top50":right_common[w]["top50"]/max(1,countR),
            "preceding_rare":left_common[w]["rare"]/max(1,countL),
            "following_rare":right_common[w]["rare"]/max(1,countR),
            "preceding_freq_mass":math.log1p(left_common[w]["freq_mass"]/
                                             max(1,countL))/math.log1p(n_tokens),
            "following_freq_mass":math.log1p(right_common[w]["freq_mass"]/
                                             max(1,countR))/math.log1p(n_tokens),
            "self_repeat_before":L[w]/max(1,countL),
            "self_repeat_after":R[w]/max(1,countR),
            "left_right_diversity_ratio":math.log1p(len(L))-math.log1p(len(R))
        }
        rows.append(z)
    return rows


def tag_gold(sequences):
    import nltk
    tally=defaultdict(Counter)
    for _,words in sequences:
        for word,tag in nltk.pos_tag(words):
            if tag.startswith("NN"):tally[word]["N"]+=1
            elif tag.startswith("VB"):tally[word]["V"]+=1
            else:tally[word]["other"]+=1
    return tally


def annotate(rows,tagcounts,min_content=2,dominance=.7):
    chosen=[]
    for row in rows:
        c=tagcounts[row["word"]]
        n,v=c["N"],c["V"]
        if n+v<min_content or max(n,v)<dominance*(n+v):continue
        # Restrict to words with at least 60% total noun/verb usages.
        if (n+v)/max(1,sum(c.values()))<.6:continue
        z=dict(row);z["noun_label"]=int(n>v)
        z["noun_tagged"]=n;z["verb_tagged"]=v
        chosen.append(z)
    return chosen


def fit_and_validate(austen,darwin):
    all_a=count_features(austen)
    all_d=count_features(darwin)
    gold_a=annotate(all_a,tag_gold(austen))
    gold_d=annotate(all_d,tag_gold(darwin))
    if len({r["noun_label"] for r in gold_a})<2:
        raise ValueError("English POS control lacks both content classes")
    def features(rows):
        return np.asarray([[r[k] for k in FEATURES] for r in rows],dtype=float)
    X=features(gold_a); y=np.asarray([r["noun_label"] for r in gold_a])
    train,test=train_test_split(np.arange(len(y)),test_size=.3,
                                stratify=y,random_state=20261028)
    def trainmodel(x,yy):
        scaler=StandardScaler()
        trainx=scaler.fit_transform(x)
        model=LogisticRegression(max_iter=3000,C=1.,class_weight="balanced")
        model.fit(trainx,yy)
        return model,scaler
    m_inner,sc_inner=trainmodel(X[train],y[train])
    p_inner=m_inner.predict_proba(sc_inner.transform(X[test]))[:,1]
    m,sc=trainmodel(X,y)
    def metrics(yy,p):
        return {
            "word_types":len(yy),"noun_fraction":round(float(np.mean(yy)),5),
            "auc":round(float(roc_auc_score(yy,p)),5),
            "average_precision":round(float(average_precision_score(yy,p)),5),
            "balanced_accuracy":round(float(balanced_accuracy_score(
                yy,np.asarray(p)>=.5)),5)}
    out={"austen_independent_word_type_holdout":metrics(y[test],p_inner),
         "austen_training_word_types":len(y),
         "darwin_independent_book":None,
         "english_label_source":"NLTK averaged perceptron POS tagger",
         "feature_set":"21 grammar/position/frequency/count-context features; no word identity, glyph or suffix features"}
    if gold_d and len({r["noun_label"] for r in gold_d})==2:
        yd=np.asarray([r["noun_label"] for r in gold_d])
        pd=m.predict_proba(sc.transform(features(gold_d)))[:,1]
        out["darwin_independent_book"]=metrics(yd,pd)
    return m,sc,out,all_a,all_d


def apply_probe(model,scaler,rows,reference):
    if not rows:return {"n_words":0},[]
    X=np.asarray([[row[k] for k in FEATURES] for row in rows],dtype=float)
    Z=scaler.transform(X)
    p=model.predict_proba(Z)[:,1]
    reference_z=scaler.transform(np.asarray(
        [[r[k] for k in FEATURES] for r in reference],dtype=float))
    reference_dist=np.sqrt(np.mean(reference_z**2,axis=1))
    threshold=np.quantile(reference_dist,.95)
    deviation=np.sqrt(np.mean(Z**2,axis=1))
    enriched=[]
    for row,prob,dist in zip(rows,p,deviation):
        r=dict(row)
        r["english_probe_noun_like"]=round(float(prob),6)
        r["english_feature_ood"]=int(dist>threshold)
        r["english_feature_distance"]=round(float(dist),5)
        enriched.append(r)
    # Token frequencies are weights, not a declaration that most word
    # forms in a language must have a given POS class.
    summary={"n_words":len(rows),"token_count":sum(r["count"] for r in rows),
             "mean_noun_like_prob_type":round(float(np.mean(p)),5),
             "pct_type_noun_like_over_08":round(float(np.mean(p>.8)),5),
             "pct_type_verb_like_below_02":round(float(np.mean(p<.2)),5),
             "pct_english_feature_ood":round(float(np.mean(deviation>threshold)),5),
             "reference_english_ood_95pct_distance":round(float(threshold),4),
             "top_noun_like":[[x["word"],x["english_probe_noun_like"],
                               x["english_feature_ood"]]
                              for x in sorted(enriched,
                               key=lambda x:-x["english_probe_noun_like"])[:15]],
             "top_verb_like":[[x["word"],x["english_probe_noun_like"],
                               x["english_feature_ood"]]
                              for x in sorted(enriched,
                               key=lambda x:x["english_probe_noun_like"])[:15]]}
    return summary,enriched


def unsupervised_roles(rows,k=5):
    if len(rows)<k:return {"available":False},[]
    X=np.asarray([[r[f] for f in FEATURES] for r in rows],dtype=float)
    scaler=StandardScaler();Z=scaler.fit_transform(X)
    model=KMeans(n_clusters=k,random_state=20261028,n_init=15)
    labels=model.fit_predict(Z)
    report={}
    enriched=[]
    for n in range(k):
        idx=np.where(labels==n)[0]
        exemplar=sorted([rows[i] for i in idx],
                        key=lambda r:(-r["count"],r["word"]))[:15]
        report[str(n)]={"n_types":len(idx),
            "tokens":int(sum(rows[i]["count"] for i in idx)),
            "mean_start_rate":round(float(np.mean(
                [rows[i]["relative_start"] for i in idx])),5),
            "mean_end_rate":round(float(np.mean(
                [rows[i]["relative_end"] for i in idx])),5),
            "top_words":[x["word"] for x in exemplar]}
    for r,label in zip(rows,labels):
        enriched.append({"word":r["word"],"role_cluster":int(label),
                         "count":r["count"]})
    return {"clusters":report,
            "interpretation":"Unsupervised contextual roles; cluster IDs have NO grammatical meaning"},enriched


def run(zl,austen,darwin,naibbe,generated,out):
    a=prefix_budget(english_sentences(
        Path(austen).read_text(encoding="utf-8-sig"),"austen"),22843)
    d=prefix_budget(english_sentences(
        Path(darwin).read_text(encoding="utf-8-sig"),"darwin"),22843)
    model,scaler,english,english_a,english_d=fit_and_validate(a,d)
    text=Path(zl).read_text(encoding="utf-8-sig")
    corpora={"Voynich_A":voynich_lines(text,"A"),
             "Voynich_B":voynich_lines(text,"B"),
             "Austen":a,"Darwin":d}
    if naibbe:
        words=re.findall(r"[a-z]+",Path(naibbe).read_text(
            encoding="utf-8-sig").lower())
        # Pseudo-line budget matches Voynich typical paragraph-line size,
        # not a true sentence or section in Naibbe.
        corpora["Naibbe"]=[(f"page{i//25}",words[i:i+10])
                           for i in range(0,min(len(words),22840),10)]
    if generated:
        txt=Path(generated).read_text(encoding="utf-8-sig")
        lines=[]
        for i,ln in enumerate(txt.splitlines()):
            if ln.strip().startswith("#"):continue
            w=re.findall(r"[a-z]+",ln.lower())
            if len(w)>=3:lines.append((f"p{i//29}",w))
        corpora["Timm"]=lines
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    summary={"english_positive_control":english,"corpora":{}}
    for name,seqs in corpora.items():
        rows=count_features(seqs,min_count=3)
        stat,annotated=apply_probe(model,scaler,rows,english_a)
        roles,clusters=unsupervised_roles(rows)
        summary["corpora"][name]={"probe":stat,"unsupervised":roles}
        with (out/f"{name}_structural_roles.csv").open("w",newline="") as f:
            if annotated:
                w=csv.DictWriter(f,fieldnames=list(annotated[0]))
                w.writeheader();w.writerows(annotated)
        with (out/f"{name}_clusters.csv").open("w",newline="") as f:
            if clusters:
                w=csv.DictWriter(f,fieldnames=list(clusters[0]))
                w.writeheader();w.writerows(clusters)
        print(name,json.dumps({
            "types":stat["n_words"],
            "feature_ood":stat["pct_english_feature_ood"],
            "role_groups":{k:len(v["top_words"]) for k,v
                           in roles["clusters"].items()},
        }),flush=True)
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("zl","austen","darwin","out"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--naibbe")
    p.add_argument("--generated")
    args=p.parse_args()
    run(args.zl,args.austen,args.darwin,args.naibbe,args.generated,args.out)
