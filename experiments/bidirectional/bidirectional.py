#!/usr/bin/env python3
"""Bidirectional per-position and fixed-slot predictions on full held-out lines.

Direction is a property of observation and conditioning, not writing direction.
All forms share a frozen train-only vocabulary with one <UNK> bucket.
Voynich sequences are manuscript LINES, never assumed grammatical sentences.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.context.slot_filling import (
    voynich_lines, english_sentences, prefix_budget, split_sequences,
    shuffle_within_group_position,
)
from experiments.latent_grammar.fixed_target import copy_edit_lines
from experiments.latent_grammar.word_lattice import WordLattice, tune_smoothing


def backwards(sequences):
    return [(group, list(reversed(words))) for group, words in sequences]


class CachedWalk:
    """Exact sparse forward propagation for any word state / number of steps."""
    def __init__(self, model):
        self.model = model
        self.cache = {}
        self.empty = [model.start_probs.copy()]

    def distribution(self, observed, target_index):
        if len(observed) > target_index:
            raise ValueError("Target must not have been observed")
        if not observed:
            while len(self.empty) <= target_index:
                self.empty.append(self.model.forward(self.empty[-1]))
            return self.empty[target_index]
        wid = self.model.index.get(observed[-1], 0)
        steps = target_index - len(observed) + 1
        if wid not in self.cache:
            unit = np.zeros(self.model.V)
            unit[wid] = 1.
            self.cache[wid] = [unit]
        values = self.cache[wid]
        while len(values) <= steps:
            values.append(self.model.forward(values[-1]))
        return values[steps]


def nll(dist, word_id):
    return -math.log(max(float(dist[word_id]), 1e-300))


def entropy(dist):
    return float(-(dist * np.log(np.clip(dist, 1e-300, None))).sum())


def fused(pf, pb, prior, strength=.5):
    # Calibrated product of conditional experts:
    # q(w) * (P(w|left)/q(w))^s * (P(w|right)/q(w))^s.
    logp = (1-2*strength)*np.log(prior) + strength*(
        np.log(pf) + np.log(pb))
    logp -= logp.max()
    arr = np.exp(logp)
    return arr / arr.sum()


def bins(i, n):
    if i == 0:
        return "first"
    if i == n-1:
        return "last"
    position = i/(n-1)
    return "early" if position < 1/3 else ("middle" if position <= 2/3 else "late")


def select_fusion(train, alpha_f, alpha_b, max_vocab=700, seed=20261028):
    inner, validation = split_sequences(train, .85)
    fm = WordLattice(inner, max_vocab, alpha_f)
    bm = WordLattice(backwards(inner), max_vocab, alpha_b)
    events=[(line, i) for _,line in validation for i in range(1,len(line)-1)]
    random.Random(seed).shuffle(events)
    events=events[:650]
    fw, bw = CachedWalk(fm), CachedWalk(bm)
    out=[]
    for strength in (0., .25, .5, .75, 1.):
        loss=0.
        for line, i in events:
            pf=fw.distribution(line[:i], i)
            reverse=list(reversed(line))
            index_b=len(line)-1-i
            pb=bw.distribution(reverse[:index_b],index_b)
            p=fused(pf,pb,fm.q,strength)
            loss+=nll(p,fm.index.get(line[i],0))
        out.append({"strength":strength,"validation_nll":round(
            loss/len(events),6) if events else None})
    if not events:
        return .5, out
    return min(out,key=lambda x:x["validation_nll"])["strength"],out


def evaluate(fwd, bwd, heldout, max_lines=55, seed=20261028,
             fusion_strength=.5, include_traces=True):
    randomizer=random.Random(seed)
    lines=[(group,words) for group,words in heldout if len(words)>=4]
    randomizer.shuffle(lines)
    selected=lines[:max_lines]
    wf,wb=CachedWalk(fwd),CachedWalk(bwd)
    # All positions from all selected complete lines; no cherry-picked slots.
    aggregation=defaultdict(lambda: defaultdict(float))
    fixed=defaultdict(lambda: defaultdict(float))
    examples=[]
    line_losses=[]
    seen=0
    total=0

    def add(key, label, dist, word_id):
        r=aggregation[(key,label)]
        r["n"]+=1
        r["nll"]+=nll(dist,word_id)
        r["top5"]+=int(word_id in np.argsort(-dist)[:5])
        r["entropy"]+=entropy(dist)

    for group,line in selected:
        losses={"group":group,"tokens":len(line),
                "forward":0.,"backward":0.,"both":0.}
        rev=list(reversed(line))
        n=len(line)
        for i,truth in enumerate(line):
            total+=1
            tid=fwd.index.get(truth,0)
            seen+=int(tid != 0)
            pf=wf.distribution(line[:i],i)
            ri=n-1-i
            pb=wb.distribution(rev[:ri],ri)
            pbi=fused(pf,pb,fwd.q,fusion_strength)
            region=bins(i,n)
            for label,d in (("forward",pf),("backward",pb),("both",pbi)):
                add("all",label,d,tid)
                add("all_"+("seen" if tid else "unknown"),label,d,tid)
                losses[label]+=nll(d,tid)
                add("region:"+region,label,d,tid)
                add("from_start:"+str(min(i,9)),label,d,tid)
                add("from_end:"+str(min(n-1-i,9)),label,d,tid)
                add("position_fraction:"+str(min(4,int(i*5/n))),label,d,tid)
                # Track how the same target changes as more boundary words
                # are revealed; every allowed reveal length is evaluated.
            for direction,walker,ordered,j in (
                ("from_start",wf,line,i),
                ("from_end",wb,rev,ri)):
                first_p=None
                samples=[]
                for t in range(j+1):
                    dist=walker.distribution(ordered[:t],j)
                    prob=float(dist[tid])
                    if first_p is None:
                        first_p=prob
                    stat=fixed[(direction,t)]
                    stat["n"]+=1
                    stat["nll"]+=-math.log(max(prob,1e-300))
                    stat["gain"]+=math.log(max(prob,1e-300))-math.log(
                        max(first_p,1e-300))
                    stat["entropy"]+=entropy(dist)
                    if include_traces and len(examples)<8 and i>=3 and ri>=3:
                        choices=np.argsort(-dist)[:5]
                        samples.append({"revealed":t,
                          "gap":j-t, "truth_probability":round(prob,7),
                          "entropy":round(entropy(dist),5),
                          "top5":[[walker.model.vocab[int(k)],
                            round(float(dist[k]),6)] for k in choices]})
                if include_traces and len(examples)<8 and i>=3 and ri>=3:
                    if direction=="from_start":
                        examples.append({"group":group,"line":line,
                          "target_index":i,"target_word":truth,"direction":direction,
                          "predictions":samples})
                    else:
                        examples.append({"group":group,"line":line,
                          "target_index":i,"target_word":truth,"direction":direction,
                          "predictions":samples})
        line_losses.append(losses)
    summary={}
    for (region,label),value in sorted(aggregation.items()):
        nn=value["n"]
        summary.setdefault(region,{})[label]={
            "n":int(nn),"nll":round(value["nll"]/nn,6),
            "top5":round(value["top5"]/nn,6),
            "entropy":round(value["entropy"]/nn,6)}
    curves={}
    for (direction,t),v in sorted(fixed.items()):
        nn=v["n"]
        curves.setdefault(direction,{})[str(t)]={
            "n":int(nn),"mean_nll":round(v["nll"]/nn,6),
            "mean_information_gain":round(v["gain"]/nn,6),
            "mean_entropy":round(v["entropy"]/nn,6)}
    return {
        "selected_lines":len(selected),"evaluated_positions":total,
        "target_seen_rate":round(seen/total,6) if total else None,
        "metrics":summary,"fixed_target_reveal_curves":curves,
        "example_traces":examples,
        "per_line_losses":line_losses}



def paired_cluster_bootstrap(real_lines, shuffled_lines,
                             seed=20261028, reps=700):
    """Resample folios/pseudoblocks, respecting within-group dependence.

    Positive backward_minus_forward means the forward model performs better.
    Positive real_vs_shuffle means real order improves log likelihood.
    """
    if len(real_lines) != len(shuffled_lines):
        raise ValueError("Unpaired evaluation lines")
    groups=defaultdict(list)
    for a,b in zip(real_lines,shuffled_lines):
        if a["group"] != b["group"] or a["tokens"] != b["tokens"]:
            raise ValueError("Real/shuffle evaluation mismatch")
        groups[a["group"]].append((a,b))
    labels=sorted(groups)
    rng=np.random.default_rng(seed)
    metrics=("backward_minus_forward",
             "backward_minus_forward_shuffle_advantage",
             "real_vs_shuffle_forward",
             "real_vs_shuffle_backward",
             "real_vs_shuffle_both")
    def calculate(draw):
        total=sum(a["tokens"] for key in draw for a,_ in groups[key])
        if not total:
            return {k:None for k in metrics}
        return {
            "backward_minus_forward":sum(
                a["backward"]-a["forward"] for key in draw
                for a,_ in groups[key])/total,
            "backward_minus_forward_shuffle_advantage":sum(
                (b["backward"]-a["backward"])-(b["forward"]-a["forward"])
                for key in draw for a,b in groups[key])/total,
            **{f"real_vs_shuffle_{k}":sum(
                b[k]-a[k] for key in draw for a,b in groups[key])/total
               for k in ("forward","backward","both")}}
    point=calculate(labels)
    trials={key:[] for key in metrics}
    for _ in range(reps):
        draw=rng.choice(labels,size=len(labels),replace=True)
        v=calculate(draw)
        for key in metrics:
            trials[key].append(v[key])
    return {"clusters":len(labels),"resamples":reps,
            "estimates":{key:{"point":round(point[key],6),
                              "ci95":[round(float(np.quantile(trials[key],q)),6)
                                      for q in (.025,.975)]}
                         for key in metrics}}


def run(zl, austen, darwin, generated, out, max_lines=55,
        max_vocab=700):
    z=Path(zl).read_text(encoding="utf-8-sig")
    corpora={"Voynich_A":voynich_lines(z,"A"),
             "Voynich_B":voynich_lines(z,"B")}
    for name,path in (("austen",austen),("darwin",darwin)):
        seq=english_sentences(Path(path).read_text(encoding="utf-8-sig"),name)
        corpora[name]=prefix_budget(seq,22843)
    if generated:
        corpora["TimmSchinner_sample"]=copy_edit_lines(
            Path(generated).read_text(encoding="utf-8-sig"))
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    report={}
    for name,seq in corpora.items():
        train,test=split_sequences(seq,.7)
        af,forward_grid=tune_smoothing(train,max_vocab)
        ab,backward_grid=tune_smoothing(backwards(train),max_vocab)
        strength,grid=select_fusion(train,af,ab,max_vocab)
        forward=WordLattice(train,max_vocab,af)
        backward=WordLattice(backwards(train),max_vocab,ab)
        real=evaluate(forward,backward,test,max_lines,
                      fusion_strength=strength)
        control=shuffle_within_group_position(train,20261028)
        null=evaluate(WordLattice(control,max_vocab,af),
                      WordLattice(backwards(control),max_vocab,ab),
                      test,max_lines,fusion_strength=strength,
                      include_traces=False)
        uncertainty=paired_cluster_bootstrap(
            real["per_line_losses"],null["per_line_losses"])
        report[name]={
            "uncertainty":uncertainty,
            "train_tokens":sum(len(line) for _,line in train),
            "heldout_tokens":sum(len(line) for _,line in test),
            "model_vocab":max_vocab+1,
            "selected_smoothing_forward":af,
            "selected_smoothing_backward":ab,
            "fusion_strength":strength,"fusion_validation_grid":grid,
            "forward_validation":forward_grid,
            "backward_validation":backward_grid,
            "real":{"selected_lines":real["selected_lines"],
                    "evaluated_positions":real["evaluated_positions"],
                    "target_seen_rate":real["target_seen_rate"],
                    "metrics":real["metrics"]},
            "position_preserving_shuffle":{"metrics":null["metrics"]}}
        (out/f"{name}_reveal.json").write_text(json.dumps({
            "real_fixed_target_curves":real["fixed_target_reveal_curves"],
            "shuffle_fixed_target_curves":null["fixed_target_reveal_curves"],
            "examples":real["example_traces"]},indent=2)+"\n")
        m=real["metrics"]["all"]
        ms=null["metrics"]["all"]
        print(name,json.dumps({
            "n":real["evaluated_positions"],
            "forward_nll":m["forward"]["nll"],
            "backward_nll":m["backward"]["nll"],
            "both_nll":m["both"]["nll"],
            "forward_vs_shuffle_gain":round(
                ms["forward"]["nll"]-m["forward"]["nll"],5),
            "backward_vs_shuffle_gain":round(
                ms["backward"]["nll"]-m["backward"]["nll"],5),
            "both_vs_shuffle_gain":round(
                ms["both"]["nll"]-m["both"]["nll"],5)}),flush=True)
    (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--zl",required=True)
    p.add_argument("--austen",required=True)
    p.add_argument("--darwin",required=True)
    p.add_argument("--generated")
    p.add_argument("--out",default="results/bidirectional")
    p.add_argument("--max-lines",type=int,default=55)
    p.add_argument("--max-vocab",type=int,default=700)
    a=p.parse_args()
    run(a.zl,a.austen,a.darwin,a.generated,a.out,
        a.max_lines,a.max_vocab)
