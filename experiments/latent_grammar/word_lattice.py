#!/usr/bin/env python3
"""Exactly marginalized word-transition lattice for one fixed future position.

The state is an observed word type (or <UNK>), not a latent grammatical role.
Intermediate words are summed over by sparse Markov forward propagation, NOT
pruned by a beam. More distant context beyond the last observed word requires
a higher-order state model and is not claimed here.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.context.slot_filling import (
    voynich_lines, english_sentences, prefix_budget,
    split_sequences, shuffle_within_group_position,
)
from experiments.latent_grammar.fixed_target import copy_edit_lines


class WordLattice:
    def __init__(self, sequences, max_vocab=700, smoothing=10.0):
        counts = Counter(w for _, line in sequences for w in line)
        self.vocab = ["<UNK>"] + [w for w, _ in counts.most_common(max_vocab)]
        self.index = {w: i for i, w in enumerate(self.vocab)}
        self.V = len(self.vocab)
        self.smoothing = float(smoothing)
        self.count = np.zeros(self.V)
        self.start = np.zeros(self.V)
        link = Counter()
        for _, line in sequences:
            ids = [self.index.get(w, 0) for w in line]
            if not ids:
                continue
            self.start[ids[0]] += 1
            for i, wid in enumerate(ids):
                self.count[wid] += 1
                if i:
                    link[ids[i-1], wid] += 1
        self.q = (self.count + .1) / (self.count.sum() + .1 * self.V)
        self.start_probs = (self.start + 12 * self.q) / (
            self.start.sum() + 12)
        rows, cols, data = [], [], []
        source_totals = np.zeros(self.V)
        for (a,b), n in link.items():
            source_totals[a] += n
        for (a,b), n in link.items():
            rows.append(a)
            cols.append(b)
            data.append(n / source_totals[a])
        self.empirical = csr_matrix(
            (np.asarray(data, dtype=float), (rows,cols)),
            shape=(self.V,self.V))
        self.mix = source_totals / (source_totals + self.smoothing)
        self.mix[0] = 0.0

    def forward(self, dist):
        nxt = np.asarray((dist * self.mix) @ self.empirical).reshape(-1)
        nxt += float(np.dot(dist, 1-self.mix)) * self.q
        # All nodes and paths retained, including the OOV state.
        if abs(nxt.sum()-1) > 1e-8:
            raise AssertionError(f"Non-normalized forward result: {nxt.sum()}")
        return nxt

    def forecast(self, prefix, target_index):
        if len(prefix)>target_index:
            raise ValueError("No observed target leakage")
        if not prefix:
            current = self.start_probs.copy()
            steps = target_index
        else:
            current = np.zeros(self.V)
            current[self.index.get(prefix[-1],0)] = 1
            steps = target_index-len(prefix)+1
        for _ in range(steps):
            current = self.forward(current)
        return current

    def lattice(self, prefix, target_index, examples=7):
        if not prefix:
            start = self.start_probs.copy()
            steps=target_index
        else:
            start=np.zeros(self.V)
            start[self.index.get(prefix[-1],0)]=1
            steps=target_index-len(prefix)+1
        dist=start
        layers=[]
        for depth in range(steps+1):
            best = np.argsort(-dist)[:examples]
            layers.append({"relative_depth":depth,"total_mass":round(float(dist.sum()),8),
                           "top_states":[{"word":self.vocab[int(i)],
                                          "mass":round(float(dist[i]),6)}
                                         for i in best],
                           "other_states_mass":round(float(1-dist[best].sum()),6)})
            if depth<steps:
                dist=self.forward(dist)
        return layers


def tune_smoothing(train, max_vocab=700):
    inner, val = split_sequences(train, .85)
    observed=[(line[i-1], line[i]) for _,line in val for i in range(1,len(line))]
    if not observed:
        return 10., []
    observed=observed[:2200]
    values=[]
    for alpha in (.5, 2, 8, 32, 128):
        m=WordLattice(inner, max_vocab, alpha)
        cache={}
        nll=0
        for prev, actual in observed:
            idx=m.index.get(prev,0)
            if idx not in cache:
                unit=np.zeros(m.V);unit[idx]=1.
                cache[idx]=m.forward(unit)
            nll+=-math.log(max(cache[idx][m.index.get(actual,0)],1e-300))
        values.append({"smoothing":alpha,"validation_nll":round(nll/len(observed),5)})
    best=min(values,key=lambda r:r["validation_nll"])
    return best["smoothing"],values


def evaluate(model, test, max_targets=300, seed=20261028):
    tasks=[(group,line,T) for group,line in test for T in (3,5,7,9)
           if T<len(line)]
    random.Random(seed).shuffle(tasks)
    tasks=tasks[:max_targets]
    stats=defaultdict(lambda:{"n":0, "nll":0., "entropy":0.,
                              "gain":0., "seen_n":0, "seen_nll":0., "top5":0})
    traces=[]
    for group,line,T in tasks:
        target=model.index.get(line[T],0)
        base=None
        track=[]
        for t in range(T+1):
            probs=model.forecast(line[:t],T)
            p=float(probs[target])
            H=float(-(probs*np.log(np.clip(probs,1e-300,None))).sum())
            if base is None:
                base=p
            key=(T,t)
            d=stats[key]
            d["n"]+=1
            d["nll"]+=-math.log(max(p,1e-300))
            d["entropy"]+=H
            d["gain"]+=math.log(max(p,1e-300)/max(base,1e-300))
            d["top5"]+=int(target in np.argsort(-probs)[:5])
            if target:
                d["seen_n"]+=1
                d["seen_nll"]+=-math.log(max(p,1e-300))
            if len(traces)<4:
                best=np.argsort(-probs)[:5]
                track.append({"revealed":t,"remaining_gap":T-t,
                              "actual_probability":round(p,7),
                              "full_entropy_nats":round(H,5),
                              "top5":[[model.vocab[int(i)],round(float(probs[i]),6)]
                                      for i in best]})
        if len(traces)<4:
            traces.append({"group":group,"target_index":T,
                           "true_word":line[T],"train_vocab_member":bool(target),
                           "observed_line_prefix":line[:T+1],
                           "trace":track,
                           "full_state_lattice":model.lattice(line[:T-3],T)})
    out={}
    for (T,t),x in sorted(stats.items()):
        n=x["n"]
        out[f"T{T}_t{t}"]={"T":T,"revealed":t,"remaining_gap":T-t,
            "n":n,"nll":round(x["nll"]/n,5),
            "entropy":round(x["entropy"]/n,5),
            "gain":round(x["gain"]/n,5),
            "top5":round(x["top5"]/n,5),
            "seen_n":x["seen_n"],
            "seen_only_nll":round(x["seen_nll"]/x["seen_n"],5)
                            if x["seen_n"] else None}
    return {"targets":len(tasks),"by_target_and_prefix":out,
            "examples":traces}


def aggregate_by_gap(result):
    grouped=defaultdict(list)
    for v in result["by_target_and_prefix"].values():
        grouped[v["remaining_gap"]].append(v)
    out={}
    for gap,rows in grouped.items():
        n=sum(x["n"] for x in rows)
        out[str(gap)]={"n":n,**{
            field:round(sum(x[field]*x["n"] for x in rows)/n,5)
            for field in ("nll","gain","entropy","top5")}}
    return out


def run(zl, austen, darwin, generated, out, max_vocab=700,max_targets=300):
    txt=Path(zl).read_text(encoding="utf-8-sig")
    sources={"Voynich_A":voynich_lines(txt,"A"),
             "Voynich_B":voynich_lines(txt,"B")}
    for name,source in (("austen",austen),("darwin",darwin)):
        parsed=english_sentences(Path(source).read_text(encoding="utf-8-sig"),name)
        sources[name]=prefix_budget(parsed,22843)
    if generated:
        sources["TimmSchinner_sample"]=copy_edit_lines(
            Path(generated).read_text(encoding="utf-8-sig"))
    results={}
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    for name,seq in sources.items():
        train,test=split_sequences(seq)
        alpha,grid=tune_smoothing(train,max_vocab)
        model=WordLattice(train,max_vocab,alpha)
        shuffled=WordLattice(shuffle_within_group_position(train,20261028),
                             max_vocab,alpha)
        real=evaluate(model,test,max_targets)
        null=evaluate(shuffled,test,max_targets)
        gain=aggregate_by_gap(real)
        null_gain=aggregate_by_gap(null)
        results[name]={"train_tokens":sum(len(s) for _,s in train),
            "test_tokens":sum(len(s) for _,s in test),
            "vocab_including_unknown":model.V,
            "heldout_target_count":real["targets"],
            "selection_grid":grid, "selected_smoothing":alpha,
            "real_by_remaining_gap":gain,
            "position_shuffled_by_remaining_gap":null_gain,
            "real_context_vs_shuffle_nll_gain_at_gap1":
                round(null_gain["1"]["nll"]-gain["1"]["nll"],5)}
        (out/f"{name}_traces.json").write_text(json.dumps(
            {"real":real,"shuffled":null},indent=2)+"\n")
        print(name,json.dumps(results[name]),flush=True)
    (out/"summary.json").write_text(json.dumps(results,indent=2)+"\n")
    return results


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--zl",required=True)
    ap.add_argument("--austen",required=True)
    ap.add_argument("--darwin",required=True)
    ap.add_argument("--generated",default=None)
    ap.add_argument("--out",default="results/word_lattice")
    ap.add_argument("--max-vocab",type=int,default=700)
    ap.add_argument("--max-targets",type=int,default=300)
    a=ap.parse_args()
    run(a.zl,a.austen,a.darwin,a.generated,a.out,a.max_vocab,a.max_targets)
