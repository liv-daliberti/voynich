#!/usr/bin/env python3
"""Unsupervised structural graph matching with a known-cipher positive control.

The mapping uses *only* token-order statistics and folio/line position:
word strings, spelling distances and meanings play NO role.
English↔Voynich correspondences are unnamed structural analogies, never
translations. This is a pilot with no cross-language ground-truth anchors.
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
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.context.slot_filling import (
    voynich_lines, english_sentences, prefix_budget,
    split_sequences, shuffle_within_group_position,
)


class ContextGraph:
    def __init__(self, sequences, top_k=70):
        count=Counter(w for _,line in sequences for w in line)
        self.words=[w for w,_ in count.most_common(top_k)]
        self.index={w:i for i,w in enumerate(self.words)}
        self.total=sum(count.values())
        self.count=count
        self.start=Counter()
        self.end=Counter()
        self.out=defaultdict(Counter)
        self.ins=defaultdict(Counter)
        for _,words in sequences:
            if not words:continue
            self.start[words[0]]+=1
            self.end[words[-1]]+=1
            for a,b in zip(words,words[1:]):
                self.out[a][b]+=1
                self.ins[b][a]+=1
        self.matrix=np.zeros((len(self.words),len(self.words)))
        for a in self.words:
            for b,n in self.out[a].items():
                if b in self.index:
                    self.matrix[self.index[a],self.index[b]]=n
        row=self.matrix.sum(axis=1,keepdims=True)
        self.matrix=np.divide(self.matrix,row,out=np.zeros_like(self.matrix),
                              where=row>0)
        self.features=self.extract_features()

    @staticmethod
    def entropy(c):
        n=sum(c.values())
        if not n:return 0.
        return -sum(v/n*math.log(v/n) for v in c.values())

    def extract_features(self):
        feats=[]
        N=len(self.words)
        for rank,w in enumerate(self.words):
            n=self.count[w]
            before,after=self.ins[w],self.out[w]
            f=[
                math.log1p(n)/math.log1p(self.total),
                1-rank/max(1,N-1),
                self.start[w]/n,self.end[w]/n,
                math.log1p(len(before))/math.log1p(max(2,len(self.count))),
                math.log1p(len(after))/math.log1p(max(2,len(self.count))),
                self.entropy(before)/math.log1p(max(1,len(before))),
                self.entropy(after)/math.log1p(max(1,len(after))),
                self.out[w][w]/n,
                sum(self.count[x]*v for x,v in before.items())/
                    (n*max(1,self.total)),
                sum(self.count[x]*v for x,v in after.items())/
                    (n*max(1,self.total)),
            ]
            feats.append(f)
        return np.asarray(feats)

    def top_neighbors(self,w,k=5):
        return self.out[w].most_common(k)


def align(left:ContextGraph,right:ContextGraph, graph_weight=20.,
          steps=5):
    if len(left.words)!=len(right.words):
        raise ValueError("Equal number of words required")
    a,b=left.features,right.features
    std=np.maximum(np.std(np.vstack((a,b)),axis=0),.05)
    costs=np.square((a[:,None,:]-b[None,:,:])/std).sum(2)
    li,rj=linear_sum_assignment(costs)
    perm=rj[np.argsort(li)]
    # Optional iterative graph structural refinement of the initial match.
    # Frequency/position signatures are the dominant initialization.
    for _ in range(steps):
        P=np.zeros_like(costs)
        P[np.arange(len(perm)),perm]=1.
        adj=left.matrix@P@right.matrix.T
        reverse=left.matrix.T@P@right.matrix
        new_cost=costs-graph_weight*(adj+reverse)
        li,rj=linear_sum_assignment(new_cost)
        updated=rj[np.argsort(li)]
        if np.array_equal(updated,perm):break
        perm=updated
    graph_sim=left.matrix@np.eye(len(perm))[:,perm].T
    # Compare transition patterns after aligning donor states to source.
    reordered=right.matrix[np.ix_(perm,perm)]
    return perm,{
        "feature_cost_mean":round(float(costs[np.arange(len(perm)),perm].mean()),5),
        "edge_pattern_cosine":round(float(
            (left.matrix*reordered).sum()/
            max(1e-12,np.linalg.norm(left.matrix)*np.linalg.norm(reordered))),5),
        "random_match_edge_cosine":None
    }


def known_renaming_test(sequences,k=70,seed=20261028):
    # Two nonidentical portions of *the same* English book.
    cut=len(sequences)//2
    left=sequences[:cut]
    right=sequences[cut:]
    allwords=sorted({w for _,line in sequences for w in line})
    rng=random.Random(seed)
    shuffled=list(allwords)
    rng.shuffle(shuffled)
    labels={w:f"c{idx:06d}" for idx,w in enumerate(shuffled)}
    coded=[(group,[labels[w] for w in words]) for group,words in right]
    gleft=ContextGraph(left,k)
    gright=ContextGraph(coded,k)
    perm,quality=align(gleft,gright)
    matches=sum(gright.words[perm[i]]==labels[w]
                for i,w in enumerate(gleft.words))
    possible=sum(labels[w] in gright.index for w in gleft.words)
    freq_match=sum(gright.words[i]==labels[w]
                   for i,w in enumerate(gleft.words))
    return {"evaluated_left_words":k,
            "ground_truth_present_in_right_topk":possible,
            "correct_graph_matches":matches,
            "accuracy_over_all_left_words":round(matches/k,5),
            "accuracy_among_feasible_left_words":round(matches/possible,5)
                                                 if possible else None,
            "frequency_rank_only_correct":freq_match,
            "random_expectation_on_shared_words":round(possible/k,5),
            "alignment_quality":quality}


def cross_structure(source, target, k=70):
    a,b=ContextGraph(source,k),ContextGraph(target,k)
    perm,quality=align(a,b)
    mapping=[]
    for i in range(min(18,k)):
        j=int(perm[i])
        mapping.append({
            "source_word":a.words[i],
            "voynich_word":b.words[j],
            "source_frequency":a.count[a.words[i]],
            "voynich_frequency":b.count[b.words[j]],
            "source_position_start_pct":round(
                100*a.start[a.words[i]]/a.count[a.words[i]],2),
            "voynich_position_start_pct":round(
                100*b.start[b.words[j]]/b.count[b.words[j]],2),
            "interpretation":"structural analogy only, NOT a translation"
        })
    return {"top_k":k,"alignment":quality,
            "ranked_structural_analogies":mapping}


def run(zl_file,austen_file,darwin_file,out,k=70):
    raw=Path(zl_file).read_text(encoding="utf-8-sig")
    corpora={s:voynich_lines(raw,s) for s in []}
    corpora={"Voynich_A":voynich_lines(raw,"A"),
             "Voynich_B":voynich_lines(raw,"B")}
    english={}
    for name,path in (("austen",austen_file),("darwin",darwin_file)):
        seq=english_sentences(Path(path).read_text(encoding="utf-8-sig"),name)
        english[name]=prefix_budget(seq,22843)
    report={"known_renaming":{},"structural_analogies":{}}
    for name,seq in english.items():
        train,_=split_sequences(seq,.7)
        report["known_renaming"][name]=known_renaming_test(train,k)
        for target,lines in corpora.items():
            target_train,_=split_sequences(lines,.7)
            real=cross_structure(train,target_train,k)
            # Identical marginal distributions and positional start/end
            # counts, but shuffled local adjacency.
            shuffled=shuffle_within_group_position(target_train,20261028)
            null=cross_structure(train,shuffled,k)
            report["structural_analogies"][f"{name}_to_{target}"]={
                "actual":real,"position_preserving_shuffle":null["alignment"]}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] if k=="known_renaming" else {
        z:{"actual":v["actual"]["alignment"],
           "shuffle":v["position_preserving_shuffle"]}
        for z,v in report[k].items()}
        for k in report},indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--zl",required=True)
    p.add_argument("--austen",required=True)
    p.add_argument("--darwin",required=True)
    p.add_argument("--out",default="results/structural_alignment")
    p.add_argument("--k",type=int,default=70)
    a=p.parse_args()
    run(a.zl,a.austen,a.darwin,a.out,a.k)
