#!/usr/bin/env python3
"""Voynich word-form stability over many 75% vocabulary deletion masks.

'Robust' means frequently recoverable from existing spellings. It does not mean
that a word is genuine, meaningful, or grammatically well-formed.

Inputs ZL3b-n.txt, IT2a-n.txt are NOT distributed with this code.
Uses authoritative Currier language labels in each folio's header.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import re
from pathlib import Path
import numpy as np

COMPOUND = ('ckh','cth','cph','cfh','ch','sh')


def glyphs(w: str) -> tuple[str, ...]:
    out=[];i=0
    while i<len(w):
        x=next((x for x in COMPOUND if w.startswith(x,i)), None)
        if x: out.append(x); i+=len(x)
        else: out.append(w[i]);i+=1
    return tuple(out)


def parse(text: str) -> list[dict]:
    result=[];page=None
    for line in text.splitlines():
        h=re.match(r'^<([^.,>]+)>\s*<!\s*(.*?)>',line)
        if h:
            m=re.search(r'\$L=([AB])(?:\s|$)',h[2]);page={'folio':h[1], 'language':m[1] if m else '?','words':[]}
            result.append(page);continue
        row=re.match(r'^<([^>,]+),([^>]+)>\s*(.*)$',line)
        if not row or not re.search(r'[+@*=]P',row[2]) or page is None:continue
        clean=re.sub(r'<![^>]*>|<%>|<\$>|\{[^}]*\}','.',row[3]).replace('<->','.')
        page['words'].extend(w for w in re.split(r'[.,\s]+',clean) if re.fullmatch(r'[a-z]+',w))
    return [p for p in result if p['words'] and p['language'] in ('A','B')]


def canonical(units: tuple[str,...],word: str) -> bool:
    # Compound EVA units can be resegmented by adjoining glyphs: do not allow it.
    return glyphs(word) == units


def one_edit_forms(units: tuple[str,...], alphabet: list[str]):
    # No feedback from any held-out manuscript word is used here.
    found=set()
    def keep(v):
        if v:
            w=''.join(v)
            if canonical(tuple(v),w):found.add(w)
    for i in range(len(units)):
        keep(units[:i]+units[i+1:])
        for u in alphabet:
            if u!=units[i]: keep(units[:i]+(u,)+units[i+1:])
    for i in range(len(units)+1):
        for u in alphabet:keep(units[:i]+(u,)+units[i:])
    return found


def make_candidate_universe(training: set[str],alphabet: list[str],min_degree:int=2):
    observed=sorted(training)
    degree=Counter()
    for w in observed:
        for v in one_edit_forms(glyphs(w),alphabet): degree[v]+=1
    tracked=sorted(w for w,d in degree.items() if d>=min_degree)
    idx={w:i for i,w in enumerate(tracked)}
    neighbors=[[] for _ in tracked]
    for j,w in enumerate(observed):
        for v in one_edit_forms(glyphs(w),alphabet):
            k=idx.get(v)
            if k is not None:neighbors[k].append(j)
    ptr=np.empty(len(tracked)+1,dtype=np.int64);ptr[0]=0
    for i,a in enumerate(neighbors):ptr[i+1]=ptr[i]+len(a)
    flat=np.fromiter((j for a in neighbors for j in a),dtype=np.int32,count=int(ptr[-1]))
    train_ids={w:i for i,w in enumerate(observed)}
    self_id=np.fromiter((train_ids.get(w,-1) for w in tracked),dtype=np.int32,count=len(tracked))
    return observed,degree,tracked,ptr,flat,self_id


def quantile(arr, q):
    return float(np.quantile(arr,q)) if len(arr) else None


def execute_language(zl_pages,it_pages,language:str,alphabet:list[str],seed:int=20261028,
                     trials:int=500,survive:float=.25,top_k:int=500):
    pages=[p for p in zl_pages if p['language']==language]
    cut=int(len(pages)*.7)
    training={w for p in pages[:cut] for w in p['words']}
    heldout={w for p in pages[cut:] for w in p['words']} - training
    full={w for p in pages for w in p['words']}
    it={w for p in it_pages if p['language']==language for w in p['words']}
    observed,degree,tracked,ptr,flat,self_id=make_candidate_universe(training,alphabet)
    types=len(tracked);total=len(observed)
    allowed=int(round(total*survive));top=np.zeros(types,dtype=np.int32)
    support_sum=np.zeros(types,dtype=np.float64)
    reached=np.zeros(types,dtype=np.int32)
    reached2=np.zeros(types,dtype=np.int32)
    eligible_sum=np.zeros(types,dtype=np.int32)
    kinds=np.array([0 if w in training else 1 if w in heldout else
                    2 if w not in full and w in it else 3 for w in tracked],dtype=np.int8)
    composition=np.zeros((trials,4),dtype=np.int32)
    cutoff=[]
    generator=np.random.default_rng(seed+ord(language)*191)
    for rep in range(trials):
        chosen=generator.choice(total,size=allowed,replace=False)
        visible=np.zeros(total,dtype=np.int8);visible[chosen]=1
        # Each row of this sparse, ragged support matrix counts independent
        # original training types surviving the current deletion mask.
        support=np.add.reduceat(visible[flat],ptr[:-1]) if types else np.empty(0,dtype=np.int64)
        present=(self_id>=0)&(visible[self_id.clip(min=0)]>0)
        eligible=(~present)
        eligible_sum+=eligible.astype(np.int32)
        support_sum+=support*eligible
        reached+=((support>=1)&eligible).astype(np.int32)
        reached2+=((support>=2)&eligible).astype(np.int32)
        possible=np.flatnonzero(eligible&(support>0))
        if not len(possible):cutoff.append(0);continue
        # Uniform random ties prevent alphabetic bias and make repeated
        # nominations a meaningful stability measurement. No heldout labels.
        rank=support[possible]+generator.random(len(possible))*.499
        k=min(top_k,len(possible));ids=np.argpartition(rank,-k)[-k:]
        finalists=possible[ids]
        cutoff.append(int(support[finalists].min()))
        top[finalists]+=1
        composition[rep]=np.bincount(kinds[finalists],minlength=4)
    den=np.maximum(1,eligible_sum)
    scores=[]
    for i,w in enumerate(tracked):
        scores.append({'word':w,'kind':('deleted_training','zl_holdout','it_only','neither')[int(kinds[i])],
                       'full_training_degree':degree[w],
                       'eligible_trials':int(eligible_sum[i]),'top500_trials':int(top[i]),
                       'top500_pct':float(100*top[i]/den[i]),
                       'reachable_pct':float(100*reached[i]/den[i]),
                       'two_witness_pct':float(100*reached2[i]/den[i]),
                       'mean_visible_witnesses':float(support_sum[i]/den[i])})
    scores.sort(key=lambda q:(-q['top500_pct'],-q['mean_visible_witnesses'],q['word']))
    kindnames=['deleted_training','zl_holdout','it_only','neither']
    comp={kindnames[i]:float(composition[:,i].mean()) for i in range(4)}
    bins=defaultdict(lambda:{'n':0,'zl_holdout':0,'it_only':0})
    for w,d in degree.items():
        if w in training:continue
        k=min(d,10);bins[k]['n']+=1
        bins[k]['zl_holdout']+=int(w in heldout)
        bins[k]['it_only']+=int(w not in full and w in it)
    binsout=[{'degree':k,**x,'later_attestation_pct':100*x['zl_holdout']/x['n']}
             for k,x in sorted(bins.items())]
    return {'summary':{'language':language,'train_pages':cut,'train_types':len(training),
                       'later_new_types':len(heldout),'candidate_universe':len(degree),
                       'tracked_min_degree':2,'tracked_types':types,'survivor_count':allowed,
                       'survivor_fraction':survive,'trials':trials,'top_k':top_k,
                       'top500_composition_mean':comp,'top500_cutoff_min':min(cutoff),
                       'top500_cutoff_max':max(cutoff),
                       'exact_topk':min(cutoff)>=2,
                       'holdout_types_reachable_from_full_training':sum(w in degree for w in heldout)},
            'scores':scores,'degree_calibration':binsout,
            'composition':composition}


def run(zl_path,it_path,out,trials=500,top_k=500):
    z=parse(Path(zl_path).read_text(encoding='utf-8-sig'))
    it=parse(Path(it_path).read_text(encoding='utf-8-sig'))
    # Predeclared EVA alphabet across the ZL corpus; no novel heldout words
    # are used to define candidate edit neighbors, but units are prespecified.
    alphabet=sorted({g for p in z for w in p['words'] for g in glyphs(w)})
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    report=[]
    for language in 'AB':
        r=execute_language(z,it,language,alphabet,trials=trials,top_k=top_k)
        report.append(r['summary'])
        print(language,r['summary'],flush=True)
        with (out/f'candidate_stability_{language}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(r['scores'][0]))
            writer.writeheader();writer.writerows(r['scores'])
        with (out/f'top100_unseen_{language}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(r['scores'][0]));writer.writeheader()
            writer.writerows([row for row in r['scores'] if row['kind']!='deleted_training'][:100])
        (out/f'degree_calibration_{language}.json').write_text(json.dumps(r['degree_calibration'],indent=2)+'\n')
        with (out/f'top500_composition_{language}.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['trial','deleted_training','zl_holdout','it_only','neither'])
            for i,v in enumerate(r['composition']):writer.writerow([i,*v])
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--zl',required=True);p.add_argument('--it',required=True)
    p.add_argument('--out',default='results/75pct_robustness')
    p.add_argument('--trials',type=int,default=500);p.add_argument('--top-k',type=int,default=500)
    a=p.parse_args()
    run(a.zl,a.it,a.out,a.trials,a.top_k)