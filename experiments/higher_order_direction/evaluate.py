#!/usr/bin/env python3
"""Position-aware 1–4-gram directional comparison on four disjoint folio folds.

True held-out word targets, same word vocabulary for real and shuffled training,
no target information in feature fitting. Page-block folds do NOT establish
manuscript writing chronology.
"""
from __future__ import annotations
import argparse, json, math, random, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.context.slot_filling import voynich_lines, english_sentences, prefix_budget, shuffle_within_group_position
from experiments.latent_grammar.fixed_target import copy_edit_lines


def groups_in_order(lines):
    return list(dict.fromkeys(g for g,_ in lines))


def blocked_folds(lines,k=4):
    names=groups_in_order(lines)
    folds=np.array_split(np.arange(len(names)), k)
    for f,inds in enumerate(folds):
        test_ids={names[i] for i in inds}
        yield f, [(g,x) for g,x in lines if g not in test_ids], [(g,x) for g,x in lines if g in test_ids]


def posbin(i,n,reverse=False):
    if reverse: i=n-1-i
    if i==0:return 'first'
    if i==n-1:return 'last'
    if i==1:return 'second'
    if i==n-2:return 'penultimate'
    q=i/max(n-1,1)
    return 'early' if q<1/3 else ('middle' if q<=2/3 else 'late')


class ConditionalNgram:
    def __init__(self,seqs, direction='forward',alpha=64, max_vocab=1200,
                 pos_alpha=50):
        self.direction=direction;self.alpha=float(alpha);self.pos_alpha=pos_alpha
        counts=Counter(w for _,line in seqs for w in line)
        self.vocab=['<UNK>']+[w for w,_ in sorted(counts.items(),key=lambda it:(-it[1],it[0]))[:max_vocab]]
        self.word_set=set(self.vocab);self.V=len(self.vocab)
        self.u=Counter();self.p=defaultdict(Counter);self.context={j:defaultdict(Counter) for j in (2,3,4)}
        self.ntotal={j:Counter() for j in (2,3,4)}
        for _,line in seqs:
            if direction=='backward':line=list(reversed(line))
            hist=['<s>']*3
            n=len(line)
            for i,token in enumerate(line):
                w=token if token in self.word_set else '<UNK>'
                self.u[w]+=1;self.p[posbin(i,n)][w]+=1
                for order in (2,3,4):
                    context=tuple(hist[-(order-1):])
                    self.context[order][context][w]+=1;self.ntotal[order][context]+=1
                hist.append(w)
        self.total=sum(self.u.values())
        self.unknown=lambda w:w if w in self.word_set else '<UNK>'

    def probability(self,target,history,i,n,mode):
        w=self.unknown(target)
        p=(self.u[w]+.1)/(self.total+.1*self.V)
        if mode=='unigram':return p
        pos=posbin(i,n)
        category=self.p[pos]
        p=(category[w]+self.pos_alpha*p)/(sum(category.values())+self.pos_alpha)
        if mode=='position':return p
        limit={'bigram':2,'trigram':3,'fourgram':4}[mode]
        hist=['<s>']*3+[self.unknown(t) for t in history]
        for order in range(2,limit+1):
            key=tuple(hist[-(order-1):])
            cnt=self.context[order][key]
            p=(cnt[w]+self.alpha*p)/(self.ntotal[order][key]+self.alpha)
        return p


MODES=('unigram','position','bigram','trigram','fourgram')


def eval_model(model,lines,modes=MODES):
    accum=defaultdict(lambda:[0,0.])
    per_page=defaultdict(lambda:defaultdict(lambda:[0,0.]))
    for g,original in lines:
        line=list(reversed(original)) if model.direction=='backward' else original
        n=len(line)
        hist=[]
        for i,word in enumerate(line):
            q='start' if i==0 else 'end' if i==n-1 else ('middle' if i>1 and i<n-2 else 'boundary_adjacent')
            known=word in model.word_set
            for mode in modes:
                p=model.probability(word,hist,i,n,mode)
                ll=-math.log(max(p,1e-300))
                for label in ('all',q,'known' if known else 'unknown'):
                    v=accum[(mode,label)];v[0]+=1;v[1]+=ll
                pg=per_page[g][mode];pg[0]+=1;pg[1]+=ll
            hist.append(word)
    return accum,per_page


def choose_alpha(seqs,direction, max_vocab=1200):
    groups=groups_in_order(seqs)
    if len(groups)<6:return 64,{}
    val=set(groups[::5])
    train=[(g,w) for g,w in seqs if g not in val]
    dev=[(g,w) for g,w in seqs if g in val]
    scores={}
    for alpha in (4,16,64,256,1024):
        m=ConditionalNgram(train,direction,alpha,max_vocab)
        result,_=eval_model(m,dev,('fourgram',))
        n,ll=result[('fourgram','all')]
        scores[str(alpha)]=round(ll/n,6)
    return min(scores,key=lambda x:scores[x]) and int(min(scores,key=scores.get)),scores


def one_corpus(lines, folds=4, max_vocab=1200, seed=20261028):
    fold_reports=[]
    folio_results=defaultdict(lambda:defaultdict(dict))
    for f,train,test in blocked_folds(lines,folds):
        if not test or not train:continue
        shuffled=shuffle_within_group_position(train,seed+f)
        direction_details={}
        for direction in ('forward','backward'):
            alpha,grid=choose_alpha(train,direction,max_vocab)
            real=ConditionalNgram(train,direction,alpha,max_vocab)
            null=ConditionalNgram(shuffled,direction,alpha,max_vocab)
            a,pages=eval_model(real,test)
            b,null_pages=eval_model(null,test)
            direction_details[direction]={'alpha':alpha,'validation_grid':grid,
                'metrics':{mode:{part:{'n':a[(mode,part)][0],
                    'real_nll':round(a[(mode,part)][1]/a[(mode,part)][0],6) if a[(mode,part)][0] else None,
                    'shuffle_nll':round(b[(mode,part)][1]/b[(mode,part)][0],6) if b[(mode,part)][0] else None}
                    for part in ('all','start','end','middle','known','unknown')}
                    for mode in MODES}}
            for page in pages:
                for mode in MODES:
                    n,l=pages[page][mode];n2,l2=null_pages[page][mode]
                    assert n==n2
                    folio_results[page][direction][mode]={
                        'n':n,'nll':l/n,'shuffle_nll':l2/n}
        fold_reports.append({'fold':f,'heldout_folios':len(set(g for g,_ in test)),
            'heldout_tokens':sum(len(w) for _,w in test),'directions':direction_details})

    rows=[]
    for folio,d in folio_results.items():
        if 'forward' not in d or 'backward' not in d:continue
        for mode in MODES:
            f=d['forward'][mode];b=d['backward'][mode]
            rows.append({'folio':folio,'mode':mode,'n':f['n'],
                'forward':f['nll'],'backward':b['nll'],
                'forward_gain':f['shuffle_nll']-f['nll'],
                'backward_gain':b['shuffle_nll']-b['nll'],
                'reverse_specific_gain':(b['shuffle_nll']-b['nll'])-(f['shuffle_nll']-f['nll'])})
    aggregated={}
    rng=np.random.default_rng(seed)
    for mode in MODES:
        z=[r for r in rows if r['mode']==mode]
        if not z:continue
        weights=np.asarray([r['n'] for r in z],float)
        keys=('forward','backward','forward_gain','backward_gain','reverse_specific_gain')
        vals={k:np.asarray([r[k] for r in z]) for k in keys}
        draws={k:[] for k in keys}
        for _ in range(1500):
            ids=rng.integers(0,len(z),size=len(z))
            for key in keys:
                draws[key].append(float(np.average(vals[key][ids],weights=weights[ids])))
        aggregated[mode]={key:{'mean':round(float(np.average(vals[key],weights=weights)),6),
            'ci95':[round(float(np.quantile(draws[key],q)),6) for q in (.025,.975)]}
            for key in keys}
        aggregated[mode]['folios']=len(z);aggregated[mode]['tokens']=int(weights.sum())
    return {'folds':fold_reports,'aggregate':aggregated,'per_folio':rows}


def run(zl,austen,darwin,out):
    raw=Path(zl).read_text(encoding='utf-8-sig')
    z={'Voynich_A':voynich_lines(raw,'A'),'Voynich_B':voynich_lines(raw,'B')}
    for name,path in (('austen',austen),('darwin',darwin)):
        text=Path(path).read_text(encoding='utf-8-sig')
        z[name]=prefix_budget(english_sentences(text,name),22843)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    results={}
    for name,lines in z.items():
        result=one_corpus(lines)
        results[name]={'aggregate':result['aggregate'],'folds':result['folds']}
        (out/f'{name}_per_folio.json').write_text(json.dumps(result['per_folio'],indent=2)+'\n')
        print(name, 'fourgram',json.dumps(result['aggregate'].get('fourgram')),flush=True)
    (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--zl',required=True);p.add_argument('--austen',required=True)
    p.add_argument('--darwin',required=True)
    p.add_argument('--out',default='results/higher_order_direction')
    a=p.parse_args();run(a.zl,a.austen,a.darwin,a.out)
