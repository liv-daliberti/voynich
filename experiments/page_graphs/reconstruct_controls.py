#!/usr/bin/env python3
"""Page-local 75%-deletion reconstruction and nearby-vs-distant graph controls.

No part of any deleted type is supplied to the score: the full-page
incidence matrix is a precomputation only, its masked rows have no effect.
Compare like-for-like pseudo-pages from Austen and Naibbe by using the exact
Voynich folio token budgets, *not* arbitrary default book pagination.
"""
from __future__ import annotations
import argparse, csv, json, math, re, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.robustness75.robustness import glyphs, one_edit_forms
from experiments.page_graphs.compare import parse_voynich,graph_stats,build_adjacency,shape_cos
from experiments.english_controls.benchmark import clean_gutenberg, character_edits


def matched_pages(voynich_pages,control_tokens,name):
    """Independent pseudo-pages using the observed folio token counts.
    The corpus is truncated to these budgets and is not recycled."""
    total=sum(len(p['words']) for p in voynich_pages)
    if len(control_tokens)<total:
        raise ValueError(f'{name} has only {len(control_tokens)} tokens, need {total}')
    out=[];pos=0
    for p in voynich_pages:
        n=len(p['words'])
        out.append({'folio':f'{name}_{p["folio"]}',
            'language':p['language'], 'section_code':name,
            'hand_code':'?', 'words':control_tokens[pos:pos+n]})
        pos+=n
    return out


def graph_adjacency(page,efa):
    vocab=sorted(set(page['words']))
    units=(lambda w:glyphs(w)) if efa else (lambda w:tuple(w))
    abc=sorted({unit for w in vocab for unit in units(w)})
    universe={}
    proposals=[]
    src=[]
    for j,w in enumerate(vocab):
        found=(one_edit_forms(units(w),abc) if efa else
               character_edits(w,tuple(abc)))
        for candidate in found:
            if candidate==w: continue
            i=universe.setdefault(candidate,len(universe))
            proposals.append(i);src.append(j)
    mat=coo_matrix((np.ones(len(src),dtype=np.int8),
                    (proposals,src)),shape=(len(universe),len(vocab)),
                   dtype=np.int8).tocsr()
    return vocab,universe,mat


def reconstruct_page(page,efa,trials=500,topk=50,seed=20261028):
    observed,lookup,incidence=graph_adjacency(page,efa)
    n=len(observed)
    if n<4:
        return {'folio':page['folio'],'types':n,'trials':0}
    retained=max(1,round(n*.25))
    rng=np.random.default_rng(seed)
    ids=np.asarray([lookup.get(w,-1) for w in observed],dtype=int)
    # True full-page edges are used only for EVALUATION, never prediction.
    reverse_row={int(ids[j]):j for j in range(n) if ids[j]>=0}
    rr,cc=incidence.nonzero()
    full_edges=np.asarray(sorted({
        tuple(sorted((reverse_row[int(r)],int(col))))
        for r,col in zip(rr,cc) if int(r) in reverse_row
        and reverse_row[int(r)]!=int(col)}),dtype=np.int32).reshape(-1,2)
    # The row-map array is used for evaluation labels or self-exclusion,
    # never for ranking a deleted word on whether it is actually attested.
    total=Counter()
    for trial in range(trials):
        survivors=rng.choice(n,size=retained,replace=False)
        visible=np.zeros(n,dtype=np.int8);visible[survivors]=1
        support=np.asarray(incidence @ visible).reshape(-1)
        surviving_ids=set(ids[survivors])
        eligible=np.flatnonzero(support>0)
        eligible=np.asarray([j for j in eligible if j not in surviving_ids],dtype=np.int32)
        removed=np.flatnonzero(visible==0)
        removed_forms={int(ids[j]) for j in removed if ids[j]>=0}
        possible=len(eligible)
        recoverable=sum(x in removed_forms for x in eligible)
        k=min(topk,possible)
        hits=0
        chosen=np.empty(0,dtype=np.int32)
        if k:
            tie=rng.random(possible)*.499
            rank=support[eligible].astype(float)+tie
            chosen=eligible[np.argpartition(rank,-k)[-k:]]
            hits=sum(int(x in removed_forms) for x in chosen)
        if len(full_edges):
            keep=np.zeros(n,dtype=bool)
            keep[survivors]=True
            before=np.mean(keep[full_edges[:,0]] & keep[full_edges[:,1]])
            restored_rows=set(chosen.tolist())
            for j in removed:
                if int(ids[j]) in restored_rows:keep[j]=True
            after=np.mean(keep[full_edges[:,0]] & keep[full_edges[:,1]])
            total['edge_recall_before']+=before
            total['edge_recall_after']+=after
            total['edge_trials']+=1
        total['eligible']+=possible
        total['removed']+=len(removed)
        total['recoverable']+=recoverable
        total['hits']+=hits
        total['selected']+=k
        total['random_expected_hits']+=k*recoverable/max(possible,1)
        total['trial_n']+=1
    def ratio(a,b):return round(total[a]/total[b],6) if total[b] else None
    return {'folio':page['folio'],'language':page['language'],
        'tokens':len(page['words']),'types':n,'trials':trials,'topk':topk,
        'removed_types_per_mask':ratio('removed','trial_n'),
        'mean_candidate_pool':ratio('eligible','trial_n'),
        'mean_recoverable_deleted':ratio('recoverable','trial_n'),
        'mean_hits_topk':ratio('hits','trial_n'),
        'mean_random_expected_hits_topk':ratio('random_expected_hits','trial_n'),
        'deleted_recall':ratio('hits','removed'),
        'deleted_reachable_fraction':ratio('recoverable','removed'),
        'original_edges':len(full_edges),
        'edge_recall_survivors_only':ratio('edge_recall_before','edge_trials'),
        'edge_recall_after_topk':ratio('edge_recall_after','edge_trials'),
        'edge_recall_gain':round(
            (total['edge_recall_after']-total['edge_recall_before'])/
            total['edge_trials'],6) if total['edge_trials'] else None,
        'precision_topk':ratio('hits','selected'),
        'random_precision_topk':ratio('random_expected_hits','selected'),
        'precision_lift_over_random':round(total['hits']/total['random_expected_hits'],5)
        if total['random_expected_hits'] else None}


def pair_metrics(pages,efa):
    """Pair all pages once, compare both form identity and unlabeled topology."""
    # Build the GLOBAL (within-corpus) edit adjacency just once.
    # Word edges on one page depend only on its own vocabulary, not
    # whether another page happens to contain that type.
    if efa:
        adjacency=build_adjacency(pages)
    else:
        alltypes={w for p in pages for w in p['words']}
        abc=tuple(sorted(set(''.join(alltypes))))
        adjacency={w:character_edits(w,abc)&alltypes for w in alltypes}
    g=[graph_stats(p,adjacency) for p in pages]
    nodes=[s for _,s,_ in g];edges=[s for _,_,s in g]
    attrs=[x for x,_,_ in g]
    n=len(pages)
    word=np.zeros((n,n),dtype=float);edge=np.zeros((n,n),dtype=float)
    topology=np.zeros((n,n),dtype=float)
    for i in range(n):
        word[i,i]=edge[i,i]=topology[i,i]=1.
        for j in range(i+1,n):
            a=nodes[i];b=nodes[j]
            ni=len(a&b);den=len(a)+len(b)-ni
            e=edges[i];f=edges[j];en=len(e&f);ed=len(e)+len(f)-en
            word[i,j]=word[j,i]=ni/den if den else 0.
            edge[i,j]=edge[j,i]=en/ed if ed else 0.
            topology[i,j]=topology[j,i]=shape_cos(
                attrs[i]['degree_histogram'],attrs[j]['degree_histogram'])
    return attrs,{'word_jaccard':word,'edge_jaccard':edge,
                  'topology_cosine':topology}


def summarize_pages(pages,efa,shuffle_trials=500,seed=20261028):
    attrs,matrices=pair_metrics(pages,efa)
    n=len(pages)
    subset={}
    rng=np.random.default_rng(seed)
    labels=[x['language'] for x in pages]
    for lang in sorted(set(labels)):
        ids=np.array([i for i,x in enumerate(labels) if x==lang],dtype=int)
        if len(ids)<25:continue
        ids=np.sort(ids)
        sub={}
        # Near pages are 1,2,3 pages apart in this variety's folio order.
        x,y=np.triu_indices(len(ids),1)
        close=(y-x<=3);far=(y-x>=20)
        for metric,matrix in matrices.items():
            data=matrix[np.ix_(ids,ids)]
            near=float(data[x[close],y[close]].mean())
            distant=float(data[x[far],y[far]].mean())
            baseline=data[x[~(close|far)],y[~(close|far)]].mean()
            actual=near-distant
            null=[]
            for _ in range(shuffle_trials):
                perm=rng.permutation(len(ids))
                rearr=data[np.ix_(perm,perm)]
                null.append(float(rearr[x[close],y[close]].mean()-
                                  rearr[x[far],y[far]].mean()))
            v=np.asarray(null)
            sub[metric]={'all_pairs_mean':round(float(data[x,y].mean()),6),
                'near_1_to_3':round(near,6),'far_20_plus':round(distant,6),
                'near_minus_far':round(actual,6),
                'page_order_null_mean':round(float(v.mean()),6),
                'page_order_null_ci95':[round(float(q),6)
                                        for q in np.quantile(v,[.025,.975])],
                'p_one_sided_near_advantage':round(
                    float((1+np.sum(v>=actual))/(len(v)+1)),6),
                'trials':shuffle_trials}
        subset[lang]={'pages':len(ids),'median_types':round(float(np.median(
            [attrs[i]['types'] for i in ids])),2),
            'median_edges':round(float(np.median([attrs[i]['edges'] for i in ids])),2),
            'median_isolated_fraction':round(float(np.median(
            [attrs[i]['isolated_fraction'] for i in ids])),5),
            'metrics':sub}
    return subset,attrs


def aggregate_recovery(results):
    grouped=defaultdict(list)
    for x in results:
        if x.get('trials',0):
            grouped[x['language']].append(x)
    out={}
    for l,rows in grouped.items():
        w=np.asarray([r['trials'] for r in rows],float)
        def avg(field):
            values=np.asarray([r[field] for r in rows],float)
            return round(float(np.average(values,weights=w)),6)
        out[l]={'pages':len(rows),
                'mean_deleted_recall':avg('deleted_recall'),
                'mean_deleted_reachable_fraction':avg('deleted_reachable_fraction'),
                'mean_precision_topk':avg('precision_topk'),
                'mean_random_precision_topk':avg('random_precision_topk'),
                'mean_hits_topk':avg('mean_hits_topk'),
                'edge_recall_survivors_only':round(float(np.nanmean([
                    r['edge_recall_survivors_only'] if r['edge_recall_survivors_only'] is not None else np.nan
                    for r in rows])),6),
                'edge_recall_after_topk':round(float(np.nanmean([
                    r['edge_recall_after_topk'] if r['edge_recall_after_topk'] is not None else np.nan
                    for r in rows])),6),
                'mean_random_expected_hits_topk':avg('mean_random_expected_hits_topk'),
                'mean_candidate_pool':avg('mean_candidate_pool')}
        out[l]['lift_vs_random_topk']=round(
            out[l]['mean_hits_topk']/max(1e-12,out[l]['mean_random_expected_hits_topk']),4)
    return out


def evaluate(name,pages,efa,output,trials=500,seed=20261028,shuffle_trials=500):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    near,graph= summarize_pages(pages,efa,shuffle_trials,seed)
    reconstruct=[]
    for i,p in enumerate(pages):
        rec=reconstruct_page(p,efa,trials,50,seed+i)
        reconstruct.append(rec)
        if (i+1)%50==0:print(name,'reconstructed',i+1,'of',len(pages),flush=True)
    summary={'name':name,'pages':len(pages),'page_graphs':near,
             'reconstruction':aggregate_recovery(reconstruct),
             'comparison_design':{
                'retained_word_type_fraction':.25,
                'independent_masks_per_page':trials,'nomination_cap':50,
                'random_baseline':'uniform over candidate forms with at least one surviving edit witness',
                'proximity_null':'whole-page-order permutation within A/B matched cohort',
                'controls':'same page token budget, same EVA glyph segmentation for Voynich/Naibbe; ASCII letters for Austen'}}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (output/'pages.csv').open('w',newline='') as f:
        fields=['folio','language','tokens','types','edges','mean_degree',
                'isolated_fraction','component_count',
                'largest_component_fraction','transitivity']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in graph:writer.writerow({k:r[k] for k in fields})
    with (output/'reconstruction.csv').open('w',newline='') as f:
        fields=list(reconstruct[0])
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows(reconstruct)
    print(name,json.dumps({'page_graphs':near,'recovery':summary['reconstruction']}),
          flush=True)
    return summary


def run(zl,austen,naibbe,output,trials=500,shuffle_trials=500,only=None):
    real=parse_voynich(Path(zl).read_text(encoding='utf-8-sig'))
    template={l:[p for p in real if p['language']==l] for l in ('A','B')}
    austen_words=clean_gutenberg(Path(austen).read_text(encoding='utf-8-sig'),'austen')
    naibbe_words=re.findall(r'[a-z]+',Path(naibbe).read_text(encoding='utf-8-sig').lower())
    total=sum(len(p['words']) for p in real)
    cohorts={
        'Voynich':(real,True),
        'Austen':(matched_pages(real,austen_words,'austen'),False),
        'Naibbe':(matched_pages(real,naibbe_words,'naibbe'),True)
    }
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    summaries={}
    for name,(pages,eva) in cohorts.items():
        if only and name!=only: continue
        summaries[name]=evaluate(name,pages,eva,out/name,
                                 trials,seed=20261028,shuffle_trials=shuffle_trials)
    (out/'summary.json').write_text(json.dumps(
        {'source_token_budget':total,'summaries':summaries},indent=2)+'\n')
    return summaries


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--zl',required=True)
    ap.add_argument('--austen',required=True)
    ap.add_argument('--naibbe',required=True)
    ap.add_argument('--out',default='results/page_reconstruction_controls')
    ap.add_argument('--masks',type=int,default=500)
    ap.add_argument('--page-shuffles',type=int,default=500)
    ap.add_argument('--cohort',choices=['Voynich','Austen','Naibbe'])
    a=ap.parse_args()
    run(a.zl,a.austen,a.naibbe,a.out,a.masks,a.page_shuffles,a.cohort)
