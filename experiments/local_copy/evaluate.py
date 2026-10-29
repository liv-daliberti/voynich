#!/usr/bin/env python3
"""Direct source-availability and near-copy tests on physical manuscript lines.

Compare candidate sources BEFORE each token vs AFTER (placebo), horizontal
and vertical, against within-page and line-position-preserving permutations.
Morphological similarity is one *canonical* EVA-glyph insertion, deletion
or substitution; exact copies are separately measured.
"""
from __future__ import annotations
import argparse, json, math, random, sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.context.slot_filling import (voynich_lines, english_sentences,
                                                prefix_budget, shuffle_within_group_position)
from experiments.latent_grammar.fixed_target import copy_edit_lines
from experiments.robustness75.robustness import glyphs,one_edit_forms
from experiments.english_controls.benchmark import character_edits

WINDOWS=('same_line_before5','same_line_after5',
         'previous_line_same_column','next_line_same_column',
         'previous_two_lines_same_column','next_two_lines_same_column',
         'previous_five_lines_any','next_five_lines_any')
SUBSETS=('all','first_page_occurrence')


def neighborhood_map(sequences,eva):
    vocab={w for _,line in sequences for w in line}
    units=(lambda w:glyphs(w)) if eva else (lambda w:tuple(w))
    alphabet=sorted({c for w in vocab for c in units(w)})
    mapping={}
    for w in vocab:
        edits=(one_edit_forms(units(w),alphabet) if eva else
               character_edits(w,tuple(alphabet)))
        mapping[w]=edits & vocab
    return mapping


def page_groups(sequences):
    grouped=defaultdict(list)
    for folio,line in sequences:
        grouped[folio].append(line)
    return grouped


def sources(lines,row,column):
    n=len(lines);line=lines[row]
    ans={}
    ans['same_line_before5']=line[max(0,column-5):column]
    ans['same_line_after5']=line[column+1:column+6]
    for label,delta in (('previous_line_same_column',-1),
                        ('next_line_same_column',1)):
        y=row+delta
        ans[label]=lines[y][max(0,column-2):column+3] if 0<=y<n else []
    for label,delta in (('previous_two_lines_same_column',-1),
                        ('next_two_lines_same_column',1)):
        out=[]
        for step in (1,2):
            y=row+delta*step
            if 0<=y<n:
                out.extend(lines[y][max(0,column-2):column+3])
        ans[label]=out
    for label,delta in (('previous_five_lines_any',-1),
                        ('next_five_lines_any',1)):
        out=[]
        for step in (1,2,3,4,5):
            y=row+delta*step
            if 0<=y<n:out.extend(lines[y])
        ans[label]=out
    return ans


def measure_page(lines,neighbors):
    aggregate=defaultdict(lambda:np.zeros(4,dtype=float))
    previously_seen=set()
    for row,line in enumerate(lines):
        for col,word in enumerate(line):
            newly_seen=word not in previously_seen
            previously_seen.add(word)
            neighborhoods=neighbors[word]
            for name,source in sources(lines,row,col).items():
                if not source:
                    continue
                equal=int(word in source)
                edit=int(not neighborhoods.isdisjoint(source))
                for subgroup in ('all', 'first_page_occurrence') if newly_seen else ('all',):
                    v=aggregate[(subgroup,name)]
                    v[0]+=1
                    v[1]+=equal
                    v[2]+=edit
                    v[3]+=int(bool(equal or edit))
    return dict(aggregate)


def total_rates(page_metrics):
    total=defaultdict(lambda:np.zeros(4,dtype=float))
    for metrics in page_metrics.values():
        for k,v in metrics.items():total[k]+=v
    return total


def shuffled_folio(lines,seed):
    """Frequency, word identity, page composition, line boundaries and
    start/middle/end token counts are invariant. Exact token ordering is not."""
    flat=[('samefolio',list(line)) for line in lines]
    return [w for _,w in shuffle_within_group_position(flat,seed)]



def shuffled_line_order(lines,seed):
    """Strong null for *vertical* proximity: retain all words and their
    within-line positions/order, shuffle only physical line sequence."""
    out=[list(line) for line in lines]
    random.Random(seed).shuffle(out)
    return out

def run_corpus(sequences,eva,permutations=150,seed=20261028,null_mode='token'):
    pages=page_groups(sequences)
    neighbors=neighborhood_map(sequences,eva)
    actual={folio:measure_page(lines,neighbors) for folio,lines in pages.items()}
    observed=total_rates(actual)
    nulldraws=defaultdict(list)
    page_null=defaultdict(lambda:defaultdict(lambda:np.zeros(4,dtype=float)))
    for trial in range(permutations):
        simulated={}
        for index,(folio,lines) in enumerate(pages.items()):
            new=(shuffled_line_order(lines,seed+trial*137+index)
                 if null_mode=='line' else
                 shuffled_folio(lines,seed+trial*137+index))
            simulated[folio]=measure_page(new,neighbors)
            for key,v in simulated[folio].items():
                page_null[folio][key]+=v / permutations
        totals=total_rates(simulated)
        for subgroup in SUBSETS:
            for win in WINDOWS:
                k=(subgroup,win)
                data=totals[k]
                for metric,id in (('exact',1),('one_edit',2),('either',3)):
                    nulldraws[(subgroup,win,metric)].append(
                        float(data[id]/data[0]) if data[0] else float('nan'))

    summary={}
    for subgroup in SUBSETS:
        summary[subgroup]={}
        for win in WINDOWS:
            key=(subgroup,win)
            data=observed[key]
            measures={}
            for metric,id in (('exact',1),('one_edit',2),('either',3)):
                observed_rate=float(data[id]/data[0]) if data[0] else None
                null=np.asarray(nulldraws[subgroup,win,metric],dtype=float)
                null=null[np.isfinite(null)]
                if observed_rate is None or not len(null):continue
                measures[metric]={
                    'n_eligible_targets':int(data[0]),
                    'observed_rate':round(observed_rate,6),
                    'shuffled_mean':round(float(null.mean()),6),
                    'excess_pp':round(100*(observed_rate-null.mean()),4),
                    'one_sided_permutation_p':round(float(
                        (1+np.sum(null>=observed_rate))/(1+len(null))),5),
                    'shuffled_95_range':[round(float(np.quantile(null,q)),6)
                                         for q in (.025,.975)]}
            summary[subgroup][win]=measures
    contrasts={}
    for subgroup in SUBSETS:
        for kind in ('one_edit','exact'):
            for left,right,name in (
                ('previous_line_same_column','next_line_same_column','vertical_one_line'),
                ('previous_two_lines_same_column','next_two_lines_same_column','vertical_two_lines'),
                ('same_line_before5','same_line_after5','horizontal_five_words'),
                ('previous_five_lines_any','next_five_lines_any','five_lines_any')):
                l=summary[subgroup][left][kind]
                r=summary[subgroup][right][kind]
                actual_difference=l['observed_rate']-r['observed_rate']
                null_left=np.asarray(nulldraws[subgroup,left,kind])
                null_right=np.asarray(nulldraws[subgroup,right,kind])
                contrasts[f'{subgroup}:{name}:{kind}']={
                    'directional_difference_pp':round(actual_difference*100,4),
                    'shuffle_expected_difference_pp':round(
                        100*float(np.mean(null_left-null_right)),4),
                    'excess_directional_pp':round(
                        100*(actual_difference-float(np.mean(null_left-null_right))),4),
                    'one_sided_permutation_p':round(float(
                        (1+np.sum((null_left-null_right)>=actual_difference))/
                        (1+permutations)),5)}
    return {'pages':len(pages),'tokens':sum(len(line) for lines in pages.values() for line in lines),
            'type_count':len(neighbors),'trials':permutations,'null_mode':null_mode,
            'metrics':summary,'directional_contrasts':contrasts}


def run(zl,austen,darwin,generated,out,permutations=150,null_mode='token'):
    z=Path(zl).read_text(encoding='utf-8-sig')
    corpora={'Voynich_A':(voynich_lines(z,'A'),True),
             'Voynich_B':(voynich_lines(z,'B'),True)}
    for name,path in (('austen',austen),('darwin',darwin)):
        s=english_sentences(Path(path).read_text(encoding='utf-8-sig'),name)
        corpora[name]=(prefix_budget(s,22843),False)
    if generated:
        seq=copy_edit_lines(Path(generated).read_text(encoding='utf-8-sig'))
        corpora['TimmSchinner_sample']=(seq,True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    report={}
    for name,(seq,eva) in corpora.items():
        result=run_corpus(seq,eva,permutations,null_mode=null_mode)
        report[name]=result
        print(name,json.dumps({
            'pages':result['pages'],'tokens':result['tokens'],
            'vertical_prior_edit':result['metrics']['all']['previous_line_same_column']['one_edit'],
            'vertical_future_edit':result['metrics']['all']['next_line_same_column']['one_edit'],
            'vertical_directional':result['directional_contrasts']['all:vertical_one_line:one_edit'],
            'horizontal_directional':result['directional_contrasts']['all:horizontal_five_words:one_edit']}),
            flush=True)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--zl',required=True);p.add_argument('--austen',required=True)
    p.add_argument('--darwin',required=True)
    p.add_argument('--generated')
    p.add_argument('--permutations',type=int,default=150)
    p.add_argument('--null-mode',choices=['token','line'],default='token')
    p.add_argument('--out',default='results/local_copy')
    a=p.parse_args()
    run(a.zl,a.austen,a.darwin,a.generated,a.out,a.permutations,a.null_mode)
