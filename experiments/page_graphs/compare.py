#!/usr/bin/env python3
"""One independent one-EVA-glyph edit graph per manuscript folio.

Compare word identity, shared edges, unlabeled graph topology, and local
page-order similarity. A graph is not a translation or direct copying proof.
Page-level lexical null randomly reallocates token positions among folios
*within Currier A/B* while retaining folio token counts and global token
frequencies. Page-order null permutes whole pages within variety.
"""
from __future__ import annotations
import argparse, csv, json, math, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.robustness75.robustness import parse, glyphs, one_edit_forms


def parse_voynich(text):
    base=parse(text)
    metadata={}
    for line in text.splitlines():
        m=re.match(r'^<([^.,>]+)>\s*<!\s*(.*?)>',line)
        if not m: continue
        fields={key:val for key,val in re.findall(
            r'\$([A-Z])=([^\s>]+)',m.group(2))}
        metadata[m.group(1)]=fields
    for p in base:
        fields=metadata.get(p['folio'],{})
        p['section_code']=fields.get('F','?')
        p['hand_code']=fields.get('H','?')
        p['quire_code']=fields.get('Q','?')
    return base


def parse_generated(text,lines_per_page=29):
    lines=[]
    for line in text.splitlines():
        line=line.strip()
        if not line or line.startswith('#'): continue
        words=re.findall(r'[a-z]+',line.lower())
        if words: lines.append(words)
    pages=[]
    for j in range(0,len(lines),lines_per_page):
        words=[w for line in lines[j:j+lines_per_page] for w in line]
        pages.append({'folio':f'generated_{j//lines_per_page:03d}',
                      'words':words,'language':'generated',
                      'section_code':'generated','hand_code':'?',
                      'quire_code':'?'})
    return pages


def build_adjacency(pages):
    types={w for p in pages for w in p['words']}
    alpha=sorted({g for word in types for g in glyphs(word)})
    return {w:set(one_edit_forms(glyphs(w),alpha))&types for w in types}


def degree_histogram(degrees):
    counts=np.zeros(8)
    for d in degrees:counts[min(d,7)]+=1
    return (counts/max(1,len(degrees))).tolist()


def graph_stats(page,adjacency):
    nodes=set(page['words'])
    neighborhood={w:adjacency[w]&nodes for w in nodes}
    degrees=[len(v) for v in neighborhood.values()]
    edges={ (w,v) if w<v else (v,w)
            for w,nn in neighborhood.items() for v in nn if w!=v}
    remaining=set(nodes)
    components=[]
    while remaining:
        start=remaining.pop()
        stack=[start]
        size=1
        while stack:
            w=stack.pop()
            linked=neighborhood[w]&remaining
            remaining.difference_update(linked)
            stack.extend(linked)
            size+=len(linked)
        components.append(size)
    triangles=sum(len(neighborhood[w]&neighborhood[v])
                  for w,v in edges)/3.
    wedges=sum(d*(d-1)/2 for d in degrees)
    clustering=(3*triangles/wedges) if wedges else 0.
    n=len(nodes)
    out={
        'folio':page['folio'],
        'language':page['language'],
        'section_code':page.get('section_code','?'),
        'hand_code':page.get('hand_code','?'),
        'tokens':len(page['words']),
        'types':n,
        'edges':len(edges),
        'mean_degree':2*len(edges)/max(n,1),
        'isolated_fraction':sum(d==0 for d in degrees)/max(n,1),
        'component_count':len(components),
        'largest_component_fraction':max(components)/n if n else 0,
        'transitivity':clustering,
        'degree_histogram':degree_histogram(degrees),
    }
    return out,nodes,edges


def jac(a,b):
    if not a and not b:return 1.0
    return len(a&b)/len(a|b)


def shape_cos(a,b):
    aa=np.asarray(a);bb=np.asarray(b)
    den=float(np.linalg.norm(aa)*np.linalg.norm(bb))
    return float(aa@bb/den) if den else 0.


def pairwise(pages,graphs):
    results=[]
    bylang=defaultdict(list)
    for i,p in enumerate(pages):
        bylang[p['language']].append(i)
    indexin={idx:i for ids in bylang.values() for i,idx in enumerate(ids)}
    for i in range(len(pages)):
        ai,na,ea=graphs[i]
        for j in range(i+1,len(pages)):
            aj,nb,eb=graphs[j]
            kind=ai['language'] if ai['language']==aj['language'] else '-'.join(
                sorted((ai['language'],aj['language'])))
            dist=abs(indexin[i]-indexin[j]) if ai['language']==aj['language'] else -1
            results.append({'a':ai['folio'],'b':aj['folio'],
                'pair_kind':kind,
                'folio_index_gap':dist,
                'same_section_code':int(ai['section_code']==aj['section_code']),
                'types_a':ai['types'],'types_b':aj['types'],
                'word_jaccard':jac(na,nb),
                'edge_jaccard':jac(ea,eb),
                'topology_degree_cosine':shape_cos(
                    ai['degree_histogram'],aj['degree_histogram']),
                'mean_degree_difference':abs(ai['mean_degree']-aj['mean_degree']),
                'giant_fraction_difference':abs(
                    ai['largest_component_fraction']-aj['largest_component_fraction'])})
    return results


def summary_pairs(rows):
    out={}
    for kind in sorted(set(row['pair_kind'] for row in rows)):
        sub=[r for r in rows if r['pair_kind']==kind]
        close=[r for r in sub if 0<r['folio_index_gap']<=3]
        far=[r for r in sub if r['folio_index_gap']>=20]
        def avg(items,k):
            return round(sum(x[k] for x in items)/len(items),6) if items else None
        out[kind]={
            'pairs':len(sub),
            'mean_word_jaccard':avg(sub,'word_jaccard'),
            'mean_edge_jaccard':avg(sub,'edge_jaccard'),
            'mean_unlabeled_degree_histogram_cosine':avg(sub,'topology_degree_cosine'),
            'mean_word_jaccard_near3':avg(close,'word_jaccard'),
            'mean_word_jaccard_far20':avg(far,'word_jaccard'),
            'near_pairs':len(close),'far_pairs':len(far)}
        if close and far:
            out[kind]['near_minus_far_word_jaccard']=round(
                out[kind]['mean_word_jaccard_near3']-
                out[kind]['mean_word_jaccard_far20'],6)
    return out


def permute_tokens_within_variety(pages,seed):
    randomizer=random.Random(seed)
    pool=defaultdict(list)
    for p in pages:pool[p['language']].extend(p['words'])
    for lang in pool:randomizer.shuffle(pool[lang])
    cursor=defaultdict(int)
    shuffled=[]
    for p in pages:
        lang=p['language'];n=len(p['words']);start=cursor[lang]
        q=dict(p);q['words']=pool[lang][start:start+n]
        cursor[lang]+=n
        shuffled.append(q)
    return shuffled


def null_distribution(pages,original,adjacency,trials=100,seed=20261028):
    # The shared universe and its edit graph are held constant under nulls.
    observed=summary_pairs(original)
    draws=defaultdict(list)
    for t in range(trials):
        surrogate=permute_tokens_within_variety(pages,seed+t)
        graphs=[graph_stats(p,adjacency) for p in surrogate]
        pairs=pairwise(surrogate,graphs)
        summary=summary_pairs(pairs)
        for lang in ('A','B','generated'):
            if lang not in summary or lang not in observed:continue
            for k in ('mean_word_jaccard','mean_edge_jaccard',
                      'mean_unlabeled_degree_histogram_cosine',
                      'near_minus_far_word_jaccard'):
                if k in summary[lang] and summary[lang][k] is not None:
                    draws[(lang,k)].append(summary[lang][k])
        if (t+1)%20==0:print('null',t+1,'/',trials,flush=True)
    results={}
    for (lang,k),scores in draws.items():
        y=np.asarray(scores)
        x=observed[lang][k]
        results.setdefault(lang,{})[k]={
            'observed':x,
            'null_mean':round(float(y.mean()),6),
            'excess':round(float(x-y.mean()),6),
            'null_95_interval':[round(float(np.quantile(y,z)),6)
                                for z in (.025,.975)],
            'one_sided_excess_p':round(float((1+sum(y>=x))/(len(y)+1)),6),
            'trials':trials}
    return results


def compare_real_and_generated(real,generated):
    # Cross-corpus shape similarities are descriptive, not size-controlled.
    output={}
    for label in ('A','B'):
        a=real[label];b=generated
        keys=('mean_degree','isolated_fraction',
              'largest_component_fraction','transitivity')
        output[label]={k:{
           'real_median':round(float(np.median([x[k] for x in a])),5),
           'generated_median':round(float(np.median([x[k] for x in b])),5)}
           for k in keys}
    return output


def run(zl_path,generated_path,out,trials=100):
    real=parse_voynich(Path(zl_path).read_text(encoding='utf-8-sig'))
    pages=list(real)
    generated=[]
    if generated_path:
        generated=parse_generated(Path(generated_path).read_text(
            encoding='utf-8-sig'))
        pages+=generated
    adjacency=build_adjacency(pages)
    graphs=[graph_stats(p,adjacency) for p in pages]
    page_rows=[g[0] for g in graphs]
    pairs=pairwise(pages,graphs)
    summary=summary_pairs(pairs)
    nulls=null_distribution(pages,pairs,adjacency,trials)
    grouped=defaultdict(list)
    for row in page_rows:grouped[row['language']].append(row)
    result={
      'page_counts':{k:len(v) for k,v in grouped.items()},
      'pairwise':summary,'page_token_type_medians':{k:{
          'tokens':float(np.median([x['tokens'] for x in v])),
          'types':float(np.median([x['types'] for x in v])),
          'edges':float(np.median([x['edges'] for x in v])),
          'isolated_fraction':float(np.median(
              [x['isolated_fraction'] for x in v]))}
          for k,v in grouped.items()},
      'token_reallocation_null':nulls,
      'same_original_folio_order_not_verified_chronology':True,
    }
    if generated:
        result['real_generated_shape_comparison']=compare_real_and_generated(
            {'A':grouped['A'],'B':grouped['B']},grouped['generated'])
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    with (out/'pages.csv').open('w',newline='') as f:
        fields=[k for k in page_rows[0] if k!='degree_histogram']
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for row in page_rows:
            writer.writerow({k:row[k] for k in fields})
    with (out/'pairs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(pairs[0]))
        writer.writeheader();writer.writerows(pairs)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--zl',required=True)
    ap.add_argument('--generated')
    ap.add_argument('--out',default='results/page_graphs')
    ap.add_argument('--null-trials',type=int,default=100)
    args=ap.parse_args()
    run(args.zl,args.generated,args.out,args.null_trials)
