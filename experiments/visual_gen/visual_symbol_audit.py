#!/usr/bin/env python3
"""Evaluation-only visual pseudo-symbol distributions, NOT EVA OCR.

Fit codewords to connected ink components observed in real TRAIN line
images, then compare heldout real/GENERATED/IID fragments in that fixed
visual vocabulary. Neither cluster IDs nor component groupings are
Voynich graphemes or verified words.
"""
from __future__ import annotations
import argparse,json,math
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from sklearn.cluster import MiniBatchKMeans


def components(line):
    """Connected marks at 64x256. These are *not* necessarily full glyphs."""
    gray=np.asarray(line,dtype=np.uint8)
    # Binary-like model inputs and recon outputs use this common threshold.
    mask=np.uint8(gray<165)
    n,labels,stats,c=cv2.connectedComponentsWithStats(mask,8)
    items=[]
    for k in range(1,n):
        x,y,w,h,area=map(int,stats[k])
        if 1<=w<=28 and 2<=h<=31 and 4<=area<=380:
            sub=np.uint8(labels[y:y+h,x:x+w]==k)*255
            # Preserve glyph geometry and standardize for comparable clusters.
            side=np.zeros((20,20),dtype=np.uint8)
            factor=min(18/w,18/h)
            sw=max(1,int(round(w*factor)));sh=max(1,int(round(h*factor)))
            small=cv2.resize(sub,(sw,sh),interpolation=cv2.INTER_NEAREST)
            side[(20-sh)//2:(20-sh)//2+sh,(20-sw)//2:(20-sw)//2+sw]=small
            desc=np.concatenate([
                side.ravel().astype("float32")/255,
                np.asarray([w/28,h/31,area/380],dtype=np.float32)])
            items.append((x,y,w,h,desc))
    return sorted(items,key=lambda v:v[0])


def sequences(lines,model):
    types=Counter();trans=Counter();word_lengths=[];total=0;line_counts=[]
    for line in lines:
        comps=components(line)
        line_counts.append(len(comps))
        if not comps:continue
        data=np.stack([x[-1] for x in comps])
        ids=model.predict(data)
        total+=len(ids)
        types.update(map(int,ids))
        # Approximate word gaps from horizontal distance between components.
        runs=[];current=[int(ids[0])]
        last=comps[0][0]+comps[0][2]
        for idx,(x,y,w,h,_) in zip(ids[1:],comps[1:]):
            gap=x-last
            if gap>max(7,1.5*np.median([c[2] for c in comps])):
                runs.append(current);current=[]
            current.append(int(idx))
            last=x+w
        if current:runs.append(current)
        for seq in runs:
            word_lengths.append(len(seq))
            trans.update(zip(seq,seq[1:]))
    return {
        "n_lines":len(lines),"n_components":total,
        "mean_components_per_line":round(float(np.mean(line_counts)),4),
        "lines_with_components":int(sum(x>0 for x in line_counts)),
        "n_pseudo_word_groups":len(word_lengths),
        "mean_pseudo_word_components":round(float(np.mean(word_lengths)),4)
                                      if word_lengths else None,
        "type_counts":types,"adjacent_component_pairs":trans,
    }


def js_count(a,b,n):
    aa=np.asarray([a.get(i,0) for i in range(n)],float)+.5
    bb=np.asarray([b.get(i,0) for i in range(n)],float)+.5
    aa/=aa.sum();bb/=bb.sum();mm=(aa+bb)/2
    return round(float(.5*np.sum(aa*np.log(aa/mm))+
                       .5*np.sum(bb*np.log(bb/mm))),6)


def montage_lines(file):
    data=np.asarray(Image.open(file).convert("L"))
    if data.shape[1]!=256 or data.shape[0]%64:
        raise ValueError("Expected stacked 64x256 line montage")
    return data.reshape(-1,64,256)


def run(dataset,generated,iid,out,k=24,seed=20261028):
    data=np.load(dataset)
    train=data["train"];real=data["test"]
    synth=montage_lines(generated)
    baseline=montage_lines(iid)
    strokes=[x for line in train for x in components(line)]
    if len(strokes)<k:
        raise ValueError("Insufficient real training components for shape codebook")
    X=np.stack([x[-1] for x in strokes])
    model=MiniBatchKMeans(n_clusters=k,random_state=seed,
        batch_size=min(1024,len(X)),n_init=3,max_iter=100)
    model.fit(X)
    summary={}
    for name,lines in (("heldout_real",real),("generated",synth),
                       ("independent_code_null",baseline)):
        s=sequences(lines,model)
        summary[name]={x:y for x,y in s.items()
                       if x not in ("type_counts","adjacent_component_pairs")}
        summary[name]["_types"]=s["type_counts"]
        summary[name]["_pairs"]=s["adjacent_component_pairs"]
    ref=summary["heldout_real"]
    result={"visual_symbol_prototypes":k,
            "training_component_examples":len(X),
            "input_line_count":len(train),
            "metrics":{},
            "semantic_status":"NO EVA: visual components and word-gaps only"}
    for name,s in summary.items():
        result["metrics"][name]={k:v for k,v in s.items()
                                   if not k.startswith("_")}
        result["metrics"][name]["visual_component_unigram_js_heldout"]=(
            js_count(ref["_types"],s["_types"],k) if name!="heldout_real"
            else 0.)
        rpair=Counter({i*k+j:v for (i,j),v in ref["_pairs"].items()})
        spair=Counter({i*k+j:v for (i,j),v in s["_pairs"].items()})
        result["metrics"][name]["visual_component_bigram_js_heldout"]=(
            js_count(rpair,spair,k*k) if name!="heldout_real" else 0.)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    print("VISUAL_SYMBOL_AUDIT",json.dumps(result),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True)
    p.add_argument("--generated",required=True)
    p.add_argument("--iid",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    run(a.dataset,a.generated,a.iid,a.out)
