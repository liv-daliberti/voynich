#!/usr/bin/env python3
"""Build page/quire-heldout SCRIPT LINE image datasets, not whole-page patches.

Line detection is conservative and unsupervised: remove parchment background
with local contrast, keep small ink components, identify horizontal rows,
and reject unusually short/large/drawing-like blobs. The recognized crop
is an unverified line image, not an EVA text label.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def folio_from_path(path):
    match=re.search(r"f(\d+)([rv])",Path(path).stem.lower())
    return ("f"+match.group(1)+match.group(2)) if match else Path(path).stem


def read_quire_lookup(text):
    out={}
    folio=None
    for row in text.splitlines():
        h=re.match(r'^<([^.,>]+)>\s*<!\s*(.*?)>',row)
        if h:
            q=re.search(r'\$Q=([^\s>]+)',h.group(2))
            out[h.group(1)]=q.group(1) if q else "unknown"
    return out


def resize_image(path,width=1024):
    im=Image.open(path).convert("L")
    h=round(im.height*width/im.width)
    im=im.resize((width,h),Image.Resampling.LANCZOS)
    return np.asarray(im,dtype=np.uint8)


def segment_lines(image,min_width=180,max_candidates=90):
    """Detect a handwriting line by peaks in SMALL COMPONENT centroids.

    Previous row-run merging accepted illustrations/edges and discarded
    most script because touching adjacent lines were merged. Here we
    identify thin ink components and explicitly enforce line-spacing
    before cropping. This still requires a manually checked sample.
    """
    from scipy.signal import find_peaks
    h,w=image.shape
    left,right=int(.085*w),int(.93*w)
    background=cv2.GaussianBlur(image,(0,0),sigmaX=19,sigmaY=19)
    contrast=(background.astype("float32")-image.astype("float32"))
    mask=np.uint8((contrast>10)&(image<205))*255
    n,labels,stats,cent=cv2.connectedComponentsWithStats(mask,8)
    comps=[]
    for i in range(1,n):
        x,y,cw,ch,area=(int(v) for v in stats[i])
        if (left<=x<right and 2<=cw<=30 and 5<=ch<=31
            and 10<=area<=210 and area/(cw*ch)>=.12):
            comps.append((x,y,cw,ch,area))
    density=np.zeros(h,dtype=np.float32)
    for x,y,cw,ch,area in comps:
        center=int(y+ch/2)
        if x<int(.70*w):
            density[max(0,center-4):min(h,center+5)]+=1
    smooth=cv2.GaussianBlur(density.reshape(-1,1),(1,9),0).ravel()
    peaks,_=find_peaks(smooth,distance=max(22,int(h/55)),
                       prominence=2,height=4)
    result=[]
    for center in peaks:
        nearby=[(x,cw) for x,y,cw,ch,area in comps
                if abs(y+ch/2-center)<13]
        if len(nearby)<10:continue
        x0=min(x for x,cw in nearby)
        x1=max(x+cw for x,cw in nearby)
        if x1-x0<min_width:continue
        # Stop before illustrations that often occupy the other half
        # of the page; the extracted crop can be a script line fragment.
        x0=max(left,x0-8)
        x1=min(right,x1+8,x0+int(.72*w))
        y0=max(0,int(center)-17)
        y1=min(h,int(center)+19)
        crop=image[y0:y1,x0:x1]
        if not crop.size:continue
        dark_fraction=float((crop<155).mean())
        if not .008<=dark_fraction<=.39:continue
        result.append({"bbox":[x0,y0,x1,y1],
                       "ink_fraction":round(dark_fraction,4),
                       "image":crop})
    return sorted(result,key=lambda x:x["bbox"][1])


def normalize_line(crop,height=64,width=256):
    """Preserve aspect ratio with padding instead of stretching glyph widths."""
    h,w=crop.shape
    factor=min((height-8)/h,(width-8)/w)
    out_w=max(1,min(width-8,int(round(w*factor))))
    out_h=max(1,min(height-8,int(round(h*factor))))
    image=cv2.resize(crop,(out_w,out_h),interpolation=cv2.INTER_AREA)
    result=np.full((height,width),255,dtype=np.uint8)
    top=(height-out_h)//2
    result[top:top+out_h,4:4+out_w]=image
    return result


def build(sources,zl_path,out,heldout_quires=("M","O"),max_lines_per_page=24):
    files=[Path(p) for p in sources]
    lookup=read_quire_lookup(Path(zl_path).read_text(encoding="utf-8-sig"))
    dest=Path(out);dest.mkdir(parents=True,exist_ok=True)
    images=[];source_rows=[]
    held=set(heldout_quires)
    for path in files:
        folio=folio_from_path(path)
        quire=lookup.get(folio,"unknown")
        subset="test" if quire in held else "train"
        image=resize_image(path)
        lines=segment_lines(image)[:max_lines_per_page]
        for idx,rec in enumerate(lines):
            normalized=normalize_line(rec["image"])
            images.append(normalized)
            source_rows.append({
                "filename":str(path),"folio":folio,"quire":quire,
                "split":subset,"bbox":rec["bbox"],"ink_fraction":rec["ink_fraction"],
                "line_index":idx,
                "source_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    if not images:raise ValueError("No script-looking lines detected")
    train=[i for i,r in enumerate(source_rows) if r["split"]=="train"]
    test=[i for i,r in enumerate(source_rows) if r["split"]=="test"]
    if len(train)<12 or len(test)<4:
        raise ValueError(f"Insufficient segmented lines: train {len(train)}, test {len(test)}")
    # Source pages in the same quire cannot be in train and test.
    assert not ({r["quire"] for r in source_rows if r["split"]=="train"}&
                {r["quire"] for r in source_rows if r["split"]=="test"})
    data=np.stack(images)
    np.savez_compressed(dest/"line_images.npz",
                        train=data[train],test=data[test])
    (dest/"line_manifest.json").write_text(json.dumps({
        "rows":source_rows,"train_lines":len(train),"test_lines":len(test),
        "heldout_quires":list(heldout_quires),
        "method":"heuristic connected-components, not manually verified"},
        indent=2)+"\n")
    for split,idxs in (("train",train),("test",test)):
        tile=np.concatenate([data[i] for i in idxs[:min(24,len(idxs))]])
        Image.fromarray(tile).save(dest/f"{split}_line_montage.png")
    print(json.dumps({"train":len(train),"test":len(test),
                      "train_folios":len(set(source_rows[i]["folio"] for i in train)),
                      "heldout_folios":len(set(source_rows[i]["folio"] for i in test))}))
    return len(train),len(test)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--images",nargs="+",required=True)
    p.add_argument("--zl",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--heldout-quires",nargs="+",default=["M","O"])
    a=p.parse_args()
    build(a.images,a.zl,a.out,tuple(a.heldout_quires))
