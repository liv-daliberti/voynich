#!/usr/bin/env python3
"""An honest, low-resource visual-only Voynich patch generation PILOT.

This builds a contrastive visual encoder, a quantized patch dictionary and
a spatial autoregressive generator. It samples new pixel pages WITHOUT EVA
text or word constraints. We separately measure generated visuals and a
weak, automatically extracted pseudo-script; pseudo-script is NOT EVA/OCR.

Pilot is intentionally small and imperfect, not a claim of learned language.
For a serious result, replace this patch grammar with MAGE/VQ-MAE/diffusion,
use quire-blocked full scan training, and train/calibrate Voynich HTR.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter
from sklearn.cluster import MiniBatchKMeans
import torch
from torch import nn
import torch.nn.functional as F

torch.set_num_threads(2)


class TinyPatchEncoder(nn.Module):
    def __init__(self, dim=48):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(1,20,3,padding=1),nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20,40,3,padding=1),nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(40,48,3,padding=1),nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(48,dim))
    def forward(self,x):
        return F.normalize(self.layers(x),dim=1)


def get_page(path,width=384,height=512):
    im=Image.open(path).convert("L")
    # Uniform size is a PILOT concession: page layout may be distorted.
    im=im.resize((width,height),Image.Resampling.BICUBIC)
    return np.asarray(im,dtype=np.uint8)


def patches(image,side=32):
    h,w=image.shape
    return image.reshape(h//side,side,w//side,side).transpose(0,2,1,3)


def augment(batch,rng):
    """Ink-preserving low-amplitude augmentations, not arbitrary rotations."""
    shifted=batch.copy()
    for i in range(len(shifted)):
        x=int(rng.integers(-2,3))
        y=int(rng.integers(-2,3))
        shifted[i]=np.roll(shifted[i],(y,x),axis=(-2,-1))
    a=np.asarray(shifted,dtype=np.float32)/255.0
    a=a*rng.uniform(.87,1.13,(len(a),1,1))+rng.normal(0,.025,a.shape)
    return torch.from_numpy(np.clip(a,0,1).astype(np.float32))[:,None]


def train_contrastive(flat_patches,steps=160,batch=64,seed=29):
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    model=TinyPatchEncoder()
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    losses=[]
    model.train()
    for step in range(steps):
        idx=rng.choice(len(flat_patches),batch,replace=len(flat_patches)<batch)
        img=flat_patches[idx]
        a=augment(img,rng);b=augment(img,rng)
        z=model(torch.cat([a,b]))
        sim=z@z.T/.13
        sim.fill_diagonal_(-1e6)
        label=torch.cat([torch.arange(batch,2*batch),
                          torch.arange(batch)])
        loss=F.cross_entropy(sim,label)
        opt.zero_grad(set_to_none=True)
        loss.backward();opt.step()
        if step%20==0 or step==steps-1:
            losses.append({"step":step,"nt_xent":round(float(loss.detach()),4)})
    model.eval()
    return model,losses


def embed(model,pixels,batch=512):
    results=[]
    with torch.no_grad():
        for i in range(0,len(pixels),batch):
            x=torch.from_numpy(pixels[i:i+batch].astype(
                np.float32)/255.0)[:,None]
            results.append(model(x).cpu().numpy())
    return np.concatenate(results)


class SpatialTokenModel:
    def __init__(self,num_codes,alpha=8):
        self.num_codes=num_codes
        self.alpha=alpha
        self.global_counts=Counter()
        self.row_counts=defaultdict(Counter)
        self.left=defaultdict(Counter)
        self.top=defaultdict(Counter)
        self.both=defaultdict(Counter)
        self.occupancies=[]
    def fit(self,grids):
        for grid in grids:
            h,w=grid.shape
            self.occupancies.append(grid)
            for y in range(h):
                for x in range(w):
                    t=int(grid[y,x]);self.global_counts[t]+=1
                    row=min(3,int(4*y/h))
                    self.row_counts[row][t]+=1
                    if x:self.left[int(grid[y,x-1])][t]+=1
                    if y:self.top[int(grid[y-1,x])][t]+=1
                    if x and y:self.both[int(grid[y,x-1]),int(grid[y-1,x])][t]+=1
        self.base=np.asarray([self.global_counts[i]+.2
                              for i in range(self.num_codes)],float)
        self.base/=self.base.sum()
    def probs(self,grid,y,x):
        h,w=grid.shape
        row=min(3,int(4*y/h))
        row_counts=self.row_counts[row]
        p=(np.asarray([row_counts[i] for i in range(self.num_codes)],float)
           +40*self.base)
        p/=p.sum()
        histories=[]
        if x:histories.append(self.left[int(grid[y,x-1])])
        if y:histories.append(self.top[int(grid[y-1,x])])
        if x and y:histories.append(self.both[int(grid[y,x-1]),int(grid[y-1,x])])
        for history in histories:
            n=sum(history.values())
            p=(np.asarray([history[i] for i in range(self.num_codes)],float)
               +self.alpha*p)/(n+self.alpha)
        return p
    def sample(self,shape,seed=20,iid=False):
        rng=np.random.default_rng(seed)
        grid=np.zeros(shape,dtype=np.int16)
        for y in range(shape[0]):
            for x in range(shape[1]):
                prob=self.base if iid else self.probs(grid,y,x)
                grid[y,x]=rng.choice(self.num_codes,p=prob)
        return grid


def reconstruct(code_grid,template,side=32):
    img=template[code_grid].transpose(0,2,1,3).reshape(
        code_grid.shape[0]*side,code_grid.shape[1]*side)
    # Small smoothing solely for joining visible patch edges.
    return Image.fromarray(np.clip(img,0,255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(.35))


def distribution_metrics(codes,baseline,model):
    # Image code histogram/2-D transitions are NOT glyph distributions.
    K=model.num_codes
    def js(a,b):
        a=a+.25;b=b+.25;a=a/a.sum();b=b/b.sum();m=(a+b)/2
        return float(.5*np.sum(a*np.log(a/m))+.5*np.sum(b*np.log(b/m)))
    real=np.concatenate([g.ravel() for g in baseline])
    sim=codes.ravel()
    hist_r=np.bincount(real,minlength=K).astype(float)
    hist_s=np.bincount(sim,minlength=K).astype(float)
    def transitions(grids):
        counts=np.zeros((K,K),float)
        for a in grids:
            y=a[:,:-1].reshape(-1);z=a[:,1:].reshape(-1)
            np.add.at(counts,(y,z),1)
        return counts
    real_edges=transitions(baseline)
    synth_edges=transitions([codes])
    return {"visual_token_unigram_js":round(js(hist_r,hist_s),5),
            "horizontal_visual_transition_js":round(js(real_edges,synth_edges),5),
            "real_token_support":int((hist_r>0).sum()),
            "generated_token_support":int((hist_s>0).sum())}


def pseudo_transcribe(image):
    """Weak geometry-only proxy, no actual Voynich grapheme recognition.
    Extract horizontal foreground runs and connected-components, using
    component widths/gaps to approximate glyph-like and word-like groups.
    """
    arr=np.asarray(image.convert("L"))
    # Robust independent extraction on both real and synthetic image.
    ink=(arr<125).astype(np.uint8)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    ink=cv2.morphologyEx(ink,cv2.MORPH_OPEN,kernel)
    n,labels,stats,cent=cv2.connectedComponentsWithStats(ink,8)
    kept=[]
    for i in range(1,n):
        x,y,w,h,area=map(int,stats[i])
        if 2<=w<=35 and 3<=h<=65 and 6<=area<=500:
            kept.append((x,y,w,h,area))
    if not kept:
        return {"glyph_components":0,"component_sequences":0,
                "pseudo_word_length_mean":None,"pseudo_word_types":0,
                "repeated_pseudo_forms_fraction":None}
    rows=defaultdict(list)
    for x,y,w,h,area in kept:
        row=int((y+h/2)//30)
        rows[row].append((x,y,w,h,area))
    # Shape signatures (NOT glyph IDs, insufficient for EVA).
    words=[]
    for row,items in rows.items():
        items.sort()
        current=[];last=None
        for x,y,w,h,area in items:
            if last is not None and x-last>max(10,1.3*w):
                if current:words.append(tuple(current))
                current=[]
            sign=(min(3,w//7),min(3,h//10),min(3,area//40))
            current.append(sign)
            last=x+w
        if current:words.append(tuple(current))
    cnt=Counter(words)
    return {"glyph_components":len(kept),
            "component_sequences":len(words),
            "pseudo_word_length_mean":round(float(
                np.mean([len(w) for w in words])),3) if words else None,
            "pseudo_word_types":len(cnt),
            "repeated_pseudo_forms_fraction":round(float(
                sum(v for v in cnt.values() if v>=2)/len(words)),4) if words else None,
            "note":"NO EVA transcription; shape/gap proxy only"}


def run(files,out,steps=120,side=32,K=40,seed=29,test_count=2):
    np.random.seed(seed);random.seed(seed)
    p=Path(out);p.mkdir(exist_ok=True,parents=True)
    images=[get_page(file) for file in files]
    if len(images)<test_count+3:raise ValueError("Need at least 3 training pages")
    # Split by source IMAGE not random image patches to prevent patch leakage.
    train=images[:-test_count];test=images[-test_count:]
    blocks=[patches(a,side) for a in train]
    shape=blocks[0].shape[:2]
    training=np.concatenate([x.reshape(-1,side,side) for x in blocks])
    contrast,trace=train_contrastive(training,steps,seed=seed)
    embeddings=embed(contrast,training)
    kmeans=MiniBatchKMeans(n_clusters=K,random_state=seed,n_init=3,
                           batch_size=256,max_iter=80)
    code=kmeans.fit_predict(embeddings)
    # Visual centroid decoder is a PILOT only, not a learned VQ decoder.
    renderer=np.zeros((K,side,side),np.float32)
    for i in range(K):
        idx=np.flatnonzero(code==i)
        if len(idx):
            renderer[i]=training[idx].mean(axis=0)
        else:renderer[i]=255
    grids=[];pos=0
    for b in blocks:
        count=b.shape[0]*b.shape[1]
        grids.append(code[pos:pos+count].reshape(shape))
        pos+=count
    gen=SpatialTokenModel(K);gen.fit(grids)
    # The genuine heldout images are embedded/quantized using the FROZEN
    # encoder and codebook. The generator never saw their patches.
    test_patches=np.concatenate([
        patches(a,side).reshape(-1,side,side) for a in test])
    test_ids=kmeans.predict(embed(contrast,test_patches))
    test_grids=[test_ids[i*shape[0]*shape[1]:
                         (i+1)*shape[0]*shape[1]].reshape(shape)
                for i in range(len(test))]
    generated=gen.sample(shape,seed+20)
    iid=gen.sample(shape,seed+20,iid=True)
    genimg=reconstruct(generated,renderer,side)
    iidimg=reconstruct(iid,renderer,side)
    genimg.save(p/"generated_pilot.png")
    iidimg.save(p/"iid_patch_null.png")
    for i,image in enumerate(test):
        Image.fromarray(image).save(p/f"heldout_real_{i}.png")
    stats={
        "source_scan_files":[str(x) for x in files],
        "source_scan_sha256":[hashlib.sha256(Path(x).read_bytes()).hexdigest()
                              for x in files],
        "pages_train":len(train),"pages_heldout":len(test),
        "contrastive_training_steps":steps,"patch_size":side,
        "visual_codebook_size":K,
        "contrastive_loss_trace":trace,
        "generated_pseudo_script":pseudo_transcribe(genimg),
        "iid_pseudo_script":pseudo_transcribe(iidimg),
        "heldout_pseudo_script":[pseudo_transcribe(Image.fromarray(x))
                                 for x in test],
        "generated_vs_heldout_visual_distribution":distribution_metrics(
            generated,test_grids,gen),
        "iid_vs_heldout_visual_distribution":distribution_metrics(
            iid,test_grids,gen),
        "generated_visual_distribution":distribution_metrics(
            generated,grids,gen),
        "iid_visual_distribution":distribution_metrics(iid,grids,gen),
        "scientific_status":"CPU image-space pilot; NOT fluent page writing, EVA/OCR or learned meaning",
        "description":"contrastive patch CNN + kmeans visual tokens + smoothed spatial token generator",
        "limitations":["few folios, image layout may contain illustrations",
                        "patch means blur actual handwriting",
                        "pseudo-script is not EVA and cannot validate Voynich word distributions",
                        "real quire heldout images required for substantial followup"]}
    (p/"summary.json").write_text(json.dumps(stats,indent=2)+"\n")
    print(json.dumps({
        "train":len(train),"heldout":len(test),"steps":steps,
        "conditional_visual_js":stats["generated_visual_distribution"],
        "iid_visual_js":stats["iid_visual_distribution"],
        "generated_vs_heldout":stats["generated_vs_heldout_visual_distribution"],
        "iid_vs_heldout":stats["iid_vs_heldout_visual_distribution"],
        "pseudo_script":stats["generated_pseudo_script"]}),flush=True)
    return stats


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--images",nargs="+",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--steps",type=int,default=120)
    parser.add_argument("--patch",type=int,default=32)
    parser.add_argument("--codes",type=int,default=40)
    a=parser.parse_args()
    run(a.images,a.out,steps=a.steps,side=a.patch,K=a.codes)
