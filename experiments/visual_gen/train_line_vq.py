#!/usr/bin/env python3
"""VQ masked visual generation trained on SCRIPT LINE images, no EVA labels.

Contrastive + weighted pixel reconstruction learn stroke-preserving visual
tokens. Masked-token Transformer is trained to predict codebook indices from
other code positions; iterative MaskGIT-style sampling generates complete
new lines. A learned decoder reconstructs pixels. This is image synthesis,
NOT Voynich transcription, natural language or historical decipherment.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch import nn
import torch.nn.functional as F

torch.set_num_threads(2)


class VQLine(nn.Module):
    def __init__(self,codes=96,dim=64):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(1,32,4,2,1),nn.GELU(),
            nn.Conv2d(32,64,4,2,1),nn.GELU(),
            nn.Conv2d(64,dim,4,2,1),
            nn.Conv2d(dim,dim,3,padding=1))
        self.codebook=nn.Embedding(codes,dim)
        nn.init.uniform_(self.codebook.weight,-1/codes,1/codes)
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(dim,64,4,2,1),nn.GELU(),
            nn.ConvTranspose2d(64,32,4,2,1),nn.GELU(),
            nn.ConvTranspose2d(32,16,4,2,1),nn.GELU(),
            nn.Conv2d(16,1,3,padding=1),nn.Sigmoid())
        self.codes=codes
        self.dim=dim
    def quantize(self,z):
        B,D,H,W=z.shape
        values=z.permute(0,2,3,1).reshape(-1,D)
        emb=self.codebook.weight
        dist=(values**2).sum(1,keepdim=True)+(
            emb**2).sum(1)[None,:]-2*values@emb.T
        ids=dist.argmin(1)
        q=self.codebook(ids).reshape(B,H,W,D).permute(0,3,1,2)
        commit=F.mse_loss(z,q.detach())*.25+F.mse_loss(q,z.detach())
        zq=z+(q-z).detach()
        return zq,ids.reshape(B,H,W),commit
    def forward(self,x):
        z=self.encoder(x)
        q,ids,commit=self.quantize(z)
        recon=self.decoder(q)
        return recon,ids,commit,z
    def from_codes(self,ids):
        ids=torch.as_tensor(ids,dtype=torch.long,
                            device=self.codebook.weight.device)
        values=self.codebook(ids).permute(0,3,1,2)
        return self.decoder(values)


class MaskedCodeTransformer(nn.Module):
    def __init__(self,k=96,seq_len=256,width=96,layers=2,heads=4):
        super().__init__()
        self.k=k;self.seq_len=seq_len
        self.embed=nn.Embedding(k+1,width)
        self.pos=nn.Parameter(torch.randn(1,seq_len,width)*.01)
        layer=nn.TransformerEncoderLayer(width,heads,dim_feedforward=width*3,
            dropout=.1,batch_first=True,norm_first=True,activation="gelu")
        self.net=nn.TransformerEncoder(layer,layers)
        self.out=nn.Linear(width,k)
    def forward(self,tokens):
        x=self.embed(tokens)+self.pos[:,:tokens.shape[1]]
        return self.out(self.net(x))


def image_tensor(arr):
    return torch.from_numpy(1.-arr.astype(np.float32)/255.)[:,None]


def augment(tensor,rng):
    y=tensor.clone()
    for i in range(len(y)):
        dx=int(rng.integers(-3,4));dy=int(rng.integers(-2,3))
        y[i]=torch.roll(y[i],(dy,dx),(1,2))
    noise=torch.from_numpy(rng.normal(0,.025,size=tuple(y.shape)).astype("float32"))
    return torch.clamp(y*float(rng.uniform(.9,1.10))+noise,0.,1.)


def nt_xent(features,temperature=.2):
    if len(features)%2:return features.sum()*0
    n=len(features)//2
    z=F.normalize(features,dim=1)
    sim=z@z.T/temperature
    sim.fill_diagonal_(-1e6)
    labels=torch.cat([torch.arange(n,2*n),torch.arange(0,n)])
    return F.cross_entropy(sim,labels)


def train_vq(train,steps=360,seed=20261028,codes=96,batch_size=12):
    """Continuous-autoencoder warmup avoids pathological VQ codebook collapse.

    The previous no-warmup run used only 9 of 96 codes and generated
    essentially blank lines. Now learn a real stroke decoder BEFORE
    initializing VQ centroids from encoder feature vectors.
    """
    from sklearn.cluster import MiniBatchKMeans
    rng=np.random.default_rng(seed)
    torch.manual_seed(seed)
    model=VQLine(codes)
    warmup_steps=140
    opt=torch.optim.AdamW(model.parameters(),lr=.0007)
    log=[]
    def losses(x,rec):
        # Foreground occupies very little of a manuscript line.
        # Heavy foreground reweighting prevents empty-page solutions.
        pixel=F.binary_cross_entropy(rec,x,weight=1.+12.*x)
        edge=F.l1_loss((rec[:,:,:,1:]-rec[:,:,:,:-1]).abs(),
                       (x[:,:,:,1:]-x[:,:,:,:-1]).abs())
        return pixel,edge
    for step in range(warmup_steps):
        idx=rng.choice(len(train),batch_size,
                       replace=len(train)<batch_size)
        x=train[idx]
        z=model.encoder(x)
        rec=model.decoder(z)
        pixel,stroke=losses(x,rec)
        aug=augment(x,rng)
        ssl=nt_xent(torch.cat([z.mean((2,3)),
                               model.encoder(aug).mean((2,3))]))
        loss=pixel+.30*stroke+.025*ssl
        opt.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if step%35==0 or step==warmup_steps-1:
            log.append({"phase":"decoder_warmup","step":step,
                        "loss":round(float(loss.detach()),5),
                        "pixel":round(float(pixel.detach()),5),
                        "contrastive":round(float(ssl.detach()),5)})
            print("WARM",log[-1],flush=True)
    # Initialize codebook from actual latent activations; random near-zero
    # code vectors previously attracted only the parchment/background.
    with torch.no_grad():
        pieces=[]
        for i in range(0,len(train),20):
            z=model.encoder(train[i:i+20])
            pieces.append(z.permute(0,2,3,1).reshape(-1,model.dim))
        features=torch.cat(pieces).cpu().numpy()
    select=rng.choice(len(features),min(18000,len(features)),replace=False)
    km=MiniBatchKMeans(n_clusters=codes,random_state=seed,batch_size=1024,
                       max_iter=80,n_init=2)
    km.fit(features[select])
    with torch.no_grad():
        model.codebook.weight.copy_(
            torch.from_numpy(km.cluster_centers_).to(
                model.codebook.weight.dtype))
    opt=torch.optim.AdamW(model.parameters(),lr=.00025)
    for step in range(steps):
        idx=rng.choice(len(train),batch_size,
                       replace=len(train)<batch_size)
        x=train[idx]
        rec,ids,commit,latent=model(x)
        pixel,stroke=losses(x,rec)
        aug=augment(x,rng)
        ssl=nt_xent(torch.cat([
            latent.mean((2,3)),
            model.encoder(aug).mean((2,3))]))
        loss=pixel+.30*stroke+.20*commit+.025*ssl
        opt.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if step%40==0 or step==steps-1:
            log.append({"phase":"vq_finetuning","step":step,
                "loss":round(float(loss.detach()),5),
                "pixel":round(float(pixel.detach()),5),
                "stroke":round(float(stroke.detach()),5),
                "vq":round(float(commit.detach()),5),
                "contrastive":round(float(ssl.detach()),5),
                "minibatch_codes":int(ids.unique().numel())})
            print("VQ",log[-1],flush=True)
    model.eval()
    return model,log


def encode_lines(model,lines,batch_size=32):
    enc=[]
    with torch.no_grad():
        for i in range(0,len(lines),batch_size):
            x=lines[i:i+batch_size]
            _,ids,_,_=model(x)
            enc.append(ids.cpu())
    return torch.cat(enc)


def train_masked(codes,k,steps=220,batch=12,seed=20261028):
    rng=np.random.default_rng(seed+3)
    torch.manual_seed(seed+3)
    seq=codes.reshape(len(codes),-1)
    L=seq.shape[1]
    m=MaskedCodeTransformer(k,L)
    opt=torch.optim.AdamW(m.parameters(),lr=.0008)
    logs=[]
    for step in range(steps):
        indices=rng.choice(len(seq),size=batch,replace=len(seq)<batch)
        actual=seq[indices]
        prob=float(rng.uniform(.25,.87))
        mask=torch.from_numpy(rng.random((len(actual),L))<prob)
        tokens=actual.clone();tokens[mask]=k
        logits=m(tokens)
        loss=F.cross_entropy(logits[mask],actual[mask])
        opt.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(),1.)
        opt.step()
        if step%40==0 or step==steps-1:
            logs.append({"step":step,"masked_ce":round(float(loss.detach()),5),
                         "mask_rate":round(prob,3)})
            print("MASK",logs[-1],flush=True)
    m.eval()
    return m,logs


@torch.no_grad()
def sample_codes(model,shape,seed=21,rounds=10,temperature=1.):
    """Parallel iterative noncausal masked sampling, then confidence freeze."""
    H,W=shape
    L=H*W
    gen=torch.Generator().manual_seed(seed)
    codes=torch.full((1,L),model.k,dtype=torch.long)
    masked=torch.ones((1,L),dtype=torch.bool)
    for step in range(1,rounds+1):
        logits=model(codes)/temperature
        probs=F.softmax(logits,dim=-1)
        drawn=torch.multinomial(probs.reshape(-1,model.k),1,
                                generator=gen).reshape(1,L)
        conf=probs.gather(-1,drawn.unsqueeze(-1)).squeeze(-1)
        conf[~masked]=-1.
        # Revisions cannot erase already high-confidence filled tokens.
        remain=int(round(L*(1-step/rounds)))
        chosen=max(0,int(masked.sum().item())-remain)
        if chosen>0:
            pos=conf.flatten().topk(chosen).indices
            codes.view(-1)[pos]=drawn.view(-1)[pos]
            masked.view(-1)[pos]=False
    return codes.reshape(1,H,W)


def page_image(lines,seed=20,width=768,height=1050):
    """Arrange synthetic script lines, not copied manuscript illustrations."""
    rng=np.random.default_rng(seed)
    page=Image.new("L",(width,height),248)
    y=85
    for line in lines:
        glyph=Image.fromarray(np.clip(255*(1-line[0,0].numpy()),
                                     0,255).astype("uint8"),"L")
        # Keep blank page margins and original learned pixel strokes.
        px=min(width-145,608)
        glyph=glyph.resize((px,66),Image.Resampling.BICUBIC)
        x=int(rng.integers(70,100))
        page.paste(glyph,(x,y))
        y+=int(rng.integers(49,58))
        if y+80>height-60:break
    ImageDraw.Draw(page).text((28,height-28),"SYNTHETIC / NOT MS 408",
                              fill=90)
    return page


def js(a,b):
    a=np.asarray(a,float)+1e-5
    b=np.asarray(b,float)+1e-5
    a/=a.sum();b/=b.sum();m=(a+b)/2.
    return float(.5*np.sum(a*np.log(a/m))+
                 .5*np.sum(b*np.log(b/m)))


def evaluate(vq,transformer,train,test,train_ids,generated_ids,iid_ids):
    with torch.no_grad():
        x=test
        rebuilt,held_codes,_,_=vq(x)
    def masked_recon():
        ids=held_codes.reshape(len(held_codes),-1)
        rng=np.random.default_rng(20261028)
        mask=torch.from_numpy(rng.random(tuple(ids.shape))<.5)
        inputs=ids.clone();inputs[mask]=transformer.k
        with torch.no_grad():
            logits=transformer(inputs)
        pred=logits.argmax(-1)
        return float((pred[mask]==ids[mask]).float().mean())
    def hist(a):
        return np.bincount(np.asarray(a).reshape(-1),
                           minlength=vq.codes)
    def transitions(a):
        arr=np.asarray(a)
        H,W=arr.shape[-2:]
        hist=np.zeros((vq.codes,vq.codes),float)
        for seq in arr.reshape(-1,H,W):
            np.add.at(hist,(seq[:,:-1].reshape(-1),
                            seq[:,1:].reshape(-1)),1)
        return hist
    real=held_codes.numpy()
    g=generated_ids.numpy();z=iid_ids.numpy()
    with torch.no_grad():
        generated=vq.from_codes(g)
        independent=vq.from_codes(z)
    def darkness(images):
        return float((images>.35).float().mean())
    def nearest_train(code):
        # Overlap in discrete visual positions, not proof of no memorization.
        flat=train_ids.reshape(len(train_ids),-1)
        return float((flat==code.reshape(1,-1)).float().mean(1).max())
    return {
        "heldout_vq_reconstruction_mae":round(float(
            torch.abs(rebuilt-x).mean()),5),
        "heldout_masked_code_accuracy":round(masked_recon(),5),
        "heldout_code_frequency_js_generated":round(js(hist(real),hist(g)),5),
        "heldout_code_frequency_js_iid":round(js(hist(real),hist(z)),5),
        "heldout_horizontal_transition_js_generated":round(
            js(transitions(real),transitions(g)),5),
        "heldout_horizontal_transition_js_iid":round(
            js(transitions(real),transitions(z)),5),
        "heldout_ink_fraction":round(darkness(x),5),
        "generated_ink_fraction":round(darkness(generated),5),
        "iid_ink_fraction":round(darkness(independent),5),
        "max_token_position_agreement_with_train_line":round(
            nearest_train(generated_ids[0]),5)}


def run(dataset,out,vq_steps=360,mask_steps=220,seed=20261028,
        nlines=16,codes=96):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    data=np.load(dataset)
    train=image_tensor(data["train"]);test=image_tensor(data["test"])
    if len(train)<12 or len(test)<4:
        raise ValueError("Quire-separated dataset too small")
    model,trace=train_vq(train,steps=vq_steps,seed=seed,codes=codes)
    ids=encode_lines(model,train)
    transformer,masked_trace=train_masked(ids,codes,steps=mask_steps,seed=seed)
    h,w=ids.shape[-2:]
    samples=[sample_codes(transformer,(h,w),seed+100+i)
             for i in range(nlines)]
    # Compare same-sized IID visual-token controls.
    counts=torch.bincount(ids.flatten(),minlength=codes).float()+.1
    p=counts/counts.sum()
    generator=torch.Generator().manual_seed(seed+11)
    iid=[torch.multinomial(p,h*w,replacement=True,
                           generator=generator).reshape(1,h,w)
         for _ in range(nlines)]
    with torch.no_grad():
        output=[model.from_codes(z).detach().cpu() for z in samples]
        null=[model.from_codes(z).detach().cpu() for z in iid]
    page_image(output,seed).save(root/"generated_script_page.png")
    page_image(null,seed).save(root/"iid_token_page.png")
    def montage(examples,name):
        arr=torch.cat(examples[:min(8,len(examples))],dim=0).detach().cpu()
        Image.fromarray(np.uint8(np.clip(
            255*(1-arr[:,0].numpy()),0,255)).reshape(-1,256)).save(root/name)
    montage(output,"generated_line_montage.png")
    montage(null,"iid_line_montage.png")
    with torch.no_grad():
        recon,_,_,_=model(test[:min(8,len(test))])
    montage([recon[i:i+1] for i in range(len(recon))],
            "heldout_reconstructed_line_montage.png")
    result=evaluate(model,transformer,train,test,ids,
                    torch.cat(samples),torch.cat(iid))
    result.update({
        "vq_train_steps":vq_steps,"masked_transformer_steps":mask_steps,
        "n_source_train_lines":len(train),"n_source_test_lines":len(test),
        "generated_line_count":nlines,
        "codebook_entries":codes,
        "latent_grid":[h,w],
        "vq_loss_trace":trace,
        "masked_token_loss_trace":masked_trace,
        "method":"learned pixel VQ decoder plus contrastive masked-code Transformer",
        "eva_transcribed":False,
        "transcription_claim":"NO: image-only output, HTR accuracy must pass holdout gate",
        "output_warning":"Synthetic page images are NOT genuine Voynich folios"})
    (root/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print("RESULT",json.dumps({k:v for k,v in result.items()
                                if k not in ("vq_loss_trace",
                                             "masked_token_loss_trace")}),
          flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--vq-steps",type=int,default=360)
    p.add_argument("--mask-steps",type=int,default=220)
    p.add_argument("--codes",type=int,default=96)
    p.add_argument("--nlines",type=int,default=16)
    a=p.parse_args()
    run(a.dataset,a.out,a.vq_steps,a.mask_steps,
        nlines=a.nlines,codes=a.codes)
