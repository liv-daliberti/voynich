#!/usr/bin/env python3
"""Exact all-type-pairs EVA glyph census; stdlib only.

Usage: python experiments/all_pairs/all_pairs.py --zl data/ZL3b-n.txt --out results/all_pairs
A finite word-form graph is NOT a translation or a contextual substitution score.
"""
from __future__ import annotations
import argparse, collections, csv, json, pathlib, re

COMPOUND = ("ckh","cth","cph","cfh","ch","sh")

def glyphs(word: str) -> tuple[str,...]:
    out=[]; i=0
    while i<len(word):
        g=next((x for x in COMPOUND if word.startswith(x,i)),None)
        if g: out.append(g);i+=len(g)
        else: out.append(word[i]);i+=1
    return tuple(out)

def parse_counts(text: str) -> dict[str,collections.Counter]:
    """Strict folio-header language classification. Unknown pages excluded."""
    counts={"A":collections.Counter(),"B":collections.Counter()}
    lang=None
    for line in text.splitlines():
        h=re.match(r"^<([^.,>]+)>\\s*<!\\s*(.*?)>",line)
        if h:
            marker=re.search(r"\\$L=([AB])(?:\\s|$)",h.group(2))
            lang=marker.group(1) if marker else None
            continue
        m=re.match(r"^<([^>,]+),([^>]+)>\\s*(.*)$",line)
        if not m or not re.search(r"[+@*=]P",m.group(2)) or lang not in counts:
            continue
        payload=re.sub(r"<![^>]*>|<%>|<\\$>|\\{[^}]*\\}"," ",m.group(3)).replace("<->",".")
        counts[lang].update(w for w in re.split(r"[.,\\s]+",payload) if re.fullmatch(r"[a-z]+",w))
    return counts


def distance_one(a:tuple[str,...],b:tuple[str,...]) -> bool:
    if abs(len(a)-len(b))>1:return False
    i=j=errors=0
    while i<len(a) and j<len(b):
        if a[i]==b[j]:
            i+=1;j+=1;continue
        errors+=1
        if errors>1:return False
        if len(a)>len(b):i+=1
        elif len(b)>len(a):j+=1
        else:i+=1;j+=1
    return errors+max(len(a)-i,len(b)-j)==1

def census(counts:collections.Counter, label:str, out:pathlib.Path) -> dict:
    types=[(w,n,glyphs(w)) for w,n in counts.items()]
    n=len(types); pairs=n*(n-1)//2
    summary={"variety":label,"types":n,"tokens":sum(counts.values()),"all_type_pairs":pairs,
             "same_glyph_length":0,"same_first_last_glyphs":0,
             "same_length_and_boundaries":0,"distance_one":0,
             "distance_one_same_boundaries":0}
    edges=out/f"one_edit_{label}.csv"
    with edges.open("w",newline="",encoding="utf-8") as fh:
        writer=csv.writer(fh);writer.writerow(["left","right","freq_left","freq_right","boundary_preserved"])
        for i in range(n):
            a,na,ga=types[i]
            for j in range(i+1,n):
                b,nb,gb=types[j]
                eq_len=len(ga)==len(gb)
                eq_boundary=ga[0]==gb[0] and ga[-1]==gb[-1]
                if eq_len:summary["same_glyph_length"]+=1
                if eq_boundary:summary["same_first_last_glyphs"]+=1
                if eq_len and eq_boundary:summary["same_length_and_boundaries"]+=1
                if distance_one(ga,gb):
                    summary["distance_one"]+=1
                    if eq_boundary:summary["distance_one_same_boundaries"]+=1
                    writer.writerow([a,b,na,nb,int(eq_boundary)])
    return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--zl",type=pathlib.Path,required=True)
    ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("results/all_pairs"))
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    cs=parse_counts(args.zl.read_text(encoding="utf-8-sig"))
    results=[census(cs[l],l,args.out) for l in ("A","B")]
    (args.out/"summary.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2))

if __name__=="__main__": main()
