"""Cross-page held-out prediction invariants and distance controls."""
import numpy as np
from experiments.cross_page.reconstruct import (
    source_model,score_pair,score_matrices,ordered_diagnostics,
    match_distant_sources,generated_words,run_pages,
)


def make_page(idx,words,language="A"):
    return {"folio":f"p{idx:02d}","words":list(words),"language":language}


def test_no_self_or_target_leakage():
    src=make_page(0,["cat","bat","sat","mat","rat"])
    a=source_model(src,eva=False,top_k=5,min_support=2)
    assert not (a["candidate"] & set(src["words"]))
    target={"hat","fat","cat"}
    original=score_pair(a,target)
    assert original["novel"]==2
    assert original["hits"]<=original["reachable"]<=original["novel"]
    changed=score_pair(a,target|{"unrelated"})
    assert a["top"]==set(a["top_order"])
    assert original["hits"]==changed["hits"]
    assert changed["novel"]==3


def test_lag_direction_and_same_target_distant_source_matching():
    pages=[]
    for i in range(26):
        word="cat" if i<13 else "dog"
        pages.append(make_page(i,[word,"bat","hat","mat","rat","sat","fat"]))
    model,arr=score_matrices(pages,eva=False,top_k=5)
    stats=ordered_diagnostics(arr,list(range(26)),lags=(1,5,20),
                              randomizations=4)
    assert stats["1"]["forward"]["n_pairs"]==25
    assert stats["20"]["forward"]["n_pairs"]==6
    assert stats["1"]["reverse_placebo"]["n_pairs"]==25
    matched=match_distant_sources(pages,model,arr,list(range(26)))
    assert matched["targets"]==6
    assert all(e["target"] not in (e["close"],e["far"])
               for e in matched["examples"])


def test_permutation_near_signal_is_detectable():
    # Independent test pages: the first and second half have different
    # neighborhoods; chronological proximity should help predict them.
    pages=[]
    for i in range(32):
        variant = (["cat","bat","rat","hat","mat"] if i<16
                   else ["dog","fog","log","hog","bog"])
        pages.append(make_page(i,variant))
    result=run_pages(pages,eva=False,top_k=6,shuffles=25)
    assert result["groups"]["A"]["lag_decay"]["1"]["forward"]["n_pairs"]==31
    assert result["groups"]["A"]["paired_near_vs_size_matched_far"]["targets"]==12


def test_generated_parser_skips_stats():
    txt="#Properties\n#method.random.pseudo.seed=19\n#Statistics\n"
    txt+="\npchal shal chor\nchey daiin qokedy\n"
    assert generated_words(txt)==["pchal","shal","chor","chey","daiin","qokedy"]
