"""Invariants for distinct persistence/innovation networks and POS probes."""
import numpy as np
from experiments.persistence_innovation.graphs import (
    tally,evaluate,propose,null_order
)
from experiments.word_classes.english_pos_probe import (
    count_features,annotate,FEATURES,unsupervised_roles
)


def pg(i,words,quire="A",language="A"):
    return {"folio":f"f{i}r","words":words,"language":language,
            "quire_code":quire,"section_code":"H","hand_code":"1"}


def test_persistence_and_innovation_have_disjoint_source_conditions():
    pages=[
        pg(0,["cat","bat","hat","sat","rat","fat"]),
        pg(1,["cat","bat","hat","mouse"]),
        pg(2,["cat","bat","hat","mat"])]
    result,edges,events,rows=tally(pages,False)
    assert result["A"]["steps"]==2
    assert ("cat","bat") in edges["persistence"]
    assert ("cat","mat") in edges["innovation"]
    assert ("cat","bat") not in edges["innovation"]
    assert all(e["edge_type"] in ("persistence","innovation")
               for e in events)
    assert rows[1]["source_folio"]=="f1r"


def test_quire_boundary_never_carries_edge():
    pages=[pg(0,["cat","bat","rat"]),
           pg(1,["cat","bat","sat"]),
           pg(2,["dog","fog","hog"],quire="B")]
    data,edges,events,_=tally(pages,False)
    assert data["A"]["steps"]==1
    assert not any(x["folio"]=="f2r" for x in events)


def test_page_order_null_rearranges_only_within_quire_section_hand():
    pages=[pg(i,["cat","bat","rat"],quire="A" if i<5 else "B")
           for i in range(10)]
    draw=null_order(pages,np.random.default_rng(11))
    assert sorted(draw.tolist())==list(range(len(pages)))
    for i,j in enumerate(draw):
        assert pages[i]["quire_code"]==pages[int(j)]["quire_code"]


def test_pos_structural_features_have_no_spelling_identity():
    seq=[("x",["the","cat","runs","quickly"]),
         ("x",["a","dog","runs","slowly"]),
         ("y",["the","dog","sleeps","quietly"])]*5
    rows=count_features(seq,min_count=3)
    assert rows
    assert len(FEATURES)==21
    assert "word" not in FEATURES
    assert "glyph_length" not in FEATURES
    assert all(np.isfinite(r["left_entropy"]) for r in rows)
    roles,_=unsupervised_roles(rows,k=3)
    assert len(roles["clusters"])==3


def test_english_tag_filter_excludes_ambiguous_or_not_noun_verb():
    rows=[{"word":"cat"},{"word":"run"},{"word":"but"}]
    counts={"cat":{"N":9,"V":1,"other":0},
            "run":{"N":2,"V":8,"other":0},
            "but":{"N":0,"V":0,"other":10}}
    ann=annotate(rows,counts)
    assert {x["word"]:x["noun_label"] for x in ann}=={
        "cat":1,"run":0}
