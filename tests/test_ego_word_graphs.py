"""Word-centered ego graph tests with controlled folio permutation."""
import numpy as np
from experiments.word_egos.evaluate import (
    two_hops,ego_profiles,pair_tables,ego_order_stat,evaluate
)


def page(i,words,quire="A",section="H"):
    return {"folio":f"f{i}r","words":words,"language":"A",
            "quire_code":quire,"section_code":section,"hand_code":"1"}


def test_one_and_two_hop_ring_graph():
    adj={"cat":{"bat","hat"},"bat":{"cat","rat"},
         "hat":{"cat"},"rat":{"bat"}}
    assert two_hops("cat",adj)=={"rat"}


def test_full_word_inventory_and_local_neighbors_are_not_global_degree():
    pages=[page(0,["cat","bat","hat"]),
           page(1,["cat","rat","bat"]),
           page(2,["cat","hat"]),
           page(3,["rat","bat","mat"])]
    adj,centers,word_rows,local=ego_profiles(pages)
    assert "cat" in centers
    assert len(word_rows)==len(set(
        w for p in pages for w in p["words"]))
    cat=[x for x in local if x["word"]=="cat"]
    assert len(cat)==3
    assert cat[0]["active_one_hop"]>=1
    assert cat[0]["global_degree"]>=cat[0]["active_one_hop"]
    assert any(x["ghost_supported_pages"]>=0 for x in word_rows)


def test_quire_word_ego_order_stat_invariant_under_no_op():
    pages=[page(i,["cat","bat","hat","rat","mat"] if i%2
                else ["cat","bat","hat","fat","sat"])
           for i in range(10)]
    _,centers,summary,_=ego_profiles(pages)
    chosen,table=pair_tables(pages,centers,summary,limit_centers=3,
                             min_document_frequency=4,min_global_degree=3)
    assert len(table["similarity"])>0
    observed=ego_order_stat(pages,table,np.arange(10))
    assert observed["matched_centers"]>=1
    changed=ego_order_stat(pages,table,np.arange(10)[::-1])
    assert observed["matched_centers"]==changed["matched_centers"]


def test_quira_strict_permutation_preserves_complete_page_graph():
    pages=[]
    for i in range(16):
        if i<8: words=["cat","bat","mat","hat","rat","sat"]
        else: words=["dog","fog","hog","log","bog","cog"]
        pages.append(page(i,words,quire="A" if i<8 else "B"))
    x,words,rows,histories,adjacency,centers=evaluate(pages,trials=5,limit_centers=8)
    assert x["n_pages"]==16
    assert x["n_selected_center_words"]>=1
    assert len(histories)<=8
    assert "quire" in x["nulls"]
