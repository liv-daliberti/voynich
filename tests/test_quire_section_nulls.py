"""Quire/section metadata and conditional page-order null invariants."""
import numpy as np

from experiments.same_target_source.evaluate import apply_correct_zl_metadata
from experiments.same_target_source.quire_section_nulls import (
    cells, sample_order, movable_summary, graph_stats_in_order, evaluate
)


def page(i, language="A", section="H", quire="A", hand="1"):
    return {"folio":f"f{i}r","words":[
        "cat","bat","rat","mat","hat",f"tail{chr(97+i%26)}"],
        "language":language,"section_code":section,
        "quire_code":quire,"hand_code":hand}


def test_metadata_uses_i_as_section_and_f_as_slot():
    raw=("<f1r> <! $Q=A $P=A $F=a $I=H $L=A $H=1>\n"
         "<f1v> <! $Q=B $P=B $F=b $I=S $L=B $H=2>\n")
    pages=[{"folio":"f1r","words":["cat"],"language":"A"},
           {"folio":"f1v","words":["dog"],"language":"B"}]
    apply_correct_zl_metadata(pages,raw)
    assert pages[0]["section_code"]=="H"
    assert pages[0]["quire_code"]=="A"
    assert pages[0]["folio_slot_code"]=="a"
    assert pages[1]["section_code"]=="S"
    assert pages[1]["hand_code"]=="2"


def test_quire_section_shuffles_cannot_cross_metadata_strata():
    pages=[page(i,section="H" if i%2 else "S",
                quire="A" if i<12 else "B") for i in range(24)]
    models=[{"types":len(set(p["words"]))} for p in pages]
    group=cells(pages,models,("language","section_code","quire_code"))
    rng=np.random.default_rng(29)
    order=sample_order(group,len(pages),rng)
    assert sorted(order.tolist())==list(range(len(pages)))
    for slot,src in enumerate(order):
        for key in ("language","section_code","quire_code"):
            assert pages[slot][key]==pages[int(src)][key]
    assert movable_summary(group,len(pages))["movable_pages"]==24


def test_size_bin_is_group_internal_and_order_is_reversible():
    pages=[page(i) for i in range(18)]
    models=[{"types":3+i} for i in range(len(pages))]
    groups=cells(pages,models,("language","section_code","quire_code",
                               "hand_code","size_bin"))
    assert len(groups)==3
    order=sample_order(groups,len(pages),np.random.default_rng(8))
    for indices in groups.values():
        assert set(order[indices])==set(indices)


def test_small_real_source_and_graph_null_finite():
    pages=[page(i,section="H" if i<12 else "B",
                quire="A" if i<6 else "B" if i<12 else "C") for i in range(26)]
    # Use a small iteration to verify all null arrays and metadata are valid.
    outcome=evaluate(pages,False,permutations=3)
    assert "currier_section_quire" in outcome["nulls"]
    assert outcome["metadata_interpretation"]["$I"]=="illustration/section"
    assert outcome["nulls"]["currier"]["graph"]["A"]["word_jaccard"]["n_valid_permutations"]==3
    strict=outcome["nulls"]["currier_section_quire_hand"]
    assert strict["permutation_mobility"]["movable_pages"]==26
