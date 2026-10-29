"""Paired held-out target and source-only ranking tests."""
from experiments.same_target_source.evaluate import (
    precompute_source,score_identical_gold,matched_far_sources,
    test_conditions as run_conditions,permutation_null,choose_source_pairs,
)


def page(i,words,section="herbal",language="A",hand="1"):
    return {"folio":f"f{i}", "words":words,"section_code":section,
            "language":language,"hand_code":hand}


def test_identical_novel_gold_between_source_models():
    close=precompute_source(page(0,["cat","bat","rat","hat","mat"]),eva=False)
    far=precompute_source(page(1,["dog","fog","hog","log","bog"]),eva=False)
    target={"sat","fat","cat","dog"}
    result=score_identical_gold(close,far,target)
    assert result["n_gold"]==2
    assert abs(result["delta_recall"]-
               (result["near"]["recall"]-result["far"]["recall"]))<1e-9
    assert result["near"]["hits"]<=result["near"]["topk"]
    assert result["far"]["hits"]<=result["far"]["topk"]


def test_target_words_do_not_change_source_candidate_ranking():
    a=precompute_source(page(0,["cat","bat","mat","rat"]),eva=False)
    prior=dict(a["rank"])
    score_identical_gold(a,precompute_source(
        page(1,["dog","bog","hog"]),eva=False),{"hat","sat","pat"})
    score_identical_gold(a,precompute_source(
        page(2,["fog","log","cog"]),eva=False),{"car","bar","tar"})
    assert a["rank"]==prior


def test_section_hand_size_matching_uses_no_target_labels():
    pages=[]
    for i in range(40):
        words=["cat","hat","mat","bat","sat"]
        sec="herbal" if i<28 else "astro"
        pages.append(page(i,words,section=sec))
    models=[precompute_source(p,False) for p in pages]
    ids=list(range(len(pages)))
    best=matched_far_sources(pages,models,ids,25,20,"strict_section_hand")
    assert best
    assert all(pages[i]["section_code"]=="herbal" for i in best)
    # Near and target same section required.
    assert not matched_far_sources(pages,models,ids,28,20,
                                   "strict_section_hand")


def test_permutation_keeps_metadata_and_target_paired():
    pages=[]
    for i in range(40):
        w=["cat","bat","rat","mat","sat","fat"] if i<20 else [
            "dog","fog","hog","bog","log","cog"]
        pages.append(page(i,w+[f"tail{chr(97+i%26)}{chr(97+i//26)}"]))
    observed,rows=run_conditions(pages,False,"same_variety",repetitions=10)
    assert observed["A"]["preselected_pairs"]>=1
    result=permutation_null(pages,False,"same_variety",trials=10)
    assert result["A"]["trial_count"]==10
    assert all(r["n_gold"]>0 for r in rows["A"])
