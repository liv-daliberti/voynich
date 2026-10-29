from experiments.family_stability.predict import (
    extract_events,match_stable_reorganizing,match_report,
    summarize,cross_validate,variants,
)


def page(i,words,quire="A",language="A"):
    return {"folio":f"f{i}r","words":words,"quire_code":quire,
            "language":language,"section_code":"H","hand_code":"1"}


def test_past_only_family_stability_and_future_gold_change():
    first=[
        page(0,["cat","bat","hat","sat","rat"]),
        page(1,["cat","bat","hat","rat"]),
        page(2,["cat","bat","hat","mat"])]
    a=extract_events(first,False,lookback=4)
    assert a
    cat=[e for e in a if e["word"]=="cat" and
         e["target_index_in_quire"]==2][0]
    assert cat["new_neighbor_any"]==1
    later=[
        page(0,["cat","bat","hat","sat","rat"]),
        page(1,["cat","bat","hat","rat"]),
        page(2,["cat","bat","hat","fat"])]
    other=[e for e in extract_events(later,False,lookback=4)
           if e["word"]=="cat" and
           e["target_index_in_quire"]==2][0]
    for field in ("stability","prefix_family_degree",
                  "prior_tokens","candidate_pool_size",
                  "previous_neighbor_count"):
        assert cat[field]==other[field]
    assert other["new_neighbor_forms"]!=cat["new_neighbor_forms"]


def test_no_neighborhood_without_two_prior_observations():
    events=extract_events([
        page(0,["cat","bat"]),page(1,["dog","fog"]),
        page(2,["hog","bog"])],False)
    assert not events


def test_same_page_matching_only():
    base={"group":"A:A","folio":"f4r",
          "log_prior_tokens":2.,"log_prior_document_pages":1.2,
          "log_prefix_neighbors":1.5,"log_candidate_pool":3.,
          "glyph_length":3,"log_previous_neighbor_count":1.1,
          "log_earlier_neighbor_count":1.1,
          "new_neighbor_any":1,"retained_neighbor_any":1,
          "new_neighbor_count":1,"candidate_pool_size":18,
          "stability":.8,"class":"stable","word":"cat"}
    r=dict(base,word="dog",**{"class":"reorganizing",
                              "new_neighbor_any":0,
                              "retained_neighbor_any":0,
                              "stability":0.})
    out=match_stable_reorganizing([base,r])
    assert len(out)==1
    assert out[0]["stable_word"]=="cat"
    assert out[0]["reorganizing_word"]=="dog"
    assert match_report(out)["new_form_pair_advantage"]["point"]==1.0
    other=dict(r,folio="f9r",word="fog")
    assert match_stable_reorganizing([base,other])==[]


def test_source_alphabet_generates_only_nonidentical_forms():
    forms=variants("cat",["a","b","c","t"],eva=False)
    assert "bat" in forms
    assert "cat" not in forms
