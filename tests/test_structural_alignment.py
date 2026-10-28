from experiments.latent_grammar.structural_alignment import (
    ContextGraph,align,known_renaming_test,cross_structure,
)


def fixture():
    return [
        ("a",["the","red","bird","flies"]),
        ("a",["the","blue","bird","sings"]),
        ("b",["a","small","dog","runs"]),
        ("b",["a","big","cat","sleeps"]),
        ("b",["the","small","cat","sings"])
    ]*16


def test_alignment_uses_no_word_spelling():
    seq=fixture()
    a=ContextGraph(seq,top_k=7)
    recode={w:f"w{i:03d}" for i,w in enumerate(sorted(
        {v for _,line in seq for v in line}))}
    enc=[(g,[recode[w] for w in line]) for g,line in seq]
    b=ContextGraph(enc,top_k=7)
    perm,quality=align(a,b)
    assert len(set(perm))==7
    assert quality["edge_pattern_cosine"]>0
    assert sum(b.words[perm[i]]==recode[w]
               for i,w in enumerate(a.words))>=4


def test_positive_control_and_nontranslation_label():
    seq=fixture()
    r=known_renaming_test(seq,k=7)
    assert r["ground_truth_present_in_right_topk"]>0
    assert r["correct_graph_matches"]>=1
    c=cross_structure(seq,seq,k=7)
    assert c["top_k"]==7
    assert all("NOT a translation" in row["interpretation"]
               for row in c["ranked_structural_analogies"])
