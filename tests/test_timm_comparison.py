from experiments.timm_generator.compare import parse_generated
def test_generated_text_follows_documented_page_count():
    text="#Properties\n#text.lines_to_create=5\n#Statistics\n\n"
    text+="\n".join(["chol chor shor", "chedy sheedy qokeedy",
                     "shol shor chor", "qokain chdy cho",
                     "daiin dal qokedy"])
    pages,seq=parse_generated(text,lines_per_page=2)
    assert len(seq)==5
    assert len(pages)==3
    assert [len(p['words']) for p in pages]==[6,6,3]
    assert seq[0][0]=='gen_0000'
    assert seq[2][0]=='gen_0001'
    assert sum(len(w) for _,w in seq)==15
