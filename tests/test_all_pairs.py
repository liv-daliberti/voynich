import importlib.util
from pathlib import Path
p = Path(__file__).resolve().parents[1] / "experiments/all_pairs/all_pairs.py"
spec=importlib.util.spec_from_file_location("all_pairs",p)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_eva_compounds():
    assert m.glyphs("qokeedy")==("q","o","k","e","e","d","y")
    assert m.glyphs("cthy")==("cth","y")
def test_one_edit():
    assert m.distance_one(m.glyphs("daiin"),m.glyphs("dain"))
    assert not m.distance_one(m.glyphs("daiin"),m.glyphs("chedy"))
    assert not m.distance_one(m.glyphs("dain"),m.glyphs("dain"))
def test_parser_fixture():
    text='<f1r> <! $L=A>\n<f1r.1,@P0> daiin.chol.dain\n<f1v> <! $L=B>\n<f1v.1,@P0> chedy.qokedy\n'
    d=m.parse_counts(text)
    assert d["A"]["daiin"]==1
    assert d["B"]["chedy"]==1
