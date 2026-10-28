import importlib.util
from pathlib import Path
p=Path(__file__).resolve().parents[1]/"experiments/transcription_ambiguity/pilot.py"
spec=importlib.util.spec_from_file_location("ambiguity",p)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_compound_parse():
    assert m.glyphs("cthy")==("cth","y")
def test_conservative_alignment(tmp_path):
    head="<f1r> <! $L=A $H=1 $I=H>\n"
    z=head+"<f1r.1,@P0> daiin.chedy.chol\n"
    t=head+"<f1r.1,@P0> doiin.chedy.chol\n"
    p=tmp_path/"ZL.txt";q=tmp_path/"IT.txt"
    p.write_text(z);q.write_text(t)
    r=m.aligned(m.read_ivtff(p),m.read_ivtff(q))
    assert r and sum(item["disagreement"] for item in r)==1
