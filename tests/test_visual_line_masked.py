"""Quick synthetic checks for the quire-heldout line generator."""
import pytest
pytest.importorskip("cv2")
pytest.importorskip("PIL")
pytest.importorskip("torch")
import numpy as np
import torch
from experiments.visual_gen.line_dataset import (
    segment_lines,normalize_line,read_quire_lookup
)
from experiments.visual_gen.train_line_vq import (
    VQLine,MaskedCodeTransformer,sample_codes,image_tensor
)
from experiments.visual_gen.htr_alignment_gate import audit


def test_extract_script_like_rows():
    paper=np.full((600,768),245,dtype=np.uint8)
    for y in range(65,470,48):
        for x in range(110,560,12):
            paper[y:y+12,x:x+5]=40
    out=segment_lines(paper,min_width=130)
    assert len(out)>=5
    line=normalize_line(out[0]["image"])
    assert line.shape==(64,256)
    assert line.min()<180


def test_masked_vq_and_sample_shapes():
    torch.manual_seed(4)
    v=VQLine(codes=16,dim=32)
    x=torch.rand(2,1,64,256)
    rec,code,commit,embedding=v(x)
    assert rec.shape==x.shape
    assert code.shape==(2,8,32)
    assert commit.isfinite()
    assert int(code.max())<16
    mask=MaskedCodeTransformer(k=16,seq_len=256,width=32,heads=4,layers=1)
    scores=mask(torch.full((1,256),16,dtype=torch.long))
    assert scores.shape==(1,256,16)
    generated=sample_codes(mask,(8,32),seed=11,rounds=3)
    assert generated.shape==(1,8,32)
    assert int(generated.min())>=0 and int(generated.max())<16
    recovered=v.from_codes(generated)
    assert recovered.shape==(1,1,64,256)


def test_quire_lookup_and_transcription_gate(tmp_path):
    text=('<f1r> <! $Q=A $P=A $F=a $I=H $L=A $H=1>\n'
          '<f1r.1,+P0> chol.daiin.or\n'
          '<f76r> <! $Q=M $P=C $F=b $I=B $L=B $H=2>\n'
          '<f76r.1,+P0> chedy.qokedy.odal\n')
    assert read_quire_lookup(text)=={"f1r":"A","f76r":"M"}
    z=tmp_path/"zl.txt";z.write_text(text)
    import json
    manifest=tmp_path/"manifest.json"
    manifest.write_text(json.dumps({"rows":[
        {"folio":"f1r","quire":"A","split":"train"},
        {"folio":"f76r","quire":"M","split":"test"}]}))
    result=audit(manifest,z)
    assert result["generated_eva_text"] is None
    assert result["real_image_to_eva_status"].startswith("BLOCKED")
