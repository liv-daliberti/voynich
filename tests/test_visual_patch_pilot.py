"""Visual pilot sanity checks, no internet or manuscript data required."""
from pathlib import Path
import pytest
pytest.importorskip('PIL')
pytest.importorskip('cv2')
pytest.importorskip('torch')
import numpy as np
from PIL import Image
from experiments.visual_gen.train_pilot import (
    patches, TinyPatchEncoder,SpatialTokenModel, pseudo_transcribe
)


def test_patch_reconstruction_shape():
    x=np.arange(64*96,dtype=np.uint8).reshape((64,96))
    p=patches(x,32)
    assert p.shape==(2,3,32,32)
    assert np.array_equal(p.transpose(0,2,1,3).reshape(x.shape),x)


def test_visual_markov_generator_is_probabilistic():
    model=SpatialTokenModel(5,alpha=3)
    data=[np.array([[0,1,2],[2,3,4]],dtype=np.int16),
          np.array([[0,1,3],[2,3,1]],dtype=np.int16)]
    model.fit(data)
    for y in range(2):
        for x in range(3):
            p=model.probs(data[0],y,x)
            assert abs(float(p.sum())-1)<1e-8
            assert np.all(p>0)
    a=model.sample((2,3),seed=5)
    assert a.shape==(2,3) and a.max()<5


def test_pseudo_script_is_not_voynich_ocr():
    im=Image.fromarray(np.full((64,96),255,np.uint8))
    x=pseudo_transcribe(im)
    assert x["glyph_components"]==0
    assert "word" not in x or "pseudo_word" in str(x)
