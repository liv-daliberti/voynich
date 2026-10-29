"""Within-quire source selection, null motion and target invariants."""
import numpy as np
from experiments.within_quire.predict import (
    strata,permute,select_pairs,analyze,group_stats,
)
from experiments.same_target_source.evaluate import precompute_source


def page(i,words,quire='A',section='H',hand='1',language='A'):
    return {'folio':f'f{i}r','words':words,'language':language,
            'quire_code':quire,'section_code':section,'hand_code':hand}


def test_quire_strata_never_cross_quire_or_section():
    rows=[page(i,['cat','bat','mat','hat','rat',f'c{i}'],
               quire='A' if i<12 else 'B',
               section='H' if i%2 else 'S') for i in range(24)]
    for keys in [('language','quire_code'),
                 ('language','quire_code','section_code'),
                 ('language','quire_code','section_code','hand_code'),
                 ('language','quire_code','section_code','hand_code','size_bin')]:
        groups=strata(rows,keys)
        order=permute(rows,keys,np.random.default_rng(11))
        assert sorted(order)==list(range(24))
        for i,src in enumerate(order):
            for k in keys:
                if k!='size_bin':
                    assert rows[i][k]==rows[src][k]
        assert sum(len(v) for v in groups.values())==24


def test_future_source_cannot_influence_near_far_selection():
    pages=[page(i,['cat','bat','hat','mat','rat','sat']+
                (['dog','fog'] if i%2 else ['fat','pat']),quire='A')
           for i in range(10)]
    models=[precompute_source(p,False) for p in pages]
    pairs=select_pairs(pages,models,list(range(10)))
    assert pairs
    for near,far,target in pairs:
        assert target>near>far
        assert pages[target]['quire_code']==pages[far]['quire_code']


def test_synthetic_progression_same_quire_and_no_lag_leak():
    words=[
        ['cat','bat','mat','rat','hat','sat'],
        ['cat','bat','mat','rat','hat','fat'],
        ['cat','bat','mat','rat','hat','pat'],
        ['dog','fog','log','bog','hog','cog'],
        ['dog','fog','log','bog','hog','jog'],
        ['dog','fog','log','bog','hog','tog']
    ]
    pages=[page(i,words[i%6],quire='A' if i<6 else 'B')
           for i in range(12)]
    result=analyze(pages,False,trials=5)
    assert 'all_within_quire' in result
    assert 'within_quire_section_hand_size' in result['all_within_quire']['nulls']
    assert result['all_within_quire']['matched_targets']>=1
