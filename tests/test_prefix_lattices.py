from experiments.prefix_lattices.prefix_beam import PrefixModel, MODELS, diverse_beam, entropy

def test_normalized():
    data=[('x',['the','cat','sat']),('x',['the','dog','ran'])]*10
    m=PrefixModel(data,max_novel=8)
    for model in MODELS:
        for prefix in ([], ['the'], ['the','cat']):
            dist=m.distribution(prefix,model)
            assert abs(sum(dist.values())-1)<1e-8
            assert entropy(dist)>0
    beam=diverse_beam(m,['the'],horizon=3,groups=3,width=6)
    assert all(len(row['words'])==3 for row in beam['paths'])
    assert len({row['words'][0] for row in beam['paths']})>=2
    assert 0<beam['path_probability_lower_bound']<=1
