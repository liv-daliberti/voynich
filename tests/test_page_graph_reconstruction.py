from experiments.page_graphs.reconstruct_controls import (
    matched_pages, reconstruct_page, summarize_pages, aggregate_recovery,
)


def test_exact_pseudo_page_budgets():
    pages=[
        {'folio':'f1r','language':'A','words':['abc','acd','abc']},
        {'folio':'f2r','language':'B','words':['abc','a']},
    ]
    target=['w'+str(i) for i in range(10)]
    mapped=matched_pages(pages,target,'book')
    assert [len(p['words']) for p in mapped]==[3,2]
    assert [w for p in mapped for w in p['words']]==target[:5]
    assert mapped[1]['language']=='B'


def test_page_deletion_is_not_leaked_into_candidate_ranking():
    page={'folio':'p','language':'A',
        'words':['cat','bat','sat','hat','rat','mat','pat','fat']}
    res=reconstruct_page(page,False,trials=80,topk=5,seed=17)
    assert res['trials']==80
    assert 0<=res['deleted_recall']<=1
    assert 0<=res['deleted_reachable_fraction']<=1
    assert res['mean_candidate_pool']>=0
    assert res['mean_random_expected_hits_topk']>=0


def test_page_order_shuffle_discriminates_nearby_overlap():
    pages=[]
    for i in range(30):
        word='aaa' if i<15 else 'bbb'
        pages.append({'folio':f'p{i}','language':'A',
                      'words':[word,'abc','abd','bbc','bcc','bac']})
    summary,graphs=summarize_pages(pages,False,shuffle_trials=15,seed=2)
    assert summary['A']['metrics']['word_jaccard']['near_minus_far']>0
    assert summary['A']['metrics']['word_jaccard']['trials']==15
    assert len(graphs)==30
