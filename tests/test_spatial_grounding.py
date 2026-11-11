"""Regression tests for the audited information-handling failures."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from sclambda.analysis.gene_names import resolve_genes
from sclambda.annotation.niche.measurements import observation_record, render, expression_record, validate_annotation
from test_spatial_decisions import harness, dossier, decision, ACTOR


@pytest.fixture
def aliases(tmp_path):
    p = tmp_path / 'gene_info.tsv'
    # Synthetic symbols/aliases: no identity-specific production knowledge.
    p.write_text('#tax_id\tGeneID\tSymbol\tLocusTag\tSynonyms\n'
                 '10090\t1\tAlpha\t-\tOldA|Shared\n'
                 '10090\t2\tBeta\t-\tShared|OldB\n'
                 '10090\t3\tGamma\t-\tOldG\n'
                 '9606\t4\tHumanA\t-\tOldA\n')
    return p


def test_aliases_preserve_ambiguity_species_and_real_missingness(aliases):
    r = resolve_genes(['Alpha','OldA','Shared','OldG','invented'], ['Alpha','Beta'],
                      organism='mouse', alias_path=aliases)
    assert [x['status'] for x in r['rows']] == ['exact','alias','ambiguous','not_measured','unrecognized']
    assert r['genes'] == ['Alpha']
    assert r['rows'][2]['candidates'] == ['Alpha','Beta']
    # Restricting the panel must not resolve biological alias ambiguity.
    r2 = resolve_genes(['Shared'], ['Alpha'], organism='mouse', alias_path=aliases)
    assert r2['genes'] == [] and r2['rows'][0]['status']=='ambiguous'
    assert r['version']['sha256'] == r2['version']['sha256']
    human = resolve_genes(['OldA'], ['HumanA'], organism='human', alias_path=aliases)
    assert human['genes'] == ['HumanA']


def test_alias_unavailable_source_is_not_unmeasured(tmp_path):
    r = resolve_genes(['Alpha','Unknown'], ['Alpha'], organism='mouse', cache_dir=tmp_path, download=False)
    assert r['genes']==['Alpha']
    assert r['rows'][1]['status']=='source_unavailable'


def test_official_full_names_and_other_designations(tmp_path):
    p = tmp_path/'gene_info'
    p.write_text('10090\t1\tAlpha\t-\t-\t-\t-\t-\talpha protein 1\tprotein-coding\tAlpha\talpha protein 1\tO\tOld protein|second name\n')
    r=resolve_genes(['AlphaProtein1','old protein'], ['Alpha'], organism='mouse',alias_path=p)
    assert r['genes']==['Alpha'] and all(x['status']=='alias' for x in r['rows'])


def test_actor_and_critic_see_identical_direct_views(monkeypatch):
    node,state,calls,_=harness(monkeypatch,[ACTOR,decision()])
    m=observation_record('0','Red target; gray context',{},kind='spatial_view');m['images']=['whole','crop']
    state['niche_dossier']['measurements'].append(m)
    node.execute(state)
    assert [kw['image_base64'] for _,kw in calls]==[['whole','crop'],['whole','crop']]


def test_image_setup_never_calls_model(monkeypatch):
    from sclambda.pipelines.annotate_spatial.nodes.images import read_views
    from sclambda.pipelines.annotate_spatial.nodes.composition import ComputeSpatialProportionsNode
    monkeypatch.setattr(ComputeSpatialProportionsNode,'_generate_image',lambda *a,**kw:(['whole','crop'],None))
    class NoModel:
        def query(self,*a,**kw): raise AssertionError('Unexpected image narrator')
    state={'cluster_id':'0','slice_key':'slide'}
    records=read_views(None,None,NoModel(),state,['s1'],'Inspect boundary')
    assert records[0]['images']==['whole','crop']
    assert records[0]['kind']=='spatial_view'
    assert 'not an anatomical interpretation' in records[0]['source']
    assert read_views(None,None,NoModel(),state,[None],'Inspect boundary')==[]


def test_neutral_plot_has_full_target_and_gray_context():
    import matplotlib.pyplot as plt
    from sclambda.analysis.spatial_visualization import plot_spatial_niche
    xy=np.array([[0,0],[0,1],[1,1],[8,8],[9,8],[9,9]],dtype=float)
    labels=np.array(['T','other','T','other','T','other'])
    fig=plot_spatial_niche(xy,labels,'T',neutral_background=True,mark_target=False,show=False,clean=True)
    scatters=fig.axes[0].collections
    assert np.array_equal(scatters[1].get_offsets()[:,0],xy[labels=='T',0])
    colors=scatters[0].get_facecolors()[:,:3]
    assert np.allclose(colors[:,0],colors[:,1]) and np.allclose(colors[:,1],colors[:,2])
    assert np.allclose(scatters[1].get_facecolors()[0,:3],[.9,.1,.1])
    assert not fig.axes[0].texts
    plt.close(fig)


def test_reference_retrieval_keeps_passages_and_filters_commentary(monkeypatch,tmp_path):
    from sclambda.annotation.niche import references as module
    calls=[]
    def fetch(url):
        calls.append(url)
        return json.dumps({'resultList':{'result':[
            {'source':'MED','id':'1','title':'Commentary','pubTypeList':{'pubType':['article-commentary']},'abstractText':'Not primary'},
            {'source':'MED','id':'2','title':'Experiment','doi':'10.x/example','pubYear':'2020','pubTypeList':{'pubType':['research-article']},'abstractText':'Measured localization differs between A and B.'},
            {'source':'MED','id':'3','title':'Title without evidence'}]}}).encode()
    monkeypatch.setattr(module,'_fetch',fetch)
    r=module.lookup_reference('0','Compare A and B',organism='mouse',tissue='organ',cache_dir=tmp_path)
    assert len(r['rows'])==1 and r['rows'][0]['source_id']=='MED:2'
    assert r['rows'][0]['text']=='Measured localization differs between A and B.'
    assert 'https://europepmc.org/article/MED/2' in render(r)
    again=module.lookup_reference('1','Compare A and B',organism='mouse',tissue='organ',cache_dir=tmp_path)
    assert len(calls)==1 and again['target']=='1'
    assert again['retrieved_at']==r['retrieved_at']


def test_citation_repair_cannot_change_biological_decision(monkeypatch):
    bad=decision();bad['support']=[{'measurement_id':'missing','items':['A']}]
    replacement=decision(label='Different compartment')
    node,state,calls,_=harness(monkeypatch,[bad,replacement])
    result,attempts=node.review(state,dossier(),[])
    assert result is None
    assert any('changed biological' in issue for issue in attempts[-1]['validation_errors'])
    assert calls[-1][1]['image_base64'] is None
    assert 'Original response:' in calls[-1][0] and 'Composition and position' in calls[-1][0]


def test_reference_failure_is_visible_and_does_not_prevent_measurement(monkeypatch):
    from sclambda.annotation.niche import references
    monkeypatch.setattr(references,'lookup_reference',lambda *a,**kw: (_ for _ in ()).throw(TimeoutError('offline')))
    node,state,_,_=harness(monkeypatch,[])
    d=dossier()
    node._act(state,d,{'tool':'done','reference_queries':[{'query':'A versus B'}]})
    assert any('Reference lookup failed' in row['fact'] for row in d.evidence_log)
    assert d.critic_decision=={}


@pytest.mark.parametrize('action,expected', [
    ({'tool':'lookup_reference','query':'A versus B'}, ['A versus B']),
    ({'tool':'lookup_reference','reference_queries':[{'query':'A versus B'},{'query':'C versus D'}]}, ['A versus B','C versus D']),
    ({'tool':'done','reference_queries':[{'query':'A versus B'}]}, ['A versus B']),
])
def test_reference_action_accepts_single_or_multiple_queries(monkeypatch,action,expected):
    node,state,_,_=harness(monkeypatch,[])
    seen=[]
    monkeypatch.setattr(node,'_reference',lambda state,dossier,check:seen.append(check['query']))
    node._act(state,dossier(),action)
    assert seen==expected


@pytest.mark.parametrize('flip', [False, True])
def test_whole_and_crop_share_frame_and_preserve_explicit_orientation(flip):
    import matplotlib.pyplot as plt
    from sclambda.analysis.spatial_visualization import plot_spatial_niche
    xy=np.array([[10,20],[20,30],[30,40],[40,50],[50,60]],dtype=float)
    original=xy.copy();labels=np.array(['other','T','T','T','other'])
    kwargs=dict(target_niche='T',section_bounds=(10,50,20,60),
                neutral_background=True,mark_target=False,clean=True,show=False)
    # Omitting flip must preserve the stored coordinates.
    whole=plot_spatial_niche(xy,labels,**kwargs,**({'flip':True} if flip else {}),
                             crop_bounds=(20,40,30,50),title='Whole section')
    crop=plot_spatial_niche(xy[1:4],labels[1:4],**kwargs,flip=flip,title='Crop')
    a,b=whole.axes[0],crop.axes[0]
    for axis in ['x','y']:
        ticks=lambda ax: dict(zip(getattr(ax,f'get_{axis}ticks')(),
                                 [t.get_text() for t in getattr(ax,f'get_{axis}ticklabels')()]))
        full,part=ticks(a),ticks(b)
        assert part and all(full[k]==v for k,v in part.items())
    expected=xy[1:4].copy()
    if flip:expected[:,1]*=-1
    np.testing.assert_array_equal(a.collections[1].get_offsets(),expected)
    np.testing.assert_array_equal(b.collections[-1].get_offsets(),expected)
    np.testing.assert_array_equal(xy,original)
    assert a.get_title()=='Whole section' and b.get_title()=='Crop'
    assert a.patches[0].get_y()==(-50 if flip else 30)
    plt.close(whole);plt.close(crop)


def test_candidates_are_not_lost_between_actor_turns(monkeypatch):
    first={'hypotheses':[{'label':'Original rival','reason':'Actor argument must stay out of critic prompt'}],
           'action':{'tool':'done'}}
    second={'hypotheses':[{'label':'New candidate'}],'action':{'tool':'done'}}
    ongoing=decision(label='Provisional',question='Compare the remaining alternatives')
    node,state,calls,_=harness(monkeypatch,[first,ongoing,second,decision()],rounds=2)
    node.execute(state)
    last_prompt=calls[-1][0]
    assert 'Original rival' in last_prompt and 'New candidate' in last_prompt
    assert 'Actor argument must stay out' not in last_prompt


def test_paired_renderer_retains_all_targets_and_marks_crop(monkeypatch):
    import pandas as pd
    from sclambda.pipelines.annotate_spatial.nodes.composition import ComputeSpatialProportionsNode
    from sclambda.analysis import spatial_visualization as module
    xy=np.array([[i, i%7] for i in range(40)],dtype=float)
    labels=np.array(['T' if i in (3,4,10,11,25,26) else 'other' for i in range(40)])
    node=object.__new__(ComputeSpatialProportionsNode)
    node.accessor=SimpleNamespace(n_cells=len(xy),get_obsm=lambda key,idx:xy[idx],
        _level_column=lambda depth:'L1',_overlay=pd.DataFrame({'L1':labels}))
    node.log_info=node.log_warning=lambda *a:None
    calls=[]
    monkeypatch.setattr(module,'generate_niche_image_base64',lambda **kw:calls.append(kw) or 'image')
    images,_=node._generate_image('T','spatial',None,None)
    assert len(images)==2 and len(calls)==2
    assert calls[0]['section_bounds']==calls[1]['section_bounds']
    assert calls[0]['crop_bounds'] is not None and calls[1]['crop_bounds'] is None
    assert not calls[0]['flip'] and not calls[1]['flip']
    for c in calls:np.testing.assert_array_equal(c['coords'][c['niche_labels']=='T'],xy[labels=='T'])


def pending_check():
    return {'question':'Distinguish the remaining identities in population A',
            'action':{'tool':'test_signatures','signatures':[
                {'cell_type':'A','genes':['GENE1','GENE2']}]}}


def test_pending_check_executes_despite_empty_next_question(monkeypatch):
    first=decision(status='resolved');first['pending_checks']=[pending_check()]
    node,state,calls,_=harness(monkeypatch,[ACTOR,first,decision()],rounds=3)
    actions=[]
    monkeypatch.setattr(node,'_act',lambda state,d,action:actions.append(action))
    result=node.execute(state)
    assert actions==[ACTOR['action'],pending_check()['action']]
    assert [kw['power'] for _,kw in calls]==['medium','strong','strong']
    assert result['dossier_agent_steps']==2
    assert result['dossier_agent_transcript'][2]['source']=='critic_request'
    assert pending_check()['question'] in calls[-1][0]


def test_budget_exhaustion_keeps_checks_and_cannot_claim_resolved(monkeypatch):
    first=decision(status='resolved');first['pending_checks']=[pending_check()]
    node,state,_,_=harness(monkeypatch,[ACTOR,first],rounds=1)
    monkeypatch.setattr(node,'_act',lambda *a:None)
    result=node.execute(state)
    assert result['dossier_stop_reason']=='budget_exhausted'
    assert result['annotation_assessment']=='best_effort'
    saved=result['niche_dossier']['critic_decision']
    assert saved['pending_checks']==[pending_check()]
    assert saved['next_question']==pending_check()['question']


def scoped_dossier():
    d=dossier()
    gene=SimpleNamespace(name='GENE1',lfc=-.2,frac_non_zero_target=.2,pval=.01)
    m=expression_record('0',SimpleNamespace(genes=[gene]),{},cell_type='Population A')
    d.measurements.append(m)
    v=decision(d);v['support']=[{'measurement_id':m['id'],'genes':['GENE1'],
                               'cell_type':'Population A','reference':m['reference']}]
    return d,v


@pytest.mark.parametrize('field,wrong',[('cell_type','Population B'),('cell_type',None),
                                      ('reference','all cell types pooled outside niche')])
def test_expression_scope_cannot_be_transferred(field,wrong):
    d,v=scoped_dossier()
    assert not validate_annotation(v,d,require_checks=True)
    v['support'][0][field]=wrong
    assert any(e.startswith('Scope mismatch:') for e in validate_annotation(v,d,require_checks=True))


def test_population_mismatch_is_not_repaired_as_formatting(monkeypatch):
    d,v=scoped_dossier();v['support'][0]['cell_type']='Population B'
    node,state,calls,_=harness(monkeypatch,[v])
    verdict,attempts=node.review(state,d,[])
    assert verdict is None and len(calls)==1
    assert attempts[0]['validation_errors'][0].startswith('Scope mismatch:')


def test_pending_actions_must_measure_and_missing_fields_are_not_silently_done():
    v=decision();v['pending_checks']=[{'question':'Unresolved distinction','action':{'tool':'done'}}]
    assert validate_annotation(v,dossier(),require_checks=True)
    v.pop('pending_checks')
    assert validate_annotation(v,dossier(),require_checks=True)
    assert not validate_annotation(v,dossier())  # Legacy saved decisions remain readable.


def test_missing_execution_plan_cannot_be_repaired_to_empty(monkeypatch):
    incomplete=decision();incomplete.pop('pending_checks')
    node,state,calls,_=harness(monkeypatch,[incomplete,decision()])
    verdict,attempts=node.review(state,dossier(),[])
    assert verdict is None and len(calls)==1
    assert attempts[0]['validation_errors']


def test_failed_requested_check_is_visible_and_other_checks_still_run(monkeypatch):
    first=decision();first['pending_checks']=[pending_check(),
        {'question':'Check another population','action':{'tool':'get_de','cell_type':'B'}}]
    node,state,calls,_=harness(monkeypatch,[ACTOR,first,decision()],rounds=3)
    actions=[]
    def act(state,d,action):
        actions.append(action)
        if action==pending_check()['action']: raise ValueError('comparison unavailable')
    monkeypatch.setattr(node,'_act',act)
    result=node.execute(state)
    assert len(actions)==3 and actions[-1]['cell_type']=='B'
    assert 'comparison unavailable' in calls[-1][0]
    assert 'no result was obtained' in calls[-1][0]
