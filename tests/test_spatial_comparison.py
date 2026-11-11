"""Counterexamples for comparative decisions, using no biological marker rules."""
from copy import deepcopy
import json
import pytest

from sclambda.annotation.niche.measurements import validate_annotation, normalize_annotation
from test_spatial_decisions import dossier, harness, ACTOR, cli, spatial_run, decision
from test_spatial_grounding import scoped_dossier, pending_check


def comparison(d=None):
    d = d or dossier()
    ref = {'measurement_id': d.measurements[0]['id'], 'items': ['A']}
    return {'comparison': {'candidates': ['Identity A', 'Identity B'], 'broader_label': 'Shared compartment',
        'observations': [{'id':'o1','refs':[ref], 'observation':'Recorded distribution',
            'interpretation': {'expectation':'Pattern expected in A more than B',
                'applicability':'Recorded distribution in the relevant population',
                'limitation':'Fine boundaries remain uncertain'},
            'effect':'favors','favors':['Identity A'],'assumptions':[]}],
        'assumptions': [], 'selection': {'label':'Identity A','basis':['o1'],'tradeoffs':{},
            'review': {'strongest_alternative':'Identity B','decisive_observations':['o1'],'remaining_issue':None}}},
        'status':'best_effort','uncertainty':'Fine boundary remains uncertain', 'pending_checks':[], 'next_question':''}


def validate(v,d=None):
    return validate_annotation(v,d or dossier(),require_checks=True,require_comparison=True)


def test_final_explanation_is_derived_and_cannot_add_another_claim():
    v=comparison();n=normalize_annotation(v)
    assert not validate(v) and not validate(n)
    assert n['label']=='Identity A' and 'Recorded distribution' in n['reason']
    assert normalize_annotation(n)==n
    for key,val in [('label','Identity B'),('reason','Invented rescue claim'),('support',[])]:
        altered={**v,key:val}
        assert any('derived' in error for error in validate(altered))


def test_shared_evidence_cannot_choose_specific_candidate_but_can_choose_broad():
    v=comparison();c=v['comparison'];r=c['observations'][0]
    r.update(effect='shared',favors=[])
    assert validate(v)
    c['selection']['label']=c['broader_label']
    c['selection']['review']['decisive_observations']=[]
    assert not validate(v)
    assert normalize_annotation(v)['label']=='Shared compartment'


def test_real_opposition_requires_existing_support_not_a_rescue_sentence():
    v=comparison();c=v['comparison'];other=deepcopy(c['observations'][0])
    other.update(id='o2',observation='Contrary measured distribution',favors=['Identity B'])
    other['interpretation']['expectation']='Pattern expected in B more than A'
    c['observations'].append(other)
    assert any('tradeoffs' in error for error in validate(v))
    c['selection']['tradeoffs']={'o2':['expression can vary']}
    assert validate(v)
    c['selection']['tradeoffs']={'o2':['o1']}
    assert not validate(v)
    rendered=normalize_annotation(v)
    assert rendered['counterevidence'] and 'Contrary measured distribution' in rendered['reason']


def test_unverified_rescue_cannot_be_necessary_support_even_with_check():
    v=comparison();c=v['comparison'];c['assumptions']=[
        {'id':'a1','claim':'An unverified rescue premise','evidence':[],'check':pending_check()}]
    c['observations'][0]['assumptions']=['a1']
    assert any('unverified' in e for e in validate(v))
    # The premise can be retained as a question without determining the winner.
    c['observations'][0]['assumptions']=[]
    assert not validate(v)
    n=normalize_annotation(v)
    assert n['pending_checks']==[pending_check()]
    assert n['next_question']==pending_check()['question'] and n['status']=='best_effort'


def test_same_observation_cannot_be_shared_and_discriminating():
    v=comparison();c=v['comparison'];other=deepcopy(c['observations'][0])
    other.update(id='o2',effect='shared',favors=[])
    c['observations'].append(other)
    assert any('contradictory' in e for e in validate(v))


def test_changed_prior_effect_requires_explicit_revision():
    d=dossier();before=comparison(d);before['comparison']['observations'][0].update(effect='shared',favors=[])
    before['comparison']['selection']['label']='Shared compartment'
    before['comparison']['selection']['review']['decisive_observations']=[]
    d.critic_decision=normalize_annotation(before)
    after=comparison(d)
    assert any('revision' in e for e in validate(after,d))
    after['comparison']['observations'][0]['revision']='Prior comparison overlooked the recorded boundary'
    assert not validate(after,d)


def test_all_observation_and_assumption_refs_are_validated_even_when_unused():
    v=comparison();c=v['comparison']
    c['assumptions']=[{'id':'a1','claim':'A claim','evidence':[{'measurement_id':'invented','items':['x']}]}]
    assert validate(v)
    c['assumptions']=[]
    r=deepcopy(c['observations'][0]);r.update(id='o2',effect='uninformative',favors=[],refs=[{'measurement_id':'invented','items':['x']}])
    c['observations'].append(r)
    assert validate(v)


def test_comparison_expression_scope_is_enforced():
    d,old=scoped_dossier();v=comparison(d)
    v['comparison']['observations'][0]['refs']=old['support']
    assert not validate(v,d)
    v['comparison']['observations'][0]['refs'][0]['cell_type']='Wrong population'
    assert any('Scope mismatch' in e for e in validate(v,d))


def test_actor_plan_survives_measurements_and_pending_rounds(monkeypatch):
    first=comparison();first['pending_checks']=[pending_check()]
    final=comparison()
    actor=deepcopy(ACTOR);actor['comparison_plan']={'candidates':['Identity A','Identity B'],
        'shared':['Generic program'], 'tests':[{'question':'Differentiate A/B','predictions':[
            {'candidate':'Identity A','supports':'Signal in A','against':'Opposite signal'}]}]}
    node,state,calls,_=harness(monkeypatch,[actor,first,final])
    state.pop('spatial_resolution_comparison');state['spatial_comparison_review']=True
    actions=[];monkeypatch.setattr(node,'_act',lambda s,d,a:actions.append(a))
    result=node.execute(state)
    assert result['annotation']=='Identity A' and result['dossier_agent_steps']==2
    assert [kw['power'] for _,kw in calls]==['medium','strong','strong']
    assert 'Signal in A' in calls[1][0] and 'Signal in A' in calls[2][0]
    assert actions[-1]==pending_check()['action']
    assert result['niche_dossier']['critic_decision']['comparison']==final['comparison']


def test_logical_comparison_failure_is_not_repaired_as_formatting(monkeypatch):
    bad=comparison();bad['comparison']['selection']['label']='Identity B'
    node,state,calls,_=harness(monkeypatch,[bad])
    state.update(spatial_resolution_comparison=False,spatial_comparison_review=True)
    verdict,attempts=node.review(state,dossier(),[])
    assert verdict is None and len(calls)==1
    assert attempts[0]['validation_errors']


def test_cli_normalizes_comparison_and_commit_keeps_it(spatial_run):
    s=spatial_run;d=dossier();s.write_dossier_dict('0',d.to_dict());v=comparison(d)
    assert cli(s,'dossier','decision','--cluster','0','--json',json.dumps(v))==0
    saved=s.read_dossier_dict('0')['critic_decision']
    assert saved['label']=='Identity A' and saved['comparison']==v['comparison']
    assert cli(s,'dossier','commit','--cluster','0')==0
    assert s.read_result('0')['label']=='Identity A'


def test_comparison_citation_id_repair_cannot_rewrite_biology(monkeypatch):
    good=comparison();bad=deepcopy(good)
    bad['comparison']['observations'][0]['refs'][0]['measurement_id']='invented'
    node,state,calls,_=harness(monkeypatch,[bad,good])
    state.update(spatial_resolution_comparison=False,spatial_comparison_review=True)
    value,attempts=node.review(state,dossier(),[])
    assert value['label']=='Identity A' and len(calls)==2
    changed=deepcopy(good);changed['comparison']['observations'][0]['interpretation']['applicability']='New biological premise'
    node,state,calls,_=harness(monkeypatch,[bad,changed])
    state.update(spatial_resolution_comparison=False,spatial_comparison_review=True)
    value,attempts=node.review(state,dossier(),[])
    assert value is None and 'Format repair changed the biological comparison' in attempts[-1]['validation_errors']


def test_exact_inline_assumptions_are_lossless_but_changed_premises_rejected():
    v=comparison();c=v['comparison'];a={'id':'a1','claim':'Unverified distinction','evidence':[],'check':None}
    c['assumptions']=[a]
    other=deepcopy(c['observations'][0]);other.update(id='o2',observation='Another observation',effect='uninformative',favors=[],assumptions=[deepcopy(a)])
    c['observations'].append(other)
    assert not validate(v)
    assert normalize_annotation(v)['comparison']['observations'][1]['assumptions']==['a1']
    other['assumptions'][0]['claim']='Different premise'
    assert validate(v)


def test_shared_context_is_allowed_beside_a_real_declared_discriminator():
    v=comparison();c=v['comparison'];r=deepcopy(c['observations'][0])
    r.update(id='context',observation='Shared general program',effect='shared',favors=[])
    c['observations'].append(r);c['selection']['basis'].append('context')
    assert not validate(v)
    c['observations'][0].update(effect='shared',favors=[])
    assert validate(v)


def test_rejected_comparison_is_reconsidered_within_existing_budget(monkeypatch):
    bad=comparison();bad['comparison']['selection']['label']='Identity B'
    good=comparison()
    node,state,calls,_=harness(monkeypatch,[ACTOR,bad,good],rounds=2)
    state.update(spatial_resolution_comparison=False,spatial_comparison_review=True)
    monkeypatch.setattr(node,'_act',lambda *a:None)
    result=node.execute(state)
    assert result['annotation']=='Identity A' and result['dossier_agent_steps']==2
    assert len(calls)==3 and 'rejected_proposal' in calls[-1][0]
    assert result['dossier_agent_transcript'][1]['attempts'][0]['validation_errors']
    assert not any(e.get('source')=='comparison_feedback' for e in result['niche_dossier']['evidence_log'])


def test_persistent_logical_failure_finalizes_with_provisional_output(monkeypatch):
    bad=comparison();bad['comparison']['selection']['label']='Identity B'
    node,state,calls,labels=harness(monkeypatch,[ACTOR,bad,bad,flat_decision()],rounds=2)
    state.update(spatial_resolution_comparison=False,spatial_comparison_review=True)
    monkeypatch.setattr(node,'_act',lambda *a:None)
    result=node.execute(state)
    assert result['dossier_agent_steps']==2 and result['dossier_stop_reason']=='best_effort_finalized'
    assert labels['0']=='Compartment A' and len(calls)==4
    assert result['annotation_status']=='best_effort'


@pytest.mark.parametrize('field,bad',[('effect',{}),('assumptions',[{}]),('refs',None),('observation',None)])
def test_malformed_comparison_fields_return_errors(field,bad):
    v=comparison();v['comparison']['observations'][0][field]=bad
    assert validate(v)


def test_compiled_graph_uses_comparison_workflow(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from sclambda.core.task_runner import TaskRunner
    from sclambda.pipelines.annotate_spatial.pipeline import AnnotateSpatialPipeline
    from sclambda.pipelines.annotate_spatial.nodes import ComputeSpatialProportionsNode,SpatialImageDescriptionNode,SpatialCellTypeDENode
    d=dossier()
    monkeypatch.setattr(ComputeSpatialProportionsNode,'execute',lambda self,state:{'niche_dossier':d.to_dict()})
    monkeypatch.setattr(SpatialImageDescriptionNode,'execute',lambda *a:{})
    monkeypatch.setattr(SpatialCellTypeDENode,'execute',lambda *a:{})
    replies=[ACTOR,comparison(d)]
    def query(prompt,**kw):
        if kw['power']=='strong': assert 'comparison IS the decision' in prompt
        return SimpleNamespace(content=json.dumps(replies.pop(0)),failed_after_retries=False)
    acc=SimpleNamespace(gene_names=np.array(['GENE1']),set_annotation=lambda *a,**kw:None,set_hypotheses=lambda *a:None)
    pipeline=AnnotateSpatialPipeline(acc,SimpleNamespace(query=query))
    with TaskRunner(max_workers=1) as runner:
        state=pipeline.build_graph(runner).invoke({'cluster_id':'0','dossier_agent_max_steps':1,'spatial_comparison_review':True})
    assert state['annotation']=='Identity A' and not replies
    assert 'comparison' in state['niche_dossier']['critic_decision']


def test_known_nonexpression_item_key_is_normalized_without_changing_claim():
    d=dossier();v=comparison(d)
    ref=v['comparison']['observations'][0]['refs'][0]
    ref['genes']=ref.pop('items')
    assert not validate(v,d)
    n=normalize_annotation(v,d)
    assert n['support'][0]['items']==['A'] and 'genes' not in n['support'][0]
    assert n['comparison']['observations'][0]['interpretation']==v['comparison']['observations'][0]['interpretation']
    ref['genes']=['Not in the table']
    assert validate(v,d)


def test_new_observation_reusing_an_id_is_not_a_changed_old_observation():
    d=dossier();d.critic_decision=normalize_annotation(comparison(d))
    v=comparison(d);r=v['comparison']['observations'][0]
    r.update(observation='A different measured observation',effect='shared',favors=[])
    v['comparison']['selection']['label']='Shared compartment'
    v['comparison']['selection']['review']['decisive_observations']=[]
    assert not validate(v,d)


def test_renaming_same_observation_does_not_bypass_revision_check():
    d=dossier();d.critic_decision=normalize_annotation(comparison(d))
    v=comparison(d);r=v['comparison']['observations'][0]
    r.update(id='renamed',effect='shared',favors=[])
    v['comparison']['selection'].update(label='Shared compartment',basis=['renamed'])
    v['comparison']['selection']['review']['decisive_observations']=[]
    assert any('revision' in e for e in validate(v,d))


def test_a_changed_candidate_comparison_can_change_evidence_direction():
    d=dossier();d.critic_decision=normalize_annotation(comparison(d))
    v=comparison(d);v['comparison']['candidates']=['Identity A','Identity C']
    v['comparison']['observations'][0].update(effect='shared',favors=[])
    v['comparison']['selection']['label']='Shared compartment'
    v['comparison']['selection']['review'].update(strongest_alternative='Identity C',decisive_observations=[])
    assert not validate(v,d)


def test_shared_context_alone_cannot_outweigh_discriminating_counterevidence():
    v=comparison();c=v['comparison']
    shared=deepcopy(c['observations'][0]);shared.update(id='shared',observation='Shared program',effect='shared',favors=[])
    contrary=deepcopy(c['observations'][0]);contrary.update(id='against',observation='Contrary discriminator',favors=['Identity B'])
    c['observations'] += [shared,contrary]
    c['selection'].update(basis=['o1','shared'],tradeoffs={'against':['shared']})
    assert any('shared context alone' in e for e in validate(v))
    c['selection']['tradeoffs']={'against':['o1','shared']}
    assert not validate(v)


@pytest.mark.parametrize('interpretation', [
    'Compatible with A', {},
    {'expectation':'A more than B','applicability':'','limitation':'Unknown'},
])
def test_discriminator_needs_a_comparative_interpretation(interpretation):
    v=comparison();v['comparison']['observations'][0]['interpretation']=interpretation
    assert any('expectation, applicability and limitation' in e for e in validate(v))


def test_review_cannot_declare_shared_or_wrong_alternative_evidence_decisive():
    v=comparison();c=v['comparison'];r=deepcopy(c['observations'][0])
    r.update(id='shared',observation='Shared program',effect='shared',favors=[],interpretation='Shared context')
    c['observations'].append(r);c['selection']['basis'].append('shared')
    c['selection']['review']['decisive_observations']=['shared']
    assert any('distinguish' in e for e in validate(v))
    c['selection']['review']['decisive_observations']=['o1']
    c['candidates'].append('Identity C')
    c['observations'][0]['favors'].append('Identity B')
    assert any('distinguish' in e for e in validate(v))
    c['selection']['review']['strongest_alternative']='Identity C'
    assert not validate(v)


def test_remaining_issue_retains_label_without_requiring_a_tool_or_claiming_resolution():
    v=comparison();v.update(status='resolved',uncertainty='')
    v['comparison']['selection']['review']['remaining_issue']='Unavailable distinction could reverse A versus B'
    assert not validate(v)
    n=normalize_annotation(v)
    assert n['label']=='Identity A' and n['status']=='best_effort'
    assert not n['pending_checks'] and not n['next_question']
    assert 'could reverse' in n['uncertainty']
    assert normalize_annotation(n)==n
    v['comparison']['selection']['review']['remaining_issue']=None
    v['uncertainty']='An ordinary limitation that does not undermine the choice'
    assert not validate(v) and normalize_annotation(v)['status']=='resolved'


@pytest.mark.parametrize('review', [None, {},
    {'strongest_alternative':'Identity A','decisive_observations':['o1'],'remaining_issue':None},
    {'strongest_alternative':'Identity B','decisive_observations':['invented'],'remaining_issue':None},
    {'strongest_alternative':'Identity B','decisive_observations':['o1'],'remaining_issue':False},
])
def test_invalid_review_is_rejected(review):
    v=comparison();v['comparison']['selection']['review']=review
    assert validate(v)


def test_expression_facts_come_from_records_not_model_transcription(spatial_run):
    d,old=scoped_dossier();v=comparison(d);r=v['comparison']['observations'][0]
    r['refs']=old['support'];r['observation']='Invented measured LFC=999 and proportion=1'
    n=normalize_annotation(v,d)
    assert not validate(v,d) and not validate(n,d)
    assert '999' not in n['reason'] and '999' in n['comparison']['observations'][0]['observation']
    assert '"lfc": -0.2' in n['reason'] and '"proportion": 0.2' in n['reason']
    assert 'Population A' in n['reason'] and 'same cell type outside niche' in n['reason']
    assert 'significance_kind' in n['reason']
    assert normalize_annotation(n,d)==n
    s=spatial_run;s.write_dossier_dict('0',d.to_dict())
    assert cli(s,'dossier','decision','--cluster','0','--json',json.dumps(v))==0
    assert cli(s,'dossier','commit','--cluster','0')==0
    assert '999' not in s.read_dossier_dict('0')['critic_decision']['reason']


def test_unavailable_decisive_question_finishes_with_best_effort_label(monkeypatch):
    v=comparison();v.update(status='resolved',uncertainty='')
    v['comparison']['selection']['review']['remaining_issue']='The decisive alternative cannot be tested in this panel'
    node,state,calls,_=harness(monkeypatch,[ACTOR,v],rounds=4)
    state.update(spatial_resolution_comparison=False,spatial_comparison_review=True)
    monkeypatch.setattr(node,'_act',lambda *a:None)
    result=node.execute(state)
    assert result['annotation']=='Identity A' and result['annotation_status']=='best_effort'
    assert result['dossier_stop_reason']=='critic_complete' and len(calls)==2


def test_malformed_complete_response_gets_bounded_fresh_critic_review(monkeypatch):
    from types import SimpleNamespace
    v=comparison()
    node,state,calls,_=harness(monkeypatch,[ACTOR],rounds=2)
    state.update(spatial_resolution_comparison=False,spatial_comparison_review=True)
    actor_query=node.llm_client.query
    seen=[]
    def query(prompt,**kwargs):
        if kwargs['power']=='medium': return actor_query(prompt,**kwargs)
        seen.append(prompt)
        content=json.dumps(v)[:-1] if len(seen)==1 else json.dumps(v)
        return SimpleNamespace(content=content,failed_after_retries=False,stop_reason='end_turn')
    monkeypatch.setattr(node.llm_client,'query',query)
    monkeypatch.setattr(node,'_act',lambda *a:None)
    result=node.execute(state)
    assert result['annotation']=='Identity A' and result['dossier_agent_steps']==2
    assert len(seen)==2 and 'rejected_response' in seen[-1]
    attempts=result['dossier_agent_transcript'][1]['attempts']
    assert attempts[0]['proposal'] is None and attempts[0]['validation_errors']


@pytest.mark.parametrize('failed,stop', [(False,'max_tokens'),(True,None)])
def test_truncation_or_transport_failure_does_not_trigger_biological_retry(monkeypatch,failed,stop):
    from types import SimpleNamespace
    node,state,calls,_=harness(monkeypatch,[ACTOR],rounds=4)
    state.update(spatial_resolution_comparison=False,spatial_comparison_review=True)
    actor_query=node.llm_client.query;seen=[]
    def query(prompt,**kwargs):
        if kwargs['power']=='medium': return actor_query(prompt,**kwargs)
        seen.append(prompt)
        return SimpleNamespace(content=json.dumps(comparison())[:-1],failed_after_retries=failed,stop_reason=stop)
    monkeypatch.setattr(node.llm_client,'query',query)
    monkeypatch.setattr(node,'_act',lambda *a:None)
    result=node.execute(state)
    assert result['dossier_stop_reason']=='inference_failed' and result['dossier_agent_steps']==1
    assert result['annotation']=='Spatial niche (anatomy unassigned)'
    assert len(seen)==3


def test_exact_unique_source_passage_is_normalized_without_an_llm_repair():
    from sclambda.annotation.niche.measurements import identify
    d=dossier();m=identify({'kind':'literature','rows':[{'item':'SOURCE:1:passage1','text':'Recorded source text'}]})
    d.measurements.append(m);v=comparison(d)
    row=v['comparison']['observations'][0];row['refs']=[{'source_id':'SOURCE:1:passage1'}]
    assert not validate(v,d)
    n=normalize_annotation(v,d)
    assert n['support']==[{'measurement_id':m['id'],'items':['SOURCE:1:passage1']}]
    assert n['comparison']['observations'][0]['interpretation']==row['interpretation']
    assert row['refs']==[{'source_id':'SOURCE:1:passage1'}]
    # Repeated retrieval under another measurement is ambiguous, even if the
    # passage text agrees. Never choose a source context on the model's behalf.
    d.measurements.append(identify({**m,'query':'different retrieval'}))
    assert validate(v,d)


@pytest.mark.parametrize('ref', [
    {'source_id':'SOURCE:1'}, {'source_id':'invented'},
    {'source_id':'SOURCE:1:passage1','items':['different claim']},
])
def test_bare_paper_or_changed_source_reference_is_not_inferred(ref):
    d=dossier();d.measurements.append({'id':'lit','kind':'literature',
        'rows':[{'item':'SOURCE:1:passage1','text':'Recorded source text'}]})
    v=comparison(d);v['comparison']['observations'][0]['refs']=[ref]
    assert validate(v,d)


def flat_decision():
    value=decision()
    value.pop('resolution')
    return value


def test_conflicting_observation_can_be_weighed_without_inventing_a_rival():
    v=comparison();c=v['comparison'];r=deepcopy(c['observations'][0])
    r.update(id='o2',observation='Ambiguous boundary',effect='conflicting',favors=[])
    c['observations'].append(r);c['selection']['tradeoffs']={'o2':['o1']}
    assert not validate(v)
    c['selection']['tradeoffs']['nonexistent']=['o1']
    assert any('nonexistent' in e for e in validate(v))
    del c['selection']['tradeoffs']['nonexistent']
    c['selection']['tradeoffs']['o2']=['nonexistent']
    assert validate(v)


def test_tradeoff_error_names_missing_and_invalid_observations():
    v=comparison();c=v['comparison'];r=deepcopy(c['observations'][0])
    r.update(id='rival',observation='Opposing pattern',favors=['Identity B'])
    c['observations'].append(r);c['selection']['tradeoffs']={'o1':['o1']}
    errors=validate(v)
    assert any("missing opposing observation IDs ['rival']" in e for e in errors)
    assert any("non-opposing/non-conflicting IDs ['o1']" in e for e in errors)


def test_citation_failure_gets_substantive_review_after_repair_rejected(monkeypatch):
    bad=comparison();bad['comparison']['observations'][0]['refs'][0]['items']=['invented']
    changed=comparison();changed['comparison']['observations'][0]['observation']='Corrected observation'
    node,state,calls,_=harness(monkeypatch,[ACTOR,bad,changed,changed],rounds=2)
    state.update(spatial_resolution_comparison=False,spatial_comparison_review=True)
    result=node.execute(state)
    assert result['annotation']=='Identity A' and result['dossier_review_complete']
    assert len(calls)==4 and 'original_attempt' in calls[-1][0]
    assert 'Format repair changed' in str(result['dossier_agent_transcript'][1]['attempts'])


def test_final_review_rejects_wrong_population_and_allows_revised_best_effort(monkeypatch):
    d,old=scoped_dossier();good=deepcopy(old);good.pop('resolution',None)
    good.update(status='best_effort',uncertainty='Fine identity uncertain',pending_checks=[])
    bad=deepcopy(good);bad['support'][0]['cell_type']='Wrong population'
    node,state,calls,_=harness(monkeypatch,[bad,good])
    value,attempts=node.final_review(state,d,[],[])
    assert value['status']=='best_effort' and value['output_origin']=='final_critic'
    assert any('Scope mismatch' in e for e in attempts[0]['validation_errors'])
    assert not attempts[-1]['validation_errors'] and len(calls)==2


def test_failed_actor_action_can_be_corrected_by_critic_pending_check(monkeypatch):
    first=comparison();first['pending_checks']=[pending_check()]
    node,state,calls,_=harness(monkeypatch,[ACTOR,first,comparison()],rounds=2)
    state.update(spatial_resolution_comparison=False,spatial_comparison_review=True)
    actions=[]
    def act(s,d,a):
        actions.append(a)
        if len(actions)==1:raise ValueError('Missing tool argument')
    monkeypatch.setattr(node,'_act',act)
    result=node.execute(state)
    assert result['dossier_stop_reason']=='critic_complete' and len(actions)==2
    assert actions[1]==pending_check()['action'] and len(calls)==3


def test_lookup_reference_accepts_query_list_alias_without_changing_queries(monkeypatch):
    node,state,_,_=harness(monkeypatch,[]);calls=[]
    monkeypatch.setattr(node,'_reference',lambda s,d,q:calls.append(q))
    queries=[{'query':'Question A','terms':['A']},{'query':'Question B','terms':['B']}]
    node._act(state,dossier(),{'tool':'lookup_reference','queries':queries})
    assert calls==queries
    with pytest.raises(ValueError,match='Conflicting'):
        node._act(state,dossier(),{'tool':'lookup_reference','queries':queries,'reference_queries':[]})


def test_scalar_reference_is_only_normalized_for_exact_existing_item():
    v=comparison();ref=v['comparison']['observations'][0]['refs'][0]
    ref['items']='A'
    assert not validate(v)
    assert normalize_annotation(v,dossier())['support'][0]['items']==['A']
    ref['items']='A, B'
    assert validate(v)


def test_failure_record_exports_without_claiming_success(monkeypatch):
    from sclambda.annotation.decisions import build_commit
    from sclambda.annotation.niche.dossier import NicheDossier
    node,state,_,_=harness(monkeypatch,[ACTOR,{}, {}, {}],rounds=1)
    result=node.execute(state)
    exported=build_commit(NicheDossier.from_dict(result['niche_dossier']))
    assert exported['label']==result['annotation']
    assert exported['output_origin']=='failure_record'
    assert exported['execution_status']=='inference_failed'
    assert exported['confidence']=='uncertain' and not result['dossier_review_complete']


def test_final_review_cannot_mark_uncertain_recovery_resolved(monkeypatch):
    bad=flat_decision();bad['status']='resolved'
    node,state,_,_=harness(monkeypatch,[bad,flat_decision()])
    value,attempts=node.final_review(state,dossier(),[],[])
    assert value['status']=='best_effort'
    assert 'Final recovery must remain best_effort' in attempts[0]['validation_errors']


def test_revised_citation_can_request_measurement_before_final_decision(monkeypatch):
    bad=comparison();bad['comparison']['observations'][0]['refs'][0]['items']=['unmeasured']
    request=comparison();request['pending_checks']=[pending_check()]
    node,state,calls,_=harness(monkeypatch,[ACTOR,bad,bad,request,comparison()],rounds=3)
    state.update(spatial_resolution_comparison=False,spatial_comparison_review=True)
    actions=[]
    monkeypatch.setattr(node,'_act',lambda s,d,a:actions.append(a))
    result=node.execute(state)
    assert result['dossier_stop_reason']=='critic_complete' and result['dossier_agent_steps']==3
    assert actions==[ACTOR['action'],pending_check()['action']]
    assert len(calls)==5


def test_broad_label_may_discuss_ambiguous_evidence_with_shared_support():
    v=comparison();c=v['comparison'];c['observations'][0].update(effect='shared',favors=[])
    c['selection']['label']=c['broader_label'];c['selection']['review']['decisive_observations']=[]
    r=deepcopy(c['observations'][0]);r.update(id='o2',observation='Ambiguous spatial boundary',effect='conflicting')
    c['observations'].append(r);c['selection']['tradeoffs']={'o2':['o1']}
    assert not validate(v)
    assert normalize_annotation(v)['label']==c['broader_label']
