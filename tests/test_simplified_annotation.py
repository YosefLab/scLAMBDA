"""The simplified loop chooses a label without pretending uncertainty is resolved."""
import json
import threading
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from sclambda.annotation.celltype.dossier import CellTypeDossier
from sclambda.annotation.celltype.measurements import (
    MeasurementText, availability_block, validate_annotation,
)
from sclambda.annotation.decisions import build_commit
from sclambda.pipelines.celltype.nodes.agent import CTDossierAgentNode
from sclambda.pipelines.celltype.pipeline import route_after_ct_generation


def dossier():
    return CellTypeDossier(cluster_id='0', measurements=[{
        'id': 'm_test', 'target': '0', 'reference': 'whole dataset INCLUDING target',
        'method': 'ttest', 'significance_kind': 'adjusted_p',
        'rows': [{'gene': 'GENE1', 'lfc': 1., 'proportion': .8, 'significance': .001}],
    }])


def decision(status='resolved', question=''):
    return {'label': 'Chosen identity', 'status': status, 'reason': 'Dominant measured identity program; optional state is not required.',
            'support': [{'measurement_id': 'm_test', 'genes': ['GENE1']}],
            'counterevidence': [], 'alternatives': ['Alternative'] if status == 'best_effort' else [],
            'uncertainty': 'Fine distinction remains uncertain' if status == 'best_effort' else '',
            'next_question': question}


ACTOR = {'reasoning': 'Existing tables suffice', 'hypotheses': [{'label': 'Actor preference', 'reason': 'Provisional'}],
         'action': {'tool': 'finalize'}, 'annotation': {'label': 'Actor cannot assign'}}


def harness(monkeypatch, replies, rounds=4):
    node = object.__new__(CTDossierAgentNode)
    node._output_local = threading.local()
    node.output = {}
    node.accessor = SimpleNamespace(gene_names=np.array(['GENE1', 'GENE2']))
    calls = []
    def query(prompt, **kwargs):
        calls.append((prompt, kwargs))
        assert replies, 'Unexpected extra model call'
        response = replies.pop(0)
        if isinstance(response, Exception): raise response
        if isinstance(response, SimpleNamespace): return response
        return SimpleNamespace(content=json.dumps(response), failed_after_retries=False, stop_reason='end_turn')
    node.llm_client = SimpleNamespace(query=query)
    monkeypatch.setattr(node, '_peer_block', lambda state: '(none)')
    monkeypatch.setattr(node, 'log_info', lambda *a: None)
    monkeypatch.setattr(node, 'log_warning', lambda *a: None)
    monkeypatch.setattr(node, 'update_annotation', lambda cid, label: node.output.update(written_label=label))
    monkeypatch.setattr(node, 'update_hypotheses', lambda *a: None)
    state = {'cluster_id': '0', 'ct_dossier': dossier().to_dict(), 'dossier_agent_max_steps': rounds}
    return node, state, calls


@pytest.mark.parametrize('status', ['resolved', 'best_effort'])
def test_one_actor_one_critic_and_label_with_separate_uncertainty(monkeypatch, status):
    node, state, calls = harness(monkeypatch, [ACTOR, decision(status)])
    result = node.execute(state)
    assert result['cell_type'] == result['written_label'] == 'Chosen identity'
    assert result['annotation_assessment'] == status
    assert result['annotation_status'] == status
    assert result['dossier_review_complete']
    assert [kw['power'] for _, kw in calls] == ['medium', 'strong']
    assert 'claim_review' not in json.dumps(result)
    assert result['annotation_uncertainty'] == decision(status)['uncertainty']


def test_actor_cannot_assign_label_when_critic_fails(monkeypatch):
    node, state, calls = harness(monkeypatch, [ACTOR, {}, {}])
    result = node.execute(state)
    assert result['cell_type'] == 'Unresolved'
    assert result['annotation_assessment'] == 'inference_failed'
    assert len(calls) == 3  # one bounded critic correction, not another actor/reviewer


def test_reference_error_gets_same_critic_correction_without_new_measurements(monkeypatch):
    invalid = decision(); invalid['support'][0]['measurement_id'] = 'wrong_table'
    node, state, calls = harness(monkeypatch, [ACTOR, invalid, decision()])
    result = node.execute(state)
    assert result['cell_type'] == 'Chosen identity'
    assert [kw['power'] for _, kw in calls] == ['medium', 'strong', 'strong']
    assert 'Invalid support reference' in calls[-1][0]
    attempts = result['dossier_agent_transcript'][-1]['attempts']
    assert attempts[0]['validation_errors'] and not attempts[1]['validation_errors']
    assert len(result['ct_dossier']['measurements']) == 1


def test_last_round_selects_best_option_without_claiming_resolution(monkeypatch):
    initial = decision('best_effort', 'Test another distinguishing program')
    final = decision('best_effort', 'More data would help')
    node, state, calls = harness(monkeypatch, [ACTOR, initial, ACTOR, final], rounds=2)
    result = node.execute(state)
    assert result['cell_type'] == 'Chosen identity'
    assert result['annotation_assessment'] == 'best_effort'
    assert result['dossier_review_complete']
    assert result['dossier_stop_reason'] == 'budget_exhausted'
    assert len(calls) == 4
    assert 'FINAL REVIEW' in calls[-1][0]


def test_later_transport_failure_retains_previous_choice_and_reports_failure(monkeypatch):
    node, state, calls = harness(monkeypatch, [ACTOR, decision('best_effort', 'Test something'), ACTOR,
                                             RuntimeError('transport'), RuntimeError('transport')])
    result = node.execute(state)
    assert result['cell_type'] == 'Chosen identity'
    assert result['annotation_status'] == 'best_effort'
    assert result['annotation_assessment'] == 'inference_failed'
    assert not result['dossier_review_complete']


def test_failed_actor_still_allows_critic_to_use_real_existing_measurements(monkeypatch):
    node, state, calls = harness(monkeypatch, [RuntimeError('actor transport'), decision('best_effort')])
    result = node.execute(state)
    assert result['cell_type'] == 'Chosen identity'
    assert result['annotation_status'] == 'best_effort'
    assert result['dossier_agent_transcript'][0]['status'] == 'failed'


def test_best_effort_survives_roundtrip_commit_and_actor_update():
    d = dossier(); d.annotation = decision('best_effort')
    d.apply_update({'annotation': decision(), 'best_cell_type': 'Other', 'axes': {'lineage': {'resolved': True}}})
    restored = CellTypeDossier.from_dict(d.to_dict())
    result = build_commit(restored)
    assert result['label'] == 'Chosen identity' and result['confidence'] == 'uncertain'
    assert result['assessment_status'] == 'best_effort' and result['uncertainty']
    assert restored.to_hypotheses()[0]['state'] == 'best_effort'


def test_availability_prompt_does_not_enumerate_transcriptome():
    text = availability_block(dossier(), {f'GENE{i}' for i in range(20000)})
    assert len(text) < 500
    assert 'GENE19999' not in text and 'GENE1' in text
    assert json.loads(text)['available_unmeasured_count'] == 19999


@pytest.mark.parametrize('mutation', ['unmeasured', 'wrong_reference', 'empty_label', 'no_uncertainty'])
def test_best_effort_still_requires_real_evidence_and_honest_output(mutation):
    value = decision('best_effort')
    if mutation == 'unmeasured': value['support'][0]['genes'] = ['GENE2']
    if mutation == 'wrong_reference': value['support'][0]['measurement_id'] = 'missing'
    if mutation == 'empty_label': value['label'] = 'Unresolved'
    if mutation == 'no_uncertainty': value['uncertainty'] = ''
    assert validate_annotation(value, dossier())


def test_optional_explanatory_wording_has_no_automatic_identity_veto():
    value = decision()
    value['reason'] = 'Measured positive program supports identity; broad exclusion wording in the prior explanation was unnecessary.'
    assert not validate_annotation(value, dossier())


def test_dossier_initialization_routes_to_actor_without_generated_hypotheses():
    assert route_after_ct_generation({'use_dossier': True, 'hypotheses': [], 'ct_generation_failed': False}) == 'dossier_agent'


def test_overlapping_panels_only_measure_new_available_genes(monkeypatch):
    node, _, _ = harness(monkeypatch, [])
    d = dossier(); requests = []
    def measure(state, genes):
        requests.append(genes)
        return MeasurementText({'id': f'm_{len(requests)}', 'target': '0', 'reference': 'all other cells',
                                'method': 'ttest', 'significance_kind': 'raw_p',
                                'rows': [{'gene': g, 'lfc': 1., 'proportion': .5, 'significance': .01} for g in genes]})
    monkeypatch.setattr(node, '_measure_signature', measure)
    action = {'tool': 'test_signatures', 'signatures': [
        {'name': 'First', 'genes': ['GENE1', 'ABSENT']},
        {'name': 'Second', 'genes': ['GENE1', 'GENE2']},
        {'name': 'Repeated', 'genes': ['GENE2', 'GENE1']}]}
    node._test_signatures({}, d, action, {'GENE1', 'GENE2'}, set())
    assert requests == [['GENE1'], ['GENE2']]
    assert 'ABSENT' in d.requested_genes
    assert 'unavailable' in json.dumps(d.evidence_log)


def test_ontology_mapping_remains_available_without_cross_cluster_correction():
    from sclambda.pipelines.celltype.pipeline import CTPipeline
    from sclambda.core.task_runner import TaskRunner
    pipe = CTPipeline(accessor=SimpleNamespace(), llm_client=SimpleNamespace(), max_workers=1)
    pipe._use_dossier = True
    with TaskRunner(max_workers=1) as runner:
        assert pipe.build_post_graph(runner) is None
        graph = pipe.build_post_graph(runner, use_cell_ont=True)
        nodes = graph.get_graph().nodes
        assert 'ont_cell_type' in nodes and 'cross_cluster_correction' not in nodes


def test_explanatory_alternative_objects_do_not_trigger_another_review(monkeypatch):
    value = decision('best_effort')
    value['alternatives'] = [{'label': 'Alternative', 'reason': 'Weaker measured evidence'}]
    node, state, calls = harness(monkeypatch, [ACTOR, value])
    result = node.execute(state)
    assert result['annotation_status'] == 'best_effort'
    assert result['annotation_alternatives'] == value['alternatives']
    assert len(calls) == 2


def test_misplaced_gene_is_repaired_only_within_same_comparison_and_raw_is_preserved(monkeypatch):
    d = dossier()
    d.measurements.append({**deepcopy(d.measurements[0]), 'id': 'm_second',
                           'rows': [{'gene': 'GENE2', 'lfc': 2., 'proportion': .4, 'significance': .01}]})
    value = decision(); value['support'][0]['genes'] = ['GENE1', 'GENE2']
    node, state, calls = harness(monkeypatch, [ACTOR, value])
    state['ct_dossier'] = d.to_dict()
    result = node.execute(state)
    assert len(calls) == 2 and result['annotation_status'] == 'resolved'
    annotation = result['ct_dossier']['annotation']
    assert annotation['support'] == [{'measurement_id': 'm_test', 'genes': ['GENE1']},
                                     {'measurement_id': 'm_second', 'genes': ['GENE2']}]
    attempt = result['dossier_agent_transcript'][-1]['attempts'][0]
    assert attempt['proposal'] == value and attempt['reference_repairs'][0]['gene'] == 'GENE2'


@pytest.mark.parametrize('context_error', ['different_reference', 'different_method', 'ambiguous', 'unmeasured'])
def test_reference_repair_never_changes_comparison_or_invents_measurements(context_error):
    from sclambda.annotation.celltype.measurements import repair_annotation_refs
    d = dossier()
    other = {**deepcopy(d.measurements[0]), 'id': 'm_second',
             'rows': [{'gene': 'GENE2', 'lfc': 2., 'proportion': .4, 'significance': .01}]}
    if context_error == 'different_reference': other['reference'] = 'sibling'
    if context_error == 'different_method': other['method'] = 'ols'
    if context_error != 'unmeasured': d.measurements.append(other)
    if context_error == 'ambiguous': d.measurements.append({**deepcopy(other), 'id': 'm_third'})
    value = decision(); value['support'][0]['genes'] = ['GENE1', 'GENE2']
    fixed, changes = repair_annotation_refs(value, d)
    assert not changes and fixed == value and validate_annotation(fixed, d)
