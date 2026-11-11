"""Regression cases for measurement handling and the blood-census trace faults."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from sclambda.annotation.celltype.measurements import (
    record, render, add_measurement, availability, validate_verdict, evidence_fingerprint,
    assessment_status, MeasurementText,
)
from sclambda.annotation.decisions import apply_critic_verdict, build_commit
from sclambda.annotation.celltype.dossier import CellTypeDossier
from sclambda.analysis.targeted_de import GeneDE, GenesDEResult
from sclambda.pipelines.celltype.nodes.agent import CTDossierAgentNode
from test_critic_resolution import proposal, axis, agent_harness


def test_tables_keep_zero_values_proportions_and_significance_with_reference():
    gene = GeneDE('MARKER', 0., .4, -.4, .003, 12, 25, frac_non_zero_target=0.)
    value = record([gene], target='A', reference='parent excluding A', method='ttest')
    assert value['rows'][0] == {'gene': 'MARKER', 'lfc': -.4, 'proportion': 0., 'significance': .003}
    assert 'parent excluding A' in render(value) and 'raw_p' in render(value)
    assert '| MARKER | -0.4 | 0 | 0.003 |' in render(value)
    other = record([gene], target='A', reference='all other cells', method='ttest')
    assert other['id'] != value['id']
    d = proposal()
    add_measurement(d, value); add_measurement(d, value)
    assert len(d.measurements) == 2
    assert CellTypeDossier.from_dict(d.to_dict()).measurements == d.measurements


def test_availability_uses_data_and_not_model_claims_or_top_list_omission():
    d = proposal()
    d.requested_genes = ['ABSENT', 'GENE2']
    d.apply_update({'measurements': [], 'requested_genes': [], 'traits': [axis('invented')],
                    'confirmed': ['GENE2 unavailable']})
    assert availability(d, {'GENE1', 'GENE2'}) == {
        'measured': ['GENE1'], 'available_unmeasured': ['GENE2'], 'unavailable_requested': ['ABSENT']}
    assert not d.traits




def test_trait_survives_unresolved_finer_identity_and_roundtrip():
    d = proposal()
    payload = {'axes': {'lineage': axis('Broad lineage'), 'cell_type': {'resolved': False}},
               'traits': [axis('isotype-X')]}
    checked, issues = validate_verdict(payload, d)
    assert not issues
    apply_critic_verdict(d, checked)
    assert d.cell_type == 'Broad lineage (isotype-X)'
    assert d.committed_granularity == 'lineage'
    restored = CellTypeDossier.from_dict(d.to_dict())
    assert restored.cell_type == d.cell_type
    assert build_commit(restored)['supported_traits'][0]['label'] == 'isotype-X'
    # Reopening identity cannot turn an independent trait into an identity.
    apply_critic_verdict(d, {'axes': {'lineage': {'resolved': False}}})
    assert d.cell_type is None and d.traits




@pytest.mark.parametrize('field,value', [
    ('measurement_refs', [{'measurement_id': 'invented', 'genes': ['GENE1']}]),
    ('measurement_refs', [{'measurement_id': 'm_test', 'genes': ['UNMEASURED']}]),
    ('measurement_refs', []), ('counterevidence', None), ('reconciliation', ''),
    ('counterevidence', [{'measurement_id': 'm_test', 'genes': ['UNMEASURED'], 'interpretation': 'conflict'}]),
])
def test_critic_cannot_accept_without_real_citations_and_reconciliation(field, value):
    d = proposal()
    original = {'axes': {'lineage': {**axis('Broad'), field: value}}}
    before = deepcopy(original)
    checked, issues = validate_verdict(original, d)
    assert issues and not checked['axes']['lineage']['resolved']
    assert original == before  # raw response remains intact for diagnosis


def test_unchanged_measurements_need_explicit_revision_not_merely_finalization():
    d = proposal()
    apply_critic_verdict(d, {'axes': {'lineage': {**axis('Broad'), 'resolved': False, 'note': 'competing identity still fits'}}})
    fingerprint = evidence_fingerprint(d)
    verdict = {'axes': {'lineage': axis('Broad')}}
    checked, issues = validate_verdict(verdict, d, previous_measurements=fingerprint)
    assert issues and not checked['axes']['lineage']['resolved']
    verdict['axes']['lineage']['revision_reason'] = 'Earlier interpretation confused a shared state with a competing lineage; measured machinery distinguishes it.'
    checked, issues = validate_verdict(verdict, d, previous_measurements=fingerprint)
    assert not issues and checked['axes']['lineage']['resolved']


def test_new_measurement_allows_resolution_without_no_data_revision_gate():
    d = proposal()
    apply_critic_verdict(d, {'axes': {'lineage': {'resolved': False}}})
    old = evidence_fingerprint(d)
    add_measurement(d, {**d.measurements[0], 'id': 'm_new'})
    _, issues = validate_verdict({'axes': {'lineage': axis('Broad')}}, d, previous_measurements=old)
    assert not issues


@pytest.mark.parametrize('content,stop', [('{"axes": {', None), ('{"axes": {}}', 'max_tokens'),
                                         ('```json\n{"axes":', None)])
def test_truncated_or_malformed_json_is_not_repaired_into_a_verdict(content, stop):
    assert CTDossierAgentNode._parse_turn(SimpleNamespace(content=content, stop_reason=stop)) is None


def test_complete_fenced_verdict_with_preface_is_not_an_inference_failure():
    text = 'Analysis of the measured evidence.\n\n```json\n{"axes": {"lineage": {"resolved": false}}}\n```'
    result = CTDossierAgentNode._parse_turn(SimpleNamespace(content=text, stop_reason='end_turn'))
    assert result == {'axes': {'lineage': {'resolved': False}}}


@pytest.mark.parametrize('content', [
    'Preface\n```json\n{"axes": {}',
    'Preface\n```json\n{"axes": {}\n```',
    '```json\n{"axes": {}}\n```\n```json\n{"axes": {"lineage": {}}}\n```',
    '```json\n{"axes": {}}\n```\n```json\n{"unfinished":',
])
def test_preface_handling_never_repairs_truncation_or_selects_between_verdicts(content):
    assert CTDossierAgentNode._parse_turn(SimpleNamespace(content=content, stop_reason='end_turn')) is None


def test_parent_comparison_excludes_target_and_preserves_reference(monkeypatch):
    node, _, _ = agent_harness(monkeypatch, [], [])
    node.accessor.get_cell_indices = lambda cid: np.arange(12) if cid == 'child' else np.arange(30)
    captured = []
    def compute(acc, genes, target, reference, **kwargs):
        captured.append((target, reference))
        return GenesDEResult([GeneDE('GENE1', 1., .4, .6, .01, 12, 18)], 12, 18)
    monkeypatch.setattr('sclambda.analysis.targeted_de.compute_de_for_genes', compute)
    result = node._measure_sibling({'cluster_id': 'child'}, 'parent', ['GENE1'])
    assert isinstance(result, MeasurementText)
    np.testing.assert_array_equal(captured[0][1], np.arange(12, 30))
    assert 'parent excluding target' in result and 'n=18' in result




def test_tool_failure_stays_visible_beside_structured_measurements():
    d = proposal()
    CTDossierAgentNode._append_measurement(d, 'neighbor', 'compare vs other: insufficient cells')
    block = CTDossierAgentNode._measurement_block(d)
    assert 'Measurement m_test' in block and 'insufficient cells' in block


def test_llm_transport_preserves_truncation_metadata_sync_and_async():
    import asyncio
    from sclambda.llm.api import LLMClient, Provider, Power
    client = LLMClient(provider='anthropic', max_retries=0)
    response = SimpleNamespace(content='{"partial":', usage_metadata={'output_tokens': 4096},
                               response_metadata={'stop_reason': 'max_tokens'})
    async def ainvoke(messages):
        return response
    llm = SimpleNamespace(invoke=lambda messages: response, ainvoke=ainvoke)
    sync = client._invoke_with_retry(llm, [], Provider.ANTHROPIC, Power.MEDIUM)
    async_result = asyncio.run(client._ainvoke_with_retry(llm, [], Provider.ANTHROPIC, Power.MEDIUM))
    assert sync.stop_reason == async_result.stop_reason == 'max_tokens'
    assert sync.output_tokens == 4096
    assert CTDossierAgentNode._parse_turn(sync) is None
