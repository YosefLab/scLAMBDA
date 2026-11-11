"""Spatial decisions retain measurements and one authority across both frontends."""
from copy import deepcopy
import json
import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sclambda.annotation.decisions import build_commit
from sclambda.annotation.niche.dossier import NicheDossier
from sclambda.annotation.niche.measurements import (
    composition_record, expression_record, observation_record, validate_annotation,
)
from sclambda.cli.main import main
from sclambda.pipelines.annotate_spatial.nodes.agent import SpatialDossierAgentNode
from sclambda.storage.session import Session


def dossier():
    record = composition_record('0', [
        {'cell_type': 'A', 'n_cells': 24, 'proportion': .75, 'morans_z': None},
        {'cell_type': 'B', 'n_cells': 8, 'proportion': .25, 'morans_z': 1.2},
    ], {'expression_slides': 'all', 'spatial_slide': 's1'})
    return NicheDossier(cluster_id='0', measurements=[record])


def decision(d=None, label='Compartment A', question='', status='best_effort'):
    d = d or dossier()
    return dict(label=label, major_region='Broad compartment', status=status,
        resolution=dict(supported_label=label, fine_label=None, closest_alternative=None, fine_supported=False, comparison='No further distinction established.', discriminating_support=[]),
        reason='Composition and position favor the leading compartment; the fine boundary remains uncertain.',
        support=[{'measurement_id': d.measurements[0]['id'], 'items': ['A']}],
        counterevidence=[], alternatives=['Compartment B'], uncertainty='Fine boundary uncertain', next_question=question,
        pending_checks=[])


ACTOR = {'hypotheses': [{'label': 'Actor-only identity'}], 'action': {'tool': 'done'},
         'belief_delta': {'best_compartment': 'Actor-only identity'}, 'reasoning': 'Use existing evidence'}


def refinement(d=None, supported=False):
    d = d or dossier()
    value = decision(d, label='Shared compartment')
    value.pop('label')  # The model no longer writes a separate display label.
    value['resolution'].update(fine_label='Fine A', closest_alternative='Fine B',
        fine_supported=supported, comparison='Recorded position distinguishes A from B.' if supported else 'The observations are shared.',
        discriminating_support=value['support'] if supported else [])
    return value


@pytest.mark.parametrize('supported', [False, True])
def test_exported_label_follows_explicit_resolution(monkeypatch, supported):
    value = refinement(supported=supported)
    value['label'] = 'Independent contradictory label'
    node, state, _, labels = harness(monkeypatch, [ACTOR, value])
    result = node.execute(state)
    expected = 'Fine A' if supported else 'Shared compartment'
    assert result['annotation'] == labels['0'] == expected
    saved = result['niche_dossier']['critic_decision']
    assert saved['resolution'] == value['resolution']
    assert ('Fine A' in saved['alternatives']) is (not supported)
    assert 'Fine B' in saved['alternatives']
    assert result['dossier_agent_transcript'][1]['attempts'][0]['proposal']['label'] == value['label']
    assert build_commit(NicheDossier.from_dict(result['niche_dossier']))['label'] == expected


@pytest.mark.parametrize('change', [
    {'fine_supported': 'false'}, {'supported_label': ''}, {'comparison': ''},
    {'fine_supported': True, 'discriminating_support': []},
    {'fine_supported': True, 'closest_alternative': 'Fine A'},
    {'fine_supported': True, 'closest_alternative': None},
    {'supported_label': 'Fine A'},
    {'discriminating_support': [{'measurement_id': 'missing', 'items': ['A']}]},
])
def test_resolution_rejects_bad_shape_or_unrecorded_discriminators(change):
    value = refinement(supported=True)
    value['resolution'].update(change)
    assert validate_annotation(value, dossier(), require_resolution=True)


def test_legacy_decision_readable_but_new_model_review_requires_resolution():
    value = decision(); value.pop('resolution')
    assert not validate_annotation(value, dossier())
    assert validate_annotation(value, dossier(), require_resolution=True)


def harness(monkeypatch, replies, rounds=4):
    node = object.__new__(SpatialDossierAgentNode)
    node._output_local = threading.local()
    node.output = {}
    labels = {}
    node.accessor = SimpleNamespace(gene_names=np.array(['GENE1', 'GENE2']),
        set_annotation=lambda cid, label, **kw: labels.update({cid: label}), set_hypotheses=lambda *a: None)
    node.runner = SimpleNamespace(with_lock=lambda f, *a, **kw: f(*a, **kw))
    calls = []
    def query(prompt, **kw):
        calls.append((prompt, kw))
        assert replies, 'Unexpected model call'
        result = replies.pop(0)
        if isinstance(result, Exception): raise result
        if isinstance(result, SimpleNamespace): return result
        return SimpleNamespace(content=json.dumps(result), failed_after_retries=False, stop_reason='end_turn')
    node.llm_client = SimpleNamespace(query=query)
    node.log_info = node.log_warning = lambda *a: None
    state = {'cluster_id': '0', 'niche_dossier': dossier().to_dict(), 'dossier_agent_max_steps': rounds,
             'spatial_resolution_comparison': True, 'spatial_comparison_review': False,
             'cell_type_col': 'ct', 'slice_key': 'slide', 'cross_slide': True, 'selected_slide': 's1'}
    return node, state, calls, labels


def test_explicit_legacy_review_keeps_established_prompt_and_contract(monkeypatch):
    value = decision(); value.pop('resolution')
    node, state, calls, _ = harness(monkeypatch, [ACTOR, value])
    state.pop('spatial_resolution_comparison')
    result = node.execute(state)
    assert result['annotation'] == value['label']
    assert 'Decide identity and resolution together:' not in calls[-1][0]


@pytest.mark.parametrize('status', ['resolved', 'best_effort'])
def test_only_critic_assigns_and_uncertainty_is_preserved(monkeypatch, status):
    node, state, calls, labels = harness(monkeypatch, [ACTOR, decision(status=status)])
    result = node.execute(state)
    assert labels['0'] == result['annotation'] == 'Compartment A'
    assert result['annotation_assessment'] == status
    assert result['annotation_uncertainty'] == 'Fine boundary uncertain'
    assert result['dossier_review_complete']
    assert [kw['power'] for _, kw in calls] == ['medium', 'strong']
    d = NicheDossier.from_dict(result['niche_dossier'])
    assert build_commit(d)['major_region'] == 'Broad compartment'


@pytest.mark.parametrize('failure', [{}, RuntimeError('transport'),
    SimpleNamespace(content='{"label":"partial"}', failed_after_retries=False, stop_reason='max_tokens')])
def test_failed_or_truncated_critic_is_never_acceptance(monkeypatch, failure):
    node, state, calls, labels = harness(monkeypatch, [ACTOR, failure, failure, failure])
    result = node.execute(state)
    assert labels['0'] == result['annotation'] == 'Spatial niche (anatomy unassigned)'
    assert result['niche_dossier']['critic_decision']['output_origin'] == 'failure_record'
    assert result['annotation_assessment'] == 'inference_failed'
    assert not result['dossier_review_complete']
    assert len(calls) == 4


def test_last_measurement_is_seen_by_critic(monkeypatch):
    d = dossier()
    m = observation_record('0', 'A second view shows an interior location', {'spatial_slide': 's2'})
    final = decision(); final['support'] = [{'measurement_id': m['id'], 'items': ['observation']}]
    node, state, calls, labels = harness(monkeypatch, [ACTOR, final], rounds=1)
    monkeypatch.setattr(node, '_act', lambda state, dossier, action: dossier.measurements.append(m))
    result = node.execute(state)
    assert result['dossier_review_complete']
    assert 'interior location' in calls[-1][0] and 'FINAL REVIEW' in calls[-1][0]
    assert result['niche_dossier']['critic_decision']['support'][0]['measurement_id'] == m['id']


def test_concession_gets_review_and_transport_failure_keeps_previous_decision(monkeypatch):
    previous = decision(question='Inspect another slide')
    node, state, calls, labels = harness(monkeypatch, [ACTOR, previous, ACTOR, {}, {}, {}])
    result = node.execute(state)
    assert labels['0'] == previous['label']
    assert result['annotation_assessment'] == 'inference_failed'
    assert not result['dossier_review_complete']
    assert len(calls) == 6


def test_actor_updates_cannot_commit_identity():
    d = dossier()
    d.apply_update({'best_compartment': 'Unreviewed', 'candidates': [
        {'compartment': 'Unreviewed', 'state': 'supported', 'evidence_for': ['actor assertion']} ]})
    with pytest.raises(ValueError): build_commit(d)
    d.critic_decision = decision(d)
    d.apply_update({'best_compartment': 'Another', 'critic_decision': decision(label='Another')})
    assert build_commit(d)['label'] == 'Compartment A'


def test_multiple_signature_panels_use_distinct_cell_type_references(monkeypatch):
    import sclambda.pipelines.annotate_spatial.nodes.agent as module
    node, state, _, _ = harness(monkeypatch, [])
    seen = []
    monkeypatch.setattr(node, '_indices', lambda state, ct: (np.arange(12), np.arange(12, 24)))
    def compute(**kw):
        seen.append(kw['gene_names'])
        genes = [SimpleNamespace(name=g, lfc=1., frac_non_zero_target=.8, pval=.01) for g in kw['gene_names']]
        return SimpleNamespace(genes=genes, n_target=12, n_reference=12)
    monkeypatch.setattr(module, 'compute_de_for_genes', compute)
    d = dossier()
    action = {'tool': 'test_signatures', 'signatures': [
        {'cell_type': 'A', 'genes': ['GENE1', 'MISSING']}, {'cell_type': 'B', 'genes': ['GENE2']} ]}
    node._act(state, d, action)
    node._act(state, d, action)
    assert seen == [['GENE1'], ['GENE2']]
    assert [m['cell_type'] for m in d.measurements if m.get('kind') == 'expression'] == ['A', 'B']
    assert 'MISSING' in d.requested_genes
    assert all(m['scope']['expression_slides'] == 'all' for m in d.measurements[1:])


@pytest.mark.parametrize('expected_status', ['source_unavailable', 'unrecognized'])
def test_partial_gene_availability_reaches_first_and_global_critic(monkeypatch, tmp_path, expected_status):
    import sclambda.pipelines.annotate_spatial.nodes.agent as module
    node, state, calls, _ = harness(monkeypatch, [decision(), decision()])
    # Pin the source explicitly: a user's cached NCBI snapshot must not decide
    # which availability condition this regression exercises.
    alias_path = tmp_path / 'gene_info.tsv'
    if expected_status == 'unrecognized':
        alias_path.write_text('#tax_id\tGeneID\tSymbol\tLocusTag\tSynonyms\n9606\t1\tGENE1\t-\t-\n')
    state['gene_alias_path'] = str(alias_path)
    state['gene_alias_download'] = False
    monkeypatch.setattr(node, '_indices', lambda state, ct: (np.arange(12), np.arange(12, 24)))
    monkeypatch.setattr(module, 'compute_de_for_genes', lambda **kw: SimpleNamespace(
        genes=[SimpleNamespace(name='GENE1', lfc=0., frac_non_zero_target=0., pval=1.)],
        n_target=12, n_reference=12))
    d = dossier()
    action = {'tool': 'test_signatures', 'signatures': [
        {'cell_type': 'A', 'genes': ['GENE1', 'MISSING']}]}
    node._act(state, d, action)
    node._act(state, d, action)  # Reusing a measurement must not erase missingness.
    assert len(d.evidence_log) == 1
    assert 'signature in A' in d.evidence_log[0]['fact']
    assert d.measurements[-1]['rows'][0]['proportion'] == 0.
    assert all(r['gene'] != 'MISSING' for r in d.measurements[-1]['rows'])
    for final in (False, True):
        verdict, attempts = node.review(state, d, [], final=final)
        assert verdict and not attempts[-1]['validation_errors']
        prompt = calls[-1][0]
        assert '"requested": "MISSING"' in prompt and f'"status": "{expected_status}"' in prompt
        assert 'unresolved names are not expression zeros' in prompt
        assert '| GENE1 | 0 | 0 | 1 |' in prompt


def test_critic_sees_measurements_without_actor_identity_arguments(monkeypatch):
    actor = deepcopy(ACTOR)
    actor['hypotheses'] = [{'label': 'Z candidate', 'reason': 'Invented anatomical premise'},
                           {'label': 'A candidate', 'reason': 'Another unmeasured premise'},
                           {'label': 'Z candidate', 'confidence': 'high'}]
    node, state, calls, _ = harness(monkeypatch, [actor, decision()])
    result = node.execute(state)
    critic_prompt = calls[1][0]
    assert '["A candidate", "Z candidate"]' in critic_prompt
    assert 'Invented anatomical premise' not in critic_prompt
    assert 'Another unmeasured premise' not in critic_prompt
    assert dossier().measurements[0]['id'] in critic_prompt
    assert '| A |' in critic_prompt
    assert result['dossier_agent_transcript'][0]['proposal']['hypotheses'] == actor['hypotheses']


@pytest.mark.parametrize('hypotheses', [None, {'label': 'Malformed candidate list'}, 3])
def test_malformed_optional_candidates_do_not_block_evidence_review(monkeypatch, hypotheses):
    node, state, calls, _ = harness(monkeypatch, [decision()])
    verdict, attempts = node.review(state, dossier(), hypotheses)
    assert verdict and not attempts[-1]['validation_errors']
    assert dossier().measurements[0]['id'] in calls[0][0]
    assert 'Malformed candidate list' not in calls[0][0]


def test_critic_question_runs_another_measurement_then_can_broaden(monkeypatch):
    first = decision(label='Provisional subregion', question='Distinguish neighboring subregions')
    final = decision(label='Broad compartment', status='best_effort')
    node, state, calls, labels = harness(monkeypatch, [ACTOR, first, ACTOR, final])
    measured = []
    def act(state, d, action):
        measured.append(len(measured) + 1)
        d.measurements.append(observation_record('0', f'View {len(measured)} lacks a distinguishing boundary', {}))
    monkeypatch.setattr(node, '_act', act)
    result = node.execute(state)
    assert measured == [1, 2]
    assert result['dossier_agent_steps'] == 2
    assert labels['0'] == 'Broad compartment'
    assert result['annotation_assessment'] == 'best_effort'
    assert first['next_question'] in calls[2][0]
    assert 'View 2 lacks a distinguishing boundary' in calls[3][0]


def test_unknown_citations_and_cross_kind_references_are_rejected():
    d = dossier()
    good = decision(d)
    assert not validate_annotation(good, d)
    bad = deepcopy(good); bad['support'][0]['items'] = ['Absent type']
    assert validate_annotation(bad, d)
    bad = deepcopy(good); bad['support'][0] = {'measurement_id': d.measurements[0]['id'], 'genes': ['A']}
    assert validate_annotation(bad, d)
    bad = deepcopy(good); bad['major_region'] = ['not a string']
    assert validate_annotation(bad, d)


def test_citation_correction_explains_expression_and_image_keys():
    d = dossier()
    gene = SimpleNamespace(name='GENE1', lfc=1., frac_non_zero_target=.5, pval=.01)
    expression = expression_record('0', SimpleNamespace(genes=[gene]), {})
    observation = observation_record('0', 'Interior stripe', {})
    d.measurements.extend([expression, observation])
    bad = decision(d)
    bad['support'] = [{'measurement_id': expression['id'], 'items': ['GENE1']},
                      {'measurement_id': observation['id'], 'items': ['Interior stripe']}]
    errors = validate_annotation(bad, d)
    assert any("use key 'genes'" in e and 'GENE1' in e for e in errors)
    assert any("use key 'items'" in e and "['observation']" in e for e in errors)
    good = deepcopy(bad)
    good['support'] = [{'measurement_id': expression['id'], 'genes': ['GENE1']},
                       {'measurement_id': observation['id'], 'items': ['observation']}]
    assert not validate_annotation(good, d)


def test_expression_records_preserve_significance_type():
    gene = SimpleNamespace(name='GENE1', lfc=1., frac_non_zero_target=.5, pval=.01)
    targeted = expression_record('0', SimpleNamespace(genes=[gene]), {})
    ranked = expression_record('0', SimpleNamespace(enriched_genes=[gene], depleted_genes=[]), {})
    assert targeted['significance_kind'] == 'raw_p'
    assert ranked['significance_kind'] == 'adjusted_p'
    assert targeted['rows'] == ranked['rows']
    assert targeted['id'] != ranked['id']


@pytest.mark.parametrize('text', ['[]', 'Explanation.\n```json\n[]\n```'])
def test_global_review_accepts_complete_fenced_array(text):
    from sclambda.pipelines.annotate_spatial.spatial_consistency_review import _parse_json_array
    assert _parse_json_array(text) == []


@pytest.mark.parametrize('text', ['No issues', '[{"niche":', '{}',
    '```json\n[]\n```\n```json\n[]\n```', 'Reason [incidental brackets] no decision'])
def test_global_review_never_recovers_malformed_or_ambiguous_arrays(text):
    from sclambda.pipelines.annotate_spatial.spatial_consistency_review import _parse_json_array
    with pytest.raises(ValueError): _parse_json_array(text)


@pytest.mark.parametrize('text,valid', [('Explanation.\n\n[]', True), ('[]\n[]', False)])
def test_global_format_recovery_never_regenerates_biology(monkeypatch, text, valid):
    from sclambda.pipelines.annotate_spatial.spatial_consistency_review import _review_slide
    import sclambda.analysis.spatial_visualization as visualization
    monkeypatch.setattr(visualization, 'generate_niche_image_base64', lambda **kw: 'saved-image')
    acc = SimpleNamespace(n_cells=2, _overlay=pd.DataFrame({'L1': ['0', '0']}),
        _level_column=lambda depth: 'L1', get_obsm=lambda key, idx: np.zeros((2, 2)))
    calls = []
    def query(prompt, **kw):
        calls.append(prompt)
        return SimpleNamespace(content=text, failed_after_retries=False, stop_reason='end_turn')
    args = (acc, SimpleNamespace(query=query), [{'cluster_id': '0', 'annotation': 'Region'}], {})
    if valid:
        assert _review_slide(*args) == {}
    else:
        with pytest.raises(ValueError): _review_slide(*args)
    assert len(calls) == 1


def test_legacy_spatial_commit_is_readable():
    old = {'cluster_id': '0', 'candidates': [{'compartment': 'Legacy', 'state': 'supported'}], 'best_compartment': 'Legacy'}
    d = NicheDossier.from_dict(old)
    assert not d.decision_mode
    assert build_commit(d)['label'] == 'Legacy'


@pytest.fixture
def spatial_run(tmp_path):
    import anndata as ad
    from scipy.sparse import csr_matrix
    n = 64
    data = ad.AnnData(csr_matrix(np.random.default_rng(5).poisson(2, (n, 3)).astype('float32')),
        obs=pd.DataFrame({'niche': ['left'] * 32 + ['right'] * 32,
                          'ct': ['A', 'B'] * 32, 'slide': (['s1'] * 16 + ['s2'] * 16) * 2},
                         index=[f'c{i}' for i in range(n)]), var=pd.DataFrame(index=['GENE1', 'GENE2', 'GENE3']))
    data.obsm['spatial'] = np.random.default_rng(3).normal(size=(n, 2))
    path = tmp_path / 'data.h5ad'; data.write_h5ad(path)
    run = tmp_path / 'run'
    assert main(['init', str(path), '--run', str(run), '--obs-column', 'niche', '--cell-type-col', 'ct',
                 '--spatial-key', 'spatial', '--slice-key', 'slide', '--organism', 'human', '--tissue', 'synthetic',
                 '--cache-dir', str(tmp_path / 'cache')]) == 0
    return Session(run)


def cli(s, *args):
    # Data commands run as real CLI processes: their HDF5 handles end with each command.
    if args[0] == 'spatial':
        import os, subprocess, sys
        from pathlib import Path
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
        result = subprocess.run([sys.executable, '-m', 'sclambda.cli.main', *args, '--run', str(s.run)],
                                env=env, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode: print(result.stderr)
        return result.returncode
    return main([*args, '--run', str(s.run)])


def test_cli_measurements_decision_revision_and_export(spatial_run, tmp_path, capsys):
    s = spatial_run
    assert cli(s, 'spatial', 'composition', '--niche', '0') == 0
    assert cli(s, 'spatial', 'de', '--niche', '0') == 0
    assert cli(s, 'spatial', 'genes', '--niche', '0', '--cell-type', 'A', '--genes', 'GENE1,MISSING') == 0
    d = NicheDossier.from_dict(s.read_dossier_dict('0'))
    assert {m.get('kind') for m in d.measurements} >= {'composition', 'expression', 'slide_distribution'}
    assert any(m.get('cell_type') == 'A' for m in d.measurements)
    assert 'MISSING' in d.requested_genes
    assert 'signature in A' in d.to_markdown()
    assert 'unresolved names are not expression zeros' in d.to_markdown()
    v = decision(d)
    assert cli(s, 'dossier', 'decision', '--cluster', '0', '--json', json.dumps(v)) == 0
    assert cli(s, 'dossier', 'commit', '--cluster', '0') == 0
    assert 'decision_fingerprint' in s.read_result('0')
    s.mark_phase('export')
    v['resolution']['supported_label'] = 'Revised compartment'
    assert cli(s, 'dossier', 'decision', '--cluster', '0', '--json', json.dumps(v)) == 0
    assert s.uncommitted_decisions() == ['0']
    assert cli(s, 'export', '--csv', str(tmp_path / 'out.csv')) == 2
    assert cli(s, 'dossier', 'commit', '--cluster', '0') == 0
    assert s.phase_state('export') == 'pending'
    assert cli(s, 'export', '--csv', str(tmp_path / 'out.csv')) == 0
    exported = pd.read_csv(tmp_path / 'out.csv')
    assert exported.cell_type_L1.dropna().eq('Revised compartment').all()
    assert exported.assessment_status_L1.dropna().eq('best_effort').all()
    assert exported.major_region_L1.dropna().eq('Broad compartment').all()


def test_image_observation_retains_slide_and_original_view(spatial_run):
    s = spatial_run
    assert cli(s, 'spatial', 'image', '--niche', '0', '--slide', 's2', '--px', '300') == 0
    path = s.run / 'images/0_whole_s2.png'
    assert cli(s, 'spatial', 'observe', '--niche', '0', '--image', str(path), '--text', 'Position uncertain') == 0
    m = s.read_dossier_dict('0')['measurements'][0]
    assert m['scope']['spatial_slide'] == 's2' and m['images']
    before = s.read_dossier_dict('0')
    path.write_bytes(b'changed')
    assert cli(s, 'spatial', 'observe', '--niche', '0', '--image', str(path), '--text', 'Bad view') == 2
    assert before == s.read_dossier_dict('0')


def test_missing_within_type_reference_does_not_skip_overall_de(monkeypatch):
    from sclambda.pipelines.annotate_spatial.nodes.differential_expression import SpatialCellTypeDENode
    import sclambda.pipelines.annotate_spatial.nodes.differential_expression as module
    node = object.__new__(SpatialCellTypeDENode)
    node._output_local = threading.local(); node.output = {}
    node.log_warning = node.log_info = lambda *a: None
    node.accessor = SimpleNamespace(get_obsm=lambda *a: np.zeros((12, 2)))
    obs = pd.DataFrame({'ct': ['A'] * 12 + ['B'] * 12})
    monkeypatch.setattr(module, 'niche_and_rest_indices', lambda *a, **kw: (np.arange(12), obs, np.arange(24) >= 12))
    monkeypatch.setattr(module, '_slice_restricted_niche_indices', lambda *a: np.arange(12))
    result = SimpleNamespace(enriched_genes=[], depleted_genes=[], n_target=12, n_reference=12)
    monkeypatch.setattr(module, 'compute_pathology_de_from_indices', lambda **kw: result)
    out = node.execute({'cluster_id': '0', 'cell_type_col': 'ct', 'spatial_proportions': [{'cell_type': 'A', 'n_cells': 12}]})
    assert out['region_de'] is result


def test_uncomputed_moran_is_unavailable():
    from sclambda.analysis.spatial_metrics import compute_spatial_proportions
    coords = np.random.default_rng(1).normal(size=(10, 2))
    result = compute_spatial_proportions(coords, np.array(['A'] * 9 + ['B']), n_permutations=9)
    rare = next(m for m in result.as_dicts() if m['cell_type'] == 'B')
    assert rare['morans_z'] is None


@pytest.mark.parametrize('failure', [False, True])
@pytest.mark.parametrize('resolution_mode', [False, True])
def test_global_proposal_uses_same_critic_and_updates_all_outputs(monkeypatch, failure, resolution_mode):
    from sclambda.framework.base_pipeline import Pipeline
    from sclambda.pipelines.annotate_spatial.pipeline import AnnotateSpatialPipeline
    import sclambda.pipelines.annotate_spatial.spatial_consistency_review as global_module
    d = dossier(); d.critic_decision = decision(d)
    result = {'cluster_id': '0', 'annotation': d.annotation, 'niche_dossier': d.to_dict()}
    labels, hypotheses = {}, {}
    acc = SimpleNamespace(gene_names=np.array(['GENE1']),
        set_annotation=lambda cid, label, column: labels.update({column: label}),
        set_hypotheses=lambda cid, values: hypotheses.update({cid: values}))
    proposal = observation_record('0', 'Visible boundary challenges prior location', {'spatial_slide': 's2'}, kind='global_review')
    monkeypatch.setattr(global_module, 'review_spatial_consistency', lambda *a, **kw: {
        '0': [{'sub_region': 'Observer proposal', 'reason': 'Reconsider location', 'evidence': proposal}]})
    monkeypatch.setattr(Pipeline, 'run', lambda *a, **kw: [result])
    calls = []
    updated = decision(d, label='Critic corrected compartment')
    updated['reason'] = 'Original composition plus new view supports correction'
    def query(prompt, **kw):
        calls.append((prompt, kw))
        return SimpleNamespace(content=json.dumps({} if failure else updated), failed_after_retries=False)
    pipeline = AnnotateSpatialPipeline(acc, SimpleNamespace(query=query))
    out = pipeline._run_level(['0'], 1, spatial_resolution_comparison=resolution_mode, spatial_comparison_review=False, spatial_consistency_review=True)
    restored = NicheDossier.from_dict(out[0]['niche_dossier'])
    assert calls and all(kw['power'] == 'strong' for _, kw in calls)
    assert ('Decide identity and resolution together:' in calls[0][0]) is resolution_mode
    if failure:
        assert restored.annotation == 'Compartment A'
        assert out[0]['global_review_status'] == 'failed'
        assert out[0].get('annotation_assessment') != 'inference_failed'
    else:
        assert restored.annotation == out[0]['annotation'] == labels['region'] == 'Critic corrected compartment'
        assert hypotheses['0'][0]['justification'] == restored.critic_decision['reason']
        assert out[0]['major_region'] == labels['major_region'] == updated['major_region']
    assert proposal in restored.measurements


def test_global_observer_never_combines_coordinate_frames(monkeypatch):
    import sclambda.pipelines.annotate_spatial.spatial_consistency_review as module
    slides = np.array(['s1'] * 12 + ['s2'] * 12)
    frame = pd.DataFrame({'L1': ['0'] * 24})
    acc = SimpleNamespace(n_cells=24, _overlay=frame, _level_column=lambda d: 'L1',
        get_obs=lambda cols, idx: pd.DataFrame({'slide': slides[idx]}),
        get_obsm=lambda key, idx: np.column_stack([idx, idx]))
    seen = []
    def review(*a, **kw):
        seen.append(kw['slide'])
        return {'0': {'reason': kw['slide']}}
    monkeypatch.setattr(module, '_review_slide', review)
    result = module.review_spatial_consistency(acc, None, [], slice_key='slide')
    assert seen == ['s1', 's2']
    assert [p['reason'] for p in result['0']] == seen


def test_compiled_graph_requires_no_candidate_generation(monkeypatch):
    from sclambda.pipelines.annotate_spatial.pipeline import AnnotateSpatialPipeline
    from sclambda.pipelines.annotate_spatial.nodes import ComputeSpatialProportionsNode, SpatialImageDescriptionNode, SpatialCellTypeDENode
    from sclambda.core.task_runner import TaskRunner
    d = dossier()
    monkeypatch.setattr(ComputeSpatialProportionsNode, 'execute', lambda self, state: {'niche_dossier': d.to_dict()})
    monkeypatch.setattr(SpatialImageDescriptionNode, 'execute', lambda self, state: {})
    monkeypatch.setattr(SpatialCellTypeDENode, 'execute', lambda self, state: {})
    replies = [ACTOR, decision(d)]
    def query(*a, **kw):
        return SimpleNamespace(content=json.dumps(replies.pop(0)), failed_after_retries=False)
    acc = SimpleNamespace(gene_names=np.array(['GENE1']), set_annotation=lambda *a, **kw: None,
                          set_hypotheses=lambda *a: None)
    pipeline = AnnotateSpatialPipeline(acc, SimpleNamespace(query=query))
    with TaskRunner(max_workers=1) as runner:
        graph = pipeline.build_graph(runner)
        result = graph.invoke({'cluster_id': '0', 'dossier_agent_max_steps': 1, 'spatial_comparison_review': False})
    assert not replies
    assert result['annotation'] == 'Compartment A'
    assert result['niche_dossier']['critic_decision']['label'] == result['annotation']


def test_real_measurements_compiled_pipeline_and_python_exports(spatial_run, tmp_path):
    """Real h5ad/statistics/graph/export with deterministic model replies, no API."""
    import os, subprocess, sys
    from pathlib import Path
    source = str(Path(__file__).resolve().parents[1] / 'src')
    program = '''
import json, re, sys
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from sclambda.core.accessor import Accessor
from sclambda.pipelines.annotate_spatial.pipeline import AnnotateSpatialPipeline
root = Path(sys.argv[1])
acc = Accessor(root / 'data.h5ad', gene_filter=None, cache_dir=str(root / 'api-cache'))
class Model:
    def query(self, prompt, **kwargs):
        if kwargs['power'] == 'medium':
            value = {'hypotheses': [], 'action': {'tool': 'done'}}
        else:
            mid = re.search(r'm_[0-9a-f]{12}', prompt).group()
            value = dict(label='Synthetic compartment', major_region='Synthetic region', status='best_effort',
                resolution=dict(supported_label='Synthetic compartment', fine_label=None, closest_alternative=None, fine_supported=False, comparison='No finer distinction.', discriminating_support=[]),
                reason='Fixture composition', support=[{'measurement_id': mid, 'items': ['A']}],
                counterevidence=[], alternatives=[], uncertainty='Synthetic evidence', next_question='', pending_checks=[])
        return SimpleNamespace(content=json.dumps(value), failed_after_retries=False)
pipeline = AnnotateSpatialPipeline(acc, Model(), max_workers=2)
cluster_ids = pipeline._create_cluster_encoding(['left', 'right'])
results = pipeline.run(cluster_ids, 1, obs_columns=['niche'], cell_type_col='ct', spatial_key='spatial',
    slice_key='slide', include_spatial_image=False, dataset_profile=False, spatial_consistency_review=False,
    organism='human', tissue='synthetic', dossier_agent_max_steps=1, spatial_comparison_review=False)
assert len(results) == 2, results
for result in results:
    assert result['annotation'] == result['niche_dossier']['critic_decision']['label']
    assert result['annotation_assessment'] == 'best_effort'
    kinds = {m['kind'] for m in result['niche_dossier']['measurements']}
    assert kinds >= {'composition', 'expression', 'slide_distribution'}
pipeline.write_outputs(acc, output_path=root/'python.csv', excel_path=root/'python.xlsx')
frame = pd.read_csv(root/'python.csv')
assert frame.region.eq('Synthetic compartment').all()
assert frame.assessment_status.eq('best_effort').all()
assert frame.uncertainty.eq('Synthetic evidence').all()
print('two-niche graph and exports passed')
'''
    result = subprocess.run([sys.executable, '-c', program, str(tmp_path)],
        capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=source))
    assert result.returncode == 0, result.stdout + result.stderr


def test_global_observation_records_actual_numbered_map_and_scope():
    from sclambda.pipelines.annotate_spatial.spatial_consistency_review import _review_slide
    frame = pd.DataFrame({'L1': ['0'] * 12 + ['1'] * 12})
    coords = np.random.default_rng(12).normal(size=(24, 2))
    acc = SimpleNamespace(n_cells=24, _overlay=frame, _level_column=lambda d: 'L1',
        get_obs=lambda cols, idx: pd.DataFrame({'slide': ['s1'] * len(idx)}),
        get_obsm=lambda key, idx: coords[idx])
    images = []
    def query(prompt, **kw):
        images.append(kw['image_base64'])
        return SimpleNamespace(failed_after_retries=False, content=json.dumps([
            {'niche': '0', 'corrected_sub_region': 'Alternative', 'reason': 'Visible arrangement challenges location'}]))
    changes = _review_slide(acc, SimpleNamespace(query=query), [
        {'cluster_id': '0', 'annotation': 'First', 'major_region': 'Broad'},
        {'cluster_id': '1', 'annotation': 'Second', 'major_region': 'Broad'}],
        {'0': ['1'], '1': ['0']}, slice_key='slide', slide='s1', image_px=300)
    m = changes['0']['evidence']
    assert m['scope']['spatial_slide'] == 's1'
    assert m['images'] == images and m['views'][0]['sha256']
    assert m['kind'] == 'global_review'


def test_started_compact_review_does_not_certify_legacy_result(tmp_path):
    s = Session(tmp_path)
    s.save_manifest({'mode': 'spatial'})
    s.write_result('0', {'cluster_id': '0', 'label': 'Legacy'})
    s.write_dossier_dict('0', dossier().to_dict())
    assert s.uncommitted_decisions() == ['0']
    assert not s.current_results()


def test_small_niche_still_has_composition_evidence(monkeypatch):
    from sclambda.pipelines.annotate_spatial.nodes.composition import ComputeSpatialProportionsNode
    import sclambda.pipelines.annotate_spatial.nodes.composition as module
    acc = SimpleNamespace(get_obs=lambda cols, idx: pd.DataFrame({'ct': ['A', 'B']}))
    node = ComputeSpatialProportionsNode(acc, None)
    node.output = {}
    monkeypatch.setattr(module, '_slice_restricted_niche_indices', lambda *a: np.array([0, 1]))
    result = node.execute({'cluster_id': '0', 'cell_type_col': 'ct', 'include_spatial_image': False})
    assert len(result['spatial_proportions']) == 2
    assert all(p['proportion'] == .5 and p['morans_z'] is None for p in result['spatial_proportions'])


def test_new_spatial_cli_decisions_require_plan_and_scoped_citations(spatial_run):
    s=spatial_run
    d=dossier()
    m=expression_record('0', SimpleNamespace(genes=[SimpleNamespace(
        name='GENE1',lfc=1.0,frac_non_zero_target=.8,pval=.01)]),{},cell_type='A')
    d.measurements.append(m)
    s.write_dossier_dict('0',d.to_dict())
    value=decision(d);value.pop('pending_checks')
    assert cli(s,'dossier','decision','--cluster','0','--json',json.dumps(value))==2
    value['pending_checks']=[]
    value['support']=[{'measurement_id':m['id'],'genes':['GENE1'],
                      'cell_type':'B','reference':m['reference']}]
    assert cli(s,'dossier','decision','--cluster','0','--json',json.dumps(value))==2
    value['support'][0]['cell_type']='A'
    assert cli(s,'dossier','decision','--cluster','0','--json',json.dumps(value))==0
