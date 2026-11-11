"""The skill's failure command preserves the pipeline's export semantics."""
from copy import deepcopy
import json

import pandas as pd
import pytest

from sclambda.annotation.niche.dossier import NicheDossier
from sclambda.pipelines.annotate_spatial.nodes.agent import SpatialDossierAgentNode
from test_spatial_decisions import cli, decision, dossier, spatial_run


@pytest.mark.parametrize("has_previous", [False, True])
def test_cli_failure_matches_pipeline_and_can_be_replaced(spatial_run, tmp_path, has_previous):
    s = spatial_run
    d = dossier()
    s.write_dossier_dict('0', d.to_dict())
    if has_previous:
        value = decision(d, status='resolved')
        assert cli(s, 'dossier', 'decision', '--cluster', '0', '--json', json.dumps(value)) == 0
    before = s.read_dossier_dict('0')
    expected = NicheDossier.from_dict(deepcopy(before))
    SpatialDossierAgentNode._retain_output(expected)

    # A rejected proposal must never become the retained identity.
    bad = decision(d, label='Rejected identity')
    bad['support'][0]['measurement_id'] = 'nonexistent'
    assert cli(s, 'dossier', 'decision', '--cluster', '0', '--json', json.dumps(bad)) == 2
    assert cli(s, 'dossier', 'retain-failure', '--cluster', '0', '--reason', 'Two final reviews failed validation') == 0
    saved = s.read_dossier_dict('0')
    assert saved['critic_decision'] == expected.critic_decision
    assert saved['measurements'] == before['measurements']
    assert saved['critic_decision']['label'] != 'Rejected identity'
    assert cli(s, 'dossier', 'commit', '--cluster', '0') == 0
    result = s.read_result('0')
    assert result['execution_status'] == 'inference_failed'
    assert result['output_origin'] == ('previous_critic' if has_previous else 'failure_record')
    assert result['assessment_status'] == 'best_effort'

    out = tmp_path / 'failed.csv'
    workbook = tmp_path / 'failed.xlsx'
    report = tmp_path / 'failed.md'
    assert cli(s, 'export', '--csv', str(out), '--xlsx', str(workbook), '--md', str(report)) == 0
    frame = pd.read_csv(out)
    assert frame.execution_status_L1.dropna().eq('inference_failed').all()
    assert frame.output_origin_L1.dropna().eq(result['output_origin']).all()
    sheet = pd.read_excel(workbook, sheet_name='L1')
    assert sheet.execution_status.eq('inference_failed').all()
    assert sheet.output_origin.eq(result['output_origin']).all()
    assert 'execution_status: inference_failed' in report.read_text()
    assert f"output_origin: {result['output_origin']}" in report.read_text()
    # Retry is idempotent, including when there never was an accepted label.
    assert cli(s, 'dossier', 'retain-failure', '--cluster', '0', '--reason', 'Same failure on resume') == 0
    assert s.read_dossier_dict('0')['critic_decision'] == saved['critic_decision']

    final = decision(d, label='Recovered identity')
    final.pop('resolution')
    final['output_origin'] = 'final_critic'
    assert cli(s, 'dossier', 'decision', '--cluster', '0', '--json', json.dumps(final)) == 0
    assert cli(s, 'export', '--csv', str(out)) == 2  # Old commit cannot be silently reused.
    assert cli(s, 'dossier', 'commit', '--cluster', '0') == 0
    assert 'execution_status' not in s.read_result('0')
    assert s.read_result('0')['output_origin'] == 'final_critic'
    assert cli(s, 'export', '--csv', str(out)) == 0
    frame = pd.read_csv(out)
    assert frame.cell_type_L1.dropna().eq('Recovered identity').all()
    assert 'execution_status_L1' not in frame or frame.execution_status_L1.dropna().empty


def test_retain_failure_rejects_missing_reason_and_celltype_run(spatial_run):
    s = spatial_run
    s.write_dossier_dict('0', dossier().to_dict())
    before = s.read_dossier_dict('0')
    assert cli(s, 'dossier', 'retain-failure', '--cluster', '0', '--reason', '   ') == 2
    assert s.read_dossier_dict('0') == before
    manifest = json.loads((s.run / 'manifest.json').read_text())
    manifest['mode'] = 'celltype'
    (s.run / 'manifest.json').write_text(json.dumps(manifest))
    assert cli(s, 'dossier', 'retain-failure', '--cluster', '0', '--reason', 'Failed review') == 2
    assert s.read_dossier_dict('0') == before
