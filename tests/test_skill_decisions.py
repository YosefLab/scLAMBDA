"""Exercise the skill's recorded-table -> critic -> commit -> export workflow."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from sclambda.cli.main import main
from sclambda.storage.session import Session


@pytest.fixture
def skill_run(tmp_path, monkeypatch):
    monkeypatch.setenv('LAMBDA_CACHE_DIR', str(tmp_path / 'cache'))
    genes = ['CD3E', 'CD8A', 'CD4', 'MS4A1', 'CD19', 'LYZ', 'CD14', 'GENEX']
    x = np.random.default_rng(1).poisson(1, (60, len(genes))).astype(np.float32)
    x[:30, :2] += 15
    x[30:, 3:5] += 15
    a = ad.AnnData(sparse.csr_matrix(x), obs=pd.DataFrame({'group': ['a']*30+['b']*30},
                   index=[f'c{i}' for i in range(60)]), var=pd.DataFrame(index=genes))
    a.obsm['X_pca'] = np.random.default_rng(2).normal(size=(60, 3))
    source = tmp_path / 'data.h5ad'; a.write_h5ad(source)
    run = tmp_path / 'run'
    assert main(['init', str(source), '--run', str(run), '--organism', 'human',
                 '--tissue', 'blood', '--obs-column', 'group', '--depth', '2']) == 0
    return run


def call(run, *args):
    # The skill invokes separate CLI processes, which own independent HDF5 handles.
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    proc = subprocess.run([sys.executable, '-m', 'sclambda.cli', *args, '--run', str(run)],
                          capture_output=True, text=True, env=env)
    print(proc.stdout)
    print(proc.stderr, file=sys.stderr)
    return proc.returncode


def proposed(table):
    return dict(label='T cells', status='best_effort', reason='Measured T-cell program; subtype uncertain.',
                support=[dict(measurement_id=table['id'], genes=['CD3E'])], counterevidence=[],
                alternatives=['CD4 T cells', 'CD8 T cells'], uncertainty='Subtype remains uncertain.',
                next_question='')


def test_record_decide_resume_and_export_uncertainty(skill_run, tmp_path, capsys):
    run = skill_run
    assert call(run, 'dossier', 'init', '--cluster', '0') == 0
    assert call(run, 'de', 'rank', '--cluster', '0', '--n', '5', '--record') == 0
    assert call(run, 'de', 'genes', '--cluster', '0', '--genes', 'CD3E,CD8A,NOTREAL', '--record') == 0
    d = Session(run).read_dossier_dict('0')
    assert len(d['measurements']) == 2
    table = d['measurements'][-1]
    assert 'target excluded' in table['reference']
    assert 'NOTREAL' in d['requested_genes']
    assert 'NOTREAL' not in {g['gene'] for g in table['rows']}
    decision = proposed(table)
    path = tmp_path / 'decision.json'; path.write_text(json.dumps(decision))
    assert call(run, 'dossier', 'decision', '--cluster', '0', '--file', str(path)) == 0
    before = Session(run).read_dossier_dict('0')
    assert call(run, 'dossier', 'init', '--cluster', '0') == 0
    assert Session(run).read_dossier_dict('0') == before
    assert call(run, 'dossier', 'commit', '--cluster', '0') == 0
    r = Session(run).all_results()['0']
    assert r['label'] == decision['label'] and r['assessment_status'] == 'best_effort'
    assert r['confidence'] == 'uncertain' and r['uncertainty'] == decision['uncertainty']
    # A child without an annotation inherits both label and matching uncertainty.
    sess = Session(run); overlay = sess.read_overlay(); overlay['L2'] = None
    overlay.loc[overlay.L1 == '0', 'L2'] = '0_0'
    sess.save_overlay(SimpleNamespace(overlay=overlay))
    csv = tmp_path / 'out.csv'; md = tmp_path / 'out.md'
    assert call(run, 'export', '--csv', str(csv), '--md', str(md)) == 0
    exported = pd.read_csv(csv)
    assert exported.loc[:29, 'assessment_status_L2'].eq('best_effort').all()
    assert exported.loc[:29, 'uncertainty_L2'].eq(decision['uncertainty']).all()
    assert decision['uncertainty'] in md.read_text()
    capsys.readouterr()
    assert call(run, 'dossier', 'show', '--cluster', '0') == 0
    shown = capsys.readouterr().out
    assert 'best_effort' in shown and table['id'] in shown


def test_invalid_critic_and_actor_update_cannot_replace_valid_decision(skill_run):
    run = skill_run
    assert call(run, 'de', 'genes', '--cluster', '0', '--genes', 'CD3E', '--record') == 0
    d = Session(run).read_dossier_dict('0'); good = proposed(d['measurements'][0])
    assert call(run, 'dossier', 'decision', '--cluster', '0', '--json', json.dumps(good)) == 0
    bad = {**good, 'label': 'B cells', 'support': [{'measurement_id': 'invented', 'genes': ['MS4A1']}]}
    assert call(run, 'dossier', 'decision', '--cluster', '0', '--json', json.dumps(bad)) == 2
    assert call(run, 'dossier', 'decision', '--cluster', '0', '--json', '{broken') == 2
    assert call(run, 'dossier', 'update', '--cluster', '0', '--json', json.dumps({'annotation': bad})) == 0
    after = Session(run).read_dossier_dict('0')
    assert after['annotation'] == good and after['measurements'] == d['measurements']
    assert any(t.get('validation_errors') for t in Session(run).read_transcript('0'))


def test_parent_comparison_excludes_target_and_missing_request_is_saved(skill_run):
    run = skill_run; sess = Session(run); overlay = sess.read_overlay()
    overlay['L2'] = None; overlay.loc[overlay.index[:15], 'L2'] = '0_0'
    overlay.loc[overlay.index[15:30], 'L2'] = '0_1'
    sess.save_overlay(SimpleNamespace(overlay=overlay))
    assert call(run, 'de', 'genes', '--cluster', '0_0', '--genes', 'CD3E', '--vs', '0', '--record') == 0
    d = Session(run).read_dossier_dict('0_0')
    assert 'excluding target; n=15; target n=15' in d['measurements'][0]['reference']
    assert call(run, 'de', 'genes', '--cluster', '0_0', '--genes', 'NOTREAL', '--record') == 0
    d = Session(run).read_dossier_dict('0_0')
    assert 'NOTREAL' in d['requested_genes'] and len(d['measurements']) == 1


def test_status_does_not_require_catalog_or_cross_cluster_pass(skill_run, capsys):
    run = skill_run; capsys.readouterr()
    assert call(run, 'status') == 0
    out = capsys.readouterr().out
    assert 'Annotate 2 remaining' in out
    sess = Session(run)
    for cid in ['0', '1']:sess.write_result(cid, {'label': 'test', 'cluster_id': cid})
    assert call(run, 'status') == 0
    out = capsys.readouterr().out.split('## Next')[-1]
    assert '**export**' in out and 'cross-cluster' not in out
