"""Regression tests for revised decisions and uneven hierarchy resumption."""
import json
import os
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from sclambda.cli.main import main
from sclambda.storage.session import Session
from sclambda.annotation.celltype.dossier import CellTypeDossier


def session(tmp_path, frame=None, depth=3):
    s = Session(tmp_path / 'run')
    s.save_manifest({'mode': 'celltype', 'max_depth': depth})
    if frame is None:
        frame = pd.DataFrame({'L1': ['0', '0']}, index=['a', 'b'])
    s.save_overlay(SimpleNamespace(overlay=frame))
    return s


def cli(s, *args):
    return main([*args, '--run', str(s.run)])


def open_dossier(s):
    d = CellTypeDossier(cluster_id='0')
    d.measurements = [{'id': 'm_fixture', 'target': '0', 'reference': 'rest', 'method': 'ttest',
                      'significance_kind': 'raw_p', 'rows': [
                          {'gene': 'GENEA', 'lfc': 2., 'proportion': .8, 'significance': .001}]}]
    s.write_dossier_dict('0', d.to_dict())


def decide(s, **changes):
    value = dict(label='Old label', status='best_effort', reason='Fixture measured evidence',
                 support=[{'measurement_id': 'm_fixture', 'genes': ['GENEA']}],
                 counterevidence=[], alternatives=[], uncertainty='Fixture uncertainty', next_question='')
    value.update(changes)
    assert cli(s, 'dossier', 'decision', '--cluster', '0', '--json', json.dumps(value)) == 0


@pytest.mark.parametrize('revision', [
    {'label': 'Revised label', 'next_question': 'Test an alternative signature'},
    {'uncertainty': 'New uncertainty with unchanged label'},
])
def test_interrupted_revision_is_pending_and_cannot_export_old_label(tmp_path, capsys, revision):
    s = session(tmp_path); open_dossier(s); decide(s)
    assert cli(s, 'dossier', 'commit', '--cluster', '0') == 0
    old = s.read_result('0')
    decide(s, **revision)
    assert s.read_result('0') == old  # preserve the old commit for recovery
    capsys.readouterr()
    assert cli(s, 'status') == 0
    status = capsys.readouterr().out
    assert 'Uncommitted critic decisions' in status and 'Every cluster is annotated' not in status
    assert cli(s, 'clusters') == 0
    assert 'in progress' in capsys.readouterr().out
    csv = tmp_path / 'export.csv'; csv.write_text('previous export\n')
    assert cli(s, 'export', '--csv', str(csv)) == 2
    assert csv.read_text() == 'previous export\n'
    old_time = time.time() - 3600
    os.utime(s.run / 'dossiers/0.json', (old_time, old_time))
    assert s.stale_clusters(minutes=30) == ['0']
    decide(s, **{**revision, 'next_question': ''})
    assert cli(s, 'dossier', 'commit', '--cluster', '0') == 0
    assert s.uncommitted_decisions() == []
    assert cli(s, 'export', '--csv', str(csv)) == 0
    df = pd.read_csv(csv)
    assert df.cell_type_L1.eq(revision.get('label', 'Old label')).all()
    assert df.uncertainty_L1.eq(revision.get('uncertainty', 'Fixture uncertainty')).all()


def test_split_flag_revision_and_export_phase_are_not_lost(tmp_path, capsys):
    s = session(tmp_path); open_dossier(s); decide(s)
    assert cli(s, 'dossier', 'commit', '--cluster', '0') == 0
    s.mark_phase('export')
    assert cli(s, 'dossier', 'axes', '--cluster', '0', '--json', '{"needs_subcluster":true}') == 0
    assert s.uncommitted_decisions() == ['0']
    assert cli(s, 'subcluster', '--cluster', '0') == 2  # no accessor/data needed
    assert cli(s, 'dossier', 'commit', '--cluster', '0') == 0
    assert s.phase_state('export') == 'pending'
    # Once at the requested depth, the pending export must be the next action.
    manifest = s.manifest.copy()
    manifest['max_depth'] = 1
    s.save_manifest(manifest)
    capsys.readouterr()
    assert cli(s, 'status') == 0
    assert 'Run the **export** phase' in capsys.readouterr().out
    s.mark_phase('export')
    assert cli(s, 'dossier', 'commit', '--cluster', '0') == 0
    assert s.phase_state('export') == 'done'  # identical recommit is a no-op for exports


def test_old_compact_commit_requires_recommit_but_legacy_result_is_readable(tmp_path):
    s = session(tmp_path); open_dossier(s); decide(s)
    assert cli(s, 'dossier', 'commit', '--cluster', '0') == 0
    result = s.read_result('0'); result.pop('decision_fingerprint')
    s.write_result('0', result)
    assert s.uncommitted_decisions() == ['0']
    assert cli(s, 'dossier', 'commit', '--cluster', '0') == 0
    assert set(s.current_results()) == {'0'}
    s.write_result('1', {'cluster_id': '1', 'label': 'Legacy label'})
    assert '1' in s.current_results()


@pytest.mark.parametrize('deep_branch', [False, True])
def test_status_uses_each_parents_depth_and_ignores_completed_splits(tmp_path, capsys, deep_branch):
    frame = pd.DataFrame({'L1': ['0', '0', '1', '1'],
                          'L2': ['0_0', '0_1', None, None]}, index=list('abcd'))
    if deep_branch:frame['L3'] = ['0_0_0', None, None, None]
    s = session(tmp_path, frame)
    for cid in ['0', '1', '0_0', '0_1'] + (['0_0_0'] if deep_branch else []):
        s.write_result(cid, {'cluster_id': cid, 'label': 'Fixture', 'needs_subcluster': cid in ['0', '1']})
    capsys.readouterr()
    assert cli(s, 'status') == 0
    next_action = capsys.readouterr().out.split('## Next')[-1]
    assert 'flagged for subclustering: 1\n' in next_action
    before = s.read_overlay().copy(); saved = s.all_results()
    assert cli(s, 'subcluster', '--cluster', '0', '--force') == 0
    assert 'already has children' in capsys.readouterr().out
    pd.testing.assert_frame_equal(before, s.read_overlay())
    assert saved == s.all_results()


def test_maximum_depth_is_respected_even_for_forced_split(tmp_path, capsys):
    s = session(tmp_path, depth=1)
    assert cli(s, 'subcluster', '--cluster', '0', '--force') == 0
    assert 'maximum depth' in capsys.readouterr().out


def test_new_split_instructs_independent_child_review(tmp_path, monkeypatch, capsys):
    import sclambda.analysis.subcluster as clustering
    s = session(tmp_path)
    frame = s.read_overlay()
    class FakeAccessor:
        overlay = frame
        def get_n_cells_in_cluster(self, cid):return 100 if cid == '0' else 50
    acc = FakeAccessor()
    monkeypatch.setattr(Session, 'accessor', property(lambda self: acc))
    def split(*args, **kw):
        acc.overlay['L2'] = ['0_0', '0_1']
        return SimpleNamespace(n_clusters=2, clusters=['0_0', '0_1'])
    monkeypatch.setattr(clustering, 'subcluster', split)
    assert cli(s, 'subcluster', '--cluster', '0') == 0
    out = capsys.readouterr().out
    assert 'provisional context' in out and 'own measurements' in out and 'lineage lock' not in out
    assert s.child_clusters('0') == ['0_0', '0_1']
