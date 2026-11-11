"""Regressions for the shared engine, durable runs, and frontend contracts."""
from __future__ import annotations
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

from sclambda.annotation.celltype.dossier import CellTypeDossier
from sclambda.annotation.niche.dossier import NicheDossier
from sclambda.annotation.decisions import apply_critic_verdict, build_commit
from sclambda.core.accessor import Accessor
from sclambda.core.matrix_reader import H5adMatrixReader
from sclambda.storage.session import Session, SessionError
from sclambda.cli.main import main


def make_data(path, value=1, cells=('001', '002', '003')):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = ad.AnnData(sparse.csr_matrix(np.full((3, 2), value, dtype=np.float32)),
                     obs=pd.DataFrame({'cluster': ['a', 'a', 'b']}, index=list(cells)),
                     var=pd.DataFrame(index=['CD3E', 'MS4A1']))
    data.obsm['X_pca'] = np.arange(6).reshape(3, 2).astype(float)
    data.write_h5ad(path)
    return path


def test_same_basename_datasets_never_share_matrix_arrays(tmp_path):
    paths = [make_data(tmp_path / 'one' / 'same.h5ad', 1),
             make_data(tmp_path / 'two' / 'same.h5ad', 9)]
    readers = [H5adMatrixReader(str(p), cache_dir=str(tmp_path / 'cache')) for p in paths]
    try:
        for r, value in zip(readers, (1, 9)):
            np.testing.assert_array_equal(r.fetch_rows_dense([0, 2]), np.full((2, 2), value))
        assert readers[0]._mmap_temp_files != readers[1]._mmap_temp_files
        locations = [Path(p) for r in readers for p in r._mmap_temp_files]
    finally:
        for r in readers:
            r.close()
    assert all(p.is_file() for p in locations), 'closing a reader must retain the shared cache'


def test_dossier_reinitialization_preserves_evidence_and_results(tmp_path):
    sess = Session(tmp_path / 'run')
    sess.save_manifest({'mode': 'celltype'})
    assert main(['dossier', 'init', '--run', str(sess.run), '--cluster', '0']) == 0
    d = sess.read_dossier_dict('0')
    d['evidence_log'] = [{'source': 'measured', 'fact': 'CD3E expressed'}]
    sess.write_dossier_dict('0', d)
    sess.write_result('0', {'cluster_id': '0', 'label': 'T cell'})
    before = (sess.run / 'dossiers/0.json').read_bytes()
    assert main(['dossier', 'init', '--run', str(sess.run), '--cluster', '0']) == 0
    assert (sess.run / 'dossiers/0.json').read_bytes() == before
    assert sess.read_result('0')['label'] == 'T cell'


def test_run_init_resumes_but_rejects_new_parameters(tmp_path):
    data = make_data(tmp_path / 'cells.h5ad')
    run = tmp_path / 'run'
    args = ['init', str(data), '--run', str(run), '--obs-column', 'cluster', '--tissue', 'blood']
    assert main(args) == 0
    before = {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}
    assert main(args) == 0
    assert before == {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}
    assert main(args + ['--tissue', 'brain']) != 0
    assert before == {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}


def test_overlay_failed_write_preserves_previous_checkpoint(tmp_path, monkeypatch):
    sess = Session(tmp_path / 'run'); sess.save_manifest({})
    original = pd.DataFrame({'L1': ['0', '1']}, index=['001', '002'])
    acc = SimpleNamespace(overlay=original)
    path = sess.save_overlay(acc)
    before = path.read_bytes()
    def failed_write(frame, destination, **kwargs):
        Path(destination).write_bytes(b'partial parquet')
        raise OSError('simulated interrupted write')
    monkeypatch.setattr(pd.DataFrame, 'to_parquet', failed_write)
    with pytest.raises(OSError):
        sess.save_overlay(acc)
    assert path.read_bytes() == before
    pd.testing.assert_frame_equal(sess.read_overlay(), original)
    assert not list(sess.run.glob('.*.parquet'))


def test_csv_fallback_preserves_cell_identifiers(tmp_path, monkeypatch):
    sess = Session(tmp_path / 'run'); sess.save_manifest({})
    frame = pd.DataFrame({'L1': ['0', '1']}, index=['001', '002'])
    def missing_engine(*args, **kwargs):
        raise ImportError('no parquet engine')
    monkeypatch.setattr(pd.DataFrame, 'to_parquet', missing_engine)
    sess.save_overlay(SimpleNamespace(overlay=frame))
    pd.testing.assert_frame_equal(sess.read_overlay(), frame)
    assert sess.get('overlay_file') == 'overlay.csv.gz'


def test_overlay_reordering_is_rejected(tmp_path):
    path = make_data(tmp_path / 'cells.h5ad')
    acc = Accessor(path)
    sess = Session(tmp_path / 'run'); sess.save_manifest({'h5ad': str(path)})
    frame = pd.DataFrame({'L1': ['0', '1', '2']}, index=['003', '002', '001'])
    sess.save_overlay(SimpleNamespace(overlay=frame))
    with pytest.raises(SessionError, match='identities/order'):
        sess._load_overlay(acc)


def test_modified_dataset_cannot_resume_an_existing_run(tmp_path):
    from sclambda.core.cache import fingerprint
    path = make_data(tmp_path / 'cells.h5ad')
    sess = Session(tmp_path / 'run')
    sess.save_manifest({'h5ad': str(path), 'dataset_fingerprint': fingerprint(path)})
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(SessionError, match='source dataset changed'):
        sess.validate_source()


def test_unsupported_manifest_schema_is_always_rejected(tmp_path):
    tmp_path.joinpath('manifest.json').write_text(json.dumps({'schema_version': 100}))
    sess = Session(tmp_path)
    for _ in range(2):
        with pytest.raises(SessionError, match='Unsupported run schema'):
            _ = sess.manifest


@pytest.mark.parametrize('kind', ['celltype', 'niche'])
def test_critic_requires_current_evidence_and_is_shared_by_both_modes(kind):
    field = 'cell_type' if kind == 'celltype' else 'compartment'
    cls = CellTypeDossier if kind == 'celltype' else NicheDossier
    names = ('B cell', 'Naive B cell') if kind == 'celltype' else ('Cortex', 'Cortex layer 4')
    axis = 'lineage' if kind == 'celltype' else 'position'
    d = cls(cluster_id='0', **({'decision_mode': False} if kind == 'niche' else {}))
    d.apply_update({'candidates': [{field: names[0], 'state': 'supported', 'evidence_for': ['measured']},
                                   {field: names[1], 'state': 'candidate'}],
                    f'best_{field}': names[0], 'axes': {axis: {'resolved': True, 'note': 'old evidence'}}})
    assert apply_critic_verdict(d, {'resolved_as': names[1]}).status == 'refused'
    assert d.leading().label == names[0]
    outcome = apply_critic_verdict(d, {'resolved_as': names[1],
                                     'axes': {axis: {'resolved': True, 'label': names[0], 'note': 'new measurement', "evidence_status": "established", "defining_evidence": ['new measurement'], "specificity": "Fixture measured combination distinguishes the alternative"}}})
    assert outcome.status == 'promoted'
    assert sum(c.state == 'supported' for c in d.candidates) == 1
    assert d.leading().label == names[1]


def test_commit_cannot_bypass_supported_candidate():
    d = CellTypeDossier(cluster_id='0')
    with pytest.raises(ValueError, match='No supported candidate'):
        build_commit(d, label='invented label')
    d.apply_update({'candidates': [{'cell_type': 'B cell', 'state': 'supported'}], 'best_cell_type': 'B cell'})
    with pytest.raises(ValueError, match='No supported candidate'):
        build_commit(d)
    apply_critic_verdict(d, {'axes': {'lineage': {'resolved': True, 'label': 'B cell', 'note': 'measured B signature', "evidence_status": "established", "defining_evidence": ['measured B signature'], "specificity": "Fixture measured combination distinguishes the alternative"}}})
    with pytest.raises(ValueError, match='differs'):
        build_commit(d, label='T cell')
    assert build_commit(d)['label'] == 'B cell'
    result = build_commit(d, label='T cell', manual_override_reason='User reviewed marker panel')
    assert result['provenance'] == 'manual_override'
    assert result['dossier_label'] == 'B cell'
    assert result['confidence'] == 'unknown'
    assert d.leading().label == 'B cell'


def test_scientific_imports_do_not_load_llm_frameworks():
    script = '''
import importlib.abc, sys
class NoLLM(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("langchain", "langgraph")):
            raise AssertionError("Scientific imports tried to load " + fullname)
sys.meta_path.insert(0, NoLLM())
from sclambda.core.accessor import Accessor
from sclambda.analysis.targeted_de import compute_de_for_genes
from sclambda.annotation.celltype.dossier import CellTypeDossier
from sclambda.cli.main import build_parser
assert build_parser().parse_args(["doctor"]).group == "doctor"
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_pipeline_nodes_find_prompts_after_module_split():
    from sclambda.pipelines.celltype.nodes.common import _load_prompt as celltype_prompt
    from sclambda.pipelines.annotate_spatial.nodes.common import _load_prompt as niche_prompt
    assert len(celltype_prompt('ct_critic.md')) > 100
    assert len(niche_prompt('17_dossier_agent.md')) > 100


def test_plugin_metadata_and_cli_use_the_package_version():
    from sclambda._version import __version__
    from sclambda.plugin import plugin_directory
    from sclambda.cli.main import __version__ as cli_version
    plugin = plugin_directory()
    metadata = json.loads((plugin / '.claude-plugin/plugin.json').read_text())
    marketplace = json.loads((plugin / '.claude-plugin/marketplace.json').read_text())
    assert metadata['version'] == cli_version == __version__
    assert marketplace['metadata']['version'] == __version__
    assert all(p['version'] == __version__ for p in marketplace['plugins'])
