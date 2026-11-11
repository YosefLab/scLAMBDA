"""Spatial observations without population annotations use a synthetic spot group."""
import hashlib
import json
import re
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from sclambda.annotation.niche.populations import ensure_spatial_cell_types
from sclambda.cli.main import main
from sclambda.core.accessor import Accessor
from sclambda.storage.session import Session
from test_spatial_decisions import cli


@pytest.fixture
def spot_data(tmp_path):
    rng = np.random.default_rng(7)
    x = rng.poisson(2, (80, 3)).astype('float32')
    x[:40, 0] += 5
    data = ad.AnnData(csr_matrix(x),
        obs=pd.DataFrame({'niche': ['left'] * 40 + ['right'] * 40,
                          'slide': ['s1'] * 20 + ['s2'] * 20 + ['s1'] * 20 + ['s2'] * 20},
                         index=[f's{i}' for i in range(80)]),
        var=pd.DataFrame(index=['GENE1', 'GENE2', 'GENE3']))
    data.obsm['spatial'] = rng.normal(size=(80, 2))
    path = tmp_path / 'spots.h5ad'
    data.write_h5ad(path)
    return path


def test_fallback_preserves_existing_columns_and_reuses_its_own_column():
    obs = pd.DataFrame({'spot': ['a', 'b'], '_sclambda_spot_1': [1, 2], 'ct': ['T', 'B']})
    acc = SimpleNamespace(obs=obs.copy())
    col = ensure_spatial_cell_types(acc, None)
    assert col == '_sclambda_spot_2' and acc.obs[col].eq('spot').all()
    pd.testing.assert_frame_equal(acc.obs[obs.columns], obs)
    assert ensure_spatial_cell_types(acc, None) == col
    assert ensure_spatial_cell_types(acc, 'ct') == 'ct'
    with pytest.raises(ValueError, match='missing'):
        ensure_spatial_cell_types(acc, 'missing')


@pytest.mark.parametrize('comparative', [False, True])
def test_public_api_runs_measurements_review_and_exports_without_cell_types(spot_data, tmp_path, monkeypatch, comparative):
    import sclambda.main as api
    before = hashlib.sha256(spot_data.read_bytes()).hexdigest()
    prompts = []

    class Model:
        def __init__(self, **kwargs):
            pass

        def query(self, prompt, **kwargs):
            prompts.append(prompt)
            if kwargs['power'] == 'medium':
                value = {'comparison_plan': {'candidates': ['Region A', 'Region B'], 'tests': []},
                         'hypotheses': [], 'action': {'tool': 'done'}}
            else:
                mid = re.search(r'm_[0-9a-f]{12}', prompt).group()
                ref = {'measurement_id': mid, 'items': ['spot']}
                value = dict(status='best_effort', uncertainty='Synthetic test, not a biological inference',
                             pending_checks=[], next_question='')
                if comparative:
                    value['comparison'] = dict(candidates=['Region A', 'Region B'], broader_label='Region',
                        observations=[dict(id='o1', refs=[ref], observation='All observations are included',
                            interpretation='Shared context only; synthetic composition does not distinguish identities',
                            effect='shared', favors=[], assumptions=[])], assumptions=[],
                        selection=dict(label='Region', basis=['o1'], tradeoffs={}, review=dict(
                            strongest_alternative='Region A', decisive_observations=[], remaining_issue='No discriminator')))
                else:
                    value.update(label='Region', reason='Synthetic fixture decision', support=[ref],
                                 counterevidence=[], alternatives=[])
            return SimpleNamespace(content=json.dumps(value), failed_after_retries=False)

    monkeypatch.setattr(api, 'LLMClient', Model)
    monkeypatch.setattr(api, 'Accessor', lambda **kwargs: Accessor(**kwargs, cache_dir=str(tmp_path / 'cache')))
    lam = api.Lambda(spot_data)
    result = lam.annotate(pipeline='annotate_spatial', obs_columns='niche', spatial_key='spatial',
        slice_key='slide', include_spatial_image=False, max_workers=1, n_top_genes=3,
        dossier_agent_max_steps=1, spatial_comparison_review=comparative)
    assert result['n_clusters_processed'] == 2
    assert lam.accessor.obs['spot'].eq('spot').all()
    for niche in result['results']:
        assert niche['annotation_assessment'] == 'best_effort'
        assert niche['annotation'] == 'Region'
        assert niche['cell_type_col'] == 'spot' and niche['cell_type_col_generated']
        records = niche['niche_dossier']['measurements']
        composition = next(m for m in records if m['kind'] == 'composition')
        assert composition['rows'][0]['item'] == 'spot'
        assert composition['rows'][0]['proportion'] == 1
        assert composition['rows'][0]['morans_z'] is None
        expression = [m for m in records if m['kind'] == 'expression']
        assert {m['cell_type'] for m in expression} == {None, 'spot'}
        for m in [composition, *expression]:
            assert m['scope']['cell_type_labels_provided'] is False
            assert 'not an inferred cell type' in m['scope']['population_note']
        assert all(m['reference'] == 'all spatial observations outside niche' for m in expression)
        assert all(m['n_target'] == 40 and m['n_reference'] == 40 for m in expression)
    assert all('not an inferred cell type' in p for p in prompts)
    lam.write(output_path=tmp_path / 'annotations.csv', excel_path=tmp_path / 'annotations.xlsx')
    assert pd.read_csv(tmp_path / 'annotations.csv').region.eq('Region').all()
    assert hashlib.sha256(spot_data.read_bytes()).hexdigest() == before


def test_cli_spot_column_survives_resume_and_queries(spot_data, tmp_path):
    before = hashlib.sha256(spot_data.read_bytes()).hexdigest()
    run = tmp_path / 'run'
    args = ['init', str(spot_data), '--run', str(run), '--obs-column', 'niche',
            '--spatial-key', 'spatial', '--slice-key', 'slide', '--cache-dir', str(tmp_path / 'cache')]
    assert main(args) == 0
    manifest = (run / 'manifest.json').read_bytes()
    assert main(args) == 0
    assert (run / 'manifest.json').read_bytes() == manifest
    s = Session(run)
    assert s.get('cell_type_col') == 'spot' and s.get('cell_type_col_generated')
    # Each spatial command executes in a new Python process.
    assert cli(s, 'spatial', 'composition', '--niche', '0') == 0
    assert cli(s, 'spatial', 'de', '--niche', '0') == 0
    assert cli(s, 'spatial', 'genes', '--niche', '0', '--cell-type', 'spot', '--genes', 'GENE1') == 0
    assert cli(s, 'spatial', 'image', '--niche', '0', '--slide', 's1', '--px', '300') == 0
    records = s.read_dossier_dict('0')['measurements']
    expression = [m for m in records if m['kind'] == 'expression']
    assert expression and all(m['n_target'] == 40 and m['n_reference'] == 40 for m in expression)
    assert all(m['scope']['cell_type_labels_provided'] is False for m in expression)
    assert all(m['reference'] == 'all spatial observations outside niche' for m in expression)
    assert (run / 'images/0_whole_s1.png').is_file()
    assert hashlib.sha256(spot_data.read_bytes()).hexdigest() == before


def test_explicit_missing_population_column_is_still_an_error(spot_data, tmp_path):
    from sclambda.pipelines.annotate_spatial.pipeline import AnnotateSpatialPipeline
    acc = Accessor(spot_data, cache_dir=str(tmp_path / 'cache'))
    pipeline = AnnotateSpatialPipeline(acc, None)
    with pytest.raises(ValueError, match="cell_type_col 'typo'"):
        pipeline.preprocess(['0'], cell_type_col='typo', spatial_key='spatial')
    assert 'spot' not in acc.obs
    assert main(['init', str(spot_data), '--run', str(tmp_path / 'run'), '--obs-column', 'niche',
                 '--spatial-key', 'spatial', '--cell-type-col', 'typo',
                 '--cache-dir', str(tmp_path / 'cache')]) != 0
