import numpy as np
import pandas as pd
import pytest

from sclambda.core.accessor import Accessor


def accessor(labels):
    acc = Accessor.__new__(Accessor)
    acc._overlay = pd.DataFrame({'L3': labels})
    ids = {x for x in labels if x is not None}
    acc._hypotheses = {x: [{'source': x}] for x in ids}
    acc._ont_cell_types = {x: f'term:{x}' for x in ids}
    acc._cluster_stats_cache = {'old': 1}
    return acc


def test_lexicographic_collision_preserves_every_cell_and_metadata():
    labels = [f'5_1_{i}' for i in range(14) for _ in range(i + 1)] + ['9_0_0', None]
    acc = accessor(labels)
    ids = [f'5_1_{i}' for i in range(14)]
    mapping = acc.reindex_clusters('5_1', ids)
    expected = [mapping.get(x, x) for x in labels]
    # Missing labels stay missing (None under pandas 2, NaN under pandas 3).
    assert [None if pd.isna(x) else x for x in acc._overlay.L3] == expected
    assert acc._overlay.L3.nunique() == 15
    for old in ids:
        new = mapping.get(old, old)
        np.testing.assert_array_equal(acc._overlay.L3.eq(new), pd.Series(labels).eq(old))
        assert acc._hypotheses[new] == [{'source': old}]
        assert acc._ont_cell_types[new] == f'term:{old}'
    assert acc._hypotheses['9_0_0'] == [{'source': '9_0_0'}]
    assert not acc._cluster_stats_cache


@pytest.mark.parametrize('kept', [
    ['5_1_0', '5_1_0'], ['5_1_0', '5_1_99'], ['5_1_0', '9_0_0'],
    ['5_1_10'],  # target 5_1_0 exists outside this rename request
])
def test_invalid_reindex_leaves_data_unchanged(kept):
    acc = accessor(['5_1_0', '5_1_10', '9_0_0', None])
    before = acc._overlay.copy(deep=True)
    hypotheses = dict(acc._hypotheses)
    with pytest.raises(ValueError):
        acc.reindex_clusters('5_1', kept)
    pd.testing.assert_frame_equal(before, acc._overlay)
    assert acc._hypotheses == hypotheses


def test_reindex_rejects_existing_descendants():
    acc = accessor(['5_1_10', '5_1_11'])
    acc._overlay['L4'] = ['5_1_10_0', None]
    with pytest.raises(ValueError, match='descendants'):
        acc.reindex_clusters('5_1', ['5_1_10', '5_1_11'])


def test_contiguous_reindex_is_noop():
    acc = accessor(['5_1_0', '5_1_1', None])
    assert acc.reindex_clusters('5_1', ['5_1_0', '5_1_1']) == {}
