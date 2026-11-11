"""Cluster-name encoding: selected clusters are renamed, unselected ones are not
allowed to land on the new names.

Regression for a collision found on 2026-11-09: annotating a subset of the
brain's `niche_gmm100` niches (11, 15, 20, 23, 28, 34, 39, 65, 74, 81) renamed
them to "0".."9", while the 90 unselected niches kept their own names — so
unselected niche "9" and the encoded id of niche 81 were the same string, and
every niche was analysed together with an unrelated one (1,342 cells instead of
334 for that pair).
"""

import numpy as np
import pytest

from sclambda.framework.base_pipeline import Pipeline


class _Pipe(Pipeline):
    """Smallest concrete pipeline: the encoding lives on the base class."""

    name = "encoding-test"

    def build_graph(self, runner):  # pragma: no cover - never executed
        raise NotImplementedError


@pytest.fixture
def pipe():
    return _Pipe(accessor=None, llm_client=None, max_workers=1)


def test_subset_of_numeric_clusters_does_not_collide(pipe):
    pipe._create_cluster_encoding(["11", "81"])
    assert pipe._cluster_name_map == {"11": "0", "81": "1"}

    labels = pipe._apply_cluster_encoding(np.array(["11", "81", "0", "1", "9"]))

    assert list(labels) == ["0", "1", "other-0", "other-1", "other-9"]
    # No unselected cell may answer to a processed cluster's id.
    assert set(labels[2:]).isdisjoint(set(pipe._cluster_name_map.values()))


def test_every_cluster_selected_needs_no_encoding(pipe):
    assert pipe._create_cluster_encoding(["0", "1", "2"]) == ["0", "1", "2"]
    assert pipe._cluster_name_map == {}

    labels = np.array(["0", "1", "2"])
    assert list(pipe._apply_cluster_encoding(labels)) == ["0", "1", "2"]


def test_named_clusters_are_encoded_in_numeric_order(pipe):
    pipe._create_cluster_encoding(["10", "2", "CA1"])
    assert pipe._cluster_name_map == {"2": "0", "10": "1", "CA1": "2"}

    labels = pipe._apply_cluster_encoding(np.array(["CA1", "10", "2", "CA3"]))
    assert list(labels) == ["2", "1", "0", "other-CA3"]
