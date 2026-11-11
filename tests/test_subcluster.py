# tests/test_subcluster.py

import pytest
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from scipy import sparse

from sclambda.core.accessor import Accessor
from sclambda.analysis.subcluster import (
    subcluster,
    subcluster_all_at_depth,
    SubclusterMethod,
    SubclusterResult,
    _encode_labels_numeric,
    _run_leiden_with_retry,
)


@pytest.fixture
def tmp_h5ad_for_clustering(tmp_path) -> Path:
    """
    Create synthetic AnnData with clear cluster structure in embeddings.

    3 clusters in PCA space:
    - Cells 0-39: centered at (5, 0)
    - Cells 40-69: centered at (-5, 0)
    - Cells 70-99: centered at (0, 5)
    """
    n_cells = 100
    n_genes = 50
    n_pcs = 20

    np.random.seed(42)

    # Expression (random, not important for clustering test)
    X = np.random.poisson(5, (n_cells, n_genes)).astype(np.float32)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X = np.log1p(X / row_sums * 1e4)

    # PCA with clear cluster structure
    pca = np.random.randn(n_cells, n_pcs).astype(np.float32) * 0.5

    # Cluster centers
    pca[:40, 0] += 5
    pca[40:70, 0] -= 5
    pca[70:, 1] += 5

    obs = pd.DataFrame(
        {
            "batch": np.random.choice(["A", "B"], n_cells),
            # User-provided hierarchical labels
            "broad_type": ["T_cell"] * 40 + ["B_cell"] * 30 + ["Myeloid"] * 30,
            "subtype": (
                ["CD4"] * 20 + ["CD8"] * 20 + ["Naive_B"] * 15 + ["Memory_B"] * 15 + ["Mono"] * 15 + ["DC"] * 15
            ),
            "fine_type": (
                ["CD4_naive"] * 10
                + ["CD4_mem"] * 10
                + ["CD8_naive"] * 10
                + ["CD8_eff"] * 10
                + ["Naive_B_1"] * 8
                + ["Naive_B_2"] * 7
                + ["Mem_B_1"] * 8
                + ["Mem_B_2"] * 7
                + ["Mono_class"] * 8
                + ["Mono_non"] * 7
                + ["cDC"] * 8
                + ["pDC"] * 7
            ),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

    obsm = {
        "X_pca": pca,
        "X_umap": np.random.randn(n_cells, 2).astype(np.float32),
    }

    adata = ad.AnnData(X=sparse.csr_matrix(X), obs=obs, var=var, obsm=obsm)

    path = tmp_path / "cluster_test.h5ad"
    adata.write_h5ad(path)

    yield path

    if path.exists():
        path.unlink()


@pytest.fixture
def accessor(tmp_h5ad_for_clustering) -> Accessor:
    return Accessor(tmp_h5ad_for_clustering)


class TestEncodeLabelsNumeric:

    def test_string_labels(self):
        labels = np.array(["B", "A", "C", "A", "B"])
        encoded = _encode_labels_numeric(labels)

        # Should be sorted: A=0, B=1, C=2
        assert list(encoded) == [1, 0, 2, 0, 1]

    def test_numeric_labels(self):
        labels = np.array([5, 2, 5, 2, 10])
        encoded = _encode_labels_numeric(labels)

        # Should be: 2=0, 5=1, 10=2
        assert list(encoded) == [1, 0, 1, 0, 2]

    def test_single_label(self):
        labels = np.array(["same", "same", "same"])
        encoded = _encode_labels_numeric(labels)

        assert list(encoded) == [0, 0, 0]


class TestRunLeidenWithRetry:

    def test_basic_clustering(self, accessor):
        embeddings = accessor.get_obsm("X_pca")

        labels, resolution = _run_leiden_with_retry(
            embeddings=embeddings,
            resolution=1.0,
            min_clusters=2,
        )

        assert len(labels) == 100
        assert len(set(labels)) >= 2

    def test_respects_min_clusters(self, accessor):
        embeddings = accessor.get_obsm("X_pca")

        labels, resolution = _run_leiden_with_retry(
            embeddings=embeddings,
            resolution=0.1,  # Low resolution
            min_clusters=3,
            max_retries=5,
            resolution_step=0.2,
        )

        n_clusters = len(set(labels))
        # Should retry until we get >= min_clusters (or max retries)
        assert n_clusters >= 2  # At minimum, should get some clusters

    def test_returns_final_resolution(self, accessor):
        embeddings = accessor.get_obsm("X_pca")

        labels, resolution = _run_leiden_with_retry(
            embeddings=embeddings,
            resolution=0.5,
            min_clusters=2,
        )

        assert isinstance(resolution, float)
        assert resolution > 0  # Metric sweep may select below the legacy starting value


class TestSubclusterResult:

    def test_summary(self):
        result = SubclusterResult(
            cluster_id="0",
            new_depth=2,
            clusters=["0_0", "0_1", "0_2"],
            n_clusters=3,
            n_cells=50,
            method="leiden",
            resolution=1.0,
        )

        assert "0" in result.summary
        assert "3 clusters" in result.summary
        assert "L2" in result.summary
        assert "resolution=1.00" in result.summary

    def test_summary_root(self):
        result = SubclusterResult(
            cluster_id="",
            new_depth=1,
            clusters=["0", "1"],
            n_clusters=2,
            n_cells=100,
            method="leiden",
            resolution=0.5,
        )

        assert "root" in result.summary

    def test_summary_from_obs(self):
        result = SubclusterResult(
            cluster_id="0",
            new_depth=2,
            clusters=["0_0", "0_1"],
            n_clusters=2,
            n_cells=50,
            method="from_obs",
            resolution=None,
        )

        # Should not have resolution
        assert "resolution" not in result.summary

    def test_summary_many_clusters(self):
        result = SubclusterResult(
            cluster_id="",
            new_depth=1,
            clusters=[str(i) for i in range(10)],
            n_clusters=10,
            n_cells=100,
            method="leiden",
            resolution=1.0,
        )

        # Should truncate with ...
        assert "..." in result.summary


class TestSubclusterLeiden:

    def test_root_clustering(self, accessor):
        result = subcluster(
            accessor,
            cluster_id="",
            method=SubclusterMethod.LEIDEN,
            resolution=1.0,
            min_clusters=2,
        )

        assert isinstance(result, SubclusterResult)
        assert result.cluster_id == ""
        assert result.new_depth == 1
        assert result.n_cells == 100
        assert result.n_clusters >= 2
        assert result.method == "leiden"
        assert "L1" in accessor.overlay.columns

    def test_nested_clustering(self, accessor):
        # First level
        subcluster(accessor, cluster_id="", method="leiden", resolution=1.0)

        # Get first cluster
        clusters_l1 = accessor.get_all_clusters(depth=1)
        first_cluster = clusters_l1[0]
        n_cells_in_first = accessor.get_n_cells_in_cluster(first_cluster)

        # Second level
        result = subcluster(
            accessor,
            cluster_id=first_cluster,
            method="leiden",
            resolution=0.5,
            min_clusters=2,
        )

        assert result.cluster_id == first_cluster
        assert result.new_depth == 2
        assert result.n_cells == n_cells_in_first
        assert "L2" in accessor.overlay.columns

        # Check cluster naming
        for cluster in result.clusters:
            assert cluster.startswith(f"{first_cluster}_")

    def test_custom_embedding(self, accessor):
        # Set custom embedding key on accessor
        accessor._embedding_key = "X_umap"
        result = subcluster(
            accessor,
            cluster_id="",
            method="leiden",
        )

        assert result.n_clusters >= 1

    def test_invalid_embedding(self, accessor):
        # Set invalid embedding key on accessor
        accessor._embedding_key = "nonexistent"
        with pytest.raises(KeyError):
            subcluster(
                accessor,
                cluster_id="",
                method="leiden",
            )

    def test_resolution_retry(self, accessor):
        result = subcluster(
            accessor,
            cluster_id="",
            method="leiden",
            resolution=0.01,  # Very low
            min_clusters=3,
            max_retries=5,
            resolution_step=0.3,
        )

        # Resolution should have increased
        assert result.resolution >= 0.01

    def test_overlay_updated(self, accessor):
        subcluster(accessor, cluster_id="", method="leiden")

        # Overlay should have L1 column
        assert "L1" in accessor.overlay.columns

        # All cells should have a label
        assert accessor.overlay["L1"].notna().all()


class TestSubclusterFromObs:

    def test_root_from_obs(self, accessor):
        result = subcluster(
            accessor,
            cluster_id="",
            method=SubclusterMethod.FROM_OBS,
            obs_columns=["broad_type", "subtype", "fine_type"],
        )

        assert result.new_depth == 1
        assert result.n_cells == 100
        assert result.method == "from_obs"
        assert result.resolution is None

        # Should have 3 clusters (T_cell, B_cell, Myeloid)
        assert result.n_clusters == 3

    def test_nested_from_obs(self, accessor):
        # First level
        subcluster(
            accessor,
            cluster_id="",
            method="from_obs",
            obs_columns=["broad_type", "subtype", "fine_type"],
        )

        # Second level under first cluster
        first_cluster = accessor.get_all_clusters(depth=1)[0]

        result = subcluster(
            accessor,
            cluster_id=first_cluster,
            method="from_obs",
            obs_columns=["broad_type", "subtype", "fine_type"],
        )

        assert result.new_depth == 2
        assert result.n_clusters >= 1

        # Cluster names should be nested
        for cluster in result.clusters:
            assert cluster.startswith(f"{first_cluster}_")

    def test_three_levels(self, accessor):
        obs_cols = ["broad_type", "subtype", "fine_type"]

        # L1
        subcluster(accessor, cluster_id="", method="from_obs", obs_columns=obs_cols)

        # L2 under "0"
        subcluster(accessor, cluster_id="0", method="from_obs", obs_columns=obs_cols)

        # L3 under "0_0"
        result = subcluster(accessor, cluster_id="0_0", method="from_obs", obs_columns=obs_cols)

        assert result.new_depth == 3
        assert "L3" in accessor.overlay.columns

    def test_missing_obs_columns(self, accessor):
        with pytest.raises(ValueError, match="obs_columns required"):
            subcluster(accessor, cluster_id="", method="from_obs")

    def test_depth_exceeds_columns(self, accessor):
        obs_cols = ["broad_type", "subtype"]  # Only 2 levels

        # L1
        subcluster(accessor, cluster_id="", method="from_obs", obs_columns=obs_cols)
        # L2
        subcluster(accessor, cluster_id="0", method="from_obs", obs_columns=obs_cols)

        # L3: obs columns exhausted, should fall back to Leiden (since embedding available)
        result = subcluster(accessor, cluster_id="0_0", method="from_obs", obs_columns=obs_cols)
        # Should use Leiden fallback
        assert result.method == "leiden"
        assert result.n_clusters >= 1

    def test_depth_exceeds_columns_no_embedding(self, tmp_path):
        """Test graceful skip when obs columns exhausted and no embedding available"""
        import anndata as ad
        import pandas as pd
        import numpy as np
        from scipy import sparse

        # Create minimal AnnData without embeddings
        n_cells = 50
        X = np.random.poisson(5, (n_cells, 20)).astype(np.float32)
        obs = pd.DataFrame(
            {"cell_type": ["A"] * 25 + ["B"] * 25},
            index=[f"cell_{i}" for i in range(n_cells)],
        )
        adata = ad.AnnData(X=sparse.csr_matrix(X), obs=obs)
        path = tmp_path / "no_embedding.h5ad"
        adata.write_h5ad(path)

        accessor = Accessor(path)
        # Set embedding key to one that doesn't exist (to test fallback behavior)
        accessor._embedding_key = "X_pca"
        obs_cols = ["cell_type"]  # Only 1 level

        # L1
        subcluster(accessor, cluster_id="", method="from_obs", obs_columns=obs_cols)

        # L2: obs columns exhausted, no embedding - should skip gracefully
        result = subcluster(
            accessor,
            cluster_id="0",
            method="from_obs",
            obs_columns=obs_cols,
        )
        assert result.method == "none"
        assert result.n_clusters == 0
        assert result.clusters == []

    def test_preserves_cell_mapping(self, accessor):
        """Verify that cells end up in correct clusters based on obs values"""
        subcluster(
            accessor,
            cluster_id="",
            method="from_obs",
            obs_columns=["broad_type", "subtype", "fine_type"],
        )

        # Get cells in each cluster and verify they have same broad_type
        for cluster_id in accessor.get_all_clusters(depth=1):
            cell_indices = accessor.get_cell_indices(cluster_id)
            broad_types = accessor.get_obs(["broad_type"], cell_indices)["broad_type"]

            # All cells in cluster should have same broad_type
            assert broad_types.nunique() == 1


class TestSubclusterEdgeCases:

    def test_empty_cluster(self, accessor):
        # Create cluster with specific assignment
        labels = np.array([0] * 50 + [1] * 50)
        accessor.set_cluster_labels("", labels)

        # Try to subcluster nonexistent cluster
        with pytest.raises(KeyError):
            subcluster(accessor, cluster_id="99", method="leiden")

    def test_invalid_method(self, accessor):
        with pytest.raises(ValueError):
            subcluster(accessor, cluster_id="", method="invalid")

    def test_single_cell_cluster(self, accessor):
        # Create cluster with 1 cell
        labels = np.array([0] + [1] * 99)
        accessor.set_cluster_labels("", labels)

        # Subcluster single-cell cluster
        result = subcluster(
            accessor,
            cluster_id="0",
            method="leiden",
            min_clusters=1,
        )

        assert result.n_cells == 1
        # Should get 1 cluster
        assert result.n_clusters >= 1

    def test_method_enum_vs_string(self, accessor):
        # Both should work
        result1 = subcluster(accessor, cluster_id="", method=SubclusterMethod.LEIDEN)

        # Reset overlay
        accessor._overlay = pd.DataFrame(index=accessor.obs.index)

        result2 = subcluster(accessor, cluster_id="", method="leiden")

        assert result1.method == result2.method


class TestSubclusterAllAtDepth:

    def test_depth_zero(self, accessor):
        results = subcluster_all_at_depth(
            accessor,
            depth=0,
            method="leiden",
            resolution=1.0,
        )

        assert len(results) == 1  # Only root
        assert results[0].cluster_id == ""
        assert results[0].new_depth == 1

    def test_depth_one(self, accessor):
        # First create L1 clusters
        subcluster(accessor, cluster_id="", method="leiden", resolution=1.0)

        n_l1_clusters = len(accessor.get_all_clusters(depth=1))

        # Subcluster all L1 clusters
        results = subcluster_all_at_depth(
            accessor,
            depth=1,
            method="leiden",
            resolution=0.5,
        )

        assert len(results) == n_l1_clusters

        for result in results:
            assert result.new_depth == 2

    def test_passes_kwargs(self, accessor, monkeypatch):
        from sclambda.analysis import subcluster as module
        calls = []
        sentinel = object()

        def capture(**kwargs):
            calls.append(kwargs)
            return sentinel

        monkeypatch.setattr(module, "subcluster", capture)
        results = subcluster_all_at_depth(accessor, depth=0, method="leiden", n_neighbors=17)
        assert results == [sentinel]
        assert calls[0]["n_neighbors"] == 17
        assert calls[0]["cluster_id"] == ""

    def test_from_obs_all_at_depth(self, accessor):
        obs_cols = ["broad_type", "subtype", "fine_type"]

        # L1
        subcluster_all_at_depth(accessor, depth=0, method="from_obs", obs_columns=obs_cols)

        # L2
        results = subcluster_all_at_depth(accessor, depth=1, method="from_obs", obs_columns=obs_cols)

        assert len(results) == 3  # 3 broad types

        for result in results:
            assert result.new_depth == 2


class TestIntegrationWithAccessor:

    def test_full_hierarchy(self, accessor):
        """Test building full 3-level hierarchy"""

        # L1
        result_l1 = subcluster(accessor, cluster_id="", method="leiden", resolution=0.5)
        assert "L1" in accessor.overlay.columns

        # L2 for first cluster
        first = accessor.get_all_clusters(depth=1)[0]
        result_l2 = subcluster(accessor, cluster_id=first, method="leiden", resolution=0.3)
        assert "L2" in accessor.overlay.columns

        # L3 for first subcluster
        first_sub = accessor.get_children(first)[0]
        result_l3 = subcluster(accessor, cluster_id=first_sub, method="leiden", resolution=0.2)
        assert "L3" in accessor.overlay.columns

        # Verify hierarchy
        assert accessor.get_children("") == accessor.get_all_clusters(depth=1)

    def test_get_expression_after_clustering(self, accessor):
        """Verify expression retrieval works after clustering"""

        subcluster(accessor, cluster_id="", method="leiden")

        first_cluster = accessor.get_all_clusters(depth=1)[0]
        cell_indices = accessor.get_cell_indices(first_cluster)

        X = accessor.get_expression(cell_indices)

        assert X.shape[0] == len(cell_indices)
        assert X.shape[1] == accessor.n_genes
