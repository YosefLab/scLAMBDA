"""Tests for targeted differential expression tool."""

import pytest
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from scipy import sparse

from sclambda.core.accessor import Accessor
from sclambda.analysis.subcluster import subcluster
from sclambda.analysis.targeted_de import (
    compute_de_for_genes,
    get_reference_indices,
    targeted_de_signatures,
    targeted_de,  # alias
    classify_effect,
    _compute_extended_metrics,
    DE_METRICS_LEGEND,
    GeneDE,
    GenesDEResult,
    SignatureDE,
    SignaturesDEResult,
)


@pytest.fixture
def accessor_with_clusters(tmp_path) -> Accessor:
    """Create accessor with two clusters and some differentially expressed genes."""
    n_cells = 200
    n_genes = 30

    np.random.seed(42)

    # Create two clusters: 0 and 1
    cluster_labels = np.array([0] * 100 + [1] * 100)

    # Create expression data with differentially expressed genes
    # Genes 0-4: highly expressed in cluster 0
    # Genes 5-9: highly expressed in cluster 1
    # Genes 10-29: similar in both

    X = np.zeros((n_cells, n_genes), dtype=np.float32)

    # Background expression
    X[:, :] = np.random.poisson(5, (n_cells, n_genes)).astype(np.float32)

    # Cluster 0 markers (indices 0-4)
    X[:100, 0:5] = np.random.poisson(50, (100, 5)).astype(np.float32)

    # Cluster 1 markers (indices 5-9)
    X[100:, 5:10] = np.random.poisson(50, (100, 5)).astype(np.float32)

    # Normalize and log transform
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X = np.log1p(X / row_sums * 1e4)

    # Use gene symbols for testing
    gene_names = [f"GENE{i}" for i in range(n_genes)]

    obs = pd.DataFrame(
        {"batch": ["A"] * n_cells},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=gene_names)
    obsm = {"X_pca": np.random.randn(n_cells, 10).astype(np.float32)}

    adata = ad.AnnData(X=sparse.csr_matrix(X), obs=obs, var=var, obsm=obsm)
    path = tmp_path / "de_test.h5ad"
    adata.write_h5ad(path)

    acc = Accessor(path, gene_filter=None)

    # Create L1 clustering
    acc._overlay["L1"] = ["0"] * 100 + ["1"] * 100

    return acc


@pytest.fixture
def accessor_hierarchical(tmp_path) -> Accessor:
    """Create accessor with hierarchical clusters (2 levels)."""
    n_cells = 400
    n_genes = 20

    np.random.seed(123)

    # L1: two clusters (0, 1) with 200 cells each
    # L2: 0 splits into 0_0, 0_1; 1 splits into 1_0, 1_1

    X = np.random.poisson(10, (n_cells, n_genes)).astype(np.float32)

    # Create differential expression for L2 subclusters
    # Cluster 0_0 (cells 0-99): high expression of gene 0
    X[:100, 0] = np.random.poisson(100, 100).astype(np.float32)
    # Cluster 0_1 (cells 100-199): high expression of gene 1
    X[100:200, 1] = np.random.poisson(100, 100).astype(np.float32)

    # Normalize
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X = np.log1p(X / row_sums * 1e4)

    gene_names = [f"MARKER{i}" for i in range(n_genes)]

    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=gene_names)
    obsm = {"X_pca": np.random.randn(n_cells, 10).astype(np.float32)}

    adata = ad.AnnData(X=sparse.csr_matrix(X), obs=obs, var=var, obsm=obsm)
    path = tmp_path / "hierarchical.h5ad"
    adata.write_h5ad(path)

    acc = Accessor(path, gene_filter=None)

    # Create hierarchical overlay
    acc._overlay["L1"] = ["0"] * 200 + ["1"] * 200
    acc._overlay["L2"] = ["0_0"] * 100 + ["0_1"] * 100 + ["1_0"] * 100 + ["1_1"] * 100

    return acc


class TestClassifyEffect:
    def test_very_high(self):
        assert "very_high" in classify_effect(2.0)
        assert "LFC=2.00" in classify_effect(2.0)

    def test_high(self):
        assert "high" in classify_effect(0.6)
        assert "very_high" not in classify_effect(0.6)

    def test_medium(self):
        assert "medium" in classify_effect(0.05)
        assert "high" not in classify_effect(0.05)

    def test_depleted(self):
        assert "depleted" in classify_effect(-0.5)
        assert "depleted" in classify_effect(0.0001)


class TestGeneDE:
    def test_is_enriched(self):
        gene = GeneDE(
            name="TEST",
            mean_target=5.0,
            mean_reference=2.0,
            lfc=3.0,
            pval=0.01,
            n_target=100,
            n_reference=100,
        )
        assert gene.is_enriched
        assert not gene.is_depleted

    def test_is_depleted(self):
        gene = GeneDE(
            name="TEST",
            mean_target=2.0,
            mean_reference=5.0,
            lfc=-3.0,
            pval=0.01,
            n_target=100,
            n_reference=100,
        )
        assert gene.is_depleted
        assert not gene.is_enriched

    def test_not_significant(self):
        gene = GeneDE(
            name="TEST",
            mean_target=5.0,
            mean_reference=2.0,
            lfc=3.0,
            pval=0.1,  # Not significant
            n_target=100,
            n_reference=100,
        )
        assert not gene.is_enriched
        assert not gene.is_depleted

    def test_effect_property(self):
        gene = GeneDE(
            name="TEST",
            mean_target=5.0,
            mean_reference=2.0,
            lfc=0.6,
            pval=0.01,
            n_target=100,
            n_reference=100,
        )
        assert gene.effect == "LFC=0.60;high"
        assert "LFC=0.60" in gene.effect

    def test_str_representation(self):
        gene = GeneDE(
            name="CD8A",
            mean_target=5.0,
            mean_reference=2.0,
            lfc=0.6,
            pval=0.01,
            n_target=100,
            n_reference=100,
        )
        s = str(gene)
        assert "CD8A" in s
        assert "LFC=" in s


class TestExtendedMetrics:
    """Test the extended DE metrics computation."""

    def test_compute_extended_metrics_basic(self):
        """Test _compute_extended_metrics with known data."""
        np.random.seed(42)
        # Target: higher expression for gene 0, lower for gene 1
        X_target = np.array([
            [5.0, 0.0],
            [6.0, 0.0],
            [4.0, 1.0],
            [5.0, 0.5],
        ])
        X_reference = np.array([
            [2.0, 3.0],
            [1.0, 4.0],
            [3.0, 2.0],
            [2.0, 3.0],
        ])

        ext = _compute_extended_metrics(X_target, X_reference)

        # Gene 0: target (4,5,5,6) vs ref (1,2,2,3) - ref median = 2.0
        # All target cells > 2.0, so FAR = 1.0
        assert ext["frac_above_ref_median"][0] == 1.0

        # Gene 1: target (0,0,0.5,1) vs ref (2,3,3,4) - ref median = 3.0
        # No target cell > 3.0, so FAR = 0.0
        assert ext["frac_above_ref_median"][1] == 0.0

        # Gene 0: zero rate target = 0/4 = 0, zero rate ref = 0/4 = 0, ZRD = 0
        assert ext["zero_rate_diff"][0] == 0.0

        # Gene 1: fraction expressing target = 2/4, reference = 4/4, DFE = -0.5
        assert ext["zero_rate_diff"][1] == -0.5

        # Gene 0: all non-zero → ZE-LFC = mean(4,5,5,6) - mean(1,2,2,3) = 5.0 - 2.0 = 3.0
        assert abs(ext["zero_excl_lfc"][0] - 3.0) < 1e-5

        # Gene 1: non-zero target (0.5,1.0) mean=0.75, non-zero ref (2,3,3,4) mean=3.0
        # ZE-LFC = 0.75 - 3.0 = -2.25
        assert abs(ext["zero_excl_lfc"][1] - (-2.25)) < 1e-5

    def test_zero_excl_lfc_nan_when_all_zero(self):
        """ZE-LFC should be NaN when one group has all zeros."""
        X_target = np.array([[0.0], [0.0], [0.0]])
        X_reference = np.array([[1.0], [2.0], [3.0]])

        ext = _compute_extended_metrics(X_target, X_reference)
        assert np.isnan(ext["zero_excl_lfc"][0])

    def test_frac_above_ref_median_shift(self):
        """FAR ≈ 0.5 when distributions are the same."""
        np.random.seed(123)
        # Same distribution → FAR should be near 0.5
        X = np.random.randn(200, 1).clip(0) + 1.0
        X_target = X[:100]
        X_reference = X[100:]

        ext = _compute_extended_metrics(X_target, X_reference)
        assert 0.3 < ext["frac_above_ref_median"][0] < 0.7


class TestGeneDE_ExtendedMetrics:
    """Test GeneDE format_compact and new fields."""

    def test_format_compact_all_metrics(self):
        gene = GeneDE(
            name="CD3D", mean_target=5.0, mean_reference=2.0,
            lfc=1.5, pval=0.01, n_target=100, n_reference=100,
            frac_above_ref_median=0.85, zero_rate_diff=-0.05, zero_excl_lfc=1.2,
        )
        s = gene.format_compact()
        assert "CD3D" in s
        assert "LFC=+1.50" in s
        assert "FAR=0.85" in s
        assert "DFE=-0.05" in s
        assert "ZE-LFC=+1.20" in s

    def test_format_compact_with_direction(self):
        gene = GeneDE(
            name="CD3D", mean_target=5.0, mean_reference=2.0,
            lfc=1.5, pval=0.01, n_target=100, n_reference=100,
            frac_above_ref_median=0.85, zero_rate_diff=-0.05, zero_excl_lfc=1.2,
        )
        s = gene.format_compact(include_direction=True)
        assert "\u2191" in s  # enriched

    def test_format_compact_nan_metrics_omitted(self):
        gene = GeneDE(
            name="CD3D", mean_target=5.0, mean_reference=2.0,
            lfc=1.5, pval=0.01, n_target=100, n_reference=100,
        )  # defaults are NaN
        s = gene.format_compact()
        assert "FAR" not in s
        assert "ZRD" not in s
        assert "ZE-LFC" not in s
        assert "LFC=+1.50" in s


class TestGenesDEResult:
    def test_n_enriched(self):
        genes = [
            GeneDE("A", 5.0, 2.0, 3.0, 0.01, 100, 100),  # enriched
            GeneDE("B", 5.0, 2.0, 3.0, 0.01, 100, 100),  # enriched
            GeneDE("C", 2.0, 5.0, -3.0, 0.01, 100, 100),  # depleted
        ]
        result = GenesDEResult(genes, 100, 100)
        assert result.n_enriched == 2
        assert result.n_depleted == 1

    def test_mean_lfc(self):
        genes = [
            GeneDE("A", 5.0, 2.0, 2.0, 0.01, 100, 100),
            GeneDE("B", 4.0, 2.0, 1.0, 0.01, 100, 100),
        ]
        result = GenesDEResult(genes, 100, 100)
        assert result.mean_lfc == 1.5

    def test_get_gene(self):
        genes = [
            GeneDE("A", 5.0, 2.0, 3.0, 0.01, 100, 100),
            GeneDE("B", 4.0, 2.0, 1.0, 0.01, 100, 100),
        ]
        result = GenesDEResult(genes, 100, 100)
        assert result.get_gene("A") is not None
        assert result.get_gene("A").name == "A"
        assert result.get_gene("NONEXISTENT") is None

    def test_genes_not_found(self):
        genes = [GeneDE("A", 5.0, 2.0, 3.0, 0.01, 100, 100)]
        result = GenesDEResult(genes, 100, 100, genes_not_found=["B", "C"])
        assert result.genes_not_found == ["B", "C"]


class TestSignatureDE:
    def test_n_enriched(self):
        genes = [
            GeneDE("A", 5.0, 2.0, 3.0, 0.01, 100, 100),  # enriched
            GeneDE("B", 5.0, 2.0, 3.0, 0.01, 100, 100),  # enriched
            GeneDE("C", 2.0, 5.0, -3.0, 0.01, 100, 100),  # depleted
        ]
        sig = SignatureDE("test_sig", genes, "test justification")
        assert sig.n_enriched == 2
        assert sig.n_depleted == 1

    def test_mean_lfc(self):
        genes = [
            GeneDE("A", 5.0, 2.0, 2.0, 0.01, 100, 100),
            GeneDE("B", 4.0, 2.0, 1.0, 0.01, 100, 100),
        ]
        sig = SignatureDE("test_sig", genes, "test justification")
        assert sig.mean_lfc == 1.5

    def test_summary(self):
        genes = [
            GeneDE("A", 5.0, 2.0, 3.0, 0.01, 100, 100),  # enriched
            GeneDE("B", 2.0, 5.0, -3.0, 0.01, 100, 100),  # depleted
        ]
        sig = SignatureDE("markers", genes, "test")
        summary = sig.summary
        assert "markers" in summary
        assert "↑" in summary or "A" in summary


class TestComputeDEForGenes:
    """Test the low-level compute_de_for_genes function."""

    def test_basic_computation(self, accessor_with_clusters):
        """Test basic DE computation with explicit indices."""
        target_indices = np.arange(100)  # Cluster 0
        reference_indices = np.arange(100, 200)  # Cluster 1

        result = compute_de_for_genes(
            accessor_with_clusters,
            gene_names=["GENE0", "GENE5", "GENE10"],
            target_indices=target_indices,
            reference_indices=reference_indices,
        )

        assert result.n_target == 100
        assert result.n_reference == 100
        assert len(result.genes) == 3

    def test_enriched_detection(self, accessor_with_clusters):
        """Test that enriched genes are detected."""
        target_indices = np.arange(100)  # Cluster 0 (GENE0-4 enriched)
        reference_indices = np.arange(100, 200)

        result = compute_de_for_genes(
            accessor_with_clusters,
            gene_names=["GENE0", "GENE1"],
            target_indices=target_indices,
            reference_indices=reference_indices,
        )

        assert result.n_enriched >= 1
        assert result.mean_lfc > 0

    def test_depleted_detection(self, accessor_with_clusters):
        """Test that depleted genes are detected."""
        target_indices = np.arange(100)  # Cluster 0
        reference_indices = np.arange(100, 200)  # Cluster 1 (GENE5-9 enriched)

        result = compute_de_for_genes(
            accessor_with_clusters,
            gene_names=["GENE5", "GENE6"],
            target_indices=target_indices,
            reference_indices=reference_indices,
        )

        assert result.n_depleted >= 1
        assert result.mean_lfc < 0

    def test_missing_genes_tracked(self, accessor_with_clusters):
        """Test that missing genes are tracked."""
        target_indices = np.arange(100)
        reference_indices = np.arange(100, 200)

        result = compute_de_for_genes(
            accessor_with_clusters,
            gene_names=["GENE0", "NONEXISTENT1", "NONEXISTENT2"],
            target_indices=target_indices,
            reference_indices=reference_indices,
        )

        assert len(result.genes) == 1
        assert len(result.genes_not_found) == 2
        assert "NONEXISTENT1" in result.genes_not_found

    def test_empty_target(self, accessor_with_clusters):
        """Test handling of empty target population."""
        target_indices = np.array([], dtype=np.int64)
        reference_indices = np.arange(100)

        result = compute_de_for_genes(
            accessor_with_clusters,
            gene_names=["GENE0"],
            target_indices=target_indices,
            reference_indices=reference_indices,
        )

        assert result.n_target == 0
        assert len(result.genes) == 0

    def test_extended_metrics_populated(self, accessor_with_clusters):
        """Test that extended metrics are populated for each gene."""
        target_indices = np.arange(100)  # Cluster 0
        reference_indices = np.arange(100, 200)  # Cluster 1

        result = compute_de_for_genes(
            accessor_with_clusters,
            gene_names=["GENE0", "GENE5", "GENE10"],
            target_indices=target_indices,
            reference_indices=reference_indices,
        )

        for gene in result.genes:
            # All extended metrics should be populated (not NaN)
            assert not np.isnan(gene.frac_above_ref_median), f"{gene.name} FAR is NaN"
            assert not np.isnan(gene.zero_rate_diff), f"{gene.name} ZRD is NaN"
            # ZE-LFC may be NaN if all cells in one group are zero

        # GENE0 is enriched in cluster 0 → FAR should be > 0.5
        gene0 = result.get_gene("GENE0")
        assert gene0.frac_above_ref_median > 0.5

        # GENE5 is depleted in cluster 0 → FAR should be < 0.5
        gene5 = result.get_gene("GENE5")
        assert gene5.frac_above_ref_median < 0.5


class TestGetReferenceIndices:
    """Test the get_reference_indices helper."""

    def test_depth_1_uses_all_other(self, accessor_with_clusters):
        """Test that depth 1 clusters use all other cells."""
        target_indices = accessor_with_clusters.get_cell_indices("0")
        ref_indices, ref_type = get_reference_indices(
            accessor_with_clusters, "0", target_indices
        )

        assert ref_type == "all_other"
        assert len(ref_indices) == 100  # Other cluster

    def test_depth_2_uses_global_reference(self, accessor_hierarchical):
        """Depth 2+ uses all other cells; explicit contrasts choose their own reference."""
        target_indices = accessor_hierarchical.get_cell_indices("0_0")
        ref_indices, ref_type = get_reference_indices(
            accessor_hierarchical, "0_0", target_indices
        )

        assert ref_type == "all_other"
        assert len(ref_indices) == 300
        np.testing.assert_array_equal(ref_indices, np.arange(100, 400))


class TestTargetedDESignatures:
    def test_basic_de(self, accessor_with_clusters):
        """Test basic DE between two clusters."""
        signatures = {
            "cluster0_markers": {
                "genes": ["GENE0", "GENE1", "GENE2"],
                "justification": "confirm cluster 0 identity",
            },
            "cluster1_markers": {
                "genes": ["GENE5", "GENE6", "GENE7"],
                "justification": "rule out cluster 1 identity",
            },
        }

        result = targeted_de_signatures(accessor_with_clusters, "0", signatures)

        assert result.cluster_id == "0"
        assert result.n_target == 100
        assert result.n_reference == 100
        assert result.reference_type == "all_other"
        assert len(result.signatures) == 2

    def test_enriched_genes_detected(self, accessor_with_clusters):
        """Test that enriched genes are detected."""
        signatures = {
            "cluster0_markers": {
                "genes": ["GENE0", "GENE1"],
                "justification": "confirm",
            },
        }

        result = targeted_de_signatures(accessor_with_clusters, "0", signatures)

        sig = result.signatures[0]
        # These genes should be enriched in cluster 0
        assert sig.n_enriched >= 1
        assert sig.mean_lfc > 0

    def test_depleted_genes_detected(self, accessor_with_clusters):
        """Test that depleted genes are detected."""
        signatures = {
            "cluster1_markers": {
                "genes": ["GENE5", "GENE6"],
                "justification": "rule out",
            },
        }

        result = targeted_de_signatures(accessor_with_clusters, "0", signatures)

        sig = result.signatures[0]
        # These genes should be depleted in cluster 0 (enriched in cluster 1)
        assert sig.n_depleted >= 1
        assert sig.mean_lfc < 0

    def test_hierarchical_de(self, accessor_hierarchical):
        """Test a hierarchical target against the global reference."""
        signatures = {
            "subcluster_marker": {
                "genes": ["MARKER0"],
                "justification": "confirm subtype",
            },
        }

        result = targeted_de_signatures(accessor_hierarchical, "0_0", signatures)

        assert result.cluster_id == "0_0"
        assert result.reference_type == "all_other"
        assert result.n_target == 100
        assert result.n_reference == 300  # All cells except the current subcluster

        sig = result.signatures[0]
        assert len(sig.genes) == 1
        # MARKER0 should be enriched in 0_0
        assert sig.genes[0].lfc > 0

    def test_missing_genes_skipped(self, accessor_with_clusters):
        """Test that missing genes are silently skipped."""
        signatures = {
            "mixed": {
                "genes": ["GENE0", "NONEXISTENT_GENE", "GENE1"],
                "justification": "test",
            },
        }

        result = targeted_de_signatures(accessor_with_clusters, "0", signatures)

        sig = result.signatures[0]
        # Should only have 2 genes (GENE0, GENE1), NONEXISTENT_GENE skipped
        assert len(sig.genes) == 2
        assert all(g.name in ["GENE0", "GENE1"] for g in sig.genes)

    def test_empty_signatures(self, accessor_with_clusters):
        """Test with empty signatures dict."""
        result = targeted_de_signatures(accessor_with_clusters, "0", {})

        assert result.cluster_id == "0"
        assert len(result.signatures) == 0

    def test_empty_cluster_id_raises(self, accessor_with_clusters):
        """Test that empty cluster_id raises error."""
        with pytest.raises(ValueError, match="cluster_id required"):
            targeted_de_signatures(accessor_with_clusters, "", {"sig": {"genes": ["GENE0"]}})

    def test_depth_1_uses_all_other(self, accessor_with_clusters):
        """Test that depth 1 clusters use all other cells as reference."""
        signatures = {"test": {"genes": ["GENE0"], "justification": "test"}}

        result = targeted_de_signatures(accessor_with_clusters, "0", signatures)

        assert result.reference_type == "all_other"
        assert result.n_reference == 100  # Other cluster

    def test_insufficient_cells(self, accessor_with_clusters):
        """Test handling of insufficient cells."""
        # Create a tiny cluster
        accessor_with_clusters._overlay["L1"] = ["0"] * 5 + ["1"] * 195

        signatures = {"test": {"genes": ["GENE0"], "justification": "test"}}

        # With min_cells=10, cluster 0 should be insufficient
        result = targeted_de_signatures(accessor_with_clusters, "0", signatures, min_cells=10)

        assert result.reference_type == "insufficient_cells"
        assert len(result.signatures) == 0

    def test_alias_works(self, accessor_with_clusters):
        """Test that targeted_de alias works."""
        signatures = {"test": {"genes": ["GENE0"], "justification": "test"}}

        # targeted_de is an alias for targeted_de_signatures
        result = targeted_de(accessor_with_clusters, "0", signatures)

        assert result.cluster_id == "0"
        assert len(result.signatures) == 1


class TestAccessorGeneIndexing:
    """Test gene indexing methods in accessor."""

    def test_get_gene_indices(self, accessor_with_clusters):
        """Test getting gene indices by name."""
        indices, names = accessor_with_clusters.get_gene_indices(["GENE0", "GENE5", "GENE10"])

        assert len(indices) == 3
        assert len(names) == 3
        assert list(indices) == [0, 5, 10]
        assert names == ["GENE0", "GENE5", "GENE10"]

    def test_get_gene_indices_missing(self, accessor_with_clusters):
        """Test that missing genes are skipped."""
        indices, names = accessor_with_clusters.get_gene_indices(
            ["GENE0", "NONEXISTENT", "GENE1"]
        )

        assert len(indices) == 2
        assert len(names) == 2
        assert "NONEXISTENT" not in names

    def test_get_expression_for_genes(self, accessor_with_clusters):
        """Test fetching expression for specific genes."""
        cell_indices = np.arange(10)
        gene_names = ["GENE0", "GENE5"]

        X, found = accessor_with_clusters.get_expression_for_genes(cell_indices, gene_names)

        assert X.shape == (10, 2)
        assert len(found) == 2
        assert found == ["GENE0", "GENE5"]

    def test_get_parent_cluster_id(self, accessor_hierarchical):
        """Test getting parent cluster ID."""
        assert accessor_hierarchical.get_parent_cluster_id("0_0") == "0"
        assert accessor_hierarchical.get_parent_cluster_id("1_0") == "1"
        assert accessor_hierarchical.get_parent_cluster_id("0") == ""
        assert accessor_hierarchical.get_parent_cluster_id("") == ""


class TestFormattingWithExtendedMetrics:
    """Test that formatting includes new metrics."""

    def test_pathology_de_result_format(self):
        """PathologyDEResult.format_for_prompt includes legend + detection + metrics."""
        from sclambda.analysis.targeted_de import PathologyDEResult

        result = PathologyDEResult(
            cluster_id="0",
            cluster_cell_type="T cell",
            n_target=100,
            n_reference=200,
            enriched_genes=[
                GeneDE("CD3D", 5.0, 2.0, 1.5, 0.01, 100, 200,
                       frac_above_ref_median=0.85, zero_rate_diff=-0.05, zero_excl_lfc=1.2),
            ],
            depleted_genes=[
                GeneDE("FOXP3", 1.0, 3.0, -2.0, 0.01, 100, 200,
                       frac_above_ref_median=0.15, zero_rate_diff=0.10, zero_excl_lfc=-1.8),
            ],
            mean_genes_detected_target=1200.0,
            mean_genes_detected_reference=1450.0,
        )
        formatted = result.format_for_prompt()

        assert "DE metrics:" in formatted
        assert "Detection: 1200 genes/cell (target) vs 1450 (ref)" in formatted
        assert "FAR=0.85" in formatted
        assert "DFE=-0.05" in formatted
        assert "ZE-LFC=+1.20" in formatted

    def test_signatures_de_result_format(self):
        """SignaturesDEResult.format_for_reevaluation_prompt includes legend."""
        sde = SignaturesDEResult(
            cluster_id="0",
            reference_type="parent",
            n_target=100,
            n_reference=100,
            signatures=[
                SignatureDE(
                    signature_name="T_markers",
                    genes=[
                        GeneDE("CD3D", 5.0, 2.0, 1.5, 0.01, 100, 100,
                               frac_above_ref_median=0.85, zero_rate_diff=-0.05, zero_excl_lfc=1.2),
                    ],
                    justification="confirm T cell",
                ),
            ],
        )
        from sclambda.annotation.celltype.formatters import SignatureFormatter
        formatted = SignatureFormatter(signature_de_result_obj=sde).format()
        assert "Delta Frac Expressing" in formatted
        assert "| CD3D | 5.00 | +1.50 | 1.00 | -0.05 | T_markers |" in formatted
        assert "confirm T cell" in formatted

    def test_format_compact_used_in_compute_de(self, accessor_with_clusters):
        """Verify format_compact works on real DE results."""
        target_indices = np.arange(100)
        reference_indices = np.arange(100, 200)

        result = compute_de_for_genes(
            accessor_with_clusters,
            gene_names=["GENE0", "GENE5"],
            target_indices=target_indices,
            reference_indices=reference_indices,
        )

        for gene in result.genes:
            compact = gene.format_compact(include_direction=True)
            assert gene.name in compact
            assert "LFC=" in compact
            assert "FAR=" in compact
            assert "DFE=" in compact


class TestTotalCounts:
    """Test total counts computation."""

    def test_total_counts_computed(self, accessor_with_clusters):
        """Test that total counts are computed."""
        # Trigger computation
        _ = accessor_with_clusters.ref_mean

        assert accessor_with_clusters._total_counts is not None
        assert len(accessor_with_clusters._total_counts) == accessor_with_clusters.n_cells

    def test_total_counts_property(self, accessor_with_clusters):
        """Test total_counts property."""
        total = accessor_with_clusters.total_counts

        assert isinstance(total, np.ndarray)
        assert len(total) == accessor_with_clusters.n_cells
        assert total.dtype == np.float32
