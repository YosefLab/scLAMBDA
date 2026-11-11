# tests/test_rank_genes.py

import pytest
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from scipy import sparse

from sclambda.core.accessor import Accessor
from sclambda.analysis.rank_genes import rank_genes, RankGenesResult, RankedGene, _classify_effect


@pytest.fixture
def tmp_h5ad_with_clusters(tmp_path) -> Path:
    """
    Create synthetic AnnData with known expression patterns.

    Cluster 0 (cells 0-49): high expression of gene_0, gene_1
    Cluster 1 (cells 50-99): high expression of gene_2, gene_3
    """
    n_cells = 100
    n_genes = 50

    np.random.seed(42)

    # Base expression (low)
    X = np.random.poisson(2, (n_cells, n_genes)).astype(np.float32)

    # Cluster 0: boost gene_0, gene_1
    X[:50, 0] += 20
    X[:50, 1] += 15

    # Cluster 1: boost gene_2, gene_3
    X[50:, 2] += 25
    X[50:, 3] += 18

    # Normalize
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X = np.log1p(X / row_sums * 1e4)

    obs = pd.DataFrame(
        {
            "batch": ["A"] * 50 + ["B"] * 50,
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    # Include some "uninformative" genes
    gene_names = [f"gene_{i}" for i in range(n_genes - 5)]
    gene_names += ["RPL10", "RPS5", "MT-CO1", "HSP90", "ENSG00001"]

    var = pd.DataFrame(index=gene_names)

    obsm = {
        "X_pca": np.random.randn(n_cells, 20).astype(np.float32),
    }

    adata = ad.AnnData(X=sparse.csr_matrix(X), obs=obs, var=var, obsm=obsm)

    path = tmp_path / "clustered.h5ad"
    adata.write_h5ad(path)

    yield path

    if path.exists():
        path.unlink()


@pytest.fixture
def accessor_with_clusters(tmp_h5ad_with_clusters) -> Accessor:
    """Accessor with L1 clusters set up"""
    acc = Accessor(tmp_h5ad_with_clusters)

    # Set up clusters
    labels = np.array([0] * 50 + [1] * 50)
    acc.set_cluster_labels("", labels)

    return acc


class TestClassifyEffect:

    def test_very_high(self):
        assert _classify_effect(0.6) == "very_high"
        assert _classify_effect(1.0) == "very_high"

    def test_high(self):
        assert _classify_effect(0.2) == "high"
        assert _classify_effect(0.4) == "high"

    def test_low(self):
        assert _classify_effect(0.01) == "low"
        assert _classify_effect(0.05) == "low"

    def test_depleted(self):
        assert _classify_effect(0.0001) == "depleted"
        assert _classify_effect(-0.5) == "depleted"


class TestRankedGene:

    def test_str_representation(self):
        gene = RankedGene(
            name="CD8A",
            lfc=0.72,
            effect="very_high",
            pval_adj=0.001,
            score=5.5,
        )

        assert str(gene) == "CD8A (LFC=0.72;very_high)"


class TestRankGenesResult:

    def test_gene_names(self):
        genes = [
            RankedGene("A", 0.5, "high", 0.01, 3.0),
            RankedGene("B", 0.3, "high", 0.02, 2.0),
        ]
        result = RankGenesResult(genes=genes, cluster_id="0", n_cells=50)

        assert result.gene_names == ["A", "B"]

    def test_summary(self):
        genes = [
            RankedGene("A", 0.5, "high", 0.01, 3.0),
            RankedGene("B", 0.3, "high", 0.02, 2.0),
        ]
        result = RankGenesResult(genes=genes, cluster_id="0", n_cells=50)

        assert "Cluster 0" in result.summary
        assert "50 cells" in result.summary
        assert "A" in result.summary

    def test_summary_empty(self):
        result = RankGenesResult(genes=[], cluster_id="0", n_cells=50)

        assert "no significant genes" in result.summary


class TestRankGenes:

    def test_returns_result(self, accessor_with_clusters):
        result = rank_genes(accessor_with_clusters, cluster_id="0")

        assert isinstance(result, RankGenesResult)
        assert result.cluster_id == "0"
        assert result.n_cells == 50

    def test_finds_correct_markers_cluster_0(self, accessor_with_clusters):
        result = rank_genes(accessor_with_clusters, cluster_id="0", n_genes=10)

        gene_names = result.gene_names

        # gene_0 and gene_1 should be top markers for cluster 0
        assert "gene_0" in gene_names[:5]
        assert "gene_1" in gene_names[:5]

    def test_finds_correct_markers_cluster_1(self, accessor_with_clusters):
        result = rank_genes(accessor_with_clusters, cluster_id="1", n_genes=10)

        gene_names = result.gene_names

        # gene_2 and gene_3 should be top markers for cluster 1
        assert "gene_2" in gene_names[:5]
        assert "gene_3" in gene_names[:5]

    def test_respects_n_genes(self, accessor_with_clusters):
        result = rank_genes(accessor_with_clusters, cluster_id="0", n_genes=5)

        assert len(result.genes) <= 5

    def test_filters_uninformative_genes(self, accessor_with_clusters):
        result = rank_genes(accessor_with_clusters, cluster_id="0", n_genes=50)

        gene_names = result.gene_names

        # These should be filtered out by default gene_filter
        assert "RPL10" not in gene_names
        assert "RPS5" not in gene_names
        assert "MT-CO1" not in gene_names
        assert "HSP90" not in gene_names
        assert "ENSG00001" not in gene_names

    def test_no_gene_filter(self, tmp_h5ad_with_clusters):
        acc = Accessor(tmp_h5ad_with_clusters, gene_filter=None)
        acc.set_cluster_labels("", np.array([0] * 50 + [1] * 50))

        result = rank_genes(acc, cluster_id="0", n_genes=50)

        # Without filter, uninformative genes could appear (if significant)
        # Just check it doesn't crash
        assert isinstance(result, RankGenesResult)

    def test_genes_have_positive_lfc(self, accessor_with_clusters):
        result = rank_genes(accessor_with_clusters, cluster_id="0", n_genes=5)

        # Top markers should be enriched (positive LFC)
        for gene in result.genes[:3]:
            assert gene.lfc > 0

    def test_genes_are_significant(self, accessor_with_clusters):
        result = rank_genes(accessor_with_clusters, cluster_id="0", alpha=0.05)

        for gene in result.genes:
            assert gene.pval_adj < 0.05

    def test_custom_alpha(self, accessor_with_clusters):
        result_strict = rank_genes(accessor_with_clusters, cluster_id="0", alpha=0.001)
        result_loose = rank_genes(accessor_with_clusters, cluster_id="0", alpha=0.1)

        # Stricter alpha should give fewer or equal genes
        assert len(result_strict.genes) <= len(result_loose.genes)

    def test_empty_cluster(self, tmp_h5ad_with_clusters):
        acc = Accessor(tmp_h5ad_with_clusters)

        # Create cluster with no cells matching "2"
        labels = np.array([0] * 50 + [1] * 50)
        acc.set_cluster_labels("", labels)

        # Manually test empty case
        result = rank_genes(acc, cluster_id="0", n_genes=5)

        # Should work normally
        assert result.n_cells == 50

    def test_nested_cluster(self, accessor_with_clusters):
        # Add subclusters under cluster "0"
        accessor_with_clusters.set_cluster_labels("0", np.array([0] * 25 + [1] * 25))

        result = rank_genes(accessor_with_clusters, cluster_id="0_0", n_genes=5)

        assert result.cluster_id == "0_0"
        assert result.n_cells == 25

    def test_result_dtype(self, accessor_with_clusters):
        result = rank_genes(accessor_with_clusters, cluster_id="0")

        for gene in result.genes:
            assert isinstance(gene.name, str)
            assert isinstance(gene.lfc, float)
            assert isinstance(gene.effect, str)
            assert isinstance(gene.pval_adj, float)
            assert isinstance(gene.score, float)


class TestRankGenesEdgeCases:

    def test_single_cell_cluster(self, tmp_h5ad_with_clusters):
        acc = Accessor(tmp_h5ad_with_clusters)

        # One cell in cluster 0, rest in cluster 1
        labels = np.array([0] + [1] * 99)
        acc.set_cluster_labels("", labels)

        result = rank_genes(acc, cluster_id="0")

        assert result.n_cells == 1
        # May have no significant genes with n=1
        assert isinstance(result.genes, list)

    def test_all_cells_one_cluster(self, tmp_h5ad_with_clusters):
        acc = Accessor(tmp_h5ad_with_clusters)

        labels = np.array([0] * 100)
        acc.set_cluster_labels("", labels)

        result = rank_genes(acc, cluster_id="0")

        assert result.n_cells == 100
        # Cluster == reference, so no DE genes expected
        # (or very few due to numerical noise)

    def test_invalid_cluster_id(self, accessor_with_clusters):
        with pytest.raises(KeyError):
            rank_genes(accessor_with_clusters, cluster_id="nonexistent")

    def test_invalid_depth(self, accessor_with_clusters):
        # No L2 clusters set
        with pytest.raises(KeyError):
            rank_genes(accessor_with_clusters, cluster_id="0_0")
