import pytest
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from scipy import sparse

from sclambda.core.accessor import Accessor


@pytest.fixture
def tmp_h5ad_normalized(tmp_path) -> Path:
    """Create synthetic AnnData with normalized data (log1p, target_sum=1e4)"""
    n_cells = 100
    n_genes = 50

    np.random.seed(42)

    # Simulate raw counts, then normalize
    raw = np.random.poisson(5, (n_cells, n_genes)).astype(np.float32)
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X = np.log1p(raw / row_sums * 1e4)

    obs = pd.DataFrame(
        {
            "batch": np.random.choice(["A", "B"], n_cells),
            "sample": np.random.choice(["S1", "S2", "S3"], n_cells),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    var = pd.DataFrame(
        {
            "highly_variable": np.random.choice([True, False], n_genes),
        },
        index=[f"gene_{i}" for i in range(n_genes)],
    )

    obsm = {
        "X_pca": np.random.randn(n_cells, 20).astype(np.float32),
        "X_umap": np.random.randn(n_cells, 2).astype(np.float32),
    }

    adata = ad.AnnData(X=sparse.csr_matrix(X), obs=obs, var=var, obsm=obsm)

    path = tmp_path / "normalized.h5ad"
    adata.write_h5ad(path)

    yield path

    if path.exists():
        path.unlink()


@pytest.fixture
def tmp_h5ad_raw(tmp_path) -> Path:
    """Create synthetic AnnData with raw integer counts"""
    n_cells = 100
    n_genes = 50

    np.random.seed(42)

    # Raw integer counts
    X = np.random.poisson(5, (n_cells, n_genes)).astype(np.float32)

    obs = pd.DataFrame(
        {
            "batch": np.random.choice(["A", "B"], n_cells),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    var = pd.DataFrame(
        {
            "highly_variable": np.random.choice([True, False], n_genes),
        },
        index=[f"gene_{i}" for i in range(n_genes)],
    )

    obsm = {
        "X_pca": np.random.randn(n_cells, 20).astype(np.float32),
    }

    adata = ad.AnnData(X=sparse.csr_matrix(X), obs=obs, var=var, obsm=obsm)

    path = tmp_path / "raw.h5ad"
    adata.write_h5ad(path)

    yield path

    if path.exists():
        path.unlink()


class TestAccessorInit:

    def test_loads_dimensions(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        assert acc.n_cells == 100
        assert acc.n_genes == 50

    def test_loads_obs(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        assert isinstance(acc.obs, pd.DataFrame)
        assert len(acc.obs) == 100
        assert "batch" in acc.obs.columns

    def test_loads_var(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        assert isinstance(acc.var, pd.DataFrame)
        assert len(acc.var) == 50
        assert "highly_variable" in acc.var.columns

    def test_loads_obsm(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        assert "X_pca" in acc.obsm_keys
        assert "X_umap" in acc.obsm_keys

    def test_gene_names(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        assert len(acc.gene_names) == 50
        assert acc.gene_names[0] == "gene_0"

    def test_overlay_starts_empty(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        assert len(acc.overlay) == 100
        assert len(acc.overlay.columns) == 0


class TestNormalizationCheck:

    def test_detects_normalized_data(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        assert acc.needs_normalization is False

    def test_detects_raw_counts(self, tmp_h5ad_raw):
        acc = Accessor(tmp_h5ad_raw)

        assert acc.needs_normalization is True

    def test_normalizes_raw_on_fetch(self, tmp_h5ad_raw):
        acc = Accessor(tmp_h5ad_raw, target_sum=1e4)

        X = acc.get_expression(np.array([0, 1, 2]))

        # After normalization: non-integer, reasonable range
        assert not np.allclose(X, X.astype(int))
        assert X.min() >= 0
        assert X.max() < 15  # log1p(1e4) ≈ 9.2


class TestReferenceStats:

    def test_computes_ref_mean(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        assert acc.ref_mean.shape == (50,)
        assert acc.ref_mean.dtype == np.float32

    def test_computes_ref_var(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        assert acc.ref_var.shape == (50,)
        assert acc.ref_var.dtype == np.float32
        assert (acc.ref_var >= 0).all()

    def test_computes_ref_n(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        assert acc.ref_n == 100

    def test_set_reference_stats(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        fake_mean = np.ones(50, dtype=np.float32)
        fake_var = np.ones(50, dtype=np.float32) * 0.5

        acc.set_reference_stats(fake_mean, fake_var, n=999)

        assert np.allclose(acc.ref_mean, 1.0)
        assert np.allclose(acc.ref_var, 0.5)
        assert acc.ref_n == 999

    def test_set_reference_stats_wrong_length(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        with pytest.raises(ValueError, match="must match n_genes"):
            acc.set_reference_stats(np.ones(10), np.ones(10), n=100)

    def test_chunked_computation_matches_direct(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized, chunk_size=30)

        # Get chunked result
        chunked_mean = acc.ref_mean.copy()
        chunked_var = acc.ref_var.copy()

        # Compute directly
        X_all = acc.get_expression(np.arange(100))
        direct_mean = X_all.mean(axis=0)
        direct_var = X_all.var(axis=0)

        assert np.allclose(chunked_mean, direct_mean, rtol=1e-2)
        assert np.allclose(chunked_var, direct_var, rtol=1e-2)


class TestSubsetting:

    def test_get_expression_all_genes(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)
        idx = np.array([0, 10, 20])

        X = acc.get_expression(idx)

        assert X.shape == (3, 50)
        assert X.dtype == np.float32

    def test_get_expression_subset_genes(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)
        idx = np.array([0, 10, 20])
        genes = ["gene_0", "gene_5", "gene_10"]

        X = acc.get_expression(idx, genes=genes)

        assert X.shape == (3, 3)

    def test_get_expression_invalid_gene(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        with pytest.raises(KeyError, match="not found"):
            acc.get_expression(np.array([0]), genes=["nonexistent"])

    def test_get_obsm(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)
        idx = np.array([0, 10, 20])

        pca = acc.get_obsm("X_pca", idx)

        assert pca.shape == (3, 20)

    def test_get_obsm_all_cells(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        pca = acc.get_obsm("X_pca")

        assert pca.shape == (100, 20)

    def test_get_obsm_invalid_key(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        with pytest.raises(KeyError, match="not found"):
            acc.get_obsm("nonexistent")

    def test_get_obs_subset_columns(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)
        idx = np.array([0, 10, 20])

        df = acc.get_obs(["batch"], idx)

        assert df.shape == (3, 1)
        assert "batch" in df.columns

    def test_get_obs_all(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        df = acc.get_obs()

        assert len(df) == 100


class TestClusterStats:

    def test_compute_cluster_stats(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)
        idx = np.array([0, 1, 2, 3, 4])

        stats = acc.compute_cluster_stats(idx)

        assert "mean" in stats
        assert "var" in stats
        assert "n" in stats
        assert stats["mean"].shape == (50,)
        assert stats["var"].shape == (50,)
        assert stats["n"] == 5

    def test_cluster_stats_dtype(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)
        idx = np.array([0, 1, 2])

        stats = acc.compute_cluster_stats(idx)

        assert stats["mean"].dtype == np.float32
        assert stats["var"].dtype == np.float32


class TestOverlayAndClustering:

    def test_set_cluster_labels_root(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        labels = np.array([0] * 50 + [1] * 50)
        result = acc.set_cluster_labels("", labels)

        assert result["depth"] == 1
        assert result["n_clusters"] == 2
        assert "0" in result["clusters"]
        assert "1" in result["clusters"]
        assert "L1" in acc.overlay.columns

    def test_set_cluster_labels_nested(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        # First level
        labels_l1 = np.array([0] * 50 + [1] * 50)
        acc.set_cluster_labels("", labels_l1)

        # Second level under cluster "0"
        labels_l2 = np.array([0] * 25 + [1] * 25)
        result = acc.set_cluster_labels("0", labels_l2)

        assert result["depth"] == 2
        assert "0_0" in result["clusters"]
        assert "0_1" in result["clusters"]
        assert "L2" in acc.overlay.columns

    def test_set_cluster_labels_wrong_length(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        with pytest.raises(ValueError, match="Labels length"):
            acc.set_cluster_labels("", np.array([0, 1, 2]))

    def test_get_cell_indices_all(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        idx = acc.get_cell_indices("")

        assert len(idx) == 100

    def test_get_cell_indices_cluster(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        labels = np.array([0] * 30 + [1] * 70)
        acc.set_cluster_labels("", labels)

        idx_0 = acc.get_cell_indices("0")
        idx_1 = acc.get_cell_indices("1")

        assert len(idx_0) == 30
        assert len(idx_1) == 70

    def test_get_cell_indices_nested(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        # L1: 50/50 split
        acc.set_cluster_labels("", np.array([0] * 50 + [1] * 50))
        # L2 under "0": 20/30 split
        acc.set_cluster_labels("0", np.array([0] * 20 + [1] * 30))

        idx = acc.get_cell_indices("0_1")

        assert len(idx) == 30

    def test_get_cell_indices_invalid_level(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        with pytest.raises(KeyError, match="not computed"):
            acc.get_cell_indices("0")

    def test_get_n_cells_in_cluster(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        acc.set_cluster_labels("", np.array([0] * 40 + [1] * 60))

        assert acc.get_n_cells_in_cluster("0") == 40
        assert acc.get_n_cells_in_cluster("1") == 60

    def test_get_all_clusters(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        acc.set_cluster_labels("", np.array([0] * 30 + [1] * 30 + [2] * 40))

        clusters = acc.get_all_clusters(depth=1)

        assert clusters == ["0", "1", "2"]

    def test_get_all_clusters_empty_depth(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        clusters = acc.get_all_clusters(depth=1)

        assert clusters == []

    def test_get_children_root(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        acc.set_cluster_labels("", np.array([0] * 50 + [1] * 50))

        children = acc.get_children("")

        assert children == ["0", "1"]

    def test_get_children_nested(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        acc.set_cluster_labels("", np.array([0] * 50 + [1] * 50))
        acc.set_cluster_labels("0", np.array([0] * 20 + [1] * 15 + [2] * 15))

        children = acc.get_children("0")

        assert children == ["0_0", "0_1", "0_2"]

    def test_get_children_no_children(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        acc.set_cluster_labels("", np.array([0] * 50 + [1] * 50))

        children = acc.get_children("0")

        assert children == []


class TestGeneFilter:

    def test_default_filter(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized)

        gene_filter = acc.gene_filter

        assert gene_filter("CD8A") is True
        assert gene_filter("RPL10") is False
        assert gene_filter("RPS5") is False
        assert gene_filter("MT-CO1") is False
        assert gene_filter("HSP90") is False
        assert gene_filter("ENSG00000123") is False

    def test_custom_filter(self, tmp_h5ad_normalized):
        custom = lambda g: not g.startswith("gene_1")
        acc = Accessor(tmp_h5ad_normalized, gene_filter=custom)

        assert acc.gene_filter("gene_0") is True
        assert acc.gene_filter("gene_10") is False

    def test_no_filter(self, tmp_h5ad_normalized):
        acc = Accessor(tmp_h5ad_normalized, gene_filter=None)

        assert acc.gene_filter is None


class TestGeneCol:
    """Tests for gene_col parameter and gene name validation."""

    @pytest.fixture
    def h5ad_with_gene_symbols(self, tmp_path) -> Path:
        """Create AnnData with gene symbols in var column."""
        n_cells = 50
        n_genes = 30

        np.random.seed(42)
        X = np.random.poisson(5, (n_cells, n_genes)).astype(np.float32)
        X = np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4)

        # Use gene symbols that match the pattern
        gene_symbols = [
            "TP53", "BRCA1", "EGFR", "CD8A", "CD4", "HLA-DRB1", "FOXP3",
            "IL2RA", "GATA3", "TBX21", "BCL2", "MYC", "KRAS", "BRAF",
            "PIK3CA", "AKT1", "MTOR", "PTEN", "RB1", "CDKN2A",
            "STAT3", "JAK2", "SRC", "ABL1", "RAF1", "ERK1", "MEK1",
            "CD19", "MS4A1", "CD3E"
        ]

        obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
        var = pd.DataFrame(
            {
                "gene_symbol": gene_symbols,
                "ensembl_id": [f"ENSG{i:011d}" for i in range(n_genes)],
            },
            index=[f"ENSG{i:011d}" for i in range(n_genes)],  # Ensembl IDs as index
        )
        obsm = {"X_pca": np.random.randn(n_cells, 10).astype(np.float32)}

        adata = ad.AnnData(X=sparse.csr_matrix(X), obs=obs, var=var, obsm=obsm)

        path = tmp_path / "gene_symbols.h5ad"
        adata.write_h5ad(path)

        yield path
        if path.exists():
            path.unlink()

    @pytest.fixture
    def h5ad_with_ensembl_ids(self, tmp_path) -> Path:
        """Create AnnData with Ensembl IDs (no gene symbols column)."""
        n_cells = 50
        n_genes = 30

        np.random.seed(42)
        X = np.random.poisson(5, (n_cells, n_genes)).astype(np.float32)
        X = np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4)

        obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
        var = pd.DataFrame(
            index=[f"ENSG{i:011d}" for i in range(n_genes)],  # Ensembl IDs
        )
        obsm = {"X_pca": np.random.randn(n_cells, 10).astype(np.float32)}

        adata = ad.AnnData(X=sparse.csr_matrix(X), obs=obs, var=var, obsm=obsm)

        path = tmp_path / "ensembl_ids.h5ad"
        adata.write_h5ad(path)

        yield path
        if path.exists():
            path.unlink()

    def test_gene_col_uses_column(self, h5ad_with_gene_symbols):
        """Test that gene_col uses the specified column for gene names."""
        acc = Accessor(h5ad_with_gene_symbols, gene_col="gene_symbol")

        assert acc.gene_col == "gene_symbol"
        assert acc.gene_names[0] == "TP53"
        assert "ENSG" not in acc.gene_names[0]

    def test_gene_col_none_uses_index(self, h5ad_with_gene_symbols):
        """Test that gene_col=None uses var.index."""
        with pytest.warns(UserWarning, match="appear to be gene symbols"):
            # Should warn because index contains Ensembl IDs
            acc = Accessor(h5ad_with_gene_symbols, gene_col=None)

        assert acc.gene_col is None
        assert acc.gene_names[0].startswith("ENSG")

    def test_gene_col_invalid_column_raises(self, h5ad_with_gene_symbols):
        """Test that invalid gene_col raises ValueError."""
        with pytest.raises(ValueError, match="gene_col 'nonexistent' not found"):
            Accessor(h5ad_with_gene_symbols, gene_col="nonexistent")

    def test_gene_symbol_validation_passes(self, h5ad_with_gene_symbols):
        """Test that gene symbols pass validation without warning."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Treat warnings as errors
            # Should not warn because gene_symbol column contains valid symbols
            acc = Accessor(h5ad_with_gene_symbols, gene_col="gene_symbol")
            assert len(acc.gene_names) == 30

    def test_gene_symbol_validation_warns_for_ensembl(self, h5ad_with_ensembl_ids):
        """Test that Ensembl IDs trigger a warning."""
        with pytest.warns(UserWarning, match="appear to be gene symbols.*Ensembl IDs"):
            Accessor(h5ad_with_ensembl_ids)

    def test_gene_symbol_pattern_matching(self):
        """Test the gene symbol regex pattern matches expected formats."""
        from sclambda.core.accessor import GENE_SYMBOL_PATTERN

        # Should match gene symbols
        valid_symbols = ["TP53", "BRCA1", "CD8A", "HLA-DRB1", "FOXP3", "IL2RA", "A1BG", "MYC"]
        for symbol in valid_symbols:
            assert GENE_SYMBOL_PATTERN.search(symbol), f"{symbol} should match"

        # Should not match Ensembl/RefSeq IDs
        invalid_ids = ["ENSG00000141510", "ENSMUSG00000001", "NM_000546", "XM_001234567"]
        for id_ in invalid_ids:
            match = GENE_SYMBOL_PATTERN.search(id_)
            # The pattern might match parts - check it doesn't match the full ID
            if match:
                # Ensure the match doesn't cover the whole ID-like string
                assert match.group() != id_, f"{id_} should not fully match as gene symbol"
