"""
Tests for spatial metrics tools with synthetic data.

Validates that Moran's z-score correctly captures:
1. Spatially clustered cell types (high positive z-score)
2. Evenly distributed cell types (near-zero z-score)
3. Spatially coherent gene expression (high z-score)
4. Uniformly distributed gene expression (near-zero z-score)

Also validates the spatial proportions computation and gene spatial scores.
"""

import numpy as np
import pytest

from sclambda.analysis.spatial_metrics import (
    _build_knn_weights,
    _permutation_morans_z,
    compute_morans_i,
    compute_spatial_proportions,
    compute_gene_spatial_scores,
)


# ============ SYNTHETIC DATA GENERATORS ============


def make_clustered_niche(n_cells=200, seed=42):
    """
    Create a niche where cell types are spatially clustered.

    Layout: 2D square grid with 3 cell types occupying distinct spatial regions.
    - Cell type A: top-left quadrant
    - Cell type B: top-right quadrant
    - Cell type C: bottom half

    Expected: High positive Moran's z-score for each cell type.
    """
    rng = np.random.default_rng(seed)
    n_per_type = n_cells // 3
    remainder = n_cells - 3 * n_per_type

    # Cell type A: top-left
    coords_a = rng.uniform(0, 0.5, size=(n_per_type, 2))
    coords_a[:, 1] = rng.uniform(0.5, 1.0, n_per_type)  # top half

    # Cell type B: top-right
    coords_b = rng.uniform(0.5, 1.0, size=(n_per_type, 2))
    coords_b[:, 1] = rng.uniform(0.5, 1.0, n_per_type)  # top half

    # Cell type C: bottom half
    coords_c = rng.uniform(0, 1.0, size=(n_per_type + remainder, 2))
    coords_c[:, 1] = rng.uniform(0, 0.5, n_per_type + remainder)  # bottom half

    coords = np.vstack([coords_a, coords_b, coords_c])
    cell_types = np.array(
        ["TypeA"] * n_per_type +
        ["TypeB"] * n_per_type +
        ["TypeC"] * (n_per_type + remainder)
    )

    return coords, cell_types


def make_dispersed_niche(n_cells=200, seed=42):
    """
    Create a niche where cell types are evenly/randomly distributed.

    All cell types are randomly placed across the full spatial extent.
    Expected: Near-zero Moran's z-score for each cell type.
    """
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1, size=(n_cells, 2))

    # Randomly assign cell types
    types = ["TypeA", "TypeB", "TypeC"]
    cell_types = rng.choice(types, size=n_cells)

    return coords, cell_types


def make_gradient_expression(coords, seed=42):
    """
    Create spatially coherent gene expression (gradient along x-axis).

    Gene1: strong gradient (high x = high expression)
    Gene2: random noise (no spatial pattern)
    Gene3: radial gradient from center

    Expected:
    - Gene1: high Moran's z-score
    - Gene2: near-zero Moran's z-score
    - Gene3: high Moran's z-score
    """
    rng = np.random.default_rng(seed)
    n = len(coords)

    # Gene1: linear gradient along x
    gene1 = coords[:, 0] * 3 + rng.normal(0, 0.3, n)
    gene1 = np.maximum(gene1, 0)  # non-negative expression

    # Gene2: random noise
    gene2 = np.abs(rng.normal(1.0, 0.5, n))

    # Gene3: radial gradient from center
    center = np.array([0.5, 0.5])
    distances = np.sqrt(np.sum((coords - center) ** 2, axis=1))
    gene3 = np.maximum(3.0 - distances * 4 + rng.normal(0, 0.3, n), 0)

    expression = np.column_stack([gene1, gene2, gene3])
    gene_names = ["GRADIENT_X", "NOISE", "RADIAL"]

    return expression, gene_names


# ============ TESTS FOR MORAN'S I (raw) ============


class TestMoransI:
    """Test raw Moran's I computation (unchanged utility)."""

    def test_clustered_binary_high_morans_i(self):
        """Spatially clustered binary variable should have high positive Moran's I."""
        coords, cell_types = make_clustered_niche(n_cells=300)
        W = _build_knn_weights(coords, k=10)

        for ct in ["TypeA", "TypeB", "TypeC"]:
            indicator = (cell_types == ct).astype(float)
            mi = compute_morans_i(indicator, W)
            assert mi > 0.2, (
                f"Cell type {ct} is spatially clustered but Moran's I = {mi:.3f} "
                f"(expected > 0.2)"
            )

    def test_constant_values_zero_morans_i(self):
        """Constant values should return Moran's I = 0."""
        coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
        W = _build_knn_weights(coords, k=5)
        values = np.ones(100)
        mi = compute_morans_i(values, W)
        assert mi == 0.0

    def test_too_few_cells(self):
        """Fewer than 3 cells should return 0."""
        coords = np.array([[0, 0], [1, 1]])
        W = _build_knn_weights(coords, k=1)
        values = np.array([1.0, 0.0])
        mi = compute_morans_i(values, W)
        assert mi == 0.0


# ============ TESTS FOR PERMUTATION Z-SCORE ============


class TestPermutationZScore:
    """Test permutation-based Moran's z-score."""

    def test_clustered_high_z(self):
        """Spatially clustered binary variable should have high z-score."""
        coords, cell_types = make_clustered_niche(n_cells=300)
        W = _build_knn_weights(coords, k=10)

        for ct in ["TypeA", "TypeB", "TypeC"]:
            indicator = (cell_types == ct).astype(float)
            z = _permutation_morans_z(indicator, W, n_permutations=199, seed=42)
            assert z > 2.0, (
                f"Cell type {ct} is spatially clustered but z = {z:.2f} "
                f"(expected > 2.0)"
            )

    def test_dispersed_low_z(self):
        """Randomly distributed binary variable should have non-significant z-score."""
        coords, cell_types = make_dispersed_niche(n_cells=300)
        W = _build_knn_weights(coords, k=10)

        for ct in ["TypeA", "TypeB", "TypeC"]:
            indicator = (cell_types == ct).astype(float)
            z = _permutation_morans_z(indicator, W, n_permutations=199, seed=42)
            assert abs(z) < 3.0, (
                f"Cell type {ct} is randomly distributed but z = {z:.2f} "
                f"(expected |z| < 3.0)"
            )

    def test_gradient_expression_high_z(self):
        """Spatially coherent expression should have high z-score."""
        coords, _ = make_clustered_niche(n_cells=300)
        expression, gene_names = make_gradient_expression(coords)
        W = _build_knn_weights(coords, k=10)

        z_gradient = _permutation_morans_z(expression[:, 0], W, seed=42)
        assert z_gradient > 2.0, (
            f"Gradient gene has z = {z_gradient:.2f} (expected > 2.0)"
        )

        z_noise = _permutation_morans_z(expression[:, 1], W, seed=42)
        assert abs(z_noise) < 3.0, (
            f"Noise gene has z = {z_noise:.2f} (expected |z| < 3.0)"
        )

    def test_constant_values(self):
        """Constant values should return z = 0."""
        coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
        W = _build_knn_weights(coords, k=5)
        z = _permutation_morans_z(np.ones(100), W, seed=42)
        assert z == 0.0

    def test_too_few_cells(self):
        """Fewer than 3 cells should return 0."""
        coords = np.array([[0, 0], [1, 1]])
        W = _build_knn_weights(coords, k=1)
        z = _permutation_morans_z(np.array([1.0, 0.0]), W, seed=42)
        assert z == 0.0


# ============ TESTS FOR SPATIAL PROPORTIONS ============


class TestSpatialProportions:
    """Test spatial proportions computation."""

    def test_clustered_proportions(self):
        """Clustered niche should report correct proportions and significant z-scores."""
        coords, cell_types = make_clustered_niche(n_cells=300)
        result = compute_spatial_proportions(coords, cell_types, "niche_0", seed=42)

        assert result.n_total == 300
        assert len(result.metrics) == 3

        # Each type should be ~1/3
        for m in result.metrics:
            assert 0.2 < m.proportion < 0.5, (
                f"{m.cell_type}: proportion {m.proportion:.3f} outside expected range"
            )
            # Each type should be significantly spatially clustered
            assert m.morans_z > 2.0, (
                f"{m.cell_type}: Moran's z = {m.morans_z:.2f} "
                f"(expected > 2.0 for clustered niche)"
            )

    def test_dispersed_proportions(self):
        """Dispersed niche should report non-significant z-scores."""
        coords, cell_types = make_dispersed_niche(n_cells=300)
        result = compute_spatial_proportions(coords, cell_types, "niche_1", seed=42)

        assert result.n_total == 300
        for m in result.metrics:
            assert abs(m.morans_z) < 3.0, (
                f"{m.cell_type}: Moran's z = {m.morans_z:.2f} "
                f"(expected |z| < 3.0 for dispersed niche)"
            )

    def test_as_dicts(self):
        """Test serialization to dicts."""
        coords, cell_types = make_clustered_niche(n_cells=100)
        result = compute_spatial_proportions(coords, cell_types, seed=42)
        dicts = result.as_dicts()

        assert isinstance(dicts, list)
        assert len(dicts) == 3
        for d in dicts:
            assert "cell_type" in d
            assert "n_cells" in d
            assert "proportion" in d
            assert "morans_z" in d


# ============ TESTS FOR GENE SPATIAL SCORES ============


class TestGeneSpatialScores:
    """Test gene-level spatial coherence scoring."""

    def test_gradient_vs_noise(self):
        """Gradient genes should have higher z-score than noise genes."""
        coords, _ = make_clustered_niche(n_cells=300)
        expression, gene_names = make_gradient_expression(coords)

        result = compute_gene_spatial_scores(
            coords, expression, gene_names, "niche_test", seed=42
        )

        gradient_score = result.get_score("GRADIENT_X")
        noise_score = result.get_score("NOISE")
        radial_score = result.get_score("RADIAL")

        assert gradient_score is not None
        assert noise_score is not None
        assert radial_score is not None

        assert gradient_score > noise_score, (
            f"Gradient z={gradient_score:.2f} should be > noise z={noise_score:.2f}"
        )
        assert radial_score > noise_score, (
            f"Radial z={radial_score:.2f} should be > noise z={noise_score:.2f}"
        )

    def test_as_dict(self):
        """Test serialization."""
        coords, _ = make_clustered_niche(n_cells=100)
        expression, gene_names = make_gradient_expression(coords)

        result = compute_gene_spatial_scores(coords, expression, gene_names, seed=42)
        d = result.as_dict()

        assert isinstance(d, dict)
        assert "GRADIENT_X" in d
        assert "NOISE" in d
        assert "RADIAL" in d

    def test_empty_genes(self):
        """Empty gene list should return empty result."""
        coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
        result = compute_gene_spatial_scores(
            coords, np.zeros((100, 0)), []
        )
        assert len(result.scores) == 0

    def test_zero_variance_gene(self):
        """An undefined score must not be presented as measured zero."""
        coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
        expression = np.ones((100, 1))
        result = compute_gene_spatial_scores(
            coords, expression, ["CONST"], seed=42
        )
        assert result.get_score("CONST") is None


# ============ TEST FORMATTER INTEGRATION ============


class TestFormatterIntegration:
    """Test that formatters work with spatial metrics output."""

    def test_format_spatial_proportions(self):
        """Test formatting of spatial proportions."""
        from sclambda.annotation.niche.formatters import format_spatial_proportions

        coords, cell_types = make_clustered_niche(n_cells=150)
        result = compute_spatial_proportions(coords, cell_types, seed=42)
        formatted = format_spatial_proportions(result.as_dicts())

        assert "Cell Type" in formatted
        assert "Proportion" in formatted
        assert "Moran's Z" in formatted
        assert "TypeA" in formatted

    def test_format_spatial_proportions_empty(self):
        """Empty proportions should return informative message."""
        from sclambda.annotation.niche.formatters import format_spatial_proportions

        formatted = format_spatial_proportions([])
        assert "No cell type" in formatted


# ============ TEST SPATIAL WEIGHT MATRIX ============


class TestSpatialWeights:
    """Test spatial weight matrix construction."""

    def test_knn_weights_shape(self):
        """Weight matrix should be n x n."""
        coords = np.random.default_rng(42).uniform(0, 1, (50, 2))
        W = _build_knn_weights(coords, k=5)
        assert W.shape == (50, 50)

    def test_knn_weights_row_sum(self):
        """Row-standardized weights should sum to ~1 per row."""
        coords = np.random.default_rng(42).uniform(0, 1, (50, 2))
        W = _build_knn_weights(coords, k=5)
        row_sums = np.array(W.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_knn_k_larger_than_n(self):
        """k larger than n-1 should be clamped."""
        coords = np.random.default_rng(42).uniform(0, 1, (5, 2))
        W = _build_knn_weights(coords, k=100)
        assert W.shape == (5, 5)
        # Should still work without error

    def test_single_cell(self):
        """Single cell should produce empty weight matrix."""
        coords = np.array([[0.0, 0.0]])
        W = _build_knn_weights(coords, k=5)
        assert W.shape == (1, 1)
        assert W.nnz == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
