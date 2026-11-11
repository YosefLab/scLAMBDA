# tests/test_pipeline.py
"""
Tests for the pipeline registry, base class metadata, and LangGraph compilation.

Focus areas:
1. Pipeline registry (register, get, list, entry_points)
2. Pipeline metadata (name, required_params, validate_params)
3. LangGraph pipeline compilation
4. resolve_initial_clusters logic
"""

import pytest
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from scipy import sparse

from sclambda.core.accessor import Accessor
from sclambda.core.task_runner import TaskRunner
from sclambda.framework.base_pipeline import Pipeline
from sclambda.pipelines.registry import (
    register_pipeline,
    get_pipeline_class,
    list_pipelines,
    _REGISTRY,
)
from sclambda.pipelines.celltype.pipeline import CTPipeline, CellTypingState


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
        {"batch": ["A"] * 50 + ["B"] * 50},
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

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


# ============ REGISTRY TESTS ============


class TestPipelineRegistry:
    """Test pipeline registration and discovery."""

    def test_builtins_registered(self):
        """All built-in pipelines should be registered."""
        names = list_pipelines()
        assert "celltype" in names
        assert "annotate_spatial" in names

    def test_get_pipeline_class(self):
        """get_pipeline_class should return the correct class."""
        cls = get_pipeline_class("celltype")
        assert cls is CTPipeline

    def test_unknown_pipeline_raises(self):
        """Unknown pipeline name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown pipeline.*nonexistent"):
            get_pipeline_class("nonexistent")

    def test_register_custom_pipeline(self):
        """Should be able to register a custom pipeline class."""

        class _TestPipeline(Pipeline):
            name = "test_custom"

            def build_graph(self, runner):
                pass

        register_pipeline("test_custom", _TestPipeline)
        assert get_pipeline_class("test_custom") is _TestPipeline

        # Clean up
        _REGISTRY.pop("test_custom", None)

    def test_register_non_pipeline_raises(self):
        """Registering a non-Pipeline class should raise TypeError."""
        with pytest.raises(TypeError):
            register_pipeline("bad", dict)

    def test_register_duplicate_same_class_ok(self):
        """Re-registering the same class for the same name is idempotent."""
        register_pipeline("celltype", CTPipeline)  # Should not raise

    def test_register_duplicate_different_class_raises(self):
        """Registering a different class for an existing name should raise."""

        class _OtherPipeline(Pipeline):
            name = "celltype"

            def build_graph(self, runner):
                pass

        with pytest.raises(ValueError, match="already registered"):
            register_pipeline("celltype", _OtherPipeline)


# ============ METADATA TESTS ============


class _RequiresParams(Pipeline):
    name = "requires_params"
    required_params = ("reference_col", "healthy_value")

    def build_graph(self, runner):
        return None


class TestPipelineMetadata:
    """Test pipeline class attributes and validate_params."""

    def test_celltype_metadata(self):
        assert CTPipeline.name == "celltype"
        assert CTPipeline.required_params == ()
        assert CTPipeline.max_depth_override is None
        assert CTPipeline.requires_obs_columns is False

    def test_validate_params_passes(self):
        """validate_params should pass when all required params are present."""
        _RequiresParams.validate_params(reference_col="ref", healthy_value="healthy")

    def test_validate_params_raises(self):
        """validate_params should raise when required params are missing."""
        with pytest.raises(ValueError, match="requires.*reference_col"):
            _RequiresParams.validate_params()

    def test_default_post_graph_accepts_run_kwargs(self):
        """Pipeline.run passes every annotate() kwarg to build_post_graph."""
        pipe = _RequiresParams.__new__(_RequiresParams)
        assert pipe.build_post_graph(None, organism="human", tissue="blood") is None

    def test_celltype_validate_params_always_passes(self):
        """CTPipeline has no required params, so validate always passes."""
        CTPipeline.validate_params()  # Should not raise


# ============ LANGGRAPH COMPILATION TESTS ============


class TestLangGraphPipeline:
    """Test LangGraph pipeline compilation."""

    def test_celltype_pipeline_compiles(self, accessor_with_clusters):
        """CTPipeline should compile a LangGraph with expected nodes."""
        from sclambda.llm.api import LLMClient

        llm_client = LLMClient(provider="openai", power="medium")
        pipe = CTPipeline(
            accessor=accessor_with_clusters,
            llm_client=llm_client,
            max_workers=2,
        )

        with TaskRunner(max_workers=2) as runner:
            compiled = pipe.build_graph(runner)

            assert compiled is not None
            nodes = list(compiled.get_graph().nodes.keys())
            assert "prepare_cluster" in nodes
            assert "rank_genes" in nodes
            assert "ct_generation" in nodes
            assert "subcluster" in nodes


# ============ RESOLVE INITIAL CLUSTERS TESTS ============


class TestResolveInitialClusters:
    """Test resolve_initial_clusters logic."""

    def test_celltype_no_obs_columns_returns_root(self, accessor_with_clusters):
        """CTPipeline without obs_columns should return [''] for Leiden mode."""
        from sclambda.llm.api import LLMClient

        llm_client = LLMClient(provider="openai", power="medium")
        pipe = CTPipeline(
            accessor=accessor_with_clusters,
            llm_client=llm_client,
            max_workers=1,
        )

        kwargs = {}
        clusters = pipe.resolve_initial_clusters(
            accessor_with_clusters, None, None, kwargs,
        )
        assert clusters == [""]

    def test_celltype_with_obs_columns(self, tmp_h5ad_with_clusters):
        """CTPipeline with obs_columns should read unique values from the column."""
        from sclambda.llm.api import LLMClient

        # Create AnnData with a cluster column
        acc = Accessor(tmp_h5ad_with_clusters)
        llm_client = LLMClient(provider="openai", power="medium")
        pipe = CTPipeline(
            accessor=acc,
            llm_client=llm_client,
            max_workers=1,
        )

        kwargs = {}
        clusters = pipe.resolve_initial_clusters(
            acc, ["batch"], None, kwargs,
        )

        assert clusters == ["0", "1"]
        assert pipe._cluster_name_map == {"A": "0", "B": "1"}
        assert kwargs.get("use_leiden") is False  # CTPipeline sets this

    def test_filter_by_clusters_list(self, tmp_h5ad_with_clusters):
        """Should filter to only specified clusters."""
        from sclambda.llm.api import LLMClient

        acc = Accessor(tmp_h5ad_with_clusters)
        llm_client = LLMClient(provider="openai", power="medium")
        pipe = CTPipeline(
            accessor=acc,
            llm_client=llm_client,
            max_workers=1,
        )

        kwargs = {}
        clusters = pipe.resolve_initial_clusters(
            acc, ["batch"], ["A"], kwargs,
        )

        assert clusters == ["0"]
        assert pipe._cluster_name_map == {"A": "0"}

    def test_nonexistent_cluster_raises(self, tmp_h5ad_with_clusters):
        """Should raise when specified clusters don't exist."""
        from sclambda.llm.api import LLMClient

        acc = Accessor(tmp_h5ad_with_clusters)
        llm_client = LLMClient(provider="openai", power="medium")
        pipe = CTPipeline(
            accessor=acc,
            llm_client=llm_client,
            max_workers=1,
        )

        kwargs = {}
        with pytest.raises(ValueError, match="None of the specified clusters"):
            pipe.resolve_initial_clusters(
                acc, ["batch"], ["nonexistent"], kwargs,
            )


# ============ FORMAT RESULT TESTS ============


class TestFormatResult:
    """Test format_result method."""

    def test_base_format_result(self, accessor_with_clusters):
        """Pipeline.format_result should return expected keys."""
        from sclambda.llm.api import LLMClient

        llm_client = LLMClient(provider="openai", power="medium")
        pipe = CTPipeline(
            accessor=accessor_with_clusters,
            llm_client=llm_client,
            max_workers=1,
        )

        results = [{"cluster_id": "0", "depth": 1}]
        output = pipe.format_result(results)

        assert output["pipeline"] == "celltype"
        assert output["n_clusters_processed"] == 1
        assert output["max_depth_reached"] == 1
        assert output["results"] == results
        assert output["accessor"] is accessor_with_clusters

    def test_celltype_format_adds_full_pipeline(self, accessor_with_clusters):
        """CTPipeline.format_result should add full_pipeline key."""
        from sclambda.llm.api import LLMClient

        llm_client = LLMClient(provider="openai", power="medium")
        pipe = CTPipeline(
            accessor=accessor_with_clusters,
            llm_client=llm_client,
            max_workers=1,
        )

        output = pipe.format_result([], full_pipeline=True)
        assert output["full_pipeline"] is True

        output = pipe.format_result([], full_pipeline=False)
        assert output["full_pipeline"] is False
