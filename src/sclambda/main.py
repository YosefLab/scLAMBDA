"""
LAMBDA2.0 - Single-cell annotation framework.

Main entry point for hierarchical cell-type and spatial niche annotation.
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

from sclambda.core.accessor import Accessor, DEFAULT_GENE_FILTER
from sclambda.core.matrix_reader import H5adMatrixReader
from sclambda.llm.api import LLMClient
from sclambda.pipelines.registry import get_pipeline_class

if TYPE_CHECKING:
    from sclambda.framework.base_pipeline import Pipeline


class Lambda:
    """
    Single-cell annotation framework.

    Usage:
        lambda_ = Lambda("data.h5ad")
        results = lambda_.annotate(depth=3)
    """

    def __init__(
        self,
        path: str | Path,
        provider: str = "anthropic",
        power: str = "medium",
        embedding_key: str = "X_scVI",
        gene_col: str | None = None,
        gene_filter: Callable[[str], bool] | None = DEFAULT_GENE_FILTER,
        chunk_size: int = 50_000,
        target_sum: float = 1e4,
        n_top_genes: int = 20,
        use_mmap_csr: bool = True,
        max_retries: int = 3,
        subcluster_n_neighbors: int = 30,
        model_overrides: dict[str, str] | None = None,
        usage_log_path: str | Path | None = None,
        column_cache: bool = True,
    ):
        """
        Initialize Lambda with data.

        Args:
            path: Path to h5ad file
            provider: LLM provider (openai, anthropic, google)
            power: LLM power level (weak, medium, strong)
            embedding_key: Embedding for clustering
            gene_filter: Function to filter genes (default filters RPL, RPS, MT, etc.)
            gene_col: Column in adata.var containing gene names. If None, uses var.index.
                      Should contain gene symbols (e.g., TP53, CD8A) for LLM compatibility.
            chunk_size: Chunk size for computing reference stats
            target_sum: Target sum for normalization
            n_top_genes: Number of top genes to use for annotation
            use_mmap_csr: If True (default), create temporary memory-mapped CSR arrays
                         for faster access to sparse expression data
            column_cache: If True (default), annotate() first builds or reuses a
                         gene-major copy of a sparse matrix in the dataset cache, so
                         targeted DE reads only the requested genes. It is about the
                         matrix's size on disk and needs twice that in RAM to build.
            max_retries: Maximum retry attempts for retryable LLM errors
                         (timeouts, overload, rate limits)
            subcluster_n_neighbors: Number of neighbors for Leiden subclustering
            usage_log_path: Optional append-only JSONL log of every LLM attempt,
                including provider usage and prompt text, for recovery during a run.
        """
        self.accessor = Accessor(
            path=path,
            gene_filter=gene_filter,
            gene_col=gene_col,
            chunk_size=chunk_size,
            target_sum=target_sum,
            embedding_key=embedding_key,
            use_mmap_csr=use_mmap_csr,
        )
        self.llm_client = LLMClient(
            provider=provider, power=power, max_retries=max_retries, model_overrides=model_overrides,
            usage_log_path=usage_log_path,
        )
        self._n_top_genes = n_top_genes
        self._column_cache = column_cache
        self._subcluster_n_neighbors = subcluster_n_neighbors
        self._last_pipeline: "Pipeline | None" = None

    def annotate(
        self,
        pipeline: str = "celltype",
        depth: int = 1,
        resolution: float = 0.3,
        max_workers: int = 8,
        obs_columns: str | Sequence[str] | None = None,
        clusters: str | Sequence[str] | None = None,
        use_leiden: bool = True,
        full_pipeline: bool = True,
        naive: bool = False,
        organism: str = "human",
        tissue: str = "unknown",
        tissue_col: str | None = None,
        min_cluster_size: int = 50,
        min_subcluster_proportion: float = 0.01,
        min_reevaluation_iterations: int = 1,
        max_reevaluation_iterations: int = 2,
        use_cell_ont: bool = False,
        ontology_file: str | Path | None = None,
        # Spatial annotation pipeline parameters
        cell_type_col: str | None = None,
        k_top_cell_types: int = 5,
        spatial_key: str = "spatial",
        k_neighbors: int = 10,
        slice_key: str | None = None,
        slice: str | None = None,
        include_spatial_image: bool = True,
        spatial_image_px: int = 700,
        flip_spatial_y: bool = True,
        standardize_niches: bool = True,
        de_method: str = "ttest",
        # Third-party pipeline parameters
        **kwargs,
    ) -> dict:
        """
        Annotate clusters using the specified pipeline.

        Built-in pipelines: "celltype", "annotate_spatial". Third-party pipelines can be registered via
        entry_points or ``sclambda.pipelines.registry.register_pipeline()``.

        Args:
            pipeline: Pipeline name (built-in or registered third-party).
            depth: Maximum hierarchy depth (overridden by some pipelines).
            max_workers: Max parallel workers.
            obs_columns: Use external clustering from these columns.
            clusters: Only process these specific clusters.
            organism: Organism name for LLM context.
            tissue: Tissue type for LLM context.
            cell_type_col: Optional population column for spatial annotation.
                When omitted, uniform 'spot' labels are created in memory.
            **kwargs: Additional pipeline-specific parameters.

        Returns:
            Dict with pipeline name, results, and metadata.
        """
        # --- Normalize list arguments ---
        if obs_columns is None:
            obs_columns_list = None
        elif isinstance(obs_columns, str):
            obs_columns_list = [col.strip() for col in obs_columns.split(",")]
        else:
            obs_columns_list = list(obs_columns)

        if clusters is None:
            clusters_list = None
        elif isinstance(clusters, str):
            clusters_list = [c.strip() for c in clusters.split(",")]
        else:
            clusters_list = list(clusters)

        # --- Collect all params into a mutable dict ---
        pipeline_kwargs = {
            "resolution": resolution,
            "use_leiden": use_leiden,
            "full_pipeline": full_pipeline,
            "naive": naive,
            "organism": organism,
            "tissue": tissue,
            "tissue_col": tissue_col,
            "min_cluster_size": min_cluster_size,
            "min_subcluster_proportion": min_subcluster_proportion,
            "min_reevaluation_iterations": min_reevaluation_iterations,
            "max_reevaluation_iterations": max_reevaluation_iterations,
            "use_cell_ont": use_cell_ont,
            "ontology_file": str(ontology_file) if ontology_file else None,
            "cell_type_col": cell_type_col,
            "k_top_cell_types": k_top_cell_types,
            "spatial_key": spatial_key,
            "k_neighbors": k_neighbors,
            "slice_key": slice_key,
            "slice": slice,
            "include_spatial_image": include_spatial_image,
            "spatial_image_px": spatial_image_px,
            "flip_spatial_y": flip_spatial_y,
            "standardize_niches": standardize_niches,
            "de_method": de_method,
            "n_top_genes": self._n_top_genes,
            "subcluster_n_neighbors": self._subcluster_n_neighbors,
            **kwargs,
        }

        # --- Look up pipeline class from registry ---
        pipeline_cls = get_pipeline_class(pipeline)

        # --- Inject obs_columns before validation so validate_params can see it ---
        if "obs_columns" not in pipeline_kwargs:
            pipeline_kwargs["obs_columns"] = obs_columns_list

        # --- Validate required parameters ---
        pipeline_cls.validate_params(**pipeline_kwargs)

        # --- Instantiate pipeline ---
        pipe = pipeline_cls(
            accessor=self.accessor,
            llm_client=self.llm_client,
            max_workers=max_workers,
        )

        # --- Resolve initial clusters (may modify pipeline_kwargs) ---
        initial_clusters = pipe.resolve_initial_clusters(
            self.accessor,
            obs_columns_list,
            clusters_list,
            pipeline_kwargs,
        )

        # --- Determine max_depth ---
        max_depth = pipeline_cls.max_depth_override if pipeline_cls.max_depth_override is not None else depth

        # Before any worker thread starts: gene-panel DE reads columns, not every row.
        if self._column_cache:
            self.accessor.prepare_column_cache()

        # --- Run pipeline ---
        from sclambda.llm.usage import UsageLedger, summarize
        ledger = getattr(self.llm_client, "usage_ledger", None)
        run_id = ledger.begin_run(pipeline) if isinstance(ledger, UsageLedger) else None
        try:
            results = pipe.run(
                initial_clusters=initial_clusters,
                max_depth=max_depth,
                **pipeline_kwargs,
            )
        finally:
            self._last_pipeline = pipe
            if run_id is not None:
                pipe._usage_events = ledger.end_run(run_id)

        result = pipe.format_result(results, **pipeline_kwargs)
        if run_id is not None:
            result["llm_usage"] = {"run_id": run_id, **summarize(pipe._usage_events)}
            result["llm_usage_events"] = pipe._usage_events
        return result

    # ============ REFERENCE STATS ============

    def compute_reference(
        self,
        files: Sequence[str | Path],
        output_path: str | Path | None = None,
    ) -> dict:
        """
        Compute reference mean and variance across multiple h5ad files.

        Useful when the dataset is spread across several files and you want
        to precompute reference statistics once and cache them.

        Uses Chan's parallel algorithm to combine per-file statistics without
        loading all data into memory at once.

        Args:
            files: List of paths to h5ad files. All files must have the same
                number of genes (columns).
            output_path: Optional path to save reference stats as .npz file.

        Returns:
            Dict with 'mean' (np.ndarray), 'var' (np.ndarray), 'n' (int).
        """
        if not files:
            raise ValueError("At least one file is required")

        global_mean = None
        global_M2 = None
        global_n = 0
        n_genes = None

        for file_path in files:
            reader = H5adMatrixReader(
                str(file_path),
                x_key="X",
                use_mmap_csr=False,
            )

            # Validate gene count matches across files
            if n_genes is None:
                n_genes = reader.n_vars
            elif reader.n_vars != n_genes:
                reader.close()
                raise ValueError(f"Gene count mismatch: {file_path} has {reader.n_vars} " f"genes, expected {n_genes}")

            # Check if this file needs normalization
            needs_norm = self._check_file_normalization(reader)

            # Compute per-file column stats
            mean, var, _, _ = reader.compute_col_stats_chunked(
                chunk_size=self.accessor._chunk_size,
                normalize=needs_norm,
                target_sum=self.accessor._target_sum,
            )
            file_n = reader.n_obs
            reader.close()

            # Combine using Chan's parallel algorithm
            if global_mean is None:
                global_mean = mean.astype(np.float64)
                global_M2 = (var * file_n).astype(np.float64)
                global_n = file_n
            else:
                delta = mean.astype(np.float64) - global_mean
                new_n = global_n + file_n
                global_mean += delta * file_n / new_n
                global_M2 += var.astype(np.float64) * file_n + (delta**2 * global_n * file_n / new_n)
                global_n = new_n

        final_mean = global_mean.astype(np.float32)
        final_var = (global_M2 / global_n).astype(np.float32)

        result = {"mean": final_mean, "var": final_var, "n": global_n}

        if output_path is not None:
            np.savez(
                output_path,
                mean=final_mean,
                var=final_var,
                n=np.array(global_n),
            )

        return result

    def set_reference(
        self,
        *,
        mean: np.ndarray | None = None,
        var: np.ndarray | None = None,
        n: int | None = None,
        path: str | Path | None = None,
    ) -> None:
        """
        Set reference statistics from precomputed values or a saved file.

        Either provide (mean, var, n) directly, or a path to a .npz file
        produced by ``compute_reference()``.

        Args:
            mean: Reference mean array (n_genes,).
            var: Reference variance array (n_genes,).
            n: Number of cells used to compute the reference.
            path: Path to .npz file with saved reference stats.
        """
        if path is not None:
            data = np.load(path)
            mean = data["mean"]
            var = data["var"]
            n = int(data["n"])

        if mean is None or var is None or n is None:
            raise ValueError("Provide either (mean, var, n) or path to a .npz file")

        self.accessor.set_reference_stats(mean, var, n)

    @staticmethod
    def _check_file_normalization(
        reader: H5adMatrixReader,
        sample_size: int = 100,
    ) -> bool:
        """Check whether an h5ad file contains raw counts (needs normalization)."""
        n_sample = min(sample_size, reader.n_obs)
        sample_rows = np.random.choice(reader.n_obs, n_sample, replace=False)
        X_sample = reader.fetch_rows_dense(sample_rows)
        return bool(np.allclose(X_sample, X_sample.astype(int)))

    def get_annotations(self, depth: int | None = None) -> dict[str, str]:
        """
        Get cluster → cell_type mapping.

        Args:
            depth: Specific depth level, or None for all depths

        Returns:
            Dict mapping cluster_id to annotation label
        """
        return self.accessor.get_annotations(depth=depth)

    def get_overlay(self):
        """Get the overlay dataframe with all clustering and annotations"""
        return self.accessor.overlay.copy()

    def write(
        self,
        output_path: str | Path | None = None,
        **kwargs,
    ) -> list[Path]:
        """
        Write pipeline results to file(s).

        The output format is determined by the pipeline that was last run.
        Possible formats include xlsx, csv, json (prompt history), and markdown,
        depending on the pipeline implementation.

        Args:
            output_path: Base output path. Interpretation varies by pipeline
                (e.g. for the celltype pipeline, this is the CSV path for
                cell-level annotations).
            **kwargs: Pipeline-specific options passed through to the pipeline's
                write_outputs() method. For the celltype pipeline:
                - excel_path: Path for Excel output
                    (default: "lambda_annotations.xlsx").

        Returns:
            List of Paths to files written.

        Raises:
            RuntimeError: If annotate() has not been called yet.
        """
        if self._last_pipeline is None:
            raise RuntimeError("No pipeline has been run yet. Call annotate() first.")
        return self._last_pipeline.write_outputs(
            accessor=self.accessor,
            output_path=output_path,
            **kwargs,
        )

    def write_narrative(
        self,
        results: dict,
        output_dir: str | Path = "narrativize_output",
    ) -> Path:
        """
        Write narrativize pipeline results to a directory of files.

        Produces:
        - narrative.md           — the synthesized scientific narrative
        - dataset_description.json — structured dataset description
        - plans/<N>_<slug>.md    — execution summary for each completed plan
        - notes.md               — cross-comparison notes (if any)
        - results.json           — full results dump for debugging

        Args:
            results: Return value of ``lambda_.annotate(pipeline="narrativize")``.
            output_dir: Directory to write into (created if absent).

        Returns:
            The resolved output directory path.
        """
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        narrative = results.get("narrative", "")
        description = results.get("dataset_description", {})
        completed_plans = results.get("completed_plans", [])
        accumulated_notes = results.get("accumulated_notes", {})

        # narrative.md
        (output_dir / "narrative.md").write_text(narrative if narrative else "(no narrative generated)\n")

        # dataset_description.json
        (output_dir / "dataset_description.json").write_text(json.dumps(description, indent=2))

        # execution_summary.md — all plans in one document + plans/<N>_<slug>.md
        if completed_plans:
            from narrativizer.plan.plan import Plan

            plans_dir = output_dir / "plans"
            plans_dir.mkdir(exist_ok=True)

            summary_sections = []
            for i, plan_dict in enumerate(completed_plans):
                plan = Plan.from_dict(plan_dict)
                slug = plan.comparison[:60].replace(" ", "_").replace("/", "-")
                filename = f"{i + 1:02d}_{slug}.md"
                plan_summary = plan.execution_summary()
                (plans_dir / filename).write_text(plan_summary)
                summary_sections.append(plan_summary)

            (output_dir / "execution_summary.md").write_text("\n\n---\n\n".join(summary_sections))

        # notes.md
        if accumulated_notes:
            notes_lines = ["# Cross-Comparison Notes\n"]
            for header, text in accumulated_notes.items():
                notes_lines.append(f"## {header}\n{text}\n")
            (output_dir / "notes.md").write_text("\n".join(notes_lines))

        # results.json — full dump for downstream use / debugging
        (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

        return output_dir
