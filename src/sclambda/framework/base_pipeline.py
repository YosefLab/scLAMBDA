"""
Reusable pipeline orchestration base class.

This class handles hierarchical wave execution so concrete pipelines can focus on:
- Building a compiled LangGraph pipeline
- Defining cluster input state
- Optional preprocessing hooks
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from sclambda.reporting.pipeline import PipelineReports
from typing import TYPE_CHECKING
import logging

from sclambda.core.task_runner import TaskRunner

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path
    from sclambda.core.accessor import Accessor
    from sclambda.llm.api import LLMClient


class Pipeline(PipelineReports, ABC):
    """Base class for hierarchical pipelines executed wave-by-wave.

    Subclasses must set ``name`` and override ``build_graph``.
    Optional class attributes control registration behaviour:

    - ``required_params``: parameter names that must be non-None.
    - ``max_depth_override``: if set, forces max_depth to this value.
    - ``requires_obs_columns``: if True, ``obs_columns`` must be provided.
    - ``_prompt_history_fields``: state fields used to determine prompt
      ordering for the prompt-history markdown written by ``write_outputs()``.
    """

    # -- Plugin metadata (subclasses override) --
    name: str = ""
    required_params: tuple[str, ...] = ()
    max_depth_override: int | None = None
    requires_obs_columns: bool = False

    # -- Output configuration (subclasses override) --
    _prompt_history_fields: tuple[str, ...] = ()

    def __init__(
        self,
        accessor: "Accessor",
        llm_client: "LLMClient",
        max_workers: int = 4,
    ):
        self.accessor = accessor
        self.llm_client = llm_client
        self.max_workers = max_workers
        self._last_results: list[dict] = []
        from sclambda.llm.usage import UsageLedger
        ledger = getattr(llm_client, "usage_ledger", None)
        self._usage_start = ledger.position if isinstance(ledger, UsageLedger) else 0
        # original_name -> numeric_string ("0", "1", ...).  Empty when no
        # encoding is needed (Leiden mode, or already 0-indexed integers).
        self._cluster_name_map: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Cluster name encoding
    # ------------------------------------------------------------------

    def _create_cluster_encoding(self, cluster_names: list[str]) -> list[str]:
        """Map arbitrary cluster names to consecutive integer strings.

        Uses natural sort (numeric-aware) to determine canonical order so
        that "2" < "10" rather than "10" < "2".

        If the names are already exactly ``"0", "1", ..., "N-1"`` (sorted),
        the mapping is the identity and ``_cluster_name_map`` is left empty.

        Returns the encoded names ``["0", "1", ..., "N-1"]``.
        """

        def _sort_key(name: str):
            try:
                return (0, int(name))
            except ValueError:
                return (1, name)

        sorted_names = sorted(cluster_names, key=_sort_key)
        expected = [str(i) for i in range(len(sorted_names))]

        if sorted_names == expected:
            return expected  # Already 0-indexed integers, no mapping needed

        self._cluster_name_map = {name: str(i) for i, name in enumerate(sorted_names)}
        return expected  # Always "0", "1", ..., "N-1"

    #: Prefix for clusters a ``clusters=`` filter left out. They keep their
    #: original name for context (images, neighbours, prototypes) but must not
    #: be mistaken for a processed cluster.
    UNSELECTED_PREFIX = "other-"

    def _apply_cluster_encoding(self, labels) -> "np.ndarray":
        """Remap a label array using the encoding in ``_cluster_name_map``.

        If no encoding has been created (Leiden mode, already numeric),
        the input is returned unchanged.

        Labels outside the encoding are the clusters a ``clusters=`` filter
        left out. They are prefixed rather than passed through: the encoding
        renames the selected clusters to "0".."N-1", so an unselected cluster
        literally named "9" would otherwise land on the encoded id of some
        selected one, and every cell of both would be fetched as a single
        cluster.
        """
        if not self._cluster_name_map:
            return labels
        import numpy as np

        return np.array(
            [self._cluster_name_map.get(str(l), f"{self.UNSELECTED_PREFIX}{l}") for l in labels]
        )

    @abstractmethod
    def build_graph(self, runner: TaskRunner):
        """Return a compiled graph with an .invoke(state) interface."""

    def build_post_graph(self, runner: TaskRunner, **kwargs):
        """Return a compiled post-processing graph, or None to skip.

        The post-graph runs once per cluster **after** all waves complete,
        receiving the cluster's own result together with a summary of every
        other cluster's result (see ``make_post_state``).

        Subclasses override this to define a cross-cluster correction step
        (e.g. spatial cross-niche annotation correction).
        """
        return None

    def preprocess(self, initial_clusters: list[str], **kwargs):
        """Optional hook to prepare accessor/state before execution."""
        return None

    def make_cluster_state(
        self,
        cluster_id: str,
        depth: int,
        max_depth: int,
        **kwargs,
    ) -> dict:
        """Build initial state dict for a single cluster.

        Subclasses should call super() and extend with pipeline-specific fields.
        """
        min_iterations = max(1, int(kwargs.get("min_reevaluation_iterations", 1)))
        max_iterations = max(min_iterations, int(kwargs.get("max_reevaluation_iterations", 2)))
        return {
            "cluster_id": cluster_id,
            "depth": depth,
            "max_depth": max_depth,
            "n_top_genes": kwargs.get("n_top_genes", 20),
            "organism": kwargs.get("organism", "human"),
            "tissue": kwargs.get("tissue", "unknown"),
            "reevaluation_iteration": 0,
            "min_reevaluation_iterations": min_iterations,
            "max_reevaluation_iterations": max_iterations,
            "reevaluation_best_confidence": "unknown",
        }

    @classmethod
    def validate_params(cls, **kwargs) -> None:
        """Check that all ``required_params`` are present and non-None.

        Called by ``Lambda.annotate()`` before pipeline instantiation.
        Subclasses can override for custom validation.
        """
        missing = [p for p in cls.required_params if not kwargs.get(p)]
        if missing:
            raise ValueError(f"{cls.name} pipeline requires: {', '.join(missing)}")

    def resolve_initial_clusters(
        self,
        accessor: "Accessor",
        obs_columns_list: list[str] | None,
        clusters_list: list[str] | None,
        pipeline_kwargs: dict,
    ) -> list[str]:
        """Determine initial cluster IDs from observation data.

        ``pipeline_kwargs`` is a mutable dict that subclasses may modify
        (e.g. pathology injects ``obs_columns`` when auto-detecting).

        Default logic:
        - If ``obs_columns_list`` is provided, read unique values from the
          first column and optionally filter by ``clusters_list``.
        - If ``requires_obs_columns`` and no obs_columns, raise ValueError.
        - Otherwise return ``[""]`` (root cluster for Leiden mode).
        """
        if obs_columns_list:
            first_col = obs_columns_list[0]
            all_cell_indices = accessor.get_cell_indices("")
            obs_subset = accessor.get_obs([first_col], all_cell_indices)
            nan_mask = obs_subset.iloc[:, 0].isna()
            initial_clusters = sorted(set(obs_subset.loc[~nan_mask, first_col].values))
            initial_clusters = [str(c) for c in initial_clusters]

            if clusters_list:
                clusters_set = set(clusters_list)
                initial_clusters = [c for c in initial_clusters if c in clusters_set]

                if not initial_clusters:
                    unique_values = sorted(set(obs_subset.iloc[:, 0].values))
                    available_preview = [str(v) for v in unique_values[:20]]
                    raise ValueError(
                        f"None of the specified clusters {clusters_list} were found. "
                        f"Available: {available_preview}"
                        f"{'...' if len(unique_values) > 20 else ''}"
                    )

            return self._create_cluster_encoding(initial_clusters)

        if self.requires_obs_columns:
            raise ValueError(f"{self.name} pipeline requires obs_columns to specify " f"the clustering column")

        return [""]

    # ------------------------------------------------------------------
    # Output writing
    # ------------------------------------------------------------------





    def format_result(self, results: list[dict], **kwargs) -> dict:
        """Build the return dict from pipeline results.

        Subclasses can override to add pipeline-specific extras.
        """
        self._last_results = results
        return {
            "pipeline": self.name,
            "n_clusters_processed": len(results),
            "max_depth_reached": (max(r.get("depth", 0) for r in results) if results else 0),
            "results": results,
            "accessor": self.accessor,
        }

    def process_cluster(
        self,
        compiled_graph,
        cluster_id: str,
        depth: int,
        max_depth: int,
        **kwargs,
    ) -> dict:
        state = self.make_cluster_state(
            cluster_id=cluster_id,
            depth=depth,
            max_depth=max_depth,
            **kwargs,
        )
        return compiled_graph.invoke(state)

    def run_wave(
        self,
        cluster_ids: list[str],
        depth: int,
        max_depth: int,
        runner: TaskRunner,
        compiled_graph,
        **kwargs,
    ) -> list[dict]:
        if not cluster_ids:
            return []

        def process_func(cluster_id: str) -> dict:
            return self.process_cluster(
                compiled_graph=compiled_graph,
                cluster_id=cluster_id,
                depth=depth,
                max_depth=max_depth,
                **kwargs,
            )

        task_results = runner.map_parallel(
            process_func,
            cluster_ids,
            return_exceptions=True,
        )

        results: list[dict] = []
        for tr in task_results:
            if tr.success:
                results.append(tr.value)
            else:
                logger.error(f"Cluster failed during wave (depth={depth}): {tr.error}")
        return results

    def make_post_state(
        self,
        result: dict,
        all_results: list[dict],
        **kwargs,
    ) -> dict:
        """Build initial state for a single cluster's post-processing.

        Called once per cluster during ``postprocess``.  The default
        implementation passes the cluster result through unchanged;
        subclasses should override to inject cross-cluster context
        (e.g. neighbor annotations for spatial correction).
        """
        return result

    def postprocess(
        self,
        all_results: list[dict],
        runner: TaskRunner,
        compiled_post_graph,
        **kwargs,
    ) -> list[dict]:
        """Run the post-processing graph for every cluster result.

        Each invocation receives its own result enriched with cross-cluster
        context via ``make_post_state``.  Runs in parallel.
        On failure, the original (uncorrected) result is kept.
        """
        if not all_results:
            return all_results

        def process_func(result: dict) -> dict:
            try:
                state = self.make_post_state(result, all_results, **kwargs)
                if state.get("_skip_post", False):
                    return result
                return compiled_post_graph.invoke(state)
            except Exception as e:
                cluster_id = result.get("cluster_id", "unknown")
                logger.error(f"Post-processing failed for {cluster_id}: {e}")
                return result  # keep original on failure

        return runner.map_parallel_ordered(process_func, all_results)

    # ------------------------------------------------------------------
    # Result helpers
    # ------------------------------------------------------------------

    def _extract_cell_type(self, result: dict) -> str | None:
        """Extract the primary cell-type annotation from a cluster result.

        Tries the field names used by each built-in pipeline in order:
        ``cell_type`` (celltype), ``condition`` (pathology),
        ``annotation`` (annotate_spatial).
        """
        return result.get("cell_type") or result.get("condition") or result.get("annotation")

    def _log_summary(
        self,
        results: list[dict],
        pre_post_types: dict[str, str | None],
    ) -> None:
        """Log a final annotation summary table.

        One row per cluster showing cluster_id, original cluster name (when
        encoding was applied), the final cell type, its Cell Ontology term,
        and — when post-processing changed the annotation — the original cell
        type before correction.
        """
        if not results:
            return

        inv_map = {v: k for k, v in self._cluster_name_map.items()}

        col_id = 12
        col_name = 28
        col_ct = 38
        col_ont = 30
        header = (
            f"{'cluster_id':<{col_id}} "
            f"{'original_name':<{col_name}} "
            f"{'cell_type':<{col_ct}} "
            f"{'cell_type_ont':<{col_ont}} "
            f"corrected_from"
        )
        sep = "-" * (col_id + 1 + col_name + 1 + col_ct + 1 + col_ont + 1 + 20)

        lines = ["", "=== Final Annotation Summary ===", header, sep]

        for r in sorted(
            results,
            key=lambda x: (x.get("depth", 0), str(x.get("cluster_id", ""))),
        ):
            cid = str(r.get("cluster_id", ""))
            orig = inv_map.get(cid, cid) if inv_map else cid
            cell_type = self._extract_cell_type(r) or "unknown"
            cell_type_ont = r.get("cell_type_ont") or ""
            pre = pre_post_types.get(cid)
            corrected_from = pre if (pre and pre != cell_type) else ""
            lines.append(
                f"{cid:<{col_id}} "
                f"{orig:<{col_name}} "
                f"{cell_type:<{col_ct}} "
                f"{cell_type_ont:<{col_ont}} "
                f"{corrected_from}"
            )

        lines.append(sep)
        logger.info("\n".join(lines))

    def run(self, initial_clusters: list[str], max_depth: int, **kwargs) -> list[dict]:
        """Run the pipeline until max_depth or no children remain."""
        refined = self.preprocess(initial_clusters=initial_clusters, **kwargs)
        if refined is not None:
            initial_clusters = refined

        with TaskRunner(max_workers=self.max_workers) as runner:
            compiled_graph = self.build_graph(runner)
            all_results: list[dict] = []
            current_clusters = initial_clusters
            current_depth = 1

            while current_clusters and current_depth <= max_depth:
                wave_results = self.run_wave(
                    cluster_ids=current_clusters,
                    depth=current_depth,
                    max_depth=max_depth,
                    runner=runner,
                    compiled_graph=compiled_graph,
                    **kwargs,
                )
                all_results.extend(wave_results)

                next_clusters: list[str] = []
                for state in wave_results:
                    next_clusters.extend(state.get("children", []))

                current_clusters = next_clusters
                current_depth += 1

            # Snapshot cell types before post-processing for the summary log
            pre_post_types: dict[str, str | None] = {
                str(r.get("cluster_id", "")): self._extract_cell_type(r) for r in all_results
            }

            # Optional cross-cluster post-processing
            compiled_post_graph = self.build_post_graph(runner, **kwargs)
            if compiled_post_graph is not None:
                logger.info(f"Running post-processing for {len(all_results)} clusters")
                all_results = self.postprocess(
                    all_results,
                    runner,
                    compiled_post_graph,
                    **kwargs,
                )

            # Propagate ontology terms and justifications from parents to
            # inherited children (both small clusters and LLM same_type).
            results_by_id = {str(r.get("cluster_id", "")): r for r in all_results}
            for r in all_results:
                parent_id = r.get("parent_cluster_id")
                if not parent_id or parent_id not in results_by_id:
                    continue
                # Only propagate to clusters that inherited their annotation
                if r.get("parent_classification") != "same_type":
                    continue
                parent = results_by_id[parent_id]
                cluster_id = str(r.get("cluster_id", ""))
                # Ontology term
                if not r.get("cell_type_ont"):
                    parent_ont = parent.get("cell_type_ont") or self.accessor.get_ont_cell_type(parent_id)
                    if parent_ont:
                        r["cell_type_ont"] = parent_ont
                        self.accessor.set_ont_cell_type(cluster_id, parent_ont)
                        logger.info(f"Propagated ontology '{parent_ont}' from {parent_id} to {cluster_id}")
                # Justification: copy parent's if child only has the generic inherit reason
                hypotheses = r.get("updated_hypotheses") or r.get("hypotheses", [])
                parent_hypotheses = parent.get("updated_hypotheses") or parent.get("hypotheses", [])
                if hypotheses and parent_hypotheses:
                    child_just = hypotheses[0].get("justification") or hypotheses[0].get("reasoning", "")
                    if "inherit" in child_just.lower():
                        parent_just = parent_hypotheses[0].get("justification") or parent_hypotheses[0].get("reasoning", "")
                        if parent_just:
                            hypotheses[0]["justification"] = parent_just
                            hypotheses[0]["reasoning"] = parent_just
                            self.accessor.set_hypotheses(cluster_id, hypotheses)
                            logger.info(f"Propagated justification from {parent_id} to {cluster_id}")

            self._log_summary(all_results, pre_post_types)

            return all_results
