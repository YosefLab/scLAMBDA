"""
LangGraph-based cell-typing pipeline and orchestration.

This module now contains both:
- The LangGraph DAG builder (node routing/execution)
- A concrete CTPipeline class that runs hierarchical wave orchestration
"""

from __future__ import annotations
import logging
from typing import TypedDict, Literal, TYPE_CHECKING

import numpy as np
from langgraph.graph import StateGraph, END

from sclambda.framework.base_pipeline import Pipeline

if TYPE_CHECKING:
    from sclambda.core.accessor import Accessor
    from sclambda.core.task_runner import TaskRunner
    from sclambda.llm.api import LLMClient
else:
    from sclambda.core.accessor import Accessor

logger = logging.getLogger(__name__)


# ============ STATE DEFINITION ============


class CellTypingState(TypedDict, total=False):
    """State for celltyping pipeline."""

    # Required inputs
    cluster_id: str
    depth: int  # Computed in prepare_cluster_node
    max_depth: int

    # Cluster preparation (from prepare_cluster_node)
    parent_cluster_id: str | None
    parent_classification: str | None  # "same_type", "subtype", "unrelated"
    can_inherit_parent: bool

    # Config flags
    full_pipeline: bool
    naive: bool  # simplified single-question annotation mode
    _prompt_execution_log: list[dict]  # ordered prompt input/output history

    # Results
    top_genes: list[str]
    rank_genes_result: object  # RankGenesResult with full DE stats per gene
    cell_type: str | None
    cell_type_ont: str | None  # Cell Ontology term
    cell_types: list[str]
    children: list[str]

    # CT generation
    generation_prompt: str
    raw_llm_response: str
    parse_response: str
    ct_generation_failed: bool
    hypotheses: list[dict]

    # Marker refinement
    refinement_prompt: str
    refinement_response: str
    refinement_signatures: dict
    refinement_confidence_decision: str  # "confident" or "needs_refinement"
    refinement_best_cell_type: str | None
    refinement_reasoning: str | None

    # Signature DE
    signature_de_results: list[dict]
    signature_de_result_obj: object  # SignaturesDEResult

    # CT reevaluation
    reevaluation_prompt: str
    reevaluation_response: str
    reevaluation_parse_response: str
    updated_hypotheses: list[dict]
    reevaluation_iteration: int
    min_reevaluation_iterations: int
    max_reevaluation_iterations: int
    reevaluation_best_confidence: str
    reevaluation_confidence_decision: str  # "confident" or "needs_refinement"

    # Justification synthesis
    justification_summary: str

    # Subcluster
    subcluster_method: str  # "leiden" or "from_obs" or "none"

    # Clustering config
    resolution: float
    min_cluster_size: int
    min_subcluster_proportion: float  # Min fraction of parent cells for a subcluster to be kept
    obs_columns: list[str] | None
    use_leiden: bool

    # Gene config
    n_top_genes: int  # Number of top genes to use for annotation

    # Harmonization config
    use_cell_ont: bool  # Whether to map cell types to Cell Ontology
    ontology_file: str | None  # Path to OBO ontology file

    # Post-processing: Cross-cluster comparison
    cross_cluster_context: str | None  # formatted other cluster annotations (same depth)
    cross_cluster_suggestion_prompt: str
    cross_cluster_suggestion_response: str
    cross_cluster_suggestion_parse_response: str
    cross_cluster_target_genes: list[str]  # shared diagnostic gene panel
    cross_cluster_comparisons: list[dict]  # [{cluster_id, cell_type, reason}]
    cross_cluster_de_summary: str  # formatted DE results for all comparisons
    cross_cluster_correction_prompt: str
    cross_cluster_correction_response: str
    cross_cluster_correction_parse_response: str
    cross_cluster_updated_justification: str
    cross_cluster_changed: bool

    # Context
    organism: str
    tissue: str
    tissue_col: str | None  # obs column to compute per-cluster tissue

    # DE method
    de_method: str  # "ttest" (default) or "ols" (dropout-corrected)

    # ---- Persistent per-cluster dossier path (opt-in) ----
    use_dossier: bool
    ct_dossier: dict | None         # serialized CellTypeDossier
    # Agentic (coarse->fine) refinement controls + outputs.
    dossier_agent_max_steps: int    # actor-critic loop cap (default 4)
    dossier_agent_transcript: list  # per-turn critic/actor log
    dossier_agent_steps: int
    dossier_review_complete: bool
    dossier_stop_reason: str
    annotation_assessment: str      # resolved | biologically_unresolved | review_incomplete | inference_failed
    supported_traits: list         # independently supported observations, not identity prerequisites
    annotation_status: str  # resolved | best_effort, independent of execution status
    annotation_uncertainty: str
    annotation_alternatives: list
    critic_response_error: str
    needs_subcluster: bool          # critic handed the finer rung to subclustering


# ============ PIPELINE ============


class CTPipeline(Pipeline):
    """Cell-typing pipeline with shared orchestration from framework.Pipeline."""

    name = "celltype"
    required_params = ()
    max_depth_override = None
    requires_obs_columns = False

    _prompt_history_fields = (
        "cluster_id",
        "depth",
        "cell_type",
        "cell_type_ont",
        "ct_dossier",
        "dossier_agent_transcript",
        "top_genes",
        "naive_ct_prompt",
        "naive_ct_response",
        "generation_prompt",
        "raw_llm_response",
        "parse_response",
        "hypotheses",
        "refinement_prompt",
        "refinement_response",
        "refinement_signatures",
        "refinement_confidence_decision",
        "refinement_best_cell_type",
        "refinement_reasoning",
        "reevaluation_prompt",
        "reevaluation_response",
        "reevaluation_parse_response",
        "updated_hypotheses",
        "justification_summary",
        "ontology_canonical_prompt",
        "ontology_canonical_response",
        "ontology_selection_prompt",
        "ontology_selection_response",
        "cross_cluster_context",
        "cross_cluster_suggestion_prompt",
        "cross_cluster_suggestion_response",
        "cross_cluster_suggestion_parse_response",
        "cross_cluster_de_summary",
        "cross_cluster_correction_prompt",
        "cross_cluster_correction_response",
        "cross_cluster_correction_parse_response",
        "cross_cluster_updated_justification",
        "cross_cluster_changed",
    )

    def resolve_initial_clusters(
        self,
        accessor,
        obs_columns_list,
        clusters_list,
        pipeline_kwargs,
    ):
        """Support both Leiden mode (no obs_columns) and external clustering."""
        if obs_columns_list:
            pipeline_kwargs["use_leiden"] = False
            return super().resolve_initial_clusters(
                accessor,
                obs_columns_list,
                clusters_list,
                pipeline_kwargs,
            )
        return [""]

    def format_result(self, results, **kwargs):
        result = super().format_result(results, **kwargs)
        result["full_pipeline"] = kwargs.get("full_pipeline", True)
        return result

    def preprocess(self, initial_clusters: list[str], **kwargs):
        """Populate overlay for external clustering mode before wave execution.

        When obs_columns are provided, uses the first column as L1 labels and
        sorts the mmap by those labels for I/O locality.

        When no obs_columns are provided (Leiden mode), runs Leiden clustering
        on the full dataset embeddings, stores the labels as depth-1 clusters
        in the overlay, and sorts the mmap for I/O locality. The L1 cluster IDs
        are returned so that the pipeline starts annotating them directly.
        """
        self._use_dossier = bool(kwargs.get("use_dossier", False))

        obs_columns = kwargs.get("obs_columns")
        use_leiden = kwargs.get("use_leiden", True)

        if obs_columns and not use_leiden and len(obs_columns) > 0:
            # Find the first obs column that has more than one unique value
            all_cell_indices = np.arange(self.accessor.n_cells)
            sort_col = None
            for col in obs_columns:
                obs_subset = self.accessor.get_obs([col], all_cell_indices)
                raw_values = obs_subset.iloc[:, 0].values.astype(str)
                n_unique = len(set(raw_values))
                if n_unique > 1:
                    sort_col = col
                    raw_labels = raw_values
                    break
                logger.info(
                    "obs column '%s' is constant (%s) — skipping for pre-sort.",
                    col,
                    raw_values[0],
                )

            if sort_col is not None:
                labels = self._apply_cluster_encoding(raw_labels)

                l1_col = self.accessor._level_column(1)
                if l1_col not in self.accessor._overlay.columns:
                    self.accessor._overlay[l1_col] = None
                self.accessor._overlay.loc[self.accessor._overlay.index, l1_col] = labels
            else:
                # All obs columns are constant — fall back to Leiden pre-sort
                logger.info("All obs_columns are constant — falling back to Leiden pre-sort.")
                return self._presort_by_leiden(kwargs.get("resolution", 1.0), force_resolution=kwargs.get("force_resolution"))

        elif use_leiden:
            # Leiden mode: cluster all cells, store as L1, and sort mmap
            return self._presort_by_leiden(kwargs.get("resolution", 1.0), force_resolution=kwargs.get("force_resolution"))

    def _presort_by_leiden(self, resolution: float = 1.0, force_resolution: float | None = None) -> list[str] | None:
        """Run Leiden clustering on all cells, store labels as L1, and sort mmap.

        The resulting labels ARE stored in the overlay as depth-1 clusters so
        that the pipeline starts annotating them directly instead of running
        Leiden a second time inside SubclusterNode.

        Returns:
            List of L1 cluster IDs, or None when no embedding is available.
        """
        from sclambda.analysis.subcluster import _run_leiden_with_retry

        try:
            embeddings = self.accessor.get_obsm(self.accessor._embedding_key)
        except KeyError:
            logger.warning(
                "No embedding '%s' found — skipping mmap pre-sort.",
                self.accessor._embedding_key,
            )
            return None

        logger.info(
            "Running Leiden clustering on %d cells (resolution=%.2f)...",
            self.accessor.n_cells,
            resolution,
        )
        labels, final_res = _run_leiden_with_retry(
            embeddings=embeddings,
            resolution=resolution,
            min_clusters=2,
            force_resolution=force_resolution,
        )

        # Store as L1 clusters in the overlay
        result = self.accessor.set_cluster_labels("", labels.astype(str))
        l1_clusters = result["clusters"]

        logger.info(
            "Leiden produced %d L1 clusters (resolution=%.2f).",
            len(l1_clusters),
            final_res,
        )

        return l1_clusters

    def make_cluster_state(
        self,
        cluster_id: str,
        depth: int,
        max_depth: int,
        **kwargs,
    ) -> CellTypingState:
        state = super().make_cluster_state(cluster_id, depth, max_depth, **kwargs)
        state.update(
            {
                "resolution": kwargs.get("resolution", 1.0),
                "min_cluster_size": kwargs.get("min_cluster_size", 50),
                "min_subcluster_proportion": kwargs.get("min_subcluster_proportion", 0.01),
                "obs_columns": kwargs.get("obs_columns"),
                "use_leiden": kwargs.get("use_leiden", True),
                "full_pipeline": kwargs.get("full_pipeline", True),
                "naive": kwargs.get("naive", False),
                "use_cell_ont": kwargs.get("use_cell_ont", False),
                "ontology_file": kwargs.get("ontology_file"),
                "tissue_col": kwargs.get("tissue_col"),
                "de_method": kwargs.get("de_method", "ttest"),
                "use_dossier": getattr(self, "_use_dossier", False),
                "ct_dossier": None,
                "dossier_agent_max_steps": kwargs.get("dossier_agent_max_steps", 4),
            }
        )
        return state

    def build_graph(self, runner: "TaskRunner"):
        """
        Build and compile the celltyping LangGraph pipeline.

        All branching is handled by conditional edges based on state flags.
        Nodes are class instances; the graph structure is declarative.
        """
        from sclambda.pipelines.celltype.nodes import (
            PrepareClusterNode,
            RankGenesNode,
            NaiveCTGenerationNode,
            InheritParentAnnotationNode,
            CTGenerationNode,
            CTDossierAgentNode,
            SubstructureGateNode,
            MarkerRefinementNode,
            SignatureDENode,
            CTReevaluationNode,
            SubclusterNode,
        )

        graph = StateGraph(CellTypingState)

        # ---- Nodes ----
        graph.add_node("prepare_cluster", PrepareClusterNode(self.accessor, runner))
        graph.add_node("rank_genes", RankGenesNode(self.accessor, runner))
        graph.add_node("naive_ct_generation", NaiveCTGenerationNode(self.accessor, runner, self.llm_client))
        graph.add_node("inherit_parent", InheritParentAnnotationNode(self.accessor, runner, self.llm_client))
        graph.add_node("ct_generation", CTGenerationNode(self.accessor, runner, self.llm_client))
        # Dossier path only: agentic coarse->fine refinement replacing the linear
        # marker_refinement <-> reevaluation loop.
        graph.add_node("dossier_agent", CTDossierAgentNode(self.accessor, runner, self.llm_client))
        graph.add_node("substructure_gate", SubstructureGateNode(self.accessor, runner, self.llm_client))
        graph.add_node("marker_refinement", MarkerRefinementNode(self.accessor, runner, self.llm_client))
        graph.add_node("signature_de", SignatureDENode(self.accessor, runner))
        graph.add_node("ct_reevaluation", CTReevaluationNode(self.accessor, runner, self.llm_client))
        graph.add_node("subcluster", SubclusterNode(self.accessor, runner))

        # ---- Edges ----
        graph.set_entry_point("prepare_cluster")

        graph.add_conditional_edges(
            "prepare_cluster",
            route_after_prepare_cluster,
            {
                "rank_genes": "rank_genes",
                "inherit_parent": "inherit_parent",
                "ct_generation": "ct_generation",
                "naive_ct_generation": "naive_ct_generation",
            },
        )

        graph.add_conditional_edges(
            "rank_genes",
            route_after_rank_genes,
            {
                "inherit_parent": "inherit_parent",
                "ct_generation": "ct_generation",
                "naive_ct_generation": "naive_ct_generation",
            },
        )
        graph.add_conditional_edges("inherit_parent", route_after_inherit_parent)  # -> END|ct_generation

        graph.add_conditional_edges("ct_generation", route_after_ct_generation)  # -> dossier_agent|marker_refinement|subcluster

        # Dossier path: agentic refinement -> substructure gate (decides needs_subcluster) -> subcluster.
        graph.add_edge("dossier_agent", "substructure_gate")
        graph.add_edge("substructure_gate", "subcluster")

        graph.add_conditional_edges("marker_refinement", route_after_refinement)  # -> signature_de|subcluster

        graph.add_edge("signature_de", "ct_reevaluation")
        graph.add_conditional_edges(
            "ct_reevaluation",
            route_after_reevaluation,
        )  # -> marker_refinement|subcluster
        # Naive mode goes directly to subcluster
        graph.add_edge("naive_ct_generation", "subcluster")
        # Cell ontology mapping is intentionally deferred to post-processing
        # so cross-cluster correction can finalize the label first.
        graph.add_edge("subcluster", END)

        return graph.compile()

    def build_post_graph(self, runner: "TaskRunner", **kwargs):
        """Build cross-cluster comparison post-processing graph."""
        if not kwargs.get("use_cross_cluster", not getattr(self, "_use_dossier", False)):
            if not kwargs.get("use_cell_ont", False):
                return None
            # Ontology naming is independent of optional cross-cluster reasoning.
            from sclambda.pipelines.celltype.nodes import JustificationSynthesisNode, OntCellTypeNode
            graph = StateGraph(CellTypingState)
            graph.add_node("justification_synthesis", JustificationSynthesisNode(self.accessor, runner))
            graph.add_node("ont_cell_type", OntCellTypeNode(self.accessor, runner, self.llm_client))
            graph.set_entry_point("justification_synthesis")
            graph.add_edge("justification_synthesis", "ont_cell_type")
            graph.add_edge("ont_cell_type", END)
            return graph.compile()
        from sclambda.pipelines.celltype.nodes import (
            CrossClusterSuggestionNode,
            CrossClusterDENode,
            CrossClusterCorrectionNode,
            JustificationSynthesisNode,
            OntCellTypeNode,
        )

        graph = StateGraph(CellTypingState)
        graph.add_node(
            "cross_cluster_suggestion",
            CrossClusterSuggestionNode(self.accessor, runner, self.llm_client),
        )
        graph.add_node(
            "cross_cluster_de",
            CrossClusterDENode(self.accessor, runner),
        )
        graph.add_node(
            "cross_cluster_correction",
            CrossClusterCorrectionNode(self.accessor, runner, self.llm_client),
        )
        graph.add_node(
            "justification_synthesis",
            JustificationSynthesisNode(self.accessor, runner),
        )
        graph.add_node(
            "ont_cell_type",
            OntCellTypeNode(self.accessor, runner, self.llm_client),
        )
        graph.set_entry_point("cross_cluster_suggestion")
        graph.add_edge("cross_cluster_suggestion", "cross_cluster_de")
        graph.add_edge("cross_cluster_de", "cross_cluster_correction")
        graph.add_edge("cross_cluster_correction", "justification_synthesis")
        graph.add_edge("justification_synthesis", "ont_cell_type")
        graph.add_edge("ont_cell_type", END)
        return graph.compile()

    def make_post_state(
        self,
        result: dict,
        all_results: list[dict],
        **kwargs,
    ) -> dict:
        """Inject cross-cluster context for same-depth comparison."""
        cluster_id = result.get("cluster_id", "")
        depth = result.get("depth", 1)

        # Skip clusters that didn't go through full annotation
        if result.get("naive", False):
            result["_skip_post"] = True
            return result
        if not result.get("full_pipeline", True):
            result["_skip_post"] = True
            return result
        if result.get("can_inherit_parent", False):
            result["_skip_post"] = True
            return result
        if result.get("parent_classification") == "same_type":
            result["_skip_post"] = True
            return result
        if not result.get("hypotheses"):
            result["_skip_post"] = True
            return result

        # Collect other clusters at the same depth
        peers = []
        for r in all_results:
            other_id = r.get("cluster_id", "")
            if other_id == cluster_id:
                continue
            if r.get("depth", 1) != depth:
                continue
            if r.get("can_inherit_parent", False):
                continue
            hypotheses = r.get("updated_hypotheses") or r.get("hypotheses", [])
            if not hypotheses:
                continue
            best = hypotheses[0]
            cell_type = best.get("cell_type", r.get("cell_type", "unknown"))
            confidence = best.get("confidence", "unknown")
            peers.append(f"- **{other_id}**: {cell_type} (confidence: {confidence})")

        if not peers:
            return result

        state = dict(result)
        state["cross_cluster_context"] = "\n".join(peers)
        return state


# ============ ROUTING FUNCTIONS ============


def route_after_prepare_cluster(
    state: CellTypingState,
) -> Literal["rank_genes", "inherit_parent", "ct_generation", "naive_ct_generation"]:
    """
    Route after prepare_cluster based on available information.

    If top_genes are already available, skip rank_genes.
    In naive mode, route directly to naive_ct_generation (skip inheritance).
    If cluster has a parent, route to inherit_parent for LLM evaluation.
    Otherwise, proceed to ct_generation (root clusters).
    """
    if state.get("top_genes"):
        if state.get("naive"):
            return "naive_ct_generation"
        if state.get("parent_cluster_id"):
            return "inherit_parent"
        return "ct_generation"

    return "rank_genes"


def route_after_rank_genes(
    state: CellTypingState,
) -> Literal["inherit_parent", "ct_generation", "naive_ct_generation"]:
    """
    Route based on parent availability.

    In naive mode, route directly to naive_ct_generation (skip inheritance).
    If cluster has a parent, route to inherit_parent for LLM evaluation.
    Otherwise, proceed to CT generation (root clusters).
    """
    if state.get("naive"):
        return "naive_ct_generation"

    if state.get("parent_cluster_id"):
        return "inherit_parent"

    return "ct_generation"


def route_after_inherit_parent(
    state: CellTypingState,
) -> Literal["__end__", "ct_generation"]:
    """
    Route based on parent match classification.

    same_type: cluster matches parent (small cluster or LLM verdict) →
               annotation + ontology term already inherited, done.
    subtype/unrelated: proceed to full ct_generation pipeline.
    """
    classification = state.get("parent_classification")
    if classification == "same_type":
        return END
    return "ct_generation"


def route_after_ct_generation(
    state: CellTypingState,
) -> Literal["dossier_agent", "marker_refinement", "subcluster"]:
    """Route based on full_pipeline flag and generation success.

    When ``use_dossier`` is set, the agentic ``dossier_agent`` node replaces the
    linear marker_refinement <-> reevaluation loop.
    """
    if state.get("use_dossier") and not state.get("ct_generation_failed", False):
        # Initialization deliberately produces no model hypotheses. The actor
        # creates them; do not let the legacy hypothesis gate skip annotation.
        return "dossier_agent"
    if not state.get("full_pipeline", True):
        return "subcluster"
    if state.get("ct_generation_failed", False):
        return "subcluster"
    if not state.get("hypotheses"):
        return "subcluster"
    return "marker_refinement"


def route_after_refinement(
    state: CellTypingState,
) -> Literal["signature_de", "subcluster"]:
    """Route based on the refinement prompt's confidence decision.

    On the first pass (before any reevaluation), the refinement prompt decides
    whether signatures are needed.  On subsequent passes (after reevaluation
    loops back), the refinement prompt again decides whether to continue
    refining or exit.  Iteration limits are enforced as guardrails.
    """
    current_iteration = state.get("reevaluation_iteration", 0)
    min_iterations = state.get("min_reevaluation_iterations", 1)

    # Always run at least min_iterations rounds of signature_de + reevaluation
    if current_iteration < min_iterations:
        return "signature_de"

    # Beyond min: let the refinement prompt decide
    # (Hard cap is enforced in route_after_reevaluation, before marker_refinement runs)
    if state.get("refinement_confidence_decision") == "confident":
        return "subcluster"
    return "signature_de"


def route_after_reevaluation(state: CellTypingState) -> Literal["marker_refinement", "subcluster"]:
    """Route back to marker_refinement or stop if hard cap reached.

    The hard iteration cap is checked here to avoid running an unnecessary
    marker_refinement LLM call when the limit is already reached.
    """
    current_iteration = state.get("reevaluation_iteration", 0)
    max_iterations = state.get("max_reevaluation_iterations", 2)
    if current_iteration >= max_iterations:
        return "subcluster"
    return "marker_refinement"
