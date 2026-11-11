"""
LangGraph-based spatial transcriptomics annotation pipeline.

Operates per-niche (from external spatial clustering). With population labels,
computes cell type proportions, population-specific DE and spatial statistics.
Without them, uses a uniform spot population for pooled niche-vs-rest expression.
An LLM generates and refines spatial region annotations from the measurements.

Flat execution only (depth=1, niches processed in parallel).
"""

from __future__ import annotations
import logging
from typing import TypedDict

import numpy as np
from langgraph.graph import StateGraph, END

from sclambda.framework.base_pipeline import Pipeline

logger = logging.getLogger(__name__)


# ============ STATE DEFINITION ============


class AnnotateSpatialState(TypedDict, total=False):
    """State for spatial annotation pipeline (per niche)."""

    # ---- Identity ----
    cluster_id: str  # Niche from external spatial clustering
    depth: int  # Always 1
    max_depth: int  # Always 1

    # ---- Config ----
    _prompt_execution_log: list[dict]  # ordered prompt input/output history
    cell_type_col: str
    cell_type_col_generated: bool  # uniform spot labels, not measured cell types
    spatial_key: str  # obsm key for spatial coordinates (e.g. "spatial")
    k_top_cell_types: int  # Number of top cell types for DE (default 5)
    k_neighbors: int  # k for spatial weight matrix (default 10)
    morans_max_hops: int  # max hop radius for local Moran permutation null (default 2)
    n_top_genes: int
    organism: str
    tissue: str
    slice_key: str | None  # obs column for slice/section labels
    slice: str | None  # specific slice value to restrict analysis to
    include_spatial_image: bool  # attach niche image to LLM prompts
    spatial_image_px: int  # target pixel size of generated niche image
    flip_spatial_y: bool  # negate y coordinates in spatial image
    spatial_background_col: str | None  # obs column for colouring non-target cells

    # ---- Stage A: anatomical catalog (per-run, injected per niche) ----
    anatomical_catalog_md: str | None  # rendered catalog for prompt injection

    # ---- Slide-level dataset profile (per-run; orients every annotator) ----
    condition_col: str | None    # slide-level condition column, for the cross-slide image sub-agent

    # ---- Persistent niche dossier (threaded through every LLM step) ----
    region_de: object
    spatial_observations: list[dict]
    spatial_distribution: dict | None
    dossier_review_complete: bool
    dossier_stop_reason: str
    annotation_assessment: str
    annotation_status: str
    annotation_uncertainty: str
    annotation_alternatives: list
    niche_dossier: dict | None  # serialized NicheDossier
    niche_description: str | None  # running free-text characterization
    spatial_comparison_review: bool
    spatial_resolution_comparison: bool  # experimental comparison schema
    gene_alias_path: str | None
    evidence_cache_dir: str | None
    reference_lookup: bool
    gene_alias_download: bool


    # ---- Spatial image ----
    spatial_image_base64: str | None
    spatial_image_description_images: list[str]  # whole-section + zoom views attached to the image-description / generation
    spatial_image_description: str | None  # image-read POSITION READ (location / rules-out / ambiguity) → seeds dossier.spatial_position
    spatial_image_legend: dict | None  # {label: hex_color} for background cells
    niche_region_context: str | None  # dominant value(s) of spatial_background_col in niche cells

    # ---- Step 1: Spatial proportions + Moran's Z ----
    spatial_proportions: list[dict]
    proportion_summary: str

    # ---- Step 2: Per-cell-type DE + gene spatial scores ----
    top_k_cell_types: list[str]
    per_celltype_de: dict  # cell_type -> PathologyDEResult
    gene_spatial_scores: dict  # cell_type -> {gene_name: morans_z}
    de_summary: str  # simplified gene names only
    de_summary_detailed: str  # with metrics (for refinement)
    de_summary_enriched: str  # enriched-only with metrics (for hypothesis generation)

    # ---- Step 3: Annotation generation ----
    hypotheses: list[dict]
    annotation: str | None

    # ---- Step 4: Agentic dossier refinement ----
    # The model drives refinement: each step edits its belief and may pull data
    # (test_signature / get_celltype_de / consult_image) until axes resolve.
    dossier_agent_transcript: list[dict]  # per-step {step, tool, reasoning}
    dossier_agent_steps: int
    dossier_agent_max_steps: int
    updated_hypotheses: list[dict]        # backward-compatible view (neighbours/post-graph)

    # ---- Multi-slide ----
    selected_slide: str | None  # slide the niche is imaged on (largest-niche slide)
    image_slides: list[str | None]  # up to k slides to render for the image description
    cross_slide: bool           # pool all slides for DE + proportions (multi-slide mode)

    # ---- DE method ----
    de_method: str  # "ttest" (default) or "ols" (dropout-corrected)


    # ---- Terminal ----
    children: list[str]


# ============ PIPELINE ============


class AnnotateSpatialPipeline(Pipeline):
    """Spatial transcriptomics annotation pipeline (per-niche)."""

    name = "annotate_spatial"
    required_params = ()
    max_depth_override = 1
    requires_obs_columns = True

    DEPTH = 1
    MAX_DEPTH = 1

    _prompt_history_fields = (
        "cluster_id", "top_k_cell_types", "annotation",
        "anatomical_catalog_md", "niche_description", "niche_dossier",
        "proportion_summary", "de_summary", "de_summary_detailed", "de_summary_enriched",
        "hypotheses", "dossier_agent_transcript", "dossier_agent_steps", "updated_hypotheses",
    )

    def preprocess(self, initial_clusters: list[str], **kwargs):
        """Validate required obs columns and spatial coordinates."""
        cell_type_col = kwargs.get("cell_type_col")
        spatial_key = kwargs.get("spatial_key", "spatial")
        slice_key = kwargs.get("slice_key")
        slice_val = kwargs.get("slice")

        from sclambda.annotation.niche.populations import ensure_spatial_cell_types
        self._cell_type_col_generated = not bool(cell_type_col)
        self._cell_type_col = ensure_spatial_cell_types(self.accessor, cell_type_col)
        cell_type_col = self._cell_type_col
        obs_columns = list(self.accessor.obs.columns)

        if spatial_key not in self.accessor.obsm_keys:
            raise ValueError(f"spatial_key '{spatial_key}' not in obsm. " f"Available: {self.accessor.obsm_keys}")

        if slice_key:
            if slice_key not in obs_columns:
                raise ValueError(f"slice_key '{slice_key}' not in obs. Available: {obs_columns}")
            if slice_val:
                all_indices = np.arange(self.accessor.n_cells)
                slice_obs = self.accessor.get_obs([slice_key], all_indices)
                available_slices = sorted(slice_obs[slice_key].dropna().unique())
                if slice_val not in [str(s) for s in available_slices]:
                    raise ValueError(
                        f"slice '{slice_val}' not found in column '{slice_key}'. "
                        f"Available: {[str(s) for s in available_slices[:20]]}"
                    )

        # Populate overlay from the obs_columns used for clustering
        user_obs_columns = kwargs.get("obs_columns")
        labels = None
        if user_obs_columns and len(user_obs_columns) > 0:
            first_col = user_obs_columns[0]
            all_cell_indices = np.arange(self.accessor.n_cells)
            obs_subset = self.accessor.get_obs([first_col], all_cell_indices)
            labels = self._apply_cluster_encoding(obs_subset.iloc[:, 0].values.astype(str))

            l1_col = self.accessor._level_column(1)
            if l1_col not in self.accessor._overlay.columns:
                self.accessor._overlay[l1_col] = None
            self.accessor._overlay.loc[self.accessor._overlay.index, l1_col] = labels

        # ---- Stage A: anatomical catalog (once per run, cached) ----
        # Enumerate the canonical spatial compartments of this organism/tissue
        # so every niche is scored against a shared universe of candidate
        # labels (incl. low-prior-but-real compartments) rather than whatever a
        # single per-niche generation happens to think of.
        self._anatomical_catalog = None
        if kwargs.get("use_anatomical_catalog", False) and self.llm_client is not None:
            try:
                from sclambda.pipelines.annotate_spatial.catalog import build_catalog

                self._anatomical_catalog = build_catalog(
                    self.llm_client,
                    organism=kwargs.get("organism", "human"),
                    tissue=kwargs.get("tissue", "unknown"),
                )
            except Exception as e:
                logger.warning(f"Anatomical catalog generation failed: {e}")
                self._anatomical_catalog = None

        # ---- Per-niche slide selection (multi-slide mode) ----
        # Multi-slide: slice_key is set but no single slice_val is pinned.
        # Each niche is imaged (and Moran-scored) on the slide where it is
        # largest, while DE and proportions pool cells from every slide.
        self._niche_slide_selection: dict[str, str | None] = {}
        self._niche_slide_ranking: dict[str, list[str]] = {}
        self._cross_slide = bool(slice_key) and not slice_val
        if self._cross_slide:
            self._niche_slide_selection = self._compute_slide_selection(
                initial_clusters, slice_key,
            )
            logger.info(
                "Multi-slide mode: imaging %d niches on their largest slide; "
                "DE/proportions pooled across all slides",
                len(self._niche_slide_selection),
            )

    def _compute_slide_selection(
        self,
        cluster_ids: list[str],
        slice_key: str,
    ) -> dict[str, str | None]:
        """Pick, for each niche, the slide where it has the most cells.

        Used only to choose the slide for the niche image and spatial
        (Moran's Z) statistics; DE and proportions pool all slides.
        """
        all_indices = np.arange(self.accessor.n_cells)
        obs = self.accessor.get_obs([slice_key], all_indices)
        l1_col = self.accessor._level_column(1)
        niche_labels = self.accessor._overlay[l1_col].values.astype(str)

        selection: dict[str, str | None] = {}
        # Full slide ranking per niche (most niche cells first), used to image
        # the niche on its top-k slides for the image-description prompt.
        self._niche_slide_ranking: dict[str, list[str]] = {}
        for cluster_id in cluster_ids:
            niche_mask = niche_labels == cluster_id
            slide_counts = obs.loc[niche_mask, slice_key].astype(str).value_counts()
            ranked = [str(s) for s in slide_counts.index]
            self._niche_slide_ranking[cluster_id] = ranked
            selection[cluster_id] = ranked[0] if ranked else None

        return selection

    def make_cluster_state(
        self,
        cluster_id: str,
        depth: int,
        max_depth: int,
        **kwargs,
    ) -> AnnotateSpatialState:
        state = super().make_cluster_state(
            cluster_id,
            self.DEPTH,
            self.MAX_DEPTH,
            **kwargs,
        )
        # In multi-slide mode each niche is imaged on its largest slide;
        # in single-slide mode fall back to the pinned slice param.
        selected_slide = self._niche_slide_selection.get(cluster_id, kwargs.get("slice"))
        # Up to k slides (most niche cells first) to render for the image
        # description, so the model sees the niche across several sections.
        k_img = int(kwargs.get("image_description_max_slides", 1))
        ranking = self._niche_slide_ranking.get(cluster_id, [])
        image_slides = ranking[:k_img] if ranking else ([selected_slide] if selected_slide is not None else [None])

        state.update(
            {
                "cell_type_col": getattr(self, "_cell_type_col", kwargs.get("cell_type_col")),
                "cell_type_col_generated": getattr(self, "_cell_type_col_generated", False),
                "spatial_key": kwargs.get("spatial_key", "spatial"),
                "k_top_cell_types": kwargs.get("k_top_cell_types", 5),
                "k_neighbors": kwargs.get("k_neighbors", 10),
                "morans_max_hops": kwargs.get("morans_max_hops", 2),
                "slice_key": kwargs.get("slice_key"),
                "slice": selected_slide,
                "selected_slide": selected_slide,
                "image_slides": image_slides,
                "cross_slide": getattr(self, "_cross_slide", False),
                "de_method": kwargs.get("de_method", "ttest"),
                "include_spatial_image": kwargs.get("include_spatial_image", True),
                "spatial_image_px": kwargs.get("spatial_image_px", 700),
                "flip_spatial_y": kwargs.get("flip_spatial_y", False),
                "spatial_background_col": kwargs.get("spatial_background_col"),
                "spatial_image_base64": None,
                "anatomical_catalog_md": (
                    self._anatomical_catalog.to_markdown()
                    if getattr(self, "_anatomical_catalog", None) else None
                ),
                "niche_dossier": None,
                "niche_description": None,
                "condition_col": kwargs.get("condition_col"),
                "dossier_agent_max_steps": kwargs.get("dossier_agent_max_steps", 4),
                "spatial_comparison_review": kwargs.get("spatial_comparison_review", False),
                "spatial_resolution_comparison": kwargs.get("spatial_resolution_comparison", False),
                "gene_alias_path": kwargs.get("gene_alias_path"),
                "evidence_cache_dir": kwargs.get("evidence_cache_dir"),
                "reference_lookup": kwargs.get("reference_lookup", True),
                "gene_alias_download": kwargs.get("gene_alias_download", True),
                "children": [],
            }
        )
        return state

    def build_graph(self, runner):
        from .nodes import ComputeSpatialProportionsNode, SpatialImageDescriptionNode, SpatialCellTypeDENode, SpatialDossierAgentNode
        graph = StateGraph(AnnotateSpatialState)
        graph.add_node("composition", ComputeSpatialProportionsNode(self.accessor, runner))
        graph.add_node("images", SpatialImageDescriptionNode(self.accessor, runner, self.llm_client))
        graph.add_node("de", SpatialCellTypeDENode(self.accessor, runner))
        graph.add_node("review", SpatialDossierAgentNode(self.accessor, runner, self.llm_client))
        graph.set_entry_point("composition")
        graph.add_edge("composition", "images")
        graph.add_edge("images", "de")
        graph.add_edge("de", "review")
        graph.add_edge("review", END)
        return graph.compile()

    def write_outputs(self, accessor, output_path=None, **kwargs) -> list:
        """Write spatial annotation Excel (region + major_region) and prompt history."""
        import pandas as pd
        from pathlib import Path

        written = []

        # --- Excel ---
        excel_path = Path(kwargs.get("excel_path") or "lambda_annotations.xlsx")
        overlay = accessor._overlay
        cluster_col = accessor._level_column(self.DEPTH)
        region_col = accessor._annotation_column(self.DEPTH, "region")
        major_col = accessor._annotation_column(self.DEPTH, "major_region")

        if cluster_col not in overlay.columns:
            return written

        def _sort_key(x):
            try:
                return (0, tuple(int(p) for p in str(x).split("_")))
            except ValueError:
                return (1, (str(x),))

        unique_clusters = overlay[cluster_col].value_counts(dropna=True).index.tolist()

        inv_cluster_map = {v: k for k, v in self._cluster_name_map.items()} if self._cluster_name_map else {}

        current = {str(v["cluster_id"]): v for v in getattr(self, "_spatial_results", [])}
        rows = []
        for cid in unique_clusters:
            mask = overlay[cluster_col] == cid
            region = ""
            if region_col in overlay.columns:
                vals = overlay.loc[mask, region_col].dropna().unique()
                if len(vals) > 0:
                    region = vals[0]
            major = ""
            if major_col in overlay.columns:
                vals = overlay.loc[mask, major_col].dropna().unique()
                if len(vals) > 0:
                    major = vals[0]
            hypotheses = accessor.get_hypotheses(str(cid))
            justification = ""
            if hypotheses:
                justification = (
                    hypotheses[0].get("justification")
                    or hypotheses[0].get("reasoning")
                    or ""
                )
            original_name = inv_cluster_map.get(str(cid), str(cid)) if inv_cluster_map else str(cid)
            rows.append({
                "cluster_id": str(cid),
                "original_name": original_name,
                "region": region,
                "major_region": major,
                "justification": justification,
                "assessment_status": current.get(str(cid), {}).get("annotation_status", ""),
                "execution_status": current.get(str(cid), {}).get("annotation_assessment", ""),
                "uncertainty": current.get(str(cid), {}).get("annotation_uncertainty", ""),
            })

        depth_df = pd.DataFrame(rows).sort_values("cluster_id", key=lambda s: s.map(_sort_key))
        cols = ["cluster_id", "original_name", "major_region", "region", "justification", "assessment_status", "execution_status", "uncertainty"]
        if not inv_cluster_map:
            cols = [c for c in cols if c != "original_name"]
        depth_df = depth_df[cols]

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            depth_df.to_excel(writer, sheet_name=f"Level_{self.DEPTH}", index=False)
        written.append(excel_path)

        if output_path:
            output_path = Path(output_path)
            cell_df = pd.DataFrame(index=overlay.index)
            cell_df.index.name = "obs_index"
            cell_df["cluster_id"] = overlay[cluster_col]
            if major_col in overlay.columns:
                cell_df["major_region"] = overlay[major_col]
            if region_col in overlay.columns:
                cell_df["region"] = overlay[region_col]
            for field, source in (("assessment_status", "annotation_status"), ("execution_status", "annotation_assessment"), ("uncertainty", "annotation_uncertainty")):
                cell_df[field] = overlay[cluster_col].map(lambda cid: current.get(str(cid), {}).get(source, ""))
            cell_df.to_csv(output_path, sep="\t" if output_path.suffix == ".tsv" else ",")
            written.append(output_path)

        # --- Prompt history ---
        usage_path = self._write_usage_report(output_path, kwargs)
        if usage_path is not None:
            written.append(usage_path)
        prompts_path = self._resolve_prompts_path(output_path, kwargs)
        if prompts_path is not None:
            p = self._write_prompt_history(prompts_path)
            if p is not None:
                written.append(p)

        return written

    def run(self, initial_clusters: list[str], max_depth: int, **kwargs) -> list[dict]:
        """Two-level (coarse → fine) hierarchical annotation.

        1. Ask the model how many broad spatial domains to expect, group the fine
           niches into that many *coarse* domains (expression + spatial adjacency;
           no ground-truth region labels are used — see ``coarse.py``).
        2. Annotate the coarse domains first (level 1).
        3. Annotate the fine niches (level 2), each inheriting its coarse parent's
           annotation as a suggestion, and with the spatial images coloured /
           labelled by the level-1 coarse annotations.

        Off by default — an ablation showed the expression-based coarse
        "major region" orientation buys ~4pp gross / 0 fine on brain while the
        prototype clusters are only ~57% anatomically coherent (a noisy prior).
        Set ``hierarchical_coarse=True`` to re-enable the two-level pass.
        """
        hierarchical = kwargs.get("hierarchical_coarse", False)
        obs_cols = kwargs.get("obs_columns")
        fine_col = obs_cols[0] if obs_cols else None
        # The coarse grouping clusters EVERY distinct niche in the column (not
        # just the requested subset), so gate hierarchy on the full niche count.
        n_fine_total = (
            self.accessor._obs[fine_col].dropna().astype(str).nunique()
            if fine_col else 0
        )
        if not hierarchical or not fine_col or n_fine_total < 4:
            return self._run_level(initial_clusters, max_depth, **kwargs)

        from sclambda.pipelines.annotate_spatial.coarse import (
            compute_coarse_groups,
            estimate_n_coarse_niches,
        )

        organism = kwargs.get("organism", "human")
        tissue = kwargs.get("tissue", "unknown")
        spatial_key = kwargs.get("spatial_key", "spatial")
        slice_key = kwargs.get("slice_key")
        slice_val = kwargs.get("slice")

        # --- Level 0: how many broad domains, and which niches merge ---
        n_coarse = kwargs.get("n_coarse_niches") or estimate_n_coarse_niches(
            self.llm_client, organism, tissue, n_fine_total
        )
        fine_to_group = compute_coarse_groups(
            self.accessor, fine_col, spatial_key, n_coarse,
            slice_key=slice_key, slice_val=slice_val,
            k_neighbors=int(kwargs.get("coarse_k_neighbors", 15)),
        )
        coarse_ids = sorted(set(fine_to_group.values()), key=lambda c: int(c[1:]))
        fine_vals = self.accessor._obs[fine_col].astype(str).values
        logger.info(
            "Hierarchical: %d fine niches → %d coarse domains",
            len(initial_clusters), len(coarse_ids),
        )

        # --- Level 1: label every coarse domain in ONE hedged image prompt ---
        # (Not the full per-niche pipeline — that over-specifies and inherits the
        # deep-structure transcriptional degeneracy. A single numbered-image call
        # gives a broad, conditional "most likely X; else Z" parent per domain.)
        logger.info("=== Level 1: single-prompt coarse annotation (%d domains) ===", len(coarse_ids))
        from sclambda.pipelines.annotate_spatial.coarse import annotate_coarse_domains

        coarse_label = annotate_coarse_domains(
            self.llm_client, self.accessor, fine_to_group, fine_col, spatial_key,
            organism, tissue,
            slice_key=slice_key, slice_val=slice_val,
            flip=kwargs.get("flip_spatial_y", False),
            image_px=int(kwargs.get("coarse_image_px", 1600)),
            power=kwargs.get("coarse_power", "strong"),
            max_images=int(kwargs.get("coarse_max_images", 4)),
            max_per_image=int(kwargs.get("coarse_max_per_image", 1000)),
        )
        coarse_results = [
            {"cluster_id": cid, "annotation": coarse_label.get(cid, "")} for cid in coarse_ids
        ]

        # Per-cell coarse annotation column: drives BOTH the level-2 parent
        # suggestion ("Major region") and the spatial-image backdrop. This is the
        # pipeline's OWN level-1 output — never a ground-truth region column.
        coarse_ann_col = "_coarse_annotation"
        self.accessor._obs[coarse_ann_col] = np.array(
            [coarse_label.get(fine_to_group.get(v, ""), "") for v in fine_vals]
        )
        self._coarse_annotation_col = coarse_ann_col

        # --- Level 2: annotate the fine niches with the coarse parent ---
        logger.info("=== Level 2: annotating %d fine niches ===", len(initial_clusters))
        fine_kwargs = dict(kwargs)
        fine_kwargs.update(
            obs_columns=[fine_col],
            hierarchical_coarse=False,
            spatial_background_col=coarse_ann_col,
        )
        fine_results = self._run_level(initial_clusters, max_depth, **fine_kwargs)

        # Attach the coarse parent to each fine result for downstream use.
        # cluster_id is the encoded integer ("0".."N-1"); decode to the raw
        # niche name to look up its coarse group.
        enc_to_name = {v: k for k, v in (self._cluster_name_map or {}).items()}
        for r in fine_results:
            cid = str(r.get("cluster_id"))
            raw_name = enc_to_name.get(cid, cid)
            grp = fine_to_group.get(raw_name, "")
            r["coarse_group"] = grp
            r["coarse_annotation"] = coarse_label.get(grp, "")
        self._coarse_results = coarse_results
        return fine_results

    def _run_level(self, initial_clusters: list[str], max_depth: int, **kwargs) -> list[dict]:
        """Run local decisions and optionally review global observations with the same critic."""
        all_results = super().run(initial_clusters, max_depth, **kwargs)

        # The critic already emits display-ready names and optional major regions.
        # Global review observes contradictions; the SAME critic adjudicates them.
        from sclambda.annotation.niche.dossier import NicheDossier
        from sclambda.annotation.niche.measurements import add_measurement
        from sclambda.core.task_runner import TaskRunner
        from .nodes.agent import SpatialDossierAgentNode
        for result in all_results:
            d = NicheDossier.from_dict(result.get("niche_dossier"))
            if d.critic_decision:
                result["major_region"] = d.critic_decision.get("major_region") or d.annotation
                self.accessor.set_annotation(result["cluster_id"], result["major_region"], column="major_region")
        for result in all_results:
            result["global_review_status"] = "not_requested"
        if kwargs.get("spatial_consistency_review", False):
            from .spatial_consistency_review import review_spatial_consistency
            try:
                proposals = review_spatial_consistency(self.accessor, self.llm_client, all_results,
                    organism=kwargs.get("organism", "human"), tissue=kwargs.get("tissue", "unknown"),
                    spatial_key=kwargs.get("spatial_key", "spatial"), slice_key=kwargs.get("slice_key"),
                    selected_slides=[kwargs["slice"]] if kwargs.get("slice") else None,
                    image_px=int(kwargs.get("spatial_image_px", 1500)), flip=kwargs.get("flip_spatial_y", False))
                for result in all_results:
                    result["global_review_status"] = "complete"
                with TaskRunner(max_workers=1) as runner:
                    reviewer = SpatialDossierAgentNode(self.accessor, runner, self.llm_client)
                    for result in all_results:
                        cid = result["cluster_id"]
                        if cid not in proposals:
                            continue
                        d = NicheDossier.from_dict(result.get("niche_dossier"))
                        d.decision_mode = True
                        for proposal in proposals[cid]:
                            add_measurement(d, proposal["evidence"])
                        state = {**result, "organism": kwargs.get("organism", "human"), "tissue": kwargs.get("tissue", "unknown"),
                                 "spatial_comparison_review": kwargs.get("spatial_comparison_review", False),
                                 "spatial_resolution_comparison": kwargs.get("spatial_resolution_comparison", False)}
                        hypotheses = [{k: v for k, v in p.items() if k != "evidence"} for p in proposals[cid]]
                        decision, attempts = reviewer.review(state, d, hypotheses, final=True)
                        result.setdefault("dossier_agent_transcript", []).append({"tool": "global_critic", "attempts": attempts})
                        if decision is None:
                            result["global_review_status"] = "failed"
                            result["global_review_error"] = "Optional critic review failed; prior local decision retained"
                            result["niche_dossier"] = d.to_dict()
                            continue
                        d.critic_decision = decision
                        d.description = decision["reason"]
                        d.open_questions = [decision["next_question"]] if decision["next_question"] else []
                        reviewer.output = {}
                        reviewer.commit_dossier(d, cid)
                        result.update(reviewer.output)
                        result.update(annotation_status=decision["status"], annotation_assessment=decision["status"],
                            annotation_uncertainty=decision["uncertainty"], annotation_alternatives=decision["alternatives"],
                            dossier_review_complete=True, dossier_stop_reason="global_critic_complete",
                            major_region=decision.get("major_region") or decision["label"])
                        self.accessor.set_annotation(cid, result["major_region"], column="major_region")
            except Exception as exc:
                logger.warning("Global observation pass failed: %s", exc)
                for result in all_results:
                    result.update(global_review_status="failed", global_review_error=str(exc))
        self._last_results = all_results
        self._spatial_results = all_results
        return all_results
