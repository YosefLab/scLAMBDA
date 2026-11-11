"""annotate_spatial pipeline: differential expression."""
from __future__ import annotations
import numpy as np

from sclambda.analysis.spatial_metrics import compute_gene_spatial_scores
from sclambda.analysis.targeted_de import compute_pathology_de_from_indices
from sclambda.annotation.niche.formatters import format_spatial_celltype_de
from sclambda.annotation.niche.populations import niche_and_rest_indices
from .common import MIN_CELLS, SpatialNode, _slice_restricted_niche_indices

class SpatialCellTypeDENode(SpatialNode):
    """Compute DE for top-k cell types: niche vs rest of dataset.

    For each selected cell type:
    - Target = cells of that type WITHIN this niche
    - Reference = cells of that type OUTSIDE this niche
    Also computes Moran's Z for top DE genes within the niche to
    measure spatial coherence of gene expression.
    """

    default_output = {
        "top_k_cell_types": [],
        "per_celltype_de": {},
        "gene_spatial_scores": {},
        "de_summary": "",
        "de_summary_detailed": "",
        "de_summary_enriched": "",
    }

    def execute(self, state: "AnnotateSpatialState") -> dict:
        cluster_id = state["cluster_id"]
        cell_type_col = state["cell_type_col"]
        spatial_key = state.get("spatial_key", "spatial")
        k_neighbors = state.get("k_neighbors", 10)
        morans_max_hops = state.get("morans_max_hops", 2)
        k = state.get("k_top_cell_types", 5)
        n_top_genes = state.get("n_top_genes", 20)
        slice_key = state.get("slice_key")
        slice_val = state.get("slice")

        cross_slide = state.get("cross_slide", False)

        proportions = state.get("spatial_proportions", [])
        if not proportions:
            self.log_warning(cluster_id, "No proportion data, skipping DE")
            return self.output

        # DE pools all slides in multi-slide mode; spatial Moran's Z stays per-slide.
        niche_indices, obs_df, rest_mask = niche_and_rest_indices(
            self.accessor,
            cluster_id,
            cell_type_col,
            slice_key=slice_key,
            slice_val=slice_val,
            cross_slide=cross_slide,
        )

        # Spatial coordinates for gene Moran's Z — largest-niche slide only.
        niche_spatial_indices = _slice_restricted_niche_indices(
            self.accessor, cluster_id, slice_key, slice_val,
        )
        niche_coords = self.accessor.get_obsm(spatial_key, niche_spatial_indices)
        if niche_coords.ndim > 1 and niche_coords.shape[1] > 2:
            niche_coords = niche_coords[:, :2]

        # Select top-k cell types by niche abundance
        candidates = []
        for p in proportions:
            ct = p["cell_type"]
            n_in_niche = p["n_cells"]
            # Check rest-of-dataset has enough cells of same type
            ct_rest_mask = (obs_df[cell_type_col].values == ct) & rest_mask
            n_rest = int(ct_rest_mask.sum())
            if n_in_niche >= MIN_CELLS and n_rest >= MIN_CELLS:
                candidates.append((ct, n_in_niche))

        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [ct for ct, _ in candidates[:k]]

        if not selected:
            self.log_warning(cluster_id, "No cell types with sufficient cells for DE")

        per_celltype_de = {}
        gene_spatial_scores = {}

        for ct in selected:
            ct_mask = obs_df[cell_type_col].values == ct

            # Target: cells of this type in the niche
            ct_all_indices = np.where(ct_mask)[0]
            target_indices = ct_all_indices[np.isin(ct_all_indices, niche_indices)]

            # Reference: cells of this type outside the niche
            reference_indices = np.where(ct_mask & rest_mask)[0]

            if len(target_indices) < MIN_CELLS or len(reference_indices) < MIN_CELLS:
                self.log_info(
                    cluster_id,
                    f"Skipping {ct}: insufficient cells "
                    f"(target={len(target_indices)}, ref={len(reference_indices)})",
                )
                continue

            de_result = compute_pathology_de_from_indices(
                accessor=self.accessor,
                target_indices=target_indices,
                reference_indices=reference_indices,
                cell_type_label=ct,
                n_top_genes=n_top_genes,
            )
            per_celltype_de[ct] = de_result

            # Compute Moran's Z for top DE genes within the niche
            top_genes = []
            if de_result.enriched_genes:
                top_genes.extend(g.name for g in de_result.enriched_genes[:n_top_genes])
            if de_result.depleted_genes:
                top_genes.extend(g.name for g in de_result.depleted_genes[:n_top_genes])

            if top_genes:
                # Moran's Z uses this cell type's cells on the largest-niche
                # slide only (DE may have pooled across slides).
                in_spatial = np.isin(target_indices, niche_spatial_indices)
                spatial_ct_indices = target_indices[in_spatial]
                if len(spatial_ct_indices) >= 3:
                    order = np.argsort(niche_spatial_indices)
                    niche_ct_local = order[np.searchsorted(
                        niche_spatial_indices, spatial_ct_indices, sorter=order)]
                    ct_coords = niche_coords[niche_ct_local]
                    X_expr, found_genes = self.accessor.get_expression_for_genes(
                        spatial_ct_indices,
                        top_genes,
                    )
                    if len(found_genes) > 0:
                        spatial_result = compute_gene_spatial_scores(
                            coords=ct_coords,
                            expression_matrix=X_expr,
                            gene_names=found_genes,
                            cluster_id=cluster_id,
                            k_neighbors=min(k_neighbors, len(spatial_ct_indices) - 1),
                            max_hops=morans_max_hops,
                        )
                        gene_spatial_scores[ct] = spatial_result.as_dict()

            self.log_info(
                cluster_id,
                f"DE for {ct}: {len(de_result.enriched_genes)} enriched, "
                f"{len(de_result.depleted_genes)} depleted "
                f"(n_niche={de_result.n_target}, n_rest={de_result.n_reference})",
            )

        # ---- Region-level DE: ALL niche cells vs ALL rest of tissue ----
        # The per-cell-type DE above compares each cell type to itself elsewhere,
        # which washes out the niche's *composition* — the very signal that
        # defines many regions (brain regions differ by which neurons they
        # contain, not by within-neuron-type state). This all-cell contrast
        # surfaces the region's overall identity markers (its dominant cell-type
        # programs + state), so the model isn't forced onto position/backdrop.
        region_de = None
        niche_all_indices = np.unique(niche_indices)
        rest_all_indices = np.where(rest_mask)[0]
        if len(niche_all_indices) >= MIN_CELLS and len(rest_all_indices) >= MIN_CELLS:
            try:
                region_de = compute_pathology_de_from_indices(
                    accessor=self.accessor,
                    target_indices=niche_all_indices,
                    reference_indices=rest_all_indices,
                    cell_type_label="Niche overall — all cell types pooled (vs rest of tissue)",
                    n_top_genes=n_top_genes,
                )
                self.output["region_de"] = region_de
            except Exception as e:
                self.log_warning(cluster_id, f"Region-level DE failed: {e}")

        # Filter to cell types with significant DE
        active_cts = [ct for ct, de in per_celltype_de.items() if de.enriched_genes or de.depleted_genes]

        self.output["top_k_cell_types"] = active_cts
        self.output["per_celltype_de"] = per_celltype_de
        self.output["gene_spatial_scores"] = gene_spatial_scores

        active_de = {ct: per_celltype_de[ct] for ct in active_cts}
        active_scores = {ct: gene_spatial_scores.get(ct, {}) for ct in active_cts}
        # Surface the region-level signature FIRST — it carries the niche's
        # composition-defining identity that the per-cell-type contrast removes.
        if region_de is not None and (region_de.enriched_genes or region_de.depleted_genes):
            _region_key = "Niche overall — all cell types pooled (vs rest of tissue)"
            active_de = {_region_key: region_de, **active_de}
            active_scores = {_region_key: {}, **active_scores}
        self.output["de_summary"] = format_spatial_celltype_de(
            active_de,
            active_scores,
            detailed=False,
        )
        self.output["de_summary_detailed"] = format_spatial_celltype_de(
            active_de,
            active_scores,
            detailed=True,
        )
        self.output["de_summary_enriched"] = format_spatial_celltype_de(
            active_de,
            active_scores,
            detailed=True,
            enriched_only=True,
        )

        self.log_info(
            cluster_id,
            f"Spatial DE complete: {len(active_cts)}/{len(selected)} " f"cell types with significant DE",
        )
        return self.output
