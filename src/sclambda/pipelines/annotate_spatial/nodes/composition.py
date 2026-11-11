"""annotate_spatial pipeline: composition."""
from __future__ import annotations
import numpy as np

from sclambda.analysis.spatial_metrics import compute_spatial_proportions
from sclambda.annotation.niche.formatters import format_spatial_proportions
from .common import MIN_CELLS, SpatialNode, _slice_restricted_niche_indices, logger

class ComputeSpatialProportionsNode(SpatialNode):
    """Compute absolute cell type proportions and Moran's Z within the niche.

    Reports absolute proportions (not changes vs reference) and uses
    Moran's Z to characterize whether each cell type is spatially
    clustered or evenly distributed within the niche.

    Also generates the spatial niche image (if ``include_spatial_image``
    is enabled) so that downstream LLM nodes can attach it to their
    prompts.
    """

    default_output = {
        "spatial_proportions": [],
        "proportion_summary": "No spatial proportion data available.",
        "spatial_image_base64": None,
        "spatial_image_description_images": [],
        "spatial_image_legend": None,
        "niche_region_context": None,
    }

    def execute(self, state: "AnnotateSpatialState") -> dict:
        cluster_id = state["cluster_id"]
        cell_type_col = state["cell_type_col"]
        spatial_key = state.get("spatial_key", "spatial")
        k_neighbors = state.get("k_neighbors", 10)
        morans_max_hops = state.get("morans_max_hops", 2)
        slice_key = state.get("slice_key")
        slice_val = state.get("slice")
        cross_slide = state.get("cross_slide", False)

        # Spatial cells (image + Moran's Z) — always the largest-niche slide.
        spatial_indices = _slice_restricted_niche_indices(
            self.accessor, cluster_id, slice_key, slice_val,
        )
        # Composition cells — pool every slide in multi-slide mode, else the slide.
        if cross_slide:
            prop_indices = self.accessor.get_cell_indices(cluster_id, depth=1)
        else:
            prop_indices = spatial_indices

        if len(prop_indices) == 0:
            self.log_warning(cluster_id, f"Too few cells in niche: {len(prop_indices)}")
            return self.output

        # Cell-type composition over the (pooled) niche cells
        prop_obs = self.accessor.get_obs([cell_type_col], prop_indices)
        prop_cell_types = prop_obs[cell_type_col].values.astype(str)

        # Moran's Z characterizes spatial clustering — computed on the
        # largest-niche slide only (cross-slide neighbours are meaningless).
        morans_by_ct: dict[str, float] = {}
        if len(spatial_indices) >= MIN_CELLS:
            spatial_obs = self.accessor.get_obs([cell_type_col], spatial_indices)
            spatial_cell_types = spatial_obs[cell_type_col].values.astype(str)
            spatial_coords = self.accessor.get_obsm(spatial_key, spatial_indices)
            if spatial_coords.ndim > 1 and spatial_coords.shape[1] > 2:
                spatial_coords = spatial_coords[:, :2]
            morans_result = compute_spatial_proportions(
                coords=spatial_coords,
                cell_types=spatial_cell_types,
                cluster_id=cluster_id,
                k_neighbors=k_neighbors,
                max_hops=morans_max_hops,
            )
            morans_by_ct = {m["cell_type"]: m["morans_z"] for m in morans_result.as_dicts()}

        # Build proportion dicts from pooled composition, sorted by abundance,
        # attaching the per-slide Moran's Z (0.0 if the type is absent there).
        n_total = len(prop_cell_types)
        unique_types, counts = np.unique(prop_cell_types, return_counts=True)
        order = np.argsort(-counts)
        proportions = [
            {
                "cell_type": str(ct),
                "n_cells": int(cnt),
                "proportion": round(float(cnt) / n_total, 4),
                "morans_z": morans_by_ct.get(str(ct)),
            }
            for ct, cnt in zip(unique_types[order], counts[order])
        ]
        from sclambda.annotation.niche.measurements import distribution_record
        self.output["spatial_distribution"] = distribution_record(self.accessor, cluster_id, state)
        self.output["spatial_proportions"] = proportions
        self.output["proportion_summary"] = format_spatial_proportions(proportions)

        # Compute major brain region context from background column
        background_col = state.get("spatial_background_col")
        if background_col:
            try:
                from collections import Counter
                bg_obs = self.accessor.get_obs([background_col], prop_indices)
                bg_vals = bg_obs[background_col].values.astype(str)
                counts = Counter(bg_vals)
                top_regions = [r for r, _ in counts.most_common(3) if r and r != "nan"]
                if top_regions:
                    self.output["niche_region_context"] = ", ".join(top_regions)
            except Exception as bg_err:
                logger.warning(f"Could not compute region context from '{background_col}': {bg_err}")

        self.log_info(
            cluster_id,
            f"Spatial proportions: {len(proportions)} cell types in niche "
            f"({n_total} cells{' pooled across slides' if cross_slide else ''})",
        )
        return self.output

    def _generate_image(
        self,
        cluster_id: str,
        spatial_key: str,
        slice_key: str | None,
        slice_val: str | None,
        *,
        image_px: int = 700,
        flip: bool = False,
        background_col: str | None = None,
        proto_map: dict | None = None,
        bg_alpha: float = 1.0,
    ) -> tuple[str | None, dict | None]:
        """Generate a base64-encoded spatial niche image and its color legend.

        Returns
        -------
        (img_b64, legend) where legend is {label: hex_color} for background cells
        when background_col is provided, otherwise None.
        """
        try:
            from sclambda.analysis.spatial_visualization import (
                generate_niche_image_base64,
                build_background_color_legend,
            )

            all_indices = np.arange(self.accessor.n_cells)

            # Restrict to slice if needed
            if slice_key and slice_val is not None:
                slice_obs = self.accessor.get_obs([slice_key], all_indices)
                s_mask = slice_obs[slice_key].values.astype(str) == str(slice_val)
                vis_indices = np.where(s_mask)[0]
            else:
                vis_indices = all_indices

            vis_coords = self.accessor.get_obsm(spatial_key, vis_indices)
            if vis_coords.ndim > 1 and vis_coords.shape[1] > 2:
                vis_coords = vis_coords[:, :2]

            # Niche labels from overlay (L1 column)
            l1_col = self.accessor._level_column(1)
            niche_labels = self.accessor._overlay.iloc[vis_indices][l1_col].values.astype(str)

            # Optional alternative background colouring
            background_labels = None
            if background_col:
                try:
                    bg_obs = self.accessor.get_obs([background_col], vis_indices)
                    background_labels = bg_obs[background_col].values.astype(str)
                except Exception as bg_err:
                    logger.warning(f"Could not load background_col '{background_col}': {bg_err}")
            # Background is coloured by the fine niche label (each niche a
            # distinct colour, no names drawn) with non-target niches dimmed
            # (bg_alpha) so the red target stands out. The expression-prototype
            # backdrop (proto_map) is intentionally NOT used here — its
            # gene-named zones were being read as anatomical identity.
            _ = proto_map  # kept for signature compat; deliberately unused

            def bounds(coords):
                return (float(coords[:, 0].min()), float(coords[:, 0].max()),
                        float(coords[:, 1].min()), float(coords[:, 1].max()))

            section_bounds = bounds(vis_coords)
            tmask = niche_labels == cluster_id
            crop = None
            if tmask.sum() >= 5:
                tc = vis_coords[tmask]
                ctr = (tc.min(0) + tc.max(0)) / 2
                span = vis_coords.max(0) - vis_coords.min(0)
                win = np.maximum((tc.max(0) - tc.min(0)) * 0.9, span * 0.08)
                m = np.all(np.abs(vis_coords - ctr) <= win, axis=1)
                if m.sum() >= 10:
                    crop = m
            crop_bounds = bounds(vis_coords[crop]) if crop is not None else None
            orientation = "y reflected by configuration" if flip else "coordinates as stored"

            def _render(coords, labels, bg, mode, box=None):
                return generate_niche_image_base64(
                    coords=coords,
                    niche_labels=labels,
                    target_niche=cluster_id,
                    background_labels=bg,
                    image_px=image_px,
                    flip=flip,
                    bg_alpha=bg_alpha,
                    neutral_background=True, mark_target=False,
                    section_bounds=section_bounds, crop_bounds=box,
                    title=f"{mode} | niche {cluster_id} RED | {orientation}\n"
                          "Plot directions are not anatomical directions",
                )

            images: list[str] = []
            # 1) whole-section view
            whole = _render(vis_coords, niche_labels, background_labels, "Whole section", crop_bounds)
            if whole:
                images.append(whole)
            # 2) zoomed view cropped around the target (gross layout + local detail)
            if crop is not None:
                zoom = _render(vis_coords[crop], niche_labels[crop],
                    background_labels[crop] if background_labels is not None else None,
                    "Crop (dashed box in whole section)")
                if zoom:
                    images.append(zoom)

            legend = None
            if background_labels is not None:
                legend = build_background_color_legend(background_labels, niche_labels, cluster_id)

            self.log_info(cluster_id, f"Spatial niche image(s) generated: {len(images)} (whole + zoom)")
            return images, legend
        except Exception as e:
            logger.warning(f"Failed to generate spatial image for {cluster_id}: {e}")
            return [], None
