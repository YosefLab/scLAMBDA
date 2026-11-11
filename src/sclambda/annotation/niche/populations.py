"""Population labels for spatial observations without cell-type annotations."""
from __future__ import annotations

import numpy as np
import pandas as pd


SPOT_NOTE = (
    "Cell-type annotations were not supplied. 'spot' is a uniform placeholder "
    "for spatial observations, not an inferred cell type. Its 100% fraction is "
    "not measured cell-type composition. Expression contrasts pool these "
    "observations and do not resolve individual cell types or same-cell coexpression."
)


def ensure_spatial_cell_types(accessor, cell_type_col: str | None) -> str:
    """Resolve explicit labels or add a synthetic column to in-memory obs only.

    Never replace user metadata, including an existing column named ``spot``.
    Explicit misspellings remain errors instead of silently discarding labels.
    """
    if cell_type_col:
        if cell_type_col not in accessor.obs.columns:
            raise ValueError(f"cell_type_col '{cell_type_col}' not in obs. Available: {list(accessor.obs.columns)}")
        return cell_type_col
    previous = getattr(accessor, "_spatial_spot_column", None)
    if previous and previous in accessor.obs and accessor.obs[previous].eq("spot").all():
        return previous
    column = "spot"
    suffix = 0
    while column in accessor.obs:
        suffix += 1
        column = f"_sclambda_spot_{suffix}"
    accessor.obs[column] = pd.Categorical.from_codes(
        np.zeros(len(accessor.obs), dtype=np.int8), categories=["spot"])
    accessor._spatial_spot_column = column
    return column


def population_context(generated: bool) -> dict:
    """Keep synthetic population provenance alongside the actual measurements."""
    return {"cell_type_labels_provided": False, "population_note": SPOT_NOTE} if generated else {}


def niche_and_rest_indices(
    accessor,
    cluster_id,
    cell_type_col,
    slice_key=None,
    slice_val=None,
    cross_slide=False,
):
    """
    Get indices of cells in the niche and cells outside the niche, for DE.

    When slice_key and slice_val are provided, both niche and rest are
    restricted to cells from that slice only. Set ``cross_slide=True`` to
    ignore the slice restriction and pool cells from every slide (more
    statistical power for niche-vs-rest DE); the spatial Moran's statistics
    stay per-slide elsewhere.

    Returns:
        Tuple of (niche_indices, obs_df, rest_mask) where:
        - niche_indices: cell indices in this niche (within slice unless pooling)
        - obs_df: DataFrame with cell_type_col for ALL cells
        - rest_mask: Boolean mask for cells NOT in the niche (within slice unless pooling)
    """
    niche_indices = accessor.get_cell_indices(cluster_id, depth=1)
    all_indices = np.arange(accessor.n_cells)

    obs_cols = [cell_type_col]
    if slice_key:
        obs_cols.append(slice_key)
    obs_df = accessor.get_obs(obs_cols, all_indices)

    # Build slice mask (True for cells in the target slice). When pooling DE
    # across slides, keep every cell regardless of slice.
    if slice_key and slice_val and not cross_slide:
        slice_mask = obs_df[slice_key].values.astype(str) == str(slice_val)
        # Filter niche indices to only cells in this slice
        niche_indices = niche_indices[slice_mask[niche_indices]]
    else:
        slice_mask = np.ones(accessor.n_cells, dtype=bool)

    # Rest = cells in the slice but NOT in the niche
    rest_mask_arr = np.ones(accessor.n_cells, dtype=bool)
    rest_mask_arr[niche_indices] = False
    rest_mask_arr &= slice_mask

    return niche_indices, obs_df, rest_mask_arr
