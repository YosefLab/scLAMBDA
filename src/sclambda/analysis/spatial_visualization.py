"""
Spatial niche visualization tool.

Plots the spatial slide with all niches colored by a colorcet palette
(filtered to exclude red hues), while the target niche is always colored red.
A black cross marks the centroid of each spatial instantiation of the target
niche, where instantiations are detected via DBSCAN clustering on x,y
coordinates.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import TYPE_CHECKING

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend (safe for pipelines)
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Colour helpers
# --------------------------------------------------------------------------- #

# Red hue band to exclude (in [0, 1] HSV hue space).
# Covers roughly 330-360 and 0-30 degrees (red / orange-red / magenta-red).
_RED_HUE_LO = 0.0
_RED_HUE_HI = 0.08  # ~30 deg
_RED_HUE_WRAP_LO = 0.92  # ~330 deg
_RED_HUE_WRAP_HI = 1.0


def _is_reddish(rgb: tuple[float, ...]) -> bool:
    """Return True if *rgb* (0-1 floats) falls in the red hue band."""
    import matplotlib.colors as mcolors

    h, s, v = mcolors.rgb_to_hsv(rgb[:3])
    if s < 0.15:  # desaturated / gray – not red
        return False
    return (_RED_HUE_LO <= h <= _RED_HUE_HI) or (_RED_HUE_WRAP_LO <= h <= _RED_HUE_WRAP_HI)


def _get_niche_palette(n_niches: int) -> list[tuple[float, ...]]:
    """Return *n_niches* colours from colorcet, excluding red hues.

    Falls back to matplotlib tab20 if colorcet is unavailable.
    """
    try:
        import colorcet as cc
        import matplotlib.colors as mcolors

        # cc.glasbey is a 256-colour categorical palette (hex strings)
        pool_hex: list[str] = cc.glasbey
        pool = [mcolors.to_rgb(h) for h in pool_hex]
    except (ImportError, AttributeError):
        # Fallback: sample from tab20 + tab20b + tab20c (60 colours)
        cmap20 = plt.cm.tab20(np.linspace(0, 1, 20))
        cmap20b = plt.cm.tab20b(np.linspace(0, 1, 20))
        cmap20c = plt.cm.tab20c(np.linspace(0, 1, 20))
        pool = [tuple(c[:3]) for c in np.vstack([cmap20, cmap20b, cmap20c])]

    # Filter out reddish colours
    safe: list[tuple[float, ...]] = []
    for c in pool:
        rgb = tuple(float(x) for x in c[:3])
        if not _is_reddish(rgb):
            safe.append(rgb)

    # Cycle if we don't have enough
    while len(safe) < n_niches:
        safe = safe + safe
    return safe[:n_niches]


# --------------------------------------------------------------------------- #
#  Instantiation detection
# --------------------------------------------------------------------------- #


def _detect_instantiations(
    coords: np.ndarray,
    eps: float | None = None,
    min_samples: int = 5,
) -> list[np.ndarray]:
    """Detect spatial instantiations of a niche using DBSCAN.

    Parameters
    ----------
    coords : ndarray of shape (n, 2)
        Spatial coordinates of cells belonging to the target niche.
    eps : float, optional
        DBSCAN neighbourhood radius.  When *None* the value is estimated from
        the data as 5 % of the spatial extent diagonal.
    min_samples : int
        Minimum cluster size for DBSCAN.

    Returns
    -------
    list of ndarray
        Each element contains the row-indices (into *coords*) of cells
        belonging to one instantiation.  Noise points are folded into
        the nearest instantiation.
    """
    if len(coords) == 0:
        return []

    if eps is None:
        extent = coords.max(axis=0) - coords.min(axis=0)
        diag = np.linalg.norm(extent)
        eps = max(diag * 0.05, 1e-6)

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_

    unique_labels = set(labels)
    unique_labels.discard(-1)  # noise label

    if len(unique_labels) == 0:
        # Everything is noise – treat as a single instantiation
        return [np.arange(len(coords))]

    clusters: list[np.ndarray] = []
    centroids: list[np.ndarray] = []
    for lab in sorted(unique_labels):
        idx = np.where(labels == lab)[0]
        clusters.append(idx)
        centroids.append(coords[idx].mean(axis=0))

    # Assign noise points to nearest cluster centroid
    noise_mask = labels == -1
    if noise_mask.any():
        centroids_arr = np.array(centroids)
        noise_indices = np.where(noise_mask)[0]
        noise_coords = coords[noise_indices]
        # Compute distances from each noise point to each centroid
        dists = np.linalg.norm(
            noise_coords[:, None, :] - centroids_arr[None, :, :],
            axis=2,
        )
        nearest = dists.argmin(axis=1)
        for ni, ci in zip(noise_indices, nearest):
            clusters[ci] = np.append(clusters[ci], ni)

    return clusters


# --------------------------------------------------------------------------- #
#  Dynamic grid helper
# --------------------------------------------------------------------------- #


def _nice_step(span: float, target_ticks: int = 6) -> float:
    """Pick a human-friendly grid step for *span* with ~*target_ticks* lines."""
    raw = span / max(target_ticks, 1)
    if raw <= 0:
        return 1.0
    magnitude = 10 ** np.floor(np.log10(raw))
    residual = raw / magnitude
    if residual <= 1.5:
        nice = 1.0
    elif residual <= 3.5:
        nice = 2.0
    elif residual <= 7.5:
        nice = 5.0
    else:
        nice = 10.0
    return nice * magnitude


# --------------------------------------------------------------------------- #
#  Main plotting function
# --------------------------------------------------------------------------- #


def _auto_sizes(xy, figsize=(6, 6), k: int = 6):
    """
    Estimate marker sizes for spatial plots using density + figure size.

    Parameters
    ----------
    xy : (N,2) array
        Spatial coordinates.
    figsize : tuple(float, float)
        Matplotlib figure size in inches.
    k : int
        Neighbor index for spacing estimate.

    Returns
    -------
    point_size : float        # scatter s (area)
    cross_size : float        # markersize
    cross_linewidth : float   # marker edge width
    """

    xy = np.asarray(xy)
    if len(xy) < k + 2:
        return 5.0, 15.0, 1.0

    # --- typical spacing ---
    tree = cKDTree(xy)
    dists, _ = tree.query(xy, k=k + 1)
    spacing = np.median(dists[:, -1])

    # --- spatial extent ---
    width = xy[:, 0].max() - xy[:, 0].min()
    height = xy[:, 1].max() - xy[:, 1].min()
    extent = max(width, height)

    if extent <= 0:
        return 5.0, 15.0, 1.0

    # spacing as fraction of slide
    spacing_frac = spacing / extent

    # convert figure size to points
    fig_pts = max(figsize) * 72  # inches → points

    # marker sizes in points — ~0.8x the typical spacing so points fill the
    # tissue (a solid region the model can read) rather than sparse dots.
    point_diam = fig_pts * spacing_frac * 0.8
    cross_diam = fig_pts * spacing_frac * 50

    # scatter expects area in pt^2
    point_size = np.pi * (point_diam / 2) ** 2

    # linewidth proportional
    cross_linewidth = 5e-4 * cross_diam

    return float(point_size), float(cross_diam), float(cross_linewidth)


# Short codes drawn on regions (read far more reliably by a VLM than a
# colour -> hex legend). Target is always "T"; neighbours get A, B, C, ...
_NEIGHBOR_CODES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def _label_anchor(pts: np.ndarray) -> tuple[float, float]:
    """A point inside the densest part of a region's cells.

    The plain mean centroid lands outside non-convex or split regions (e.g. a
    cluster that wraps around its neighbour, so its mean falls in the hole).
    Instead, bin the points and return the mean of the densest bin — always a
    real, interior location.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) <= 5:
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())
    nb = int(np.clip(np.sqrt(len(pts) / 4.0), 6, 30))
    xb = np.linspace(pts[:, 0].min(), pts[:, 0].max(), nb + 1)
    yb = np.linspace(pts[:, 1].min(), pts[:, 1].max(), nb + 1)
    ix = np.clip(np.digitize(pts[:, 0], xb) - 1, 0, nb - 1)
    iy = np.clip(np.digitize(pts[:, 1], yb) - 1, 0, nb - 1)
    flat = ix * nb + iy
    vals, counts = np.unique(flat, return_counts=True)
    sel = flat == vals[int(np.argmax(counts))]
    return float(pts[sel, 0].mean()), float(pts[sel, 1].mean())


def _apply_coordinate_frame(ax, coords, *, n: int = 10, fontsize: int = 11,
                            section_bounds=None):
    """Use one section origin and scale, including when viewing a crop.

    Bounds and coordinates here are already in the displayed orientation.
    Relative section coordinates are not anatomical directions or units.
    """
    x, y = coords[:, 0], coords[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    xspan = max(xmax - xmin, 1e-9)
    yspan = max(ymax - ymin, 1e-9)
    xpad, ypad = xspan * 0.04, yspan * 0.04
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)

    # One common step for both axes so the grid cells are square (equal physical
    # spacing on x and y under equal aspect). The longer axis spans 1..n; the
    # shorter axis gets proportionally fewer ticks.
    rx0, rx1, ry0, ry1 = section_bounds or (xmin, xmax, ymin, ymax)
    step = max(rx1 - rx0, ry1 - ry0, 1e-9) / n
    nx = max(int(np.floor((rx1 - rx0) / step + 1e-6)), 1)
    ny = max(int(np.floor((ry1 - ry0) / step + 1e-6)), 1)
    xt = np.arange(1, nx + 1)
    yt = np.arange(1, ny + 1)
    xt = xt[(rx0 + xt * step >= xmin - xpad) & (rx0 + xt * step <= xmax + xpad)]
    yt = yt[(ry0 + yt * step >= ymin - ypad) & (ry0 + yt * step <= ymax + ypad)]
    ax.set_xticks(rx0 + xt * step)
    ax.set_yticks(ry0 + yt * step)
    ax.set_xticklabels(xt, fontsize=fontsize)
    ax.set_yticklabels(yt, fontsize=fontsize)
    ax.tick_params(axis="both", which="both", length=0, labelbottom=True, labelleft=True)
    ax.grid(True, which="major", linestyle=":", linewidth=0.7, color="0.6", alpha=0.8)
    ax.set_axisbelow(True)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(1.4)
        ax.spines[s].set_color("0.2")
    # arrowheads at the ends of the x and y axes
    ax.plot(1, 0, marker=">", ms=8, color="0.2", transform=ax.transAxes, clip_on=False)
    ax.plot(0, 1, marker="^", ms=8, color="0.2", transform=ax.transAxes, clip_on=False)
    # Keep true proportions but honour the limits set above so the 1..n grid
    # fills the frame edge-to-edge (adjust the box, not the data limits).
    ax.set_aspect("equal", adjustable="box")


def _draw_region_code(ax, coords, mask, code, *, fontsize=18, color="black"):
    """Draw a short code at the densest point of the masked cells, with a halo."""
    if not mask.any():
        return
    cx, cy = _label_anchor(coords[mask])
    txt = ax.text(
        cx, cy, code,
        fontsize=fontsize, fontweight="bold",
        ha="center", va="center", color=color, zorder=20,
    )
    txt.set_path_effects([pe.withStroke(linewidth=3.5, foreground="white")])


def plot_spatial_niche(
    coords: np.ndarray,
    niche_labels: np.ndarray,
    target_niche: str,
    *,
    background_labels: np.ndarray | None = None,
    cell_types: np.ndarray | None = None,
    title: str | None = None,
    point_size: float | None = None,
    figsize: tuple[float, float] = (14, 10),
    save_path: str | Path | None = None,
    dpi: int = 200,
    cross_size: float | None = None,
    cross_linewidth: float | None = None,
    dbscan_eps: float | None = None,
    dbscan_min_samples: int = 5,
    clean: bool = False,
    flip: bool = False,
    show: bool = True,
    bg_alpha: float = 1.0,
    number_niches: bool = False,
    neutral_background: bool = False,
    mark_target: bool = True,
    section_bounds: tuple | None = None,
    crop_bounds: tuple | None = None,
) -> plt.Figure:
    """Plot a spatial slide highlighting a target niche in red.

    Parameters
    ----------
    coords : ndarray of shape (n_cells, 2)
        Spatial x, y coordinates for every cell.
    niche_labels : ndarray of shape (n_cells,)
        Niche (cluster) label for every cell.  Used to identify target cells.
    target_niche : str
        The niche to highlight in red.
    background_labels : ndarray of shape (n_cells,), optional
        Alternative labels used to colour non-target cells.  When provided,
        the background (non-target) cells are coloured by these labels instead
        of ``niche_labels``.  Useful for showing coarser regions while the
        target niche is still identified by the finer ``niche_labels``.
    cell_types : ndarray, optional
        Cell type labels per cell (currently unused).
    title : str, optional
        Plot title.  Defaults to ``"Spatial Niche: <target_niche>"``.
    point_size : float or None
        Scatter point size.  ``None`` (default) = computed dynamically
        from *figsize* and number of cells.
    figsize : tuple
        Figure size in inches.
    save_path : str or Path, optional
        If given, save the figure to this path.
    dpi : int
        Resolution for saved figure.
    cross_size : float or None
        Size of the centroid cross marker.  ``None`` = dynamic.
    cross_linewidth : float or None
        Line width of the centroid cross marker.  ``None`` = dynamic.
    dbscan_eps : float, optional
        DBSCAN epsilon for instantiation detection.  ``None`` = auto.
    dbscan_min_samples : int
        DBSCAN min_samples parameter.
    clean : bool
        If True, produce a compact plot for LLM consumption: no axis labels,
        no tick labels, no legend; a light coordinate grid is added instead.
    flip : bool
        If explicitly True, negate y coordinates so that the image coordinate
        system (origin at top-left) is used.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Work on a copy so we never mutate the caller's array
    coords = np.array(coords, dtype=float, copy=True)

    niche_labels = np.asarray(niche_labels, dtype=str)

    # Determine which cells belong to the target niche
    target_mask = niche_labels == target_niche

    # Labels used to colour background (non-target) cells
    bg_labels = np.asarray(background_labels, dtype=str) if background_labels is not None else niche_labels
    other_bg = sorted(set(bg_labels[~target_mask]))

    # Dynamic sizes
    auto_ps, auto_cs, auto_clw = _auto_sizes(coords, figsize)
    if point_size is None:
        point_size = auto_ps
    if cross_size is None:
        cross_size = auto_cs
    if cross_linewidth is None:
        cross_linewidth = auto_clw

    # Build colour map: background categories get non-red colours
    palette = _get_niche_palette(len(other_bg))
    bg_color = {n: ((0.78, 0.78, 0.78) if neutral_background else palette[i])
                for i, n in enumerate(other_bg)}

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if flip:
        coords[:, 1] = -coords[:, 1]
    def display_bounds(bounds):
        if bounds is None:
            return None
        x0, x1, y0, y1 = bounds
        return (x0, x1, -y1, -y0) if flip else bounds

    section_bounds = display_bounds(section_bounds)
    crop_bounds = display_bounds(crop_bounds)

    # Draw all non-target cells in a SINGLE rasterized scatter using a
    # per-point colour array. Drawing one scatter per background label is
    # prohibitively slow when there are many niches (one full rasterization
    # pass each) and is visually identical here — the legend and the on-map
    # region codes are built separately from ``bg_color`` and never depend on
    # these scatter calls.
    bg_mask = ~target_mask
    if bg_mask.any():
        bg_pt_colors = np.array([bg_color[b] for b in bg_labels[bg_mask]])
        ax.scatter(
            coords[bg_mask, 0],
            coords[bg_mask, 1],
            s=point_size,
            c=bg_pt_colors,
            alpha=bg_alpha,
            rasterized=True,
        )

    # Draw target niche on top (foreground) – red
    if target_mask.any():
        ax.scatter(
            coords[target_mask, 0],
            coords[target_mask, 1],
            s=point_size * 1.5,
            c=[(0.9, 0.1, 0.1)],
            alpha=1.0,
            rasterized=True,
            label=target_niche,
        )

        # Detect instantiations and mark centroids
        niche_coords = coords[target_mask]
        instantiations = _detect_instantiations(
            niche_coords,
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
        ) if mark_target else []

        # Compute cross diameter in data coordinates so we can suppress
        # centroids that are too close together to be visually distinguishable.
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        extent = max(x_range, y_range)
        fig_pts = max(figsize) * 72  # figure size in points
        pts_per_data = fig_pts / extent if extent > 0 else 1.0
        cross_diam_data = 2 * np.sqrt(cross_size) / pts_per_data
        min_sep = 3 * cross_diam_data  # suppress if closer than 3× diameter

        # Greedy deduplication: keep a centroid only if it is at least
        # min_sep away from every already-kept centroid.
        centroids = [niche_coords[inst_idx][np.argmin(np.sum(
            (niche_coords[inst_idx] - niche_coords[inst_idx].mean(axis=0)) ** 2, axis=1))]
            for inst_idx in instantiations] if mark_target else []
        kept_centroids = []
        for centroid in centroids:
            if all(np.linalg.norm(centroid - k) >= min_sep for k in kept_centroids):
                kept_centroids.append(centroid)

        import matplotlib.patheffects as _pe

        # Mark each target centroid with a black arrow (white rim) pointing at it
        # from the upper-right, instead of an "X" cross.
        arrow_len = 0.06 * extent if extent > 0 else 1.0
        arrow_lw = max(cross_linewidth, 2.0)
        head = max(float(np.sqrt(cross_size)), 16.0)
        for centroid in kept_centroids:
            cx, cy = float(centroid[0]), float(centroid[1])
            ann = ax.annotate(
                "",
                xy=(cx, cy),
                xytext=(cx + arrow_len, cy + arrow_len),
                arrowprops=dict(
                    arrowstyle="-|>",
                    mutation_scale=head,
                    color="black",
                    linewidth=arrow_lw,
                    shrinkA=0,
                    shrinkB=0,
                ),
            )
            ann.arrow_patch.set_path_effects(
                [_pe.withStroke(linewidth=arrow_lw + 3.0, foreground="white")]
            )

        n_inst = len(instantiations)
        logger.info(
            f"Niche '{target_niche}': {target_mask.sum()} cells, "
            f"{n_inst} spatial instantiation(s) detected, "
            f"{len(kept_centroids)} cross(es) plotted after proximity filtering"
        )

    # ---- formatting ----
    ax.set_aspect("equal", adjustable="datalim")

    if clean:
        # LLM-friendly: an optional view title and a shared section coordinate frame
        # so the model can read the target's absolute position.
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(title or "", fontsize=11)
        _apply_coordinate_frame(ax, coords, section_bounds=section_bounds)
        if section_bounds is not None:
            ax.set_xlabel("Section x (relative coordinate)")
            ax.set_ylabel("Section y (relative coordinate)")
        # When the backdrop is a coarse grouping (background_labels provided,
        # e.g. expression prototypes), name each group on the map so the model
        # can read the gross tissue layout and place the target within it.
        if background_labels is not None and not neutral_background:
            for bg_id in other_bg:
                bmask = (~target_mask) & (bg_labels == bg_id)
                if bmask.any():
                    _draw_region_code(ax, coords, bmask, bg_id, fontsize=12, color=(0.2, 0.2, 0.2))
    else:
        ax.set_xlabel("Spatial X")
        ax.set_ylabel("Spatial Y")
        ax.set_title(title or f"Spatial Niche: {target_niche}")

        # Legend – show target niche + a subset of others
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=(0.9, 0.1, 0.1),
                markersize=8,
                label=f"{target_niche} (target niche)",
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="w",
                markerfacecolor="black",
                markeredgecolor="white",
                markersize=10,
                label="Instantiation centroid",
            ),
        ]
        for bg_id in other_bg[:15]:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=bg_color[bg_id],
                    markersize=6,
                    label=bg_id,
                ),
            )
        if len(other_bg) > 15:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="",
                    color="w",
                    label=f"... +{len(other_bg) - 15} more",
                ),
            )
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=7,
            frameon=True,
        )

    # Optionally overlay each niche's id at its centroid, in a box COLOURED to
    # match the niche (red for the target), with a visible black border — so the
    # model has discrete, colour-matched numbered landmarks to reason about.
    if number_niches:
        try:
            import matplotlib.colors as mcolors
            from scipy.ndimage import gaussian_filter as _gf
            num_fs = max(9.0, max(figsize) * 100.0 / 120.0)
            n_tot = len(niche_labels)
            for lab in np.unique(niche_labels):
                m = niche_labels == lab
                if int(m.sum()) < max(20, int(0.002 * n_tot)):
                    continue
                px, py = coords[m, 0], coords[m, 1]  # coords already flipped above
                nb = max(8, min(40, int(m.sum() ** 0.5)))
                Hh, xe, ye = np.histogram2d(px, py, bins=nb)
                Hh = _gf(Hh, 1.0)
                i, j = np.unravel_index(Hh.argmax(), Hh.shape)
                cx, cy = 0.5 * (xe[i] + xe[i + 1]), 0.5 * (ye[j] + ye[j + 1])
                if str(lab) == str(target_niche):
                    box_c = (0.9, 0.1, 0.1)
                else:
                    bgm = bg_labels[m]
                    vals, cnts = np.unique(bgm, return_counts=True)
                    box_c = bg_color.get(vals[cnts.argmax()], (0.8, 0.8, 0.8))
                r_, g_, b_ = mcolors.to_rgb(box_c)
                tc = "black" if (0.299 * r_ + 0.587 * g_ + 0.114 * b_) > 0.55 else "white"
                ax.text(
                    cx, cy, str(lab), fontsize=num_fs, ha="center", va="center",
                    color=tc, weight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor=box_c,
                              edgecolor="black", linewidth=0.8, alpha=0.95),
                )
        except Exception:  # noqa: BLE001 — numbering is best-effort
            pass

    if crop_bounds is not None:
        from matplotlib.patches import Rectangle
        x0, x1, y0, y1 = crop_bounds
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                              edgecolor="black", linestyle="--", linewidth=1, zorder=10))
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved spatial niche plot to {save_path}")

    if show:
        plt.show()

    return fig


# --------------------------------------------------------------------------- #
#  Color legend helper
# --------------------------------------------------------------------------- #


def build_background_color_legend(
    background_labels: np.ndarray,
    niche_labels: np.ndarray,
    target_niche: str,
) -> dict[str, str]:
    """Return {label: hex_color} for background cells, matching the palette used by plot_spatial_niche.

    Parameters
    ----------
    background_labels : ndarray
        Labels used to colour non-target cells (same array passed as ``background_labels``
        to ``plot_spatial_niche``).
    niche_labels : ndarray
        Niche membership labels used to identify target cells.
    target_niche : str
        The niche that is highlighted in red.

    Returns
    -------
    dict mapping label string → hex color string (e.g. "#4e79a7").
    """
    import matplotlib.colors as mcolors

    target_mask = np.asarray(niche_labels, dtype=str) == target_niche
    bg_labels = np.asarray(background_labels, dtype=str)
    other_bg = sorted(set(bg_labels[~target_mask]))
    palette = _get_niche_palette(len(other_bg))
    return {label: mcolors.to_hex(palette[i]) for i, label in enumerate(other_bg)}


# --------------------------------------------------------------------------- #
#  Image-to-base64 helper (for LLM prompt injection)
# --------------------------------------------------------------------------- #


def generate_niche_image_base64(
    coords: np.ndarray,
    niche_labels: np.ndarray,
    target_niche: str,
    *,
    background_labels: np.ndarray | None = None,
    dbscan_eps: float | None = None,
    dbscan_min_samples: int = 5,
    image_px: int = 700,
    flip: bool = False,
    bg_alpha: float = 1.0,
    number_niches: bool = False,
    neutral_background: bool = False,
    mark_target: bool = True,
    section_bounds: tuple | None = None,
    crop_bounds: tuple | None = None,
    title: str | None = None,
) -> str:
    """Generate a clean niche plot and return it as a base64-encoded PNG string.

    The image is produced with ``clean=True`` (section grid, optional title, no legend)
    so that it can be embedded directly into an LLM prompt.

    The figure dimensions are computed from the data aspect ratio so that
    no pixels are wasted on empty space.  The longer axis gets *image_px*
    pixels; the shorter axis is scaled proportionally.

    Marker sizes and cross sizes are computed dynamically from the figure
    dimensions and cell count.

    Parameters
    ----------
    background_labels : ndarray of shape (n_cells,), optional
        Alternative labels used to colour non-target cells (e.g. a coarser
        region column).  See :func:`plot_spatial_niche`.
    image_px : int
        Target size in pixels for the longer dimension.
    flip : bool
        If explicitly True, negate y coordinates.
    """
    _dpi = 100

    # Compute data extent to derive a matching aspect ratio
    _coords = np.asarray(coords, dtype=float)
    x_span = max(float(np.ptp(_coords[:, 0])), 1e-6)
    y_span = max(float(np.ptp(_coords[:, 1])), 1e-6)
    aspect = x_span / y_span  # width / height of data

    if aspect >= 1.0:
        w_px = image_px
        h_px = max(int(image_px / aspect), 1)
    else:
        h_px = image_px
        w_px = max(int(image_px * aspect), 1)

    _figsize = (w_px / _dpi, h_px / _dpi)

    fig = plot_spatial_niche(
        coords=coords,
        niche_labels=niche_labels,
        target_niche=target_niche,
        background_labels=background_labels,
        clean=True,
        figsize=_figsize,
        dpi=_dpi,
        flip=flip,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        show=False,
        bg_alpha=bg_alpha,
        number_niches=number_niches,
        neutral_background=neutral_background, mark_target=mark_target,
        section_bounds=section_bounds, crop_bounds=crop_bounds, title=title,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# --------------------------------------------------------------------------- #
#  Focused image: target niche + adjacent niches only
# --------------------------------------------------------------------------- #


def generate_focused_niche_image_base64(
    coords: np.ndarray,
    niche_labels: np.ndarray,
    target_niche: str,
    adjacent_niches: list[str],
    *,
    background_labels: np.ndarray | None = None,
    image_px: int = 700,
    flip: bool = False,
) -> tuple[str, dict[str, str], dict[str, str]]:
    """Generate a spatial image colouring target + adjacent niches over a
    coarse tissue-organization backdrop.

    The target niche is drawn in red and labelled "T"; each adjacent niche
    gets a distinct non-red colour and a 1-letter code (A, B, C, ...) drawn at
    its centroid. When ``background_labels`` is given, every other cell is
    coloured by its coarse compartment (a muted, non-red palette) and each
    compartment is named on the map — so the model can read gross tissue
    organization (which band is mucosa / muscularis / submucosa / serosa) to
    place the target, instead of a flat-gray void. Without it, other cells fall
    back to faint gray.

    Returns ``(base64_png, code_legend, region_legend)`` where ``code_legend``
    maps adjacent niche → 1-letter code and ``region_legend`` maps coarse
    compartment → hex colour.

    Parameters
    ----------
    coords : ndarray (n_cells, 2)
    niche_labels : ndarray (n_cells,)
    target_niche : str
    adjacent_niches : list[str]
        Niches to colour (everything else becomes gray).
    image_px : int
        Target pixel size (longer axis).
    flip : bool
        Negate y coordinates.

    Returns
    -------
    (base64_png, legend) where legend maps adjacent niche names to 1-letter codes.
    """
    _dpi = 100
    _coords = np.asarray(coords, dtype=float, copy=True)
    niche_labels = np.asarray(niche_labels, dtype=str)

    x_span = max(float(np.ptp(_coords[:, 0])), 1e-6)
    y_span = max(float(np.ptp(_coords[:, 1])), 1e-6)
    aspect = x_span / y_span

    if aspect >= 1.0:
        w_px, h_px = image_px, max(int(image_px / aspect), 1)
    else:
        h_px, w_px = image_px, max(int(image_px * aspect), 1)

    _figsize = (w_px / _dpi, h_px / _dpi)

    target_mask = niche_labels == target_niche
    adj_set = set(adjacent_niches)
    adj_mask = np.array([nl in adj_set for nl in niche_labels])
    other_mask = ~target_mask & ~adj_mask

    # Palette for adjacent niches
    adj_sorted = [n for n in adjacent_niches if n in set(niche_labels)]
    palette = _get_niche_palette(max(len(adj_sorted), 1))
    adj_color = {n: palette[i] for i, n in enumerate(adj_sorted)}

    auto_ps, auto_cs, auto_clw = _auto_sizes(_coords, _figsize)

    fig, ax = plt.subplots(1, 1, figsize=_figsize)

    if flip:
        _coords[:, 1] = -_coords[:, 1]

    # Background cells: coloured by coarse compartment (tissue-organization
    # map) when background_labels is given, else faint gray.
    region_legend: dict[str, str] = {}
    if background_labels is not None and other_mask.any():
        import matplotlib.colors as _mcolors

        def _lighten(c, f=0.5):
            r, g, b = _mcolors.to_rgb(c)
            return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)

        bg = np.asarray(background_labels, dtype=str)
        regions = sorted({r for r in bg[other_mask] if r and r.lower() != "nan"})
        rpal = _get_niche_palette(max(len(regions), 1))  # non-red palette
        rcolor = {r: _lighten(rpal[i]) for i, r in enumerate(regions)}
        for r in regions:
            m = other_mask & (bg == r)
            if not m.any():
                continue
            ax.scatter(
                _coords[m, 0], _coords[m, 1],
                s=auto_ps * 0.6, c=[rcolor[r]], alpha=1.0, rasterized=True,
            )
            _draw_region_code(ax, _coords, m, r, fontsize=12, color=(0.25, 0.25, 0.25))
            region_legend[r] = _mcolors.to_hex(rcolor[r])
    elif other_mask.any():
        ax.scatter(
            _coords[other_mask, 0], _coords[other_mask, 1],
            s=auto_ps * 0.5, c=[(0.8, 0.8, 0.8)], alpha=1.0,
            rasterized=True,
        )

    # Adjacent niches, each assigned a 1-letter code.
    code_map: dict[str, str] = {}
    for i, niche_id in enumerate(adj_sorted):
        m = niche_labels == niche_id
        if not m.any():
            continue
        code = _NEIGHBOR_CODES[i] if i < len(_NEIGHBOR_CODES) else f"N{i}"
        code_map[niche_id] = code
        ax.scatter(
            _coords[m, 0], _coords[m, 1],
            s=auto_ps, c=[adj_color[niche_id]], alpha=1.0,
            rasterized=True,
        )
        _draw_region_code(ax, _coords, m, code)

    # Target niche on top (red), labelled "T".
    if target_mask.any():
        ax.scatter(
            _coords[target_mask, 0], _coords[target_mask, 1],
            s=auto_ps * 1.5, c=[(0.9, 0.1, 0.1)], alpha=1.0,
            rasterized=True,
        )
        _draw_region_code(ax, _coords, target_mask, "T", fontsize=22, color=(0.6, 0.0, 0.0))

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    # Normalized 1..10 coordinate frame so the model can read absolute position.
    _apply_coordinate_frame(ax, _coords)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("ascii")

    # Legend maps each adjacent niche to its on-image 1-letter code (the codes
    # are what the model reads off the image — far more reliable than colours).
    legend = {label: code_map[label] for label in adj_sorted if label in code_map}
    return img_b64, legend, region_legend
