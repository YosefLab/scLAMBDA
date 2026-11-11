"""Coarse spatial-domain construction for hierarchical niche annotation.

The fine niches (the input spatial clustering) are first grouped into a small
number of *coarse* spatial domains, which are annotated first (level 1). Each
fine niche then inherits its coarse parent's annotation as a suggestion at
level 2.

The grouping combines **what merges** (per-niche mean expression) with **whether
a merge is allowed** (spatial adjacency of the niches), via connectivity-
constrained Ward agglomeration. This injects position into the grouping while
keeping expression as the driver, and yields contiguous, anatomically-coherent
coarse domains — robustly, even for deep-brain nuclei the per-niche LLM
annotation confuses.

No ground-truth region/major-region column is ever used as input or label —
the coarse domains come purely from expression + spatial adjacency of the input
niches, and are *named* by the annotation pipeline (level 1). This avoids
leaking any pre-existing coarse labels into the annotation.
"""

from __future__ import annotations

import logging
import re

import numpy as np

logger = logging.getLogger(__name__)


def estimate_n_coarse_niches(
    llm_client,
    organism: str,
    tissue: str,
    n_fine: int,
) -> int:
    """Ask the model how many broad spatial domains to expect in this tissue.

    Returns an integer in [2, n_fine - 1]. Falls back to a heuristic
    (~sqrt of the fine-niche count) if the LLM is unavailable or unparseable.
    """
    fallback = int(np.clip(round(np.sqrt(max(n_fine, 4))), 2, max(n_fine - 1, 2)))
    if llm_client is None:
        return fallback

    prompt = (
        f"You are a tissue anatomist. A spatial transcriptomics section of "
        f"{organism} {tissue} has been clustered into {n_fine} fine spatial "
        f"niches. We want to group these into a smaller number of BROAD, "
        f"anatomically-meaningful spatial domains (the major compartments / "
        f"regions a pathologist would name at a glance, before resolving fine "
        f"substructure).\n\n"
        f"How many such broad spatial domains would you expect to be able to "
        f"distinguish in this tissue? Consider the canonical gross anatomy of "
        f"{organism} {tissue}. Give a single integer that is meaningfully "
        f"smaller than {n_fine} but large enough to separate the principal "
        f"compartments.\n\n"
        f"Respond with ONLY the integer, nothing else."
    )
    try:
        resp = llm_client.query(prompt, power="medium")
        text = getattr(resp, "content", str(resp))
        m = re.search(r"\d+", text)
        if not m:
            return fallback
        k = int(m.group())
        k = int(np.clip(k, 2, max(n_fine - 1, 2)))
        logger.info("Estimated %d coarse spatial domains (fine niches: %d)", k, n_fine)
        return k
    except Exception as e:  # noqa: BLE001
        logger.warning("Coarse-niche estimation failed (%s); using fallback %d", e, fallback)
        return fallback


def _render_numbered_domains(
    coords: np.ndarray,
    cell_domain_num: np.ndarray,
    n_domains: int,
    organism: str,
    tissue: str,
    flip: bool,
    image_px: int,
    label_nums: set | None = None,
    title: str | None = None,
):
    """Whole-section image: each coarse domain a distinct colour, with a NUMBER
    drawn as a black-arrow (white rim) pointing at a real cell of the domain.

    ``label_nums`` restricts which domain numbers get an arrow (all cells are
    still drawn for context) — used in multi-slide mode where each section only
    highlights the domains best represented on it. Returns base64 PNG."""
    import base64
    import io

    import matplotlib

    matplotlib.use("Agg")
    import colorcet as cc
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    c = coords.astype(float).copy()
    if flip:
        c[:, 1] = -c[:, 1]
    pal = cc.glasbey_dark
    dpi = 130
    fig, ax = plt.subplots(figsize=(image_px / dpi, image_px * 0.7 / dpi))
    for n in range(1, n_domains + 1):
        m = cell_domain_num == n
        if not m.any():
            continue
        ax.scatter(c[m, 0], c[m, 1], s=2.0, c=[pal[(n - 1) % len(pal)]],
                   linewidths=0, rasterized=True)

    # Each number is drawn with an ARROW pointing at a real cell of its domain.
    # A domain can be ring-shaped or split across patches, so its centre of mass
    # may fall in empty space or inside another domain — instead we target the
    # cell at the centre of the domain's densest patch, and splay the label
    # outward from the section centre so the arrow clearly disambiguates it.
    from scipy.spatial import cKDTree

    valid = cell_domain_num > 0
    sec_ctr = c[valid].mean(0) if valid.any() else c.mean(0)
    extent = float(np.linalg.norm(c.max(0) - c.min(0)))
    # Small offset only: the digit must stay essentially ON its region (a large
    # outward splay puts the number over a neighbouring region and the model
    # reads the digit's location, not the arrow target). Short arrow disambiguates.
    off = 0.02 * extent if extent > 0 else 1.0
    for n in range(1, n_domains + 1):
        if label_nums is not None and n not in label_nums:
            continue
        mask = cell_domain_num == n
        pts = c[mask]
        if len(pts) == 0:
            continue
        # Target = the MOST INTERIOR cell: the domain cell farthest from any
        # OTHER region. This sits deep in the domain's mass, away from borders —
        # correct even for concave or ring-shaped
        # domains, where the centroid/medoid drifts onto the boundary with a
        # neighbour and the model then reads the digit as that neighbour.
        others = c[valid & ~mask]
        if len(pts) > 3 and len(others) > 0:
            dist, _ = cKDTree(others).query(pts)
            target = pts[int(np.argmax(dist))]
        else:
            cx, cy = pts.mean(0)
            target = pts[int(np.argmin((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2))]
        v = target - sec_ctr
        nrm = float(np.linalg.norm(v))
        dirv = v / nrm if nrm > 1e-9 else np.array([1.0, 0.0])
        txt_xy = target + dirv * off
        digit_color = pal[(n - 1) % len(pal)]  # match the domain's colour
        ann = ax.annotate(
            str(n), xy=(target[0], target[1]), xytext=(txt_xy[0], txt_xy[1]),
            fontsize=18, fontweight="bold", ha="center", va="center",
            color=digit_color, zorder=11,
            arrowprops=dict(arrowstyle="-|>", color="black", lw=2.0,
                            shrinkA=3, shrinkB=1, mutation_scale=16),
        )
        ann.set_path_effects([pe.withStroke(linewidth=4.0, foreground="white")])
        ann.arrow_patch.set_path_effects([pe.withStroke(linewidth=4.0, foreground="white")])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title or f"{tissue}: {n_domains} spatial domains", fontsize=13)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


_COARSE_PROMPT = """You are an expert anatomist. The image shows a {organism} {tissue} partitioned into {n} spatial domains by transcriptomic clustering. Each domain is a distinct colour with a NUMBER at its centroid. First orient the section (anatomical plane, and the main axes — e.g. anterior/posterior, dorsal/ventral).

For EVERY numbered domain (1..{n}), give its likely **major anatomical region**. Reading anatomy off coloured blobs is error-prone and many deep structures look alike, so make each label EXPLICITLY CONDITIONAL, in exactly this form (one line per number):

  N. Most likely <REGION A>; but could be confused with <REGION B> — if <distinguishing cue / neighbour relation>, then it is <REGION C> instead.

Keep regions at the level of **major divisions or large named regions** (for {tissue}, the gross compartments a pathologist names at a glance) — NOT tiny specific sub-structures. **Name each domain by the WHOLE compartment it forms, never by one of its interdigitated sub-parts.** If a domain is a mix of intimately interwoven sub-compartments — e.g. a dominant cell population together with the supporting stroma interleaved with it — name the encompassing **whole** compartment, NOT a single interleaved sub-part; naming it after a sub-part wrongly anchors the downstream fine call on a minority component. **Aim for a SMALL number of distinct regions overall: it is expected and fine for several numbered domains to share the same broad region label.** Do not over-segment or invent a unique fine structure per domain — a finer annotation is done downstream, so these labels should stay broad and few. Use each domain's position, shape, and which domains border it. Output exactly one line per number (1 to {n}), and nothing else."""


_COARSE_PROMPT_MULTI = """You are an expert anatomist. You are shown {k} views of the same {organism} {tissue}, partitioned into spatial domains by transcriptomic clustering. To keep each view readable, every view highlights only a SUBSET of the domains — and some views may be different physical sections, used where a domain is better represented. The domains are NUMBERED CONSISTENTLY across all views: domain N is the same domain wherever it appears, its number drawn in the domain's own colour with a black arrow (white rim) pointing at a real cell of it. Which domains each view highlights:
{image_desc}

First orient the tissue (anatomical plane and main axes). Then for EVERY numbered domain across all views (the full set 1..{n}), give its likely **major anatomical region**, EXPLICITLY CONDITIONAL, in exactly this form (one line per number):

  N. Most likely <REGION A>; but could be confused with <REGION B> — if <distinguishing cue / neighbour relation>, then it is <REGION C> instead.

Keep regions at the level of **major divisions or large named regions** (for {tissue}, the gross compartments a pathologist names at a glance) — NOT tiny specific sub-structures. **Name each domain by the WHOLE compartment it forms, never by one of its interdigitated sub-parts.** If a domain is a mix of intimately interwoven sub-compartments — e.g. a dominant cell population together with the supporting stroma interleaved with it — name the encompassing **whole** compartment, NOT a single interleaved sub-part; naming it after a sub-part wrongly anchors the downstream fine call on a minority component. **Aim for a SMALL number of distinct regions overall: it is expected and fine for several numbered domains to share the same broad region label.** Do not over-segment or invent a unique fine structure per domain — a finer annotation is done downstream, so these labels should stay broad and few. Use each domain's position, shape, and which domains border it. Output exactly one line for every number that appears across the views (1 to {n}), and nothing else."""


def _select_coverage_slides(
    accessor,
    fine_to_group: dict[str, str],
    fine_col: str,
    slice_key: str,
    *,
    max_images: int = 4,
    min_frac: float = 0.4,
) -> list[tuple[str, list[str]]]:
    """Choose a few slides that together cover every coarse domain.

    Greedy weighted set-cover: repeatedly take the slide holding the most cells
    of still-uncovered domains, assigning to it the domains that are abundant
    there (≥ ``min_frac`` of their best-slide count). Leftovers (if the image cap
    is hit) attach to the chosen slide where they are most abundant. Returns
    ``[(slide, [coarse_id, …]), …]`` — each domain on a slide where it peaks.
    """
    import collections

    groups = sorted(set(fine_to_group.values()), key=lambda g: int(g[1:]))
    idx = np.arange(accessor.n_cells)
    fv = accessor.get_obs([fine_col], idx).iloc[:, 0].astype(str).values
    sv = accessor.get_obs([slice_key], idx).iloc[:, 0].astype(str).values
    cell_g = np.array([fine_to_group.get(v, "") for v in fv])
    slides = sorted(set(sv))
    count = {g: collections.Counter() for g in groups}
    for g, s in zip(cell_g, sv):
        if g:
            count[g][s] += 1
    gmax = {g: (max(count[g].values()) if count[g] else 0) for g in groups}
    remaining = {g for g in groups if gmax[g] > 0}

    assignment: list[tuple[str, list[str]]] = []
    while remaining and len(assignment) < max_images:
        best_s = max(slides, key=lambda s: sum(count[g][s] for g in remaining))
        assigned = [g for g in remaining if count[g][best_s] >= min_frac * gmax[g]]
        if not assigned:
            assigned = [g for g in remaining if count[g][best_s] == gmax[g]]
        assignment.append((best_s, assigned))
        remaining -= set(assigned)
    if remaining:  # image cap hit — fold leftovers onto their best chosen slide
        chosen = [s for s, _ in assignment]
        by_slide = {s: gs for s, gs in assignment}
        for g in remaining:
            best = max(chosen, key=lambda s: count[g][s])
            by_slide[best].append(g)
    return assignment


def annotate_coarse_domains(
    llm_client,
    accessor,
    fine_to_group: dict[str, str],
    fine_col: str,
    spatial_key: str,
    organism: str,
    tissue: str,
    *,
    slice_key: str | None = None,
    slice_val: str | None = None,
    flip: bool = False,
    image_px: int = 1600,
    power: str = "strong",
    max_images: int = 4,
    max_per_image: int = 1000,
) -> dict[str, str]:
    """Label every coarse domain in ONE hedged image prompt.

    Renders the whole section with numbered coarse domains and asks the model
    for a conditional ("most likely X; could be Y; if … then Z") major-region
    label per domain — instead of running the full per-niche pipeline, which
    over-specifies and inherits the deep-structure transcriptional degeneracy.

    Returns dict mapping coarse group id ("C0"…) -> hedged label string.
    """
    groups = sorted(set(fine_to_group.values()), key=lambda g: int(g[1:]))
    gid = {g: i + 1 for i, g in enumerate(groups)}  # 1-based domain numbers
    n = len(groups)

    all_idx = np.arange(accessor.n_cells)

    def _render_slice(sidx, label_nums=None, title=None):
        fv = accessor.get_obs([fine_col], sidx).iloc[:, 0].astype(str).values
        co = accessor.get_obsm(spatial_key, sidx)
        if co.ndim > 1 and co.shape[1] > 2:
            co = co[:, :2]
        cn = np.array([gid.get(fine_to_group.get(v), 0) for v in fv])
        return _render_numbered_domains(co, cn, n, organism, tissue, flip, image_px,
                                        label_nums=label_nums, title=title)

    # Assign domains to sections, then split into PANELS of at most
    # ``max_per_image`` domains each (too many labels on one image is unreadable).
    #  - multi-slide: a few sections cover all domains, each shown where it peaks;
    #  - single section: one frame, domains chunked across views.
    # An over-full section is split into several views of the same section.
    # Numbering is consistent across every panel.
    multi = bool(slice_key) and not slice_val
    if multi:
        base = _select_coverage_slides(
            accessor, fine_to_group, fine_col, slice_key, max_images=max_images
        )
        s_all = accessor.get_obs([slice_key], all_idx).iloc[:, 0].astype(str).values
        idx_for = {sld: np.where(s_all == str(sld))[0] for sld, _ in base}
    else:
        idx = all_idx
        if slice_key and slice_val:
            s_obs = accessor.get_obs([slice_key], all_idx)
            idx = np.where(s_obs[slice_key].values.astype(str) == str(slice_val))[0]
        base = [(None, list(groups))]
        idx_for = {None: idx}

    panels: list[tuple] = []  # (slide, [coarse_id …]) each ≤ max_per_image domains
    for sld, cids in base:
        cs = sorted(cids, key=lambda c: gid[c])
        for i in range(0, len(cs), max_per_image):
            panels.append((sld, cs[i:i + max_per_image]))

    img_arg, desc = [], []
    for k, (sld, cids) in enumerate(panels, 1):
        nums = sorted(gid[c] for c in cids)
        img_arg.append(_render_slice(idx_for[sld], set(nums), f"{tissue} — view {k}"))
        desc.append(f"- Image {k}: domains {', '.join(map(str, nums))}")

    if len(img_arg) == 1:
        img_arg = img_arg[0]
        prompt = _COARSE_PROMPT.format(organism=organism, tissue=tissue, n=n)
    else:
        prompt = _COARSE_PROMPT_MULTI.format(
            organism=organism, tissue=tissue, n=n, k=len(img_arg),
            image_desc="\n".join(desc),
        )
    logger.info("Coarse: %d view(s) for %d domains (≤%d per view)",
                1 if isinstance(img_arg, str) else len(img_arg), n, max_per_image)

    try:
        resp = llm_client.query(prompt, power=power, image_base64=img_arg)
        text = getattr(resp, "content", str(resp))
    except Exception as e:  # noqa: BLE001
        logger.warning("Coarse single-prompt annotation failed: %s", e)
        return {g: "" for g in groups}

    # Parse "N. <label>" lines → {coarse_id: label}
    num_to_group = {v: k for k, v in gid.items()}
    labels: dict[str, str] = {g: "" for g in groups}
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)[\.\)]\s*(.+)", line)
        if not m:
            continue
        num = int(m.group(1))
        if num in num_to_group:
            labels[num_to_group[num]] = m.group(2).strip()
    n_found = sum(1 for v in labels.values() if v)
    logger.info("Coarse single-prompt: labelled %d/%d domains", n_found, n)
    return labels


def compute_coarse_groups(
    accessor,
    fine_col: str,
    spatial_key: str,
    n_groups: int,
    *,
    slice_key: str | None = None,
    slice_val: str | None = None,
    k_neighbors: int = 15,
) -> dict[str, str]:
    """Group fine niches into ``n_groups`` contiguous coarse domains.

    Method: per-niche z-scored mean expression (what merges) + niche-level
    spatial adjacency (whether a merge is allowed) → connectivity-constrained
    Ward agglomeration.

    Returns
    -------
    dict mapping each fine-niche label (str) -> coarse group id ("C0", "C1", …).
    Singleton/everything-in-one fallbacks return one group per niche if the
    constrained clustering cannot run.
    """
    import scipy.sparse as sp
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import kneighbors_graph

    n_cells = accessor.n_cells
    all_idx = np.arange(n_cells)

    # Optionally restrict to one slide for the adjacency graph / coordinates.
    if slice_key and slice_val:
        s_obs = accessor.get_obs([slice_key], all_idx)
        mask = s_obs[slice_key].values.astype(str) == str(slice_val)
        idx = np.where(mask)[0]
    else:
        idx = all_idx

    labels_all = accessor.get_obs([fine_col], idx).iloc[:, 0].values.astype(str)
    coords = accessor.get_obsm(spatial_key, idx)
    if coords.ndim > 1 and coords.shape[1] > 2:
        coords = coords[:, :2]

    niches = sorted(set(labels_all))
    n_niche = len(niches)
    if n_niche <= n_groups or n_niche < 3:
        # Nothing to merge — each niche is its own group.
        return {nl: f"C{i}" for i, nl in enumerate(niches)}

    ri = {nl: i for i, nl in enumerate(niches)}
    codes = np.array([ri[nl] for nl in labels_all])

    # --- Per-niche mean expression (normalized), z-scored across niches ---
    M = np.vstack([accessor.get_expression(idx[codes == i]).mean(0) for i in range(n_niche)])
    M = np.asarray(M)
    Z = (M - M.mean(0)) / (M.std(0) + 1e-9)

    # --- Niche-level spatial adjacency from a cell kNN graph ---
    # onehot(cell -> niche).T @ cell_knn @ onehot  counts cross-niche edges.
    # Computed PER SLIDE and summed: different sections have independent
    # coordinate frames, so a single pooled kNN would invent cross-section
    # neighbours. Each niche gets edges from whatever slides it appears on, while
    # the expression means above pool all of its cells.
    oh = sp.csr_matrix(
        (np.ones(len(codes)), (np.arange(len(codes)), codes)),
        shape=(len(codes), n_niche),
    )
    if slice_key:
        slide_of = accessor.get_obs([slice_key], idx).iloc[:, 0].values.astype(str)
        adj = np.zeros((n_niche, n_niche))
        for s in np.unique(slide_of):
            sm = np.where(slide_of == s)[0]
            if len(sm) <= 3:
                continue
            knn = kneighbors_graph(coords[sm], min(k_neighbors, len(sm) - 1), mode="connectivity")
            ohs = oh[sm]
            adj += np.asarray((ohs.T @ knn @ ohs).todense())
    else:
        cell_knn = kneighbors_graph(coords, min(k_neighbors, len(coords) - 1), mode="connectivity")
        adj = np.asarray((oh.T @ cell_knn @ oh).todense())
    np.fill_diagonal(adj, 0.0)
    adj = adj + adj.T
    # Binarize at the mean edge weight to keep only real spatial contacts.
    thresh = adj.sum() / max(adj.size, 1)
    niche_adj = sp.csr_matrix((adj > thresh).astype(int))

    try:
        group = AgglomerativeClustering(
            n_clusters=n_groups,
            connectivity=niche_adj,
            linkage="ward",
        ).fit_predict(Z)
    except Exception as e:  # noqa: BLE001
        logger.warning("Constrained coarse clustering failed (%s); unconstrained Ward", e)
        group = AgglomerativeClustering(n_clusters=n_groups, linkage="ward").fit_predict(Z)

    return {niches[i]: f"C{int(group[i])}" for i in range(n_niche)}
