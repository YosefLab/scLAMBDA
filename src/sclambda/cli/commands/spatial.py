"""`lambdakit spatial …` — niche composition, spatial DE, and rendered maps.

Ports the computation out of sc-lambda's `ComputeSpatialProportionsNode` and
`SpatialCellTypeDENode` into standalone commands. Two design points from the
original are preserved exactly because they change what the model concludes:

* **Composition pools slides; Moran's statistics and images do not.** One
  transcriptional niche can sit in different places on different sections, so
  pooling gives DE its statistical power, while neighbour-based statistics stay
  within a single physical section where "neighbour" means something.
* **Region-level DE comes first.** Per-cell-type DE compares each cell type to
  itself elsewhere, which washes out *composition* — the very thing that defines
  most anatomical regions. The all-cells-pooled contrast is emitted as the first
  section so identity evidence is not buried under within-type state.

The image commands write PNGs to the run directory rather than base64 into a
prompt. Claude reads them with the Read tool, which is why the vision API call
disappears entirely.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from sclambda.storage.session import Session
from sclambda.storage.atomic import write_text

MIN_CELLS = 10


def _niche_indices(acc, cid, slice_key=None, slice_val=None):
    """Niche cells, optionally restricted to one slide."""
    idx = acc.get_cell_indices(cid, depth=1)
    if slice_key and slice_val:
        obs = acc.get_obs([slice_key], np.arange(acc.n_cells))
        mask = obs[slice_key].values.astype(str) == str(slice_val)
        idx = idx[mask[idx]]
    return idx


def _plot_slide(sess, acc, cid):
    """The slide carrying most of this niche — used for images and Moran's Z."""
    slice_key = sess.get("slice_key")
    if not slice_key:
        return None
    fixed = sess.get("slice")
    if fixed:
        return str(fixed)
    idx = acc.get_cell_indices(cid, depth=1)
    obs = acc.get_obs([slice_key], idx)
    vals = obs[slice_key].values.astype(str)
    if len(vals) == 0:
        return None
    uniq, counts = np.unique(vals, return_counts=True)
    return str(uniq[np.argmax(counts)])


def cmd_composition(args) -> int:
    from sclambda.analysis.spatial_metrics import compute_spatial_proportions
    from sclambda.annotation.niche.formatters import format_spatial_proportions

    sess = Session(args.run)
    acc = sess.accessor
    cid = args.niche
    ct_col = sess.get("cell_type_col")
    slice_key = sess.get("slice_key")
    cross_slide = bool(slice_key) and not sess.get("slice")
    plot_slide = _plot_slide(sess, acc, cid)

    spatial_idx = _niche_indices(acc, cid, slice_key, plot_slide)
    prop_idx = acc.get_cell_indices(cid, depth=1) if cross_slide else spatial_idx

    if len(prop_idx) == 0:
        print(f"Niche {cid} has no cells in the requested scope.")
        return 0

    cts = acc.get_obs([ct_col], prop_idx)[ct_col].values.astype(str)

    morans = {}
    if len(spatial_idx) >= MIN_CELLS:
        s_cts = acc.get_obs([ct_col], spatial_idx)[ct_col].values.astype(str)
        coords = acc.get_obsm(sess.get("spatial_key"), spatial_idx)
        if coords.ndim > 1 and coords.shape[1] > 2:
            coords = coords[:, :2]
        res = compute_spatial_proportions(coords=coords, cell_types=s_cts, cluster_id=cid,
                                          k_neighbors=args.k, max_hops=2)
        morans = {m["cell_type"]: m["morans_z"] for m in res.as_dicts()}

    n_total = len(cts)
    uniq, counts = np.unique(cts, return_counts=True)
    order = np.argsort(-counts)
    proportions = [
        {"cell_type": str(ct), "n_cells": int(c), "proportion": round(float(c) / n_total, 4),
         "morans_z": morans.get(str(ct))}
        for ct, c in zip(uniq[order], counts[order])
    ]

    d = Path(sess.run) / "composition"
    d.mkdir(parents=True, exist_ok=True)
    write_text(d / f"{cid}.json", json.dumps(
        {"cluster_id": cid, "n_cells": n_total, "plot_slide": plot_slide,
         "cross_slide": cross_slide, "proportions": proportions}, indent=2))

    from sclambda.annotation.niche.measurements import composition_record, persist, render, distribution_record
    from sclambda.annotation.niche.populations import population_context
    context = {"slice_key": slice_key, "expression_slides": "all" if cross_slide else plot_slide,
               "spatial_slide": plot_slide, "cell_type_column": ct_col,
               **population_context(sess.get("cell_type_col_generated", False))}
    value = composition_record(cid, proportions, context)
    distribution = distribution_record(acc, cid, {"slice_key": slice_key})
    persist(sess, cid, [value] + ([distribution] if distribution else []))
    print(render(value))
    if distribution:
        print(render(distribution))
    print(f"# Niche {cid} — cell-type composition ({n_total:,} cells"
          f"{' pooled across slides' if cross_slide else ''})\n")
    print(format_spatial_proportions(proportions))
    print()
    print("Moran Z is autocorrelation relative to a local permutation null on the stated slide. "
          "Low Z does not establish uniformity or representativeness; unavailable is not zero.")
    if plot_slide:
        print(f"\nSpatial statistics and images use slide `{plot_slide}` (most niche cells).")
    return 0


def cmd_de(args) -> int:
    from sclambda.analysis.targeted_de import compute_pathology_de_from_indices
    from sclambda.analysis.spatial_metrics import compute_gene_spatial_scores
    from sclambda.annotation.niche.formatters import format_spatial_celltype_de

    sess = Session(args.run)
    acc = sess.accessor
    cid = args.niche
    ct_col = sess.get("cell_type_col")
    slice_key = sess.get("slice_key")
    cross_slide = bool(slice_key) and not sess.get("slice")
    plot_slide = _plot_slide(sess, acc, cid)
    n_top = args.n or sess.get("n_top_genes", 20)

    # DE pools slides (power); Moran's Z stays on the plotted slide (meaning).
    niche_idx = acc.get_cell_indices(cid, depth=1)
    if slice_key and sess.get("slice") and not cross_slide:
        niche_idx = _niche_indices(acc, cid, slice_key, sess.get("slice"))

    all_idx = np.arange(acc.n_cells)
    obs = acc.get_obs([ct_col] + ([slice_key] if slice_key else []), all_idx)
    if slice_key and sess.get("slice") and not cross_slide:
        slide_mask = obs[slice_key].values.astype(str) == str(sess.get("slice"))
    else:
        slide_mask = np.ones(acc.n_cells, dtype=bool)
    rest_mask = np.ones(acc.n_cells, dtype=bool)
    rest_mask[niche_idx] = False
    rest_mask &= slide_mask

    spatial_idx = _niche_indices(acc, cid, slice_key, plot_slide)
    coords = acc.get_obsm(sess.get("spatial_key"), spatial_idx)
    if coords.ndim > 1 and coords.shape[1] > 2:
        coords = coords[:, :2]

    ct_values = obs[ct_col].values.astype(str)

    # --- region-level DE first: it carries the composition-defined identity ---
    tables, scores = {}, {}
    region_key = "Niche overall — all cell types pooled (vs rest of tissue)"
    rest_idx = np.where(rest_mask)[0]
    if len(niche_idx) >= MIN_CELLS and len(rest_idx) >= MIN_CELLS:
        tables[region_key] = compute_pathology_de_from_indices(
            accessor=acc, target_indices=np.unique(niche_idx),
            reference_indices=rest_idx, cell_type_label=region_key, n_top_genes=n_top)
        scores[region_key] = {}

    # --- per-cell-type DE: this type here vs the same type elsewhere ---
    comp_file = Path(sess.run) / "composition" / f"{cid}.json"
    if comp_file.exists():
        proportions = json.loads(comp_file.read_text())["proportions"]
    else:
        print("error: run `lambdakit spatial composition` first.", file=sys.stderr)
        return 2

    wanted = [args.cell_type] if args.cell_type else None
    selected = []
    for p in proportions:
        ct = p["cell_type"]
        if wanted and ct not in wanted:
            continue
        n_rest = int(((ct_values == ct) & rest_mask).sum())
        if p["n_cells"] >= MIN_CELLS and n_rest >= MIN_CELLS:
            selected.append(ct)
    selected = selected[: (args.k if not wanted else len(selected))]

    for ct in selected:
        ct_mask = ct_values == ct
        ct_idx = np.where(ct_mask)[0]
        target = ct_idx[np.isin(ct_idx, niche_idx)]
        reference = np.where(ct_mask & rest_mask)[0]
        if len(target) < MIN_CELLS or len(reference) < MIN_CELLS:
            continue
        de = compute_pathology_de_from_indices(
            accessor=acc, target_indices=target, reference_indices=reference,
            cell_type_label=ct, n_top_genes=n_top)
        if not (de.enriched_genes or de.depleted_genes):
            continue
        tables[ct] = de
        scores[ct] = {}

        top = [g.name for g in de.enriched_genes[:n_top]] + [g.name for g in de.depleted_genes[:n_top]]
        s_idx = target[np.isin(target, spatial_idx)]
        if top and len(s_idx) >= 3:
            order = np.argsort(spatial_idx)
            local = order[np.searchsorted(spatial_idx, s_idx, sorter=order)]
            X, found = acc.get_expression_for_genes(s_idx, top)
            if len(found) > 0:
                sr = compute_gene_spatial_scores(
                    coords=coords[local], expression_matrix=X, gene_names=found,
                    cluster_id=cid, k_neighbors=min(args.k_neighbors, len(s_idx) - 1), max_hops=2)
                scores[ct] = sr.as_dict()

    if not tables:
        print(f"No cell type in niche {cid} has enough cells on both sides for DE.")
        return 0

    from sclambda.annotation.niche.measurements import expression_record, persist, render
    from sclambda.annotation.niche.populations import population_context
    context = {"slice_key": slice_key, "expression_slides": "all" if cross_slide else plot_slide,
               "spatial_slide": plot_slide, "cell_type_column": ct_col,
               **population_context(sess.get("cell_type_col_generated", False))}
    records = [expression_record(cid, result, context, None if ct == region_key else ct)
               for ct, result in tables.items()]
    from sclambda.annotation.niche.measurements import spatial_score_record
    records.extend(spatial_score_record(cid, ct, values, plot_slide) for ct, values in scores.items() if values)
    persist(sess, cid, records)
    print("\n\n".join(render(value) for value in records))
    print("Pooled DE combines composition and expression differences. Within-type DE can reflect state or spatial specialization. "
          "Interpret each contrast with its stated reference; top-list omission is not absence.")
    return 0


def cmd_image(args) -> int:
    """Render a numbered niche map to PNG. Read it with the Read tool."""
    import matplotlib
    matplotlib.use("Agg")
    from sclambda.analysis.spatial_visualization import plot_spatial_niche

    sess = Session(args.run)
    acc = sess.accessor
    # A sentinel that matches no niche renders the section with nothing in red —
    # the whole-tissue overview the consistency review reads.
    cid = "\x00__no_target__" if args.no_target else args.niche
    if not args.no_target and not args.niche:
        print("error: pass --niche ID, or --no-target for a whole-section overview", file=sys.stderr)
        return 2
    slice_key = sess.get("slice_key")
    slide = args.slide or (None if args.no_target else _plot_slide(sess, acc, cid))
    if args.no_target and slice_key and not slide:
        obs_all = acc.get_obs([slice_key], np.arange(acc.n_cells))
        vals = obs_all[slice_key].values.astype(str)
        uniq, counts = np.unique(vals, return_counts=True)
        slide = str(uniq[np.argmax(counts)])

    all_idx = np.arange(acc.n_cells)
    if slice_key and slide:
        obs = acc.get_obs([slice_key], all_idx)
        vis_idx = np.where(obs[slice_key].values.astype(str) == str(slide))[0]
    else:
        vis_idx = all_idx

    coords = acc.get_obsm(sess.get("spatial_key"), vis_idx)
    if coords.ndim > 1 and coords.shape[1] > 2:
        coords = coords[:, :2]

    section_bounds = (float(coords[:, 0].min()), float(coords[:, 0].max()),
                      float(coords[:, 1].min()), float(coords[:, 1].max()))
    l1 = acc._level_column(1)
    labels = acc._overlay.iloc[vis_idx][l1].values.astype(str)

    background = None
    if args.color_by and args.color_by in acc.obs.columns:
        background = acc.get_obs([args.color_by], vis_idx)[args.color_by].values.astype(str)

    if args.mode == "centered":
        tmask = labels == cid
        if tmask.sum() >= 5:
            tc = coords[tmask]
            ctr = (tc.min(0) + tc.max(0)) / 2
            span = coords.max(0) - coords.min(0)
            # 1.5x the target's own bounding box, so the niche fills roughly a
            # third of the frame and its surroundings stay visible. A tight crop
            # makes the neighbourhood look like the whole section.
            win = np.maximum((tc.max(0) - tc.min(0)) * 1.5, span * 0.12)
            m = (np.abs(coords[:, 0] - ctr[0]) <= win[0]) & (np.abs(coords[:, 1] - ctr[1]) <= win[1])
            if m.sum() >= 10:
                coords, labels = coords[m], labels[m]
                background = background[m] if background is not None else None

    stem = "overview" if args.no_target else f"{cid}_{args.mode}"
    out = Path(args.out) if args.out else (
        sess.image_dir() / f"{stem}{'_' + str(slide) if slide else ''}.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Figure aspect follows the data so no pixels are spent on empty space;
    # the longer axis gets --px. Above ~1600 px is wasted detail.
    x_span = max(float(np.ptp(coords[:, 0])), 1e-6)
    y_span = max(float(np.ptp(coords[:, 1])), 1e-6)
    aspect = x_span / y_span
    dpi = 100
    if aspect >= 1.0:
        w_px, h_px = args.px, max(int(args.px / aspect), 1)
    else:
        h_px, w_px = args.px, max(int(args.px * aspect), 1)

    fig = plot_spatial_niche(
        coords=coords, niche_labels=labels, target_niche=cid,
        background_labels=background, flip=bool(sess.get("flip_spatial_y", False)),
        clean=True, number_niches=bool(args.number), bg_alpha=0.5,
        neutral_background=not args.no_target and not args.color_by, mark_target=False,
        section_bounds=section_bounds,
        title=f"{'Crop' if args.mode == 'centered' else 'Whole section'} | "
              f"{'y reflected by configuration' if sess.get('flip_spatial_y', False) else 'coordinates as stored'}\n"
              "Plot directions are not anatomical directions",
        figsize=(w_px / dpi, h_px / dpi), dpi=dpi,
        save_path=out, show=False,
    )
    import matplotlib.pyplot as plt
    plt.close(fig)

    import hashlib
    metadata = {"target": None if args.no_target else cid, "slide": slide,
                "flip_y": bool(sess.get("flip_spatial_y", False)),
                "section_bounds": section_bounds,
                "slice_key": slice_key, "mode": args.mode, "path": str(out.resolve()),
                "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    write_text(out.with_suffix(out.suffix + ".json"), json.dumps(metadata))
    print(f"Wrote `{out}`")
    print()
    if args.no_target:
        print("Whole-section overview with no niche highlighted"
              + (f", coloured by `{args.color_by}`" if args.color_by else "") + ".")
        print(f"Slide: `{slide}`." if slide else "")
        print()
        print("Read this image to judge the tissue's gross architecture — which compartments "
              "are layered, which sit at an extreme, and which are enclosed. Flag a niche only "
              "when its label is contradicted by where it sits, never by markers.")
        return 0
    print(f"Target niche **{cid}** is drawn in red, larger than the other cells, with a "
          f"target mask retaining all locations. Other niches are the dimmed "
          f"background. Both axes use the same section origin and scale in whole and cropped views; "
          f"these are not anatomical directions.")
    if slide:
        print(f"Slide: `{slide}`.")
    print()
    print("Read this image now. Use it to establish **where** the niche sits — lock the frame on "
          "landmarks at opposite poles of the section before placing the target, and resolve "
          "in/out at every border. Do not read identity off position.")
    return 0


def cmd_distribution(args) -> int:
    """Where a niche sits across slides — the `niche_distribution` tool, no image."""
    sess = Session(args.run)
    acc = sess.accessor
    cid = args.niche
    slice_key = sess.get("slice_key")
    if not slice_key:
        print("Single-slide dataset — no cross-slide distribution to report.")
        return 0

    spatial_key = sess.get("spatial_key")
    all_idx = np.arange(acc.n_cells)
    cols = [slice_key] + ([args.condition_col] if args.condition_col else [])
    obs = acc.get_obs(cols, all_idx)
    slides = obs[slice_key].values.astype(str)
    coords_all = acc.get_obsm(spatial_key, all_idx)
    if coords_all.ndim > 1 and coords_all.shape[1] > 2:
        coords_all = coords_all[:, :2]

    niche_mask = np.zeros(acc.n_cells, dtype=bool)
    niche_mask[acc.get_cell_indices(cid, depth=1)] = True

    rows = []
    for slide in np.unique(slides):
        s = slides == slide
        n = int((s & niche_mask).sum())
        if n < 15:
            continue
        sec = coords_all[s]
        nic = coords_all[s & niche_mask]
        centre = sec.mean(0)
        radius = np.linalg.norm(sec - centre, axis=1).max() or 1.0
        nc = nic.mean(0)
        radial = float(np.linalg.norm(nc - centre) / radius)
        spread = float(np.linalg.norm(nic - nc, axis=1).mean() / radius)
        cond = ""
        if args.condition_col:
            vals = obs[args.condition_col].values.astype(str)[s]
            cond = vals[0] if len(vals) else ""
        rows.append((str(slide), n, float(n) / max(1, int(s.sum())), radial, spread, cond))

    if not rows:
        print(f"Niche {cid} has fewer than 15 cells on every slide.")
        return 0

    rows.sort(key=lambda r: -r[1])
    rows = rows[:16]
    lines = [f"# Niche {cid} across slides", ""]
    head = "| slide | n cells | % of slide | radial (0=centre, 1=edge) | spread |"
    sep = "|---|---:|---:|---:|---:|"
    if args.condition_col:
        head = head[:-1] + " condition |"
        sep = sep[:-1] + "---|"
    lines += [head, sep]
    for slide, n, frac, radial, spread, cond in rows:
        line = f"| {slide} | {n:,} | {frac*100:.1f}% | {radial:.2f} | {spread:.2f} |"
        if args.condition_col:
            line = line[:-1] + f" {cond} |"
        lines.append(line)
    lines += ["", "**radial** is the distance from the niche's centroid to the section centre, as a "
                  "fraction of the section radius. **spread** is how dispersed the niche's own cells are. "
                  "A niche that is central on one slide and peripheral on another is genuinely "
                  "position-variable — say so rather than picking one placement."]
    print("\n".join(lines))
    return 0


def cmd_adjacency(args) -> int:
    """Which niches border which — the consistency review's neighbour roster.

    Computed per plotted slide and never pooled: neighbours across physical
    sections do not exist, and pooling would invent them.
    """
    from sclambda.analysis.spatial_metrics import compute_niche_adjacency

    sess = Session(args.run)
    acc = sess.accessor
    slice_key = sess.get("slice_key")
    slide = args.slide or sess.get("slice")

    all_idx = np.arange(acc.n_cells)
    if slice_key and slide:
        obs = acc.get_obs([slice_key], all_idx)
        idx = np.where(obs[slice_key].values.astype(str) == str(slide))[0]
    elif slice_key:
        # No slide named: use the one with the most cells rather than pooling.
        obs = acc.get_obs([slice_key], all_idx)
        vals = obs[slice_key].values.astype(str)
        uniq, counts = np.unique(vals, return_counts=True)
        slide = str(uniq[np.argmax(counts)])
        idx = np.where(vals == slide)[0]
    else:
        idx = all_idx

    coords = acc.get_obsm(sess.get("spatial_key"), idx)
    if coords.ndim > 1 and coords.shape[1] > 2:
        coords = coords[:, :2]
    labels = acc._overlay.iloc[idx][acc._level_column(1)].values.astype(str)

    adjacency = compute_niche_adjacency(coords, labels, k=args.k)
    results = sess.current_results()

    lines = [f"# Niche adjacency{f' — slide {slide}' if slide else ''}", ""]
    lines.append("| niche | label | borders (most-shared first) |")
    lines.append("|---|---|---|")
    for niche in sorted(adjacency, key=lambda s: (len(s), s)):
        if niche in ("nan", ""):
            continue
        label = (results.get(niche) or {}).get("label", "—")
        neighbours = [n for n in adjacency[niche] if n not in ("nan", "")][:8]
        pretty = ", ".join(f"{n} ({(results.get(n) or {}).get('label', '—')})" for n in neighbours)
        lines.append(f"| {niche} | {label} | {pretty or '—'} |")
    lines += ["", "Adjacency is computed within one physical section only. Neighbor labels are "
                  "revisable predictions, and missing labels may reflect incomplete annotation. "
                  "Inspect the shared boundary and target geometry before interpreting an "
                  "unexpected neighbor; a single contact does not establish or exclude anatomy."]
    print("\n".join(lines))
    return 0


def cmd_ledger(args) -> int:
    """Composition ledger for every niche — the dominance gate's input."""
    sess = Session(args.run)
    results = sess.current_results()
    comp_dir = Path(sess.run) / "composition"
    if not comp_dir.exists():
        print("error: no composition computed yet.", file=sys.stderr)
        return 2

    lines = ["# Composition ledger — every annotated niche", ""]
    flagged = 0
    for f in sorted(comp_dir.glob("*.json")):
        data = json.loads(f.read_text())
        cid = data["cluster_id"]
        label = (results.get(cid) or {}).get("label", "(unlabelled)")
        props = data["proportions"][:8]
        lines.append(f"## niche {cid} — {label}")
        lines.append(", ".join(f"{p['cell_type']} {p['proportion']*100:.1f}%" for p in props))
        lines.append("")

    lines.append("---")
    lines.append("Composition constrains the label but does not dictate anatomy by a universal fraction. "
                 "Compare dominant populations, gene measurements and spatial organization together.")
    print("\n".join(lines))
    return 0


def cmd_genes(args):
    """Targeted panels, optionally restricted to a supplied cell type."""
    from sclambda.analysis.targeted_de import compute_de_for_genes
    from sclambda.annotation.niche.populations import niche_and_rest_indices
    from sclambda.annotation.niche.measurements import expression_record, persist, render
    sess = Session(args.run)
    acc = sess.accessor
    cid = args.niche
    cross_slide = bool(sess.get("slice_key")) and not sess.get("slice")
    target, obs, rest = niche_and_rest_indices(acc, cid, sess.get("cell_type_col"),
        slice_key=sess.get("slice_key"), slice_val=sess.get("slice"), cross_slide=cross_slide)
    if args.cell_type:
        mask = obs[sess.get("cell_type_col")].astype(str).values == args.cell_type
        target, rest = target[mask[target]], rest & mask
    reference = np.where(rest)[0]
    requested = [g.strip() for g in args.genes.split(",") if g.strip()]
    from sclambda.analysis.gene_names import resolve_genes
    from sclambda.annotation.niche.measurements import identify
    resolved = resolve_genes(requested, acc.gene_names, organism=sess.get("organism", "human"),
        alias_path=getattr(args, "alias_file", None), cache_dir=getattr(args, "evidence_cache", None))
    valid = resolved["genes"]
    naming = identify({"kind": "gene_resolution", "target": cid, "scope": {"cell_type": args.cell_type},
        "source": resolved["version"], "source_error": resolved["source_error"],
        "rows": [{"item": r["requested"], "text": json.dumps(r), **r} for r in resolved["rows"]]})
    d = persist(sess, cid, [naming])
    d.requested_genes = sorted(set(d.requested_genes) | set(requested))
    missing = [r for r in resolved["rows"] if r["feature"] is None]
    if missing:
        note = f"Requested signature in {args.cell_type or 'whole niche'}: " + json.dumps(missing) + "; unresolved names are not expression zeros."
        if not any(e.get("fact") == note for e in d.evidence_log):
            d.add_evidence("tool_status", note)
    sess.write_dossier_dict(cid, d.to_dict())
    print(render(naming))
    if not valid or len(target) < MIN_CELLS or len(reference) < MIN_CELLS:
        print(f"Comparison unavailable: n_target={len(target)}, n_reference={len(reference)}; not evidence of absence.")
        return 0
    result = compute_de_for_genes(acc, valid, target, reference, de_method=sess.get("de_method", "ttest"))
    from sclambda.annotation.niche.populations import population_context
    context = {"slice_key": sess.get("slice_key"), "expression_slides": "all" if cross_slide else sess.get("slice"),
               "cell_type_column": sess.get("cell_type_col"),
               **population_context(sess.get("cell_type_col_generated", False))}
    value = expression_record(cid, result, context, args.cell_type, sess.get("de_method", "ttest"))
    persist(sess, cid, [value])
    print(render(value))
    return 0


def cmd_observe(args):
    """Record a revisable observation linked to a tool-generated image."""
    import base64
    import hashlib
    from sclambda.annotation.niche.measurements import observation_record, persist, render
    sess = Session(args.run)
    path = Path(args.image)
    metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    if metadata.get("target") not in (None, args.niche):
        print("error: image highlights a different niche", file=sys.stderr)
        return 2
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        print("error: image changed since rendering", file=sys.stderr)
        return 2
    value = observation_record(args.niche, args.text,
        {"slice_key": metadata.get("slice_key"), "spatial_slide": metadata.get("slide")}, views=[metadata])
    value["images"] = [base64.b64encode(path.read_bytes()).decode()]
    persist(sess, args.niche, [value])
    print(render(value))
    return 0


def cmd_reference(args):
    """Record attributed passages for a specific biological discriminator."""
    from sclambda.annotation.niche.references import lookup_reference
    from sclambda.annotation.niche.measurements import persist, render
    sess = Session(args.run)
    value = lookup_reference(args.niche, args.query, organism=sess.get("organism", "human"),
        tissue=sess.get("tissue", "unknown"), terms=[s.strip() for s in (args.terms or '').split(',') if s.strip()],
        cache_dir=args.evidence_cache)
    persist(sess, args.niche, [value])
    print(render(value))
    if not value['rows']:
        print("No source passages retrieved; this does not refute the claim.")
    return 0
