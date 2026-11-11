"""`lambdakit init` / `clusters` / `catalog` / `cache`.

``init`` is the expensive step: it opens the h5ad, validates that gene names are
symbols, decides whether the matrix needs CP10k+log1p, runs the one full pass
that produces reference statistics, and encodes cluster names.

The expensive results are a pure function of the ``.h5ad``, so they go to a
dataset-scoped cache rather than the run directory (see :mod:`sclambda.core.cache`).
A second run over the same file reuses them and ``init`` returns immediately.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sclambda.core import cache as _cache
from sclambda.storage.session import Session, SessionError, default_run_dir


def _natural_key(name: str):
    try:
        return (0, int(name))
    except ValueError:
        return (1, str(name))


def encode_cluster_names(names: list[str]) -> tuple[list[str], dict[str, str]]:
    """Map arbitrary cluster names to "0","1",...,"N-1" in natural-sort order.

    This is sc-lambda's deliberate blinding step: the model must never see a
    ground-truth label like "CD8 T cell" as a cluster *name*, or benchmarking
    against an annotated column would be circular. The inverse map is stored
    and re-applied only at export time.
    """
    ordered = sorted(names, key=_natural_key)
    expected = [str(i) for i in range(len(ordered))]
    if ordered == expected:
        return expected, {}
    return expected, {name: str(i) for i, name in enumerate(ordered)}


def cmd_init(args) -> int:
    run = Path(args.run) if args.run else default_run_dir(args.h5ad)
    run.mkdir(parents=True, exist_ok=True)
    sess = Session(run)

    manifest = {
        "h5ad": str(Path(args.h5ad).resolve()),
        "organism": args.organism,
        "tissue": args.tissue,
        "gene_col": args.gene_col,
        "embedding_key": args.embedding_key,
        "obs_column": args.obs_column,
        "cell_type_col": args.cell_type_col,
        "spatial_key": args.spatial_key,
        "slice_key": args.slice_key,
        "slice": args.slice,
        "n_top_genes": args.n_top_genes,
        "de_method": args.de_method,
        "max_depth": args.depth,
        "min_cells": args.min_cells,
        "mode": "spatial" if args.spatial_key else "celltype",
        # Record the *resolved* root, not the flag: a run must keep finding the
        # cache it was built against even if the environment later changes.
        "cache_root": str(_cache.resolve_root(args.cache_dir)),
        "cluster_name_map": {},
        # Both exist for `lambdakit usage`, which finds the harness transcripts
        # by cwd and bounds the orchestrator's share by the run's time window.
        # Neither is recoverable afterwards: the run directory can be read from
        # anywhere, and the manifest's mtime only approximates the start.
        "cwd": str(Path.cwd()),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest["dataset_fingerprint"] = _cache.fingerprint(manifest["h5ad"])
    if (run / "manifest.json").exists():
        existing = sess.manifest
        ignore = {"cwd", "created_at", "cluster_name_map", "dataset_fingerprint"}
        changed = [key for key, value in manifest.items()
                   if key not in ignore and
                   (None if key == "cell_type_col" and existing.get("cell_type_col_generated")
                    else existing.get(key)) != value]
        if changed:
            raise SessionError("Run already exists with different parameters: " + ", ".join(changed)
                               + ". Use a new --run directory.")
        sess.validate_source()
        if sess._overlay_path() is not None:
            sess.read_overlay()
            print(f"Run already initialized at {run}; state preserved. Use `lambdakit status --run {run}`.")
            return 0
        if sess.all_results() or sess.all_dossiers():
            raise SessionError("Run has annotations but no overlay; restore its checkpoint before continuing")
    sess.save_manifest(manifest)

    # Sample the cache state *before* touching the accessor: the checks below
    # (normalization detection in particular) read the matrix and would build
    # the cache, making a cold start look warm.
    import time

    cache_dir = sess.dataset_cache
    had_matrix = (cache_dir / "csr_data.npy").exists()
    had_stats = sess.refstats_cached()
    t0 = time.time()

    acc = sess.accessor
    out = [f"# Run initialised at `{run}`", ""]
    out.append(f"- dataset: `{manifest['h5ad']}`")
    out.append(f"- cells: {acc.n_cells:,}   genes: {acc.n_genes:,}")
    out.append(f"- organism / tissue: {args.organism} / {args.tissue}")

    # --- gene symbols -------------------------------------------------
    genes = acc.gene_names
    symbolic = sum(1 for g in genes[:2000] if re.match(r"^[A-Za-z][A-Za-z0-9\-\.@_]*$", str(g)) and not str(g).startswith("ENS"))
    frac = symbolic / max(1, min(2000, len(genes)))
    out.append(f"- gene names look like symbols: {frac:.0%} " + ("OK" if frac >= 0.5 else "**PROBLEM**"))
    if frac < 0.5:
        out.append("")
        out.append("  **Gene names are not symbols.** The reasoning steps read gene names "
                   "(CD8A, FOXP3); Ensembl IDs cannot be interpreted. Re-run with "
                   "`--gene-col <var column holding symbols>`.")

    # --- normalization ------------------------------------------------
    out.append(f"- normalization: {'raw counts -> CP10k+log1p applied internally' if acc.needs_normalization else 'already normalized, used as-is'}")

    # --- embedding ----------------------------------------------------
    if args.embedding_key and args.embedding_key not in acc.obsm_keys:
        out.append(f"- **embedding `{args.embedding_key}` not found**; available: {list(acc.obsm_keys)}")
    elif args.embedding_key:
        out.append(f"- embedding: `{args.embedding_key}` OK (used for subclustering and the substructure probe)")

    # --- clusters -----------------------------------------------------
    if args.obs_column:
        if args.obs_column not in acc.obs.columns:
            print(f"error: obs column '{args.obs_column}' not found. Available: {list(acc.obs.columns)[:40]}", file=sys.stderr)
            return 2
        raw = acc.obs[args.obs_column].astype(str)
        keep = ~acc.obs[args.obs_column].isna()
        names = sorted(set(raw[keep].values))
        encoded, name_map = encode_cluster_names([str(n) for n in names])
        manifest["cluster_name_map"] = name_map
        labels = raw.values.copy()
        if name_map:
            labels = np.array([name_map.get(str(v), str(v)) for v in labels])
        labels[~keep.values] = "nan"
        acc.set_cluster_labels("", labels)
        out.append(f"- clusters: {len(encoded)} from obs column `{args.obs_column}` "
                   f"(names encoded to 0..{len(encoded)-1}; original names hidden from reasoning)")
    elif args.use_leiden:
        from sclambda.analysis.subcluster import subcluster
        # subcluster() writes the labels into the overlay itself.
        res = subcluster(acc, cluster_id="", method="leiden", n_neighbors=args.n_neighbors)
        res_str = f", resolution {res.resolution:.2f}" if res.resolution else ""
        out.append(f"- clusters: {res.n_clusters} from de-novo Leiden{res_str}")
    else:
        print("error: pass --obs-column <col> or --use-leiden", file=sys.stderr)
        return 2

    # --- spatial checks ----------------------------------------------
    if args.spatial_key:
        if args.spatial_key not in acc.obsm_keys:
            print(f"error: spatial key '{args.spatial_key}' not in obsm. Available: {list(acc.obsm_keys)}", file=sys.stderr)
            return 2
        out.append(f"- spatial: coords `{args.spatial_key}`, population column `{manifest['cell_type_col']}`")
        if manifest.get("cell_type_col_generated"):
            from sclambda.annotation.niche.populations import SPOT_NOTE
            out.append(f"- {SPOT_NOTE}")
        if args.slice_key:
            n_slides = acc.obs[args.slice_key].nunique()
            mode = f"single slide `{args.slice}`" if args.slice else f"cross-slide over {n_slides} slides"
            out.append(f"- slides: `{args.slice_key}` — {mode}")

    # --- dataset-scoped cache ----------------------------------------
    # The matrix copy and the reference statistics depend only on the .h5ad, so
    # they live outside the run directory and are shared by every run over this
    # file. A second analysis of the same dataset pays none of this again.
    _ = acc.ref_mean  # triggers the full chunked pass when not already cached
    # Warm the memory-mapped copy here rather than leaving it to the first DE
    # call: it is the largest fixed cost, and it belongs in the step the user is
    # already waiting on.
    acc.get_expression(np.arange(min(64, acc.n_cells)))
    if not had_stats:
        sess.save_refstats(acc)
    from sclambda.core import column_cache
    had_columns = column_cache.is_built(acc._reader)
    if not args.no_column_cache:
        acc.prepare_column_cache()
    elapsed = time.time() - t0

    sess.save_overlay(acc)
    sess.save_genes(genes)
    sess.save_manifest(manifest)

    new_columns = not had_columns and acc._reader._column_matrix is not None
    if had_matrix and had_stats and not new_columns:
        out.append(f"- dataset cache **reused** ({REF_NOTE}) — `{cache_dir}`")
    else:
        built = " + ".join(
            ([] if had_matrix else ["matrix copy"]) + ([] if had_stats else ["reference statistics"])
            + (["gene-column copy"] if new_columns else [])
        )
        out.append(f"- dataset cache: built {built} in {elapsed:.0f}s — `{cache_dir}`")
        out.append(f"  Reused by every later command and by any other run over this file.")

    out.append("")
    out.append(f"Gene panel written to `{run}/genes.txt` ({len(genes):,} measured genes). "
               "Marker requests must come from this list.")
    out.append("")
    out.append("Next: `lambdakit clusters --run " + str(run) + "`")
    print("\n".join(out))
    return 0


REF_NOTE = "per-gene mean/variance and per-cell totals; later commands reuse it"


def cmd_clusters(args) -> int:
    sess = Session(args.run)
    overlay = sess.read_overlay()
    results = sess.current_results()
    dossiers = sess.all_dossiers()

    depth = args.depth
    col = f"L{depth}"
    if col not in overlay.columns:
        avail = [c for c in overlay.columns if re.fullmatch(r"L\d+", c)]
        print(f"error: no clusters at depth {depth}; available levels: {avail}", file=sys.stderr)
        return 2

    counts = overlay[col].value_counts()
    counts = counts[[i for i in counts.index if str(i) not in ("nan", "None", "")]]

    name_map = sess.get("cluster_name_map") or {}
    inv = {v: k for k, v in name_map.items()}

    rows = []
    for cid, n in counts.sort_index(key=lambda s: s.map(_natural_key)).items():
        cid = str(cid)
        res = results.get(cid)
        dos = dossiers.get(cid)
        if res and res.get("label"):
            status = "done"
        elif dos:
            status = f"in progress (step {dos.get('iteration', 0)})"
        else:
            status = "pending"
        rows.append((cid, int(n), status, (res or {}).get("label", "—"), inv.get(cid, "")))

    lines = [f"# Clusters at depth {depth} — {len(rows)} total", ""]
    header = "| cluster_id | n_cells | status | label |"
    sep = "|---|---:|---|---|"
    if any(r[4] for r in rows) and args.show_original:
        header = "| cluster_id | n_cells | status | label | original name |"
        sep = "|---|---:|---|---|---|"
    lines += [header, sep]
    for cid, n, status, label, orig in rows:
        if args.show_original and any(r[4] for r in rows):
            lines.append(f"| {cid} | {n:,} | {status} | {label} | {orig} |")
        else:
            lines.append(f"| {cid} | {n:,} | {status} | {label} |")

    pending = [r[0] for r in rows if r[2] == "pending"]
    lines += ["", f"pending: {len(pending)}  |  done: {sum(1 for r in rows if r[2] == 'done')}"]
    if pending:
        lines.append("")
        lines.append("pending ids: " + " ".join(pending))
    print("\n".join(lines))
    return 0


def cmd_catalog_set(args) -> int:
    sess = Session(args.run)
    payload = _read_json_arg(args.json, args.file)
    if not isinstance(payload, list):
        print("error: catalog must be a JSON array", file=sys.stderr)
        return 2
    sess.save_catalog(payload)
    print(_render_catalog(sess, payload))
    return 0


def cmd_catalog_show(args) -> int:
    sess = Session(args.run)
    entries = sess.read_catalog()
    if not entries:
        print("No catalog yet. Write one with `lambdakit catalog set --json '[...]'`.")
        return 0
    print(_render_catalog(sess, entries))
    return 0


def _render_catalog(sess: Session, entries: list[dict]) -> str:
    mode = sess.get("mode", "celltype")
    if mode == "spatial":
        from sclambda.annotation.niche.catalog import AnatomicalCatalog, CatalogEntry
        cat = AnatomicalCatalog(
            entries=[c for c in (CatalogEntry.from_dict(e) for e in entries) if c],
            organism=sess.get("organism", ""), tissue=sess.get("tissue", ""),
        )
    else:
        from sclambda.annotation.celltype.catalog import CellTypeCatalog, CatalogCellType
        cat = CellTypeCatalog(
            entries=[c for c in (CatalogCellType.from_dict(e) for e in entries) if c],
            organism=sess.get("organism", ""), tissue=sess.get("tissue", ""),
        )
    return cat.to_markdown()


def cmd_cache_ls(args) -> int:
    from sclambda.core import cache

    root = cache.resolve_root(args.cache_dir)
    rows = cache.entries(args.cache_dir)
    if not rows:
        print(f"No cached datasets under `{root}`.")
        return 0
    total = sum(r["bytes"] for r in rows)
    lines = [f"# Dataset cache — `{root}`", "",
             "| dataset | size | matrix | stats | status |", "|---|---:|---|---:|---|"]
    for r in rows:
        lines.append(f"| {Path(r['source']).name} | {cache.human(r['bytes'])} | "
                     f"{'yes' if r['has_matrix'] else 'no'} | {r['n_stats']} | {r['status']} |")
    lines += ["", f"{len(rows)} dataset(s), {cache.human(total)} total.",
              "", "Entries marked stale were built from a version of the file that has since "
                  "changed; they are never reused and can be removed."]
    print("\n".join(lines))
    return 0


def cmd_cache_rm(args) -> int:
    from sclambda.core import cache

    rows = cache.entries(args.cache_dir)
    if args.run:
        target = str(Session(args.run).dataset_cache)
        rows = [r for r in rows if str(r["dir"]) == target]
    elif args.stale:
        rows = [r for r in rows if r["status"] != "current"]
    elif not args.all:
        print("error: pass --run RUN (this run's dataset), --stale, or --all", file=sys.stderr)
        return 2

    if not rows:
        print("Nothing to remove.")
        return 0
    freed = sum(cache.remove(r["dir"]) for r in rows)
    print(f"Removed {len(rows)} cached dataset(s), freeing {cache.human(freed)}.")
    print("The next command over those datasets rebuilds what it needs.")
    return 0


def cmd_cache_path(args) -> int:
    print(Session(args.run).dataset_cache)
    return 0


def _read_json_arg(json_str: str | None, file: str | None):
    if file:
        return json.loads(Path(file).read_text())
    if json_str:
        if json_str == "-":
            return json.loads(sys.stdin.read())
        return json.loads(json_str)
    raise SystemExit("error: pass --json '<...>' (or '-' for stdin) or --file <path>")


def cmd_cache_columns(args) -> int:
    """Build the gene-column cache for an existing run's dataset."""
    import time
    from sclambda.core import column_cache
    reader = Session(args.run).accessor._reader
    if column_cache.is_built(reader):
        print("The gene-column cache already exists.")
        return 0
    t0 = time.time()
    if not column_cache.build(reader):
        print("The matrix is dense; gene queries do not need a column cache.")
        return 0
    print(f"Built the gene-column cache in {time.time() - t0:.0f}s; gene queries now read "
          "only the requested genes.")
    return 0
