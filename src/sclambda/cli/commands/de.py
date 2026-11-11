"""Ranked and targeted DE, with persistent measurement tables for cell typing."""

from __future__ import annotations

import sys

import numpy as np

from sclambda.storage.session import Session

MIN_CELLS = 10  # default floor; --min-cells lowers it for pseudobulk input


def _cluster_indices(acc, cluster_id: str) -> np.ndarray:
    return acc.get_cell_indices(cluster_id)


def _rest_indices(acc, cluster_id: str) -> np.ndarray:
    return np.setdiff1d(np.arange(acc.n_cells), _cluster_indices(acc, cluster_id))


def _celltype_measurement(sess, cid, genes, *, reference, method, ranked=False,
                          persist=False, requested=()):
    from sclambda.annotation.celltype.dossier import CellTypeDossier
    from sclambda.annotation.celltype.measurements import record, render, add_measurement
    m = record(genes, target=cid, reference=reference, method=method, ranked=ranked)
    print(render(m))
    if persist:
        d = CellTypeDossier.from_dict(sess.read_dossier_dict(cid) or {"cluster_id": cid})
        add_measurement(d, m)
        d.requested_genes = sorted(set(d.requested_genes) | set(requested))
        sess.write_dossier_dict(cid, d.to_dict())
        sess.append_transcript(cid, {"tool": "get_de" if ranked else "targeted_de",
                                    "measurement": m, "requested_genes": list(requested)})
    return m


def cmd_rank(args) -> int:
    """Ranked marker genes for one cluster versus the rest of the dataset."""
    from sclambda.analysis.rank_genes import rank_genes
    from sclambda.annotation.celltype.formatters import format_top_genes_de_table

    sess = Session(args.run)
    acc = sess.accessor
    cid = args.cluster
    floor = args.min_cells or sess.get("min_cells") or MIN_CELLS
    n = acc.get_n_cells_in_cluster(cid)
    if n < floor:
        print(f"Cluster {cid} has only {n} cells (minimum {floor}) — no reliable DE.")
        return 0

    result = rank_genes(
        acc, cid,
        n_genes=args.n or sess.get("n_top_genes", 20),
        de_method=args.method or sess.get("de_method", "ttest"),
    )
    print(f"# Differential expression — cluster {cid} ({n:,} cells); reference specified below\n")
    if sess.get("mode", "celltype") == "spatial":
        print(format_top_genes_de_table(result))
    else:
        method = args.method or sess.get("de_method", "ttest")
        reference = ("all other dataset cells, target excluded; sampled up to 50000" if method == "ols"
                     else "whole dataset INCLUDING target (cached global statistics)")
        _celltype_measurement(sess, cid, result.genes, reference=reference, method=method,
                              ranked=True, persist=args.record)
    if args.record:
        sess.append_transcript(cid, {"tool": "get_de", "n": args.n, "n_genes_returned": len(result.genes)})
    return 0


def cmd_genes(args) -> int:
    """Measure named genes in one cluster, against the rest or against a peer."""
    from sclambda.analysis.targeted_de import compute_de_for_genes, format_genes_as_gene_table

    sess = Session(args.run)
    acc = sess.accessor
    cid = args.cluster
    requested = [g.strip() for g in args.genes.split(",") if g.strip()]
    if not requested:
        print("error: --genes must list at least one gene", file=sys.stderr)
        return 2

    # Genes outside the measured panel are dropped and reported back, exactly as
    # the original agent loop did — silently ignoring them would let the model
    # believe it tested a marker it never measured.
    known = set(str(g) for g in acc.gene_names)
    genes = [g for g in requested if g in known]
    dropped = [g for g in requested if g not in known]

    target = _cluster_indices(acc, cid)
    if args.vs:
        reference = np.setdiff1d(_cluster_indices(acc, args.vs), target)
        title = f"# Cluster {cid} vs cluster {args.vs} (LFC > 0 means higher in {cid})"
        label = f"cluster {cid} vs {args.vs}"
    else:
        reference = _rest_indices(acc, cid)
        title = f"# Signature test — cluster {cid} vs rest"
        label = f"cluster {cid} vs rest"

    floor = args.min_cells or sess.get("min_cells") or MIN_CELLS
    if len(target) < floor or len(reference) < floor:
        print(f"Not enough cells to measure (target={len(target)}, reference={len(reference)}, "
              f"minimum {floor} each).")
        return 0

    if not genes:
        if args.record and sess.get("mode", "celltype") != "spatial":
            from sclambda.annotation.celltype.dossier import CellTypeDossier
            d = CellTypeDossier.from_dict(sess.read_dossier_dict(cid) or {"cluster_id": cid})
            d.requested_genes = sorted(set(d.requested_genes) | set(requested))
            sess.write_dossier_dict(cid, d.to_dict())
            sess.append_transcript(cid, {"tool": "targeted_de", "requested_genes": requested,
                                        "not_found": dropped})
        print(f"None of the requested genes are in the measured panel: {', '.join(requested)}")
        return 0

    result = compute_de_for_genes(
        acc, genes, target, reference,
        de_method=args.method or sess.get("de_method", "ttest"),
    )
    print(f"{title}  —  n_target={len(target):,}, n_reference={len(reference):,}\n")
    if sess.get("mode", "celltype") == "spatial":
        print(format_genes_as_gene_table([(label, result.genes)]))
    else:
        reference_name = (f"cluster {args.vs} excluding target" if args.vs
                          else "all other dataset cells, target excluded")
        _celltype_measurement(sess, cid, result.genes,
            reference=f"{reference_name}; n={len(reference)}; target n={len(target)}",
            method=args.method or sess.get("de_method", "ttest"), persist=args.record, requested=requested)
    not_found = list(dropped) + list(result.genes_not_found or [])
    if not_found:
        print(f"\nNot measured in this dataset (ignore these, do not treat as absent): {', '.join(sorted(set(not_found)))}")
    if args.record:
        sess.append_transcript(cid, {
            "tool": "compare_to_sibling" if args.vs else "test_signature",
            "genes": genes, "vs": args.vs, "not_found": not_found,
        })
    return 0


def cmd_peers(args) -> int:
    """List annotated clusters at the same depth — the `compare_to_sibling` menu."""
    import re
    sess = Session(args.run)
    overlay = sess.read_overlay()
    results = sess.current_results()
    cid = args.cluster
    depth = cid.count("_") + 1
    col = f"L{depth}"
    if col not in overlay.columns:
        print("No clusters at this depth.")
        return 0
    ids = [str(c) for c in overlay[col].dropna().unique() if str(c) not in ("nan", "")]
    peers = []
    for other in sorted(ids):
        if other == cid:
            continue
        res = results.get(other)
        if res and res.get("label"):
            peers.append(f"- {other}: {res['label']}")
    if not peers:
        print("No annotated peers yet at this depth.")
        return 0
    print(f"# Peer clusters at depth {depth} ({len(peers)} annotated)\n")
    print("\n".join(peers[:20]))
    if len(peers) > 20:
        print(f"\n(+{len(peers) - 20} more; capped at 20)")
    return 0
