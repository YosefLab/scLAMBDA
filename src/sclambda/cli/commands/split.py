"""`lambdakit substructure` and `lambdakit subcluster` — the split gate.

sc-lambda splits a cluster in two stages, and the order matters. First a purely
statistical probe asks whether there is *anything* to split; only if there is
does a reasoning step get asked whether the something is biologically meaningful.
A cluster with no depletion pocket and no coherent expression module is declared
a leaf without any model call at all.

The probe runs two tests on the cluster's own cells:

* **self-depletion** — a spatially coherent pocket of cells that do *not* express
  the committed identity's own markers. This is the signal that separates a real
  contaminating population from ambient RNA, and the decision prompt leans on it
  hard: a foreign-lineage module is separable *only* when a depletion pocket
  exists. Otherwise the foreign transcripts are ambient and splitting would just
  return the same cell type twice.
* **structured variance** — genes with significant Moran's I on the kNN graph,
  grouped into correlated modules, each kept only if a two-component mixture on
  its score shows a genuinely discrete high mode covering enough cells.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sclambda.storage.session import Session
from sclambda.storage.atomic import write_text


def cmd_substructure(args) -> int:
    from sclambda.annotation.celltype.substructure import probe_substructure, signature_genes_from_dossier
    from sclambda.cli.commands.dossier import _load

    sess = Session(args.run)
    acc = sess.accessor
    cid = args.cluster

    depth = cid.count("_") + 1 if cid else 0
    max_depth = int(sess.get("max_depth", 1) or 1)
    if depth >= max_depth:
        print(f"Cluster {cid} is at depth {depth} of max_depth {max_depth} — "
              f"splitting is not possible, treat it as a leaf.")
        return 0

    d = _load(sess, cid)
    valid = np.asarray(acc.gene_names)
    if d is not None:
        self_genes = signature_genes_from_dossier(d, valid)
    else:
        self_genes = []
    if args.genes:
        self_genes = [g.strip() for g in args.genes.split(",") if g.strip() and g.strip() in set(map(str, valid))]

    probe = probe_substructure(
        acc, cid, self_genes,
        embedding_key=sess.get("embedding_key") or "X_pca",
    )

    dep = probe.get("depletion") or {}
    st = probe.get("structure") or {}
    modules = st.get("modules") or []

    out = [f"# Substructure probe — cluster {cid} ({probe.get('n_cells', 0):,} cells)", ""]
    if probe.get("depletion") is None and probe.get("structure") is None:
        out.append("Too few cells to probe. Treat as a leaf.")
        print("\n".join(out))
        return 0

    out.append("## A. Self-depletion (cells lacking this cluster's own identity markers)")
    if not self_genes:
        out.append("No committed signature genes available, so this test could not run.")
    elif dep:
        out.append(f"- signature genes tested: {', '.join(self_genes[:15])}"
                   + (f" (+{len(self_genes)-15} more)" if len(self_genes) > 15 else ""))
        out.append(f"- fraction of cells with the signature absent: {dep.get('frac_absent', 0):.3f}")
        out.append(f"- spatial coherence of the absent cells (Moran's I): {dep.get('morans_i', 0):.3f} "
                   f"(permutation null {dep.get('null_i', 0):.3f})")
        verdict = "**a coherent depletion pocket is present**" if dep.get("present") else "no coherent pocket"
        out.append(f"- verdict: {verdict}")
        pocket = dep.get("pocket_markers") or []
        if pocket:
            out.append(f"- genes enriched in the depleted pocket: {', '.join(pocket)}")

    out.append("")
    out.append("## B. Structured expression modules")
    if st:
        out.append(f"- genes with significant spatial structure: {st.get('n_structured', 0)} "
                   f"({st.get('frac_structured', 0):.1%} of tested; chance rate {st.get('chance_rate', 0):.1%})")
    if not modules:
        out.append("- no discrete modules found")
    else:
        for i, m in enumerate(modules, 1):
            genes = m.get("genes", [])
            out.append(f"- module {i} (mean Moran's I {m.get('mean_i', 0):.3f}, {len(genes)} genes): "
                       f"{', '.join(genes[:20])}")

    out.append("")
    if not dep.get("present") and not modules:
        out.append("## Verdict (deterministic)")
        out.append("Neither test found anything to split. **This cluster is a leaf — do not subcluster it, "
                   "and do not ask for a split decision.**")
        _write_probe(sess, cid, probe, needs_decision=False)
    else:
        out.append("## Next")
        out.append("The statistics found candidate structure. The remaining question is biological and "
                   "is yours to answer: is each module a meaningful subdivision, or a confound "
                   "(clonality, donor/sex, technical, transient activation) to ignore? "
                   "Read `references/subcluster-decision.md` before deciding.")
        if modules and not dep.get("present"):
            out.append("")
            out.append("> Note: there is **no depletion pocket**. This alone does not establish "
                       "whether another program reflects ambient RNA or a separable population. "
                       "Assess the measured structure and retain uncertainty.")
        _write_probe(sess, cid, probe, needs_decision=True)

    print("\n".join(out))
    return 0


def _write_probe(sess: Session, cid: str, probe: dict, needs_decision: bool) -> None:
    d = Path(sess.run) / "probes"
    d.mkdir(parents=True, exist_ok=True)
    payload = dict(probe)
    payload["needs_decision"] = needs_decision
    write_text(d / f"{cid or '_root_'}.json", json.dumps(payload, indent=2, default=float))


def cmd_subcluster(args) -> int:
    """Split one cluster into children. Orchestrator-only — never run in parallel."""
    sess = Session(args.run)
    cid = args.cluster
    children = sess.child_clusters(cid)
    if children:
        print(f"Cluster {cid} already has children: {', '.join(children)}. "
              "Resume their annotation; existing memberships are preserved.")
        return 0
    depth = cid.count("_") + 1 if cid else 0
    if depth >= int(sess.get("max_depth", 1) or 1):
        print(f"Cluster {cid} is at the requested maximum depth; not splitting.")
        return 0
    if cid in sess.uncommitted_decisions():
        print(f"Cluster {cid} has an uncommitted critic decision; finish review and commit before splitting.")
        return 2
    acc = sess.accessor

    result = sess.read_result(cid)
    if result is not None and not result.get("needs_subcluster") and not args.force:
        print(f"Cluster {cid} was not flagged for subclustering (`needs_subcluster: false`). "
              f"Pass --force to split anyway.")
        return 0

    n_before = acc.get_n_cells_in_cluster(cid)
    if n_before < args.min_cells:
        print(f"Cluster {cid} has {n_before} cells, below --min-cells {args.min_cells}. Not splitting.")
        return 0

    from sclambda.analysis.subcluster import subcluster

    res = subcluster(acc, cluster_id=cid, method="leiden", n_neighbors=args.n_neighbors)
    if res.n_clusters < 2:
        # Deliberately do NOT persist: subcluster() already wrote a single-child
        # level into the in-memory overlay, and saving it would leave a useless
        # depth that every later command has to skip past.
        print(f"Leiden found no split for cluster {cid} — treat it as a leaf.")
        return 0

    # Drop children too small to annotate, then renumber. sc-lambda escalates
    # the floor when a split fragments badly, and skips filtering entirely for
    # small splits where every child is worth keeping.
    kept, dropped = _filter_by_proportion(acc, res.clusters, n_before)
    if dropped:
        acc.dissolve_clusters(dropped)
        mapping = acc.reindex_clusters(cid, kept)
        kept = [mapping.get(k, k) for k in kept]

    sess.save_overlay(acc)

    lines = [f"# Cluster {cid} split into {len(kept)} children", ""]
    lines.append("| child | n_cells |")
    lines.append("|---|---:|")
    for child in kept:
        lines.append(f"| {child} | {acc.get_n_cells_in_cluster(child):,} |")
    if dropped:
        lines.append("")
        lines.append(f"Dropped {len(dropped)} child cluster(s) below the size floor; their cells "
                     f"stay with the parent.")
    lines.append("")
    lines.append("Annotate each child from its own measurements. Pass `--parent-label` to "
                 "`lambdakit dossier init` only as provisional context; children may disagree with the parent.")
    print("\n".join(lines))
    return 0


def _filter_by_proportion(acc, children, n_parent):
    if len(children) <= 10:
        return list(children), []
    floor = 0.05 if len(children) > 20 else 0.01
    kept, dropped = [], []
    for c in children:
        frac = acc.get_n_cells_in_cluster(c) / max(1, n_parent)
        (kept if frac >= floor else dropped).append(c)
    if len(kept) < 2:  # never dissolve a split down to nothing
        return list(children), []
    return kept, dropped
