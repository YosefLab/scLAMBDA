"""`lambdakit export` — write annotations out.

Three outputs, matching sc-lambda's:

* a **per-cell CSV/TSV** joining every cluster level to its committed label,
  with shallower annotations forward-filled into deeper levels so a cell
  annotated only at L1 still carries that label at L2;
* an **Excel workbook** with one sheet per depth plus the cluster-name mapping
  sheet — the only place original cluster names reappear, since they are hidden
  from every reasoning step;
* a **Markdown report** with the per-cluster transcript, which replaces the
  prompt-history file the original assembled from its node execution log.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from sclambda.storage.session import Session


def _levels(overlay: pd.DataFrame) -> list[int]:
    return sorted(int(m.group(1)) for c in overlay.columns
                  if (m := re.fullmatch(r"L(\d+)", c)))


def cmd_export(args) -> int:
    sess = Session(args.run)
    overlay = sess.read_overlay()
    pending = sess.uncommitted_decisions()
    if pending:
        print("error: uncommitted critic decisions for clusters " + ", ".join(pending)
              + ". Finish their review and run `lambdakit dossier commit` before exporting.", file=sys.stderr)
        return 2
    results = sess.all_results()
    if not results:
        print("Nothing to export — no cluster has been committed yet.", file=sys.stderr)
        return 2

    ontology = sess.current_ontology_terms(results)
    stale = sorted(set(sess.read_ontology().get("terms", {})) & set(results) - set(ontology))
    name_map = sess.get("cluster_name_map") or {}
    inv = {v: k for k, v in name_map.items()}
    levels = _levels(overlay)
    written = []

    # ---------------- per-cell table ----------------
    if args.csv:
        df = pd.DataFrame(index=overlay.index)
        last = None
        for d in levels:
            col = f"L{d}"
            df[col] = overlay[col]
            labels = overlay[col].map(lambda c: (results.get(str(c)) or {}).get("label"))
            # Forward-fill: a cell whose deeper cluster was never annotated keeps
            # the shallower call rather than going blank.
            if last is not None:
                labels = labels.fillna(df[last])
            df[f"cell_type_L{d}"] = labels
            for field in ("assessment_status", "uncertainty", "major_region", "execution_status", "output_origin"):
                values = overlay[col].map(lambda c: (results.get(str(c)) or {}).get(field))
                # Inherit metadata only when the label itself came from the parent.
                inherited = overlay[col].map(lambda c: not (results.get(str(c)) or {}).get("label"))
                previous = f"{field}_L{d-1}"
                if previous in df:
                    values = values.where(~inherited, df[previous])
                df[f"{field}_L{d}"] = values
            for key, field in (("id", "cell_ontology_id"), ("name", "cell_ontology")):
                if not ontology:
                    break
                values = overlay[col].map(lambda c: (ontology.get(str(c)) or {}).get(key))
                inherited = overlay[col].map(lambda c: not (results.get(str(c)) or {}).get("label"))
                previous = f"{field}_L{d-1}"
                if previous in df:
                    values = values.where(~inherited, df[previous])
                df[f"{field}_L{d}"] = values
            last = f"cell_type_L{d}"
        if inv:
            df["original_cluster"] = overlay[f"L{levels[0]}"].map(lambda c: inv.get(str(c), ""))
        out = Path(args.csv)
        sep = "\t" if out.suffix in (".tsv", ".txt") else ","
        df.to_csv(out, sep=sep)
        written.append(out)

    # ---------------- workbook ----------------
    if args.xlsx:
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            print("error: --xlsx needs openpyxl (`pip install openpyxl`)", file=sys.stderr)
            return 2
        out = Path(args.xlsx)
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            for d in levels:
                ids = [str(c) for c in overlay[f"L{d}"].dropna().unique() if str(c) not in ("nan", "")]
                rows = []
                for cid in sorted(ids, key=lambda s: (len(s), s)):
                    r = results.get(cid)
                    if not r:
                        continue
                    rows.append({
                        "cluster_id": cid,
                        "cell_type": r.get("label", ""),
                        "confidence": r.get("confidence", ""),
                        "assessment_status": r.get("assessment_status", ""),
                        "execution_status": r.get("execution_status", ""),
                        "output_origin": r.get("output_origin", ""),
                        "major_region": r.get("major_region", ""),
                        "uncertainty": r.get("uncertainty", ""),
                        "alternatives": str(r.get("alternatives", [])),
                        "justification": r.get("justification", ""),
                        "granularity": r.get("granularity", ""),
                        "open_questions": "; ".join(r.get("open_questions", []) or []),
                    })
                    if ontology:
                        term = ontology.get(cid) or {}
                        rows[-1].update(cell_ontology_id=term.get("id") or "",
                                        cell_ontology=term.get("name") or "")
                if rows:
                    pd.DataFrame(rows).to_excel(writer, sheet_name=f"L{d}", index=False)
            if inv:
                pd.DataFrame(
                    [{"encoded_id": k, "original_name": v} for k, v in sorted(inv.items(), key=lambda kv: (len(kv[0]), kv[0]))]
                ).to_excel(writer, sheet_name="Cluster_Mapping", index=False)
        written.append(out)

    # ---------------- report ----------------
    if args.md:
        lines = [f"# Annotation report — {sess.get('tissue', '')} ({sess.get('organism', '')})", ""]
        lines.append(f"Dataset: `{sess.get('h5ad')}`")
        lines.append(f"Clusters annotated: {len(results)}")
        lines.append("")
        for cid in sorted(results, key=lambda s: (len(s), s)):
            r = results[cid]
            lines.append(f"## Cluster {cid} — {r.get('label', '')}")
            if inv.get(cid):
                lines.append(f"*original cluster name: {inv[cid]}*")
            lines.append("")
            if cid in ontology:
                term = ontology[cid]
                lines.append(f"- cell ontology: {term['id']} {term['name']}" if term.get("id")
                             else f"- cell ontology: no term ({term.get('note', '')})")
            if r.get("confidence"):
                lines.append(f"- confidence: {r['confidence']}")
            if r.get("assessment_status"):
                lines.append(f"- assessment: {r['assessment_status']}")
            if r.get("execution_status"):
                lines.append(f"- execution_status: {r['execution_status']}")
            if r.get("output_origin"):
                lines.append(f"- output_origin: {r['output_origin']}")
            if r.get("uncertainty"):
                lines.append(f"- uncertainty: {r['uncertainty']}")
            if r.get("alternatives"):
                lines.append(f"- alternatives: {r['alternatives']}")
            if r.get("granularity"):
                lines.append(f"- resolved to: {r['granularity']}")
            if r.get("needs_subcluster"):
                lines.append("- flagged for subclustering")
            if r.get("justification"):
                lines.append("")
                lines.append(r["justification"])
            transcript = sess.read_transcript(cid)
            if transcript:
                lines.append("")
                lines.append("<details><summary>reasoning transcript "
                             f"({len(transcript)} steps)</summary>")
                lines.append("")
                for i, entry in enumerate(transcript, 1):
                    lines.append(f"**{i}. {entry.get('tool', 'step')}**")
                    for k, v in entry.items():
                        if k == "tool":
                            continue
                        lines.append(f"- {k}: `{v}`")
                    lines.append("")
                lines.append("</details>")
            lines.append("")
        Path(args.md).write_text("\n".join(lines))
        written.append(Path(args.md))

    if not written:
        print("Nothing written — pass --csv, --xlsx and/or --md.", file=sys.stderr)
        return 2

    print(f"Exported {len(results)} annotated cluster(s):")
    for p in written:
        print(f"- `{p}`")

    if stale:
        print(f"\nOntology terms left out because the label changed after mapping: {' '.join(stale)}. "
              "Map them again with `lambdakit ontology candidates`.")

    unlabelled = []
    for d in levels:
        for cid in overlay[f"L{d}"].dropna().unique():
            if str(cid) not in ("nan", "") and str(cid) not in results:
                unlabelled.append(str(cid))
    if unlabelled:
        print(f"\n{len(unlabelled)} cluster(s) have no committed label: {' '.join(sorted(set(unlabelled))[:20])}")
    return 0
