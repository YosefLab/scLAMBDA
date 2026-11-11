"""`lambdakit status` / `lambdakit phase` — where the run is, and what to do next.

A long run gets interrupted: quota runs out, a session closes, a node is
reclaimed. On restart the model needs to answer "what is already done?" without
re-reading the whole run directory or, worse, guessing and redoing work.

Almost all of that is *derivable* — a cluster with a matching result is annotated,
and a parent with child memberships has been split. What is not derivable are the
whole-run phases (the cross-cluster pass, the spatial final passes, the export),
because they leave no per-cluster artefact. Those are recorded explicitly in the
phase ledger; everything else is inferred, so it cannot drift out of step with
what is actually on disk.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from sclambda.storage.session import Session

# Phases with no per-cluster artefact, so they have to be recorded by hand.
CELLTYPE_PHASES = [
    ("cross-cluster", "compare confusable clusters on a shared diagnostic panel"),
    ("ontology", "map labels to Cell Ontology terms"),
    ("export", "write annotations out"),
]
NICHE_PHASES = [
    ("consistency-review", "optional global observations reviewed by the deciding critic"),
    ("export", "write annotations out"),
]
# Run only when the user asks; never the "next" step.
OPTIONAL_PHASES = {"cross-cluster", "consistency-review", "ontology"}


def _levels(overlay) -> list[int]:
    return sorted(int(m.group(1)) for c in overlay.columns
                  if (m := re.fullmatch(r"L(\d+)", c)))


def cmd_status(args) -> int:
    sess = Session(args.run)
    mode = sess.get("mode", "celltype")
    unit = "niche" if mode == "spatial" else "cluster"

    out = [f"# Run status — `{sess.run}`", ""]
    out.append(f"- dataset: `{sess.get('h5ad')}`")
    out.append(f"- {sess.get('organism')} / {sess.get('tissue')}  ({mode} annotation)")

    try:
        overlay = sess.read_overlay()
    except Exception:
        out.append("")
        out.append("**Not initialised.** Run `lambdakit init` first.")
        print("\n".join(out))
        return 0

    results = sess.current_results()
    dossiers = sess.all_dossiers()
    catalog = sess.read_catalog()
    levels = _levels(overlay)
    max_depth = int(sess.get("max_depth", 1) or 1)

    # ---- per-depth progress, inferred from the files that exist ----
    out += ["", "## Annotation progress", "",
            f"| depth | {unit}s | done | in progress | pending |", "|---|---:|---:|---:|---:|"]
    pending_by_depth: dict[int, list[str]] = {}
    active_ids = set()
    for d in levels:
        ids = [str(c) for c in overlay[f"L{d}"].dropna().unique() if str(c) not in ("nan", "")]
        active_ids.update(ids)
        done = [c for c in ids if results.get(c, {}).get("label")]
        started = [c for c in ids if c in dossiers and c not in done]
        pending = [c for c in ids if c not in dossiers and c not in done]
        pending_by_depth[d] = pending + started
        out.append(f"| L{d} | {len(ids)} | {len(done)} | {len(started)} | {len(pending)} |")

    # ---- one-shot phases, from the ledger ----
    phases = [("catalog", "the tissue catalog")] + (
        NICHE_PHASES if mode == "spatial" else CELLTYPE_PHASES)
    out += ["", "## Whole-run phases", "", "| phase | state |", "|---|---|"]
    for name, why in phases:
        if name == "catalog":
            state = f"done ({len(catalog)} entries)" if catalog else "optional (not configured)"
        else:
            recorded = sess.phase_state(name)
            state = recorded or ("optional (not requested)" if name in OPTIONAL_PHASES else "not started")
        out.append(f"| {name} _({why})_ | {state} |")

    # ---- interrupted work ----
    stale = sess.stale_clusters(minutes=args.stale_minutes)
    if stale:
        out += ["", f"## Interrupted {unit}s", "",
                f"{len(stale)} {unit}(s) have uncommitted work and have not been "
                f"touched in {args.stale_minutes} minutes — most likely a halted run:",
                "", "  " + " ".join(stale), "",
                f"Resume each with `lambdakit dossier show --cluster <id>`: the dossier holds its "
                f"decision, uncertainty and every measurement already taken, so continue "
                f"the loop rather than starting it over."]

    uncommitted = sess.uncommitted_decisions()
    if uncommitted:
        out += ["", "## Uncommitted critic decisions", "",
                "Review remaining questions, then commit the current decision for: " + " ".join(uncommitted),
                "Previous result files are retained but are not current annotations."]

    # ---- the single next action ----
    out += ["", "## Next"]
    todo_depth = next((d for d in levels if pending_by_depth.get(d)), None)
    if todo_depth is not None:
        ids = pending_by_depth[todo_depth]
        out.append(f"Annotate {len(ids)} remaining {unit}(s) at depth {todo_depth}:")
        out.append("")
        out.append("  " + " ".join(ids[:40]) + (" …" if len(ids) > 40 else ""))
    else:
        flagged = [c for c, r in results.items()
                   if r.get("needs_subcluster")
                   and (c.count("_") + 1 if c else 0) < max_depth
                   and (c in active_ids or c == "")
                   and not sess.child_clusters(c, overlay)]
        if flagged:
            out.append(f"Split the {len(flagged)} {unit}(s) flagged for subclustering: "
                       + " ".join(flagged[:20]))
            out.append("")
            out.append(f"  lambdakit subcluster --run {sess.run} --cluster <id>")
        else:
            nxt = next((n for n, _ in phases[1:] if n not in OPTIONAL_PHASES and sess.phase_state(n) not in {"done", "skipped"}), None)
            if nxt:
                out.append(f"Every {unit} is annotated. Run the **{nxt}** phase, then record it:")
                out.append("")
                out.append(f"  lambdakit phase done --run {sess.run} --name {nxt}")
            else:
                out.append("Everything is done.")

    print("\n".join(out))
    return 0


def cmd_phase(args) -> int:
    sess = Session(args.run)
    if args.list:
        data = sess.read_progress().get("phases", {})
        if not data:
            print("No phases recorded yet.")
            return 0
        print("| phase | state | when | note |")
        print("|---|---|---|---|")
        for name, info in data.items():
            print(f"| {name} | {info.get('state','')} | {info.get('at','')} | {info.get('note','')} |")
        return 0
    if not args.name:
        print("error: pass --name <phase> (or --list)", file=sys.stderr)
        return 2
    sess.mark_phase(args.name, args.state, args.note or "")
    print(f"Recorded phase `{args.name}` as **{args.state}**.")
    print(f"`lambdakit status --run {sess.run}` will now skip it on a restart.")
    return 0
