"""`lambdakit ontology` — map committed labels to Cell Ontology terms.

The CLI ranks candidate terms by name and synonym; the caller reads them and
records one term per cluster. A recorded term is tied to the label it was
chosen for, so relabelling a cluster makes its term stale instead of wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

from sclambda.analysis.ontology import OntologySearch, read_obo
from sclambda.storage.session import Session


def _order(ids):
    return sorted(ids, key=lambda s: (len(s), s))


def _load(sess: Session, obo_arg: str | None):
    data = sess.read_ontology()
    obo = obo_arg or data.get("obo")
    if not obo:
        raise SystemExit("error: no ontology file for this run yet; pass --obo PATH (e.g. cl.obo or cl-basic.obo)")
    obo = str(Path(obo).expanduser().resolve())
    if not Path(obo).exists():
        raise SystemExit(f"error: ontology file not found: {obo}")
    if data.get("obo") != obo:
        if data.get("terms"):
            print(f"note: switching ontology file from {data.get('obo')} to {obo}; "
                  "recorded terms are kept and re-checked against it.\n")
        data["obo"] = obo
        sess.write_ontology(data)
    return data, read_obo(obo)


def _cell(text: str, limit: int = 110) -> str:
    text = " ".join(text.split()).replace("|", "/")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def cmd_candidates(args) -> int:
    sess = Session(args.run)
    results = sess.all_results()
    if not results:
        print("Nothing to map: no cluster has a committed label yet.", file=sys.stderr)
        return 2
    data, terms = _load(sess, args.obo)
    search = OntologySearch(terms)

    if args.query and len(args.cluster or []) != 1:
        print("error: --query needs exactly one --cluster", file=sys.stderr)
        return 2
    if args.cluster:
        missing = [c for c in args.cluster if c not in results]
        if missing:
            print(f"error: no committed label for cluster(s) {', '.join(missing)}", file=sys.stderr)
            return 2
        ids = args.cluster
    else:
        done = sess.current_ontology_terms(results)
        ids = [c for c in _order(results) if c not in done]
        if not ids:
            print("Every committed label has a current ontology term. See `lambdakit ontology show`.")
            return 0

    organism = sess.get("organism", "")
    tissue = sess.get("tissue", "")
    print(f"# Cell Ontology candidates ({organism}, {tissue})\n")
    print(f"Ontology: `{data['obo']}`, {len(terms)} terms. Scores rank string similarity only.\n")
    for cid in ids:
        label = results[cid].get("label", "")
        query = args.query or label
        print(f"## Cluster {cid}: {label}")
        if args.query:
            print(f"Search: `{query}`")
        print()
        hits = search.search(query, top=args.top)
        if not hits:
            print("No candidates. Try `--query` with a broader or alternative name.\n")
            continue
        print("| term | name | matched | score | is_a | definition |")
        print("|---|---|---|---|---|---|")
        for h in hits:
            t = terms[h["id"]]
            parents = ", ".join(terms[p].name for p in t.parents if p in terms)
            matched = "name" if h["via"] == "name" else f"synonym: {h['matched']}"
            print(f"| {h['id']} | {t.name} | {_cell(matched, 40)} | {h['score']} | "
                  f"{_cell(parents, 50)} | {_cell(t.definition)} |")
        print()
    return 0


def cmd_set(args) -> int:
    sess = Session(args.run)
    result = sess.read_result(args.cluster)
    if not result or not result.get("label"):
        print(f"error: cluster {args.cluster} has no committed label", file=sys.stderr)
        return 2
    if bool(args.term) == bool(args.none):
        print("error: pass exactly one of --term ID or --none", file=sys.stderr)
        return 2
    if args.none and not args.note:
        print("error: --none needs --note saying why no term fits", file=sys.stderr)
        return 2
    data, terms = _load(sess, args.obo)
    entry = {"label": result["label"], "id": None, "name": None, "note": args.note or ""}
    if args.term:
        term_id = args.term.strip().replace("_", ":")
        if term_id not in terms:
            print(f"error: {term_id} is not a current term in {data['obo']}", file=sys.stderr)
            return 2
        entry.update(id=term_id, name=terms[term_id].name)
    data.setdefault("terms", {})[args.cluster] = entry
    sess.write_ontology(data)

    shown = f"{entry['id']} {entry['name']}" if entry["id"] else "no term"
    print(f"Cluster {args.cluster} ({result['label']}): {shown}")
    remaining = [c for c in sess.all_results() if c not in sess.current_ontology_terms()]
    print(f"{len(remaining)} committed label(s) still without a current term.")
    return 0


def cmd_show(args) -> int:
    sess = Session(args.run)
    results = sess.all_results()
    recorded = sess.read_ontology().get("terms", {})
    current = sess.current_ontology_terms(results)
    print("| cluster | label | term | name | status | note |")
    print("|---|---|---|---|---|---|")
    for cid in _order(results):
        entry = recorded.get(cid)
        if cid in current:
            status = "mapped" if entry["id"] else "no term"
        elif entry:
            status = f"stale (chosen for '{entry.get('label')}')"
        else:
            status = "missing"
        entry = entry or {}
        print(f"| {cid} | {results[cid].get('label', '')} | {entry.get('id') or ''} | "
              f"{entry.get('name') or ''} | {status} | {_cell(entry.get('note', ''), 80)} |")
    return 0
