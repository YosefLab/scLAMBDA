"""Persistent annotation decisions and legacy dossier commands for the skills."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sclambda.storage.session import Session
from sclambda.annotation.decisions import apply_critic_verdict, build_commit


def _load(sess: Session, cluster_id: str):
    mode = sess.get("mode", "celltype")
    data = sess.read_dossier_dict(cluster_id)
    if mode == "spatial":
        from sclambda.annotation.niche.dossier import NicheDossier
        return NicheDossier.from_dict(data) if data else None
    from sclambda.annotation.celltype.dossier import CellTypeDossier
    return CellTypeDossier.from_dict(data) if data else None


def _require(sess: Session, cluster_id: str):
    d = _load(sess, cluster_id)
    if d is None:
        print(f"error: no dossier for cluster {cluster_id}. Run `lambdakit dossier init --cluster {cluster_id}` first.",
              file=sys.stderr)
        raise SystemExit(2)
    return d


def cmd_init(args) -> int:
    sess = Session(args.run)
    mode = sess.get("mode", "celltype")
    cid = args.cluster
    existing = _load(sess, cid)
    if existing is not None:
        print(f"Resuming existing dossier for cluster {cid}; evidence preserved.")
        print(existing.to_markdown())
        return 0
    catalog = sess.read_catalog()

    if mode == "spatial":
        from sclambda.annotation.niche.dossier import seed_dossier
        composition = []
        comp_file = Path(sess.run) / "composition" / f"{cid}.json"
        if comp_file.exists():
            composition = json.loads(comp_file.read_text()).get("proportions", [])
        d = seed_dossier(
            cluster_id=cid,
            catalog_compartments=catalog,
            spatial_position=args.position or "",
            composition=composition,
        )
    else:
        from sclambda.annotation.celltype.dossier import seed_ct_dossier
        d = seed_ct_dossier(cluster_id=cid, catalog_cell_types=catalog)

        # Parent context is a hypothesis; only the child's critic resolves it.
        if args.parent_label:
            from sclambda.annotation.celltype.dossier import Candidate
            d.upsert(Candidate(cell_type=args.parent_label, state="candidate", source="data",
                               prior="high", likelihood="high",
                               evidence_for=[f"inherited from parent cluster {args.parent}"]))
            d.open_questions.append(f"Does this child support parent identity {args.parent_label}?")

    sess.write_dossier_dict(cid, d.to_dict())
    print(d.to_markdown())
    return 0


def cmd_show(args) -> int:
    sess = Session(args.run)
    d = _require(sess, args.cluster)
    if args.json:
        print(json.dumps(d.to_dict(), indent=2))
    else:
        print(d.to_markdown())
    return 0


def cmd_update(args) -> int:
    """Apply a belief_delta through the vendored `apply_update` contract."""
    sess = Session(args.run)
    d = _require(sess, args.cluster)
    update = _read_json(args.json, args.file)
    if not isinstance(update, dict):
        print("error: --json must be a JSON object (the belief_delta)", file=sys.stderr)
        return 2

    before = _leader_name(d)
    d.iteration = int(getattr(d, "iteration", 0)) + 1
    d.apply_update(update, default_source=args.source or "agent")
    sess.write_dossier_dict(args.cluster, d.to_dict())
    after = _leader_name(d)

    sess.append_transcript(args.cluster, {"step": d.iteration, "tool": "belief_update", "update": update})

    # Echo the post-contract state: what the model proposed and what the
    # dossier actually accepted are not always the same, and the difference is
    # exactly what it needs to see.
    print(d.to_markdown())
    if before != after:
        print(f"\n> leader changed: {before or '(none)'} -> {after or '(none)'}")
    else:
        print(f"\n> leader unchanged: {after or '(none)'}")
    return 0


def cmd_evidence(args) -> int:
    """Append an immutable measured fact to the evidence log."""
    sess = Session(args.run)
    d = _require(sess, args.cluster)
    d.add_evidence(args.source, args.fact)
    sess.write_dossier_dict(args.cluster, d.to_dict())
    print(f"Evidence recorded for cluster {args.cluster} [{args.source}].")
    print(f"Log now holds {len(d.evidence_log)} fact(s).")
    return 0


def cmd_decision(args) -> int:
    """Save the critic decision using runtime schema and evidence validation."""
    from sclambda.annotation.celltype.measurements import repair_annotation_refs, validate_annotation
    sess = Session(args.run)
    spatial = sess.get("mode", "celltype") == "spatial"
    if spatial:
        from sclambda.annotation.niche.measurements import validate_annotation, normalize_annotation
        repair_annotation_refs = lambda raw, d: (raw, [])
    d = _require(sess, args.cluster)
    try:
        raw = _read_json(args.json, args.file)
    except json.JSONDecodeError as exc:
        sess.append_transcript(args.cluster, {"tool": "critic_decision", "raw_response": exc.doc,
            "reference_repairs": [], "validation_errors": [str(exc)]})
        print(f"error: invalid critic JSON: {exc}", file=sys.stderr)
        return 2
    value, repairs = repair_annotation_refs(raw, d)
    issues = validate_annotation(value, d, **({"require_checks": True} if spatial else {}))
    sess.append_transcript(args.cluster, {"tool": "critic_decision", "proposal": raw,
        "reference_repairs": repairs, "validation_errors": issues})
    if issues:
        print("error: " + "; ".join(issues), file=sys.stderr)
        return 2
    if spatial:
        value = normalize_annotation(value, d)
        d.decision_mode = True
        d.critic_decision = value
    else:
        d.annotation = value
    d.description = value["reason"]
    d.open_questions = [value["next_question"]] if value["next_question"] else []
    d.iteration += 1
    sess.write_dossier_dict(args.cluster, d.to_dict())
    print(d.to_markdown())
    return 0


def cmd_retain_failure(args) -> int:
    """Expose the pipeline's last-resort output without accepting a new label."""
    from sclambda.annotation.niche.recovery import retain_failed_output
    sess = Session(args.run)
    if sess.get("mode", "celltype") != "spatial" or not args.reason.strip():
        print("error: retain-failure requires a spatial run and a nonempty failure reason", file=sys.stderr)
        return 2
    d = _require(sess, args.cluster)
    sess.append_transcript(args.cluster, {"tool": "retain_failure", "reason": args.reason,
                                         "previous_decision": d.critic_decision})
    retain_failed_output(d)
    d.decision_mode = True
    d.description = d.critic_decision["reason"]
    d.open_questions = [d.critic_decision["next_question"]] if d.critic_decision["next_question"] else []
    sess.write_dossier_dict(args.cluster, d.to_dict())
    print(d.to_markdown())
    print("Execution failed; this provisional output still needs `dossier commit` before export.")
    return 0


def cmd_axes(args) -> int:
    """Set the resolution ladder (critic-owned) and the subcluster flag."""
    sess = Session(args.run)
    d = _require(sess, args.cluster)
    payload = _read_json(args.json, args.file)
    if not isinstance(payload, dict) or not any(key in payload for key in ("axes", "needs_subcluster", "resolved_as")):
        print("error: pass axes, needs_subcluster and/or resolved_as in a JSON object", file=sys.stderr)
        return 2
    decision = apply_critic_verdict(d, payload)
    sess.write_dossier_dict(args.cluster, d.to_dict())
    sess.append_transcript(args.cluster, {"tool": "critic_verdict", "update": payload,
                                         "decision": decision.status, "reason": decision.reason})
    print(d.to_markdown())
    if decision.candidate:
        print(f"\n> resolved_as {decision.status}: {decision.candidate}: {decision.reason}")
    return 0


def cmd_gap(args) -> int:
    """Record a critic gap: an open question plus the measurement that closes it."""
    sess = Session(args.run)
    d = _require(sess, args.cluster)
    for issue, disc in zip(args.issue or [], args.discriminator or []):
        d.add_evidence("critic", f"GAP: {issue} — resolve via: {disc}")
        d.open_questions.append(f"CRITIC gap: {issue}")
    sess.write_dossier_dict(args.cluster, d.to_dict())
    sess.append_transcript(args.cluster, {
        "tool": "critic",
        "gaps": [{"issue": i, "discriminator": dd} for i, dd in zip(args.issue or [], args.discriminator or [])],
    })
    print(f"Recorded {len(args.issue or [])} gap(s) for cluster {args.cluster}.")
    return 0


def cmd_commit(args) -> int:
    """Commit the dossier's leading candidate as this cluster's annotation."""
    sess = Session(args.run)
    d = _require(sess, args.cluster)
    try:
        result = build_commit(d, label=args.label, justification=args.justification or "",
                              manual_override_reason=getattr(args, "manual_override_reason", "") or "")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    label = result["label"]
    sess.write_result(args.cluster, result)
    print(f"Committed cluster {args.cluster}: **{label}**")
    if result["confidence"]:
        print(f"confidence: {result['confidence']}")
    if result["needs_subcluster"]:
        print("flagged for subclustering (`needs_subcluster: true`)")
    if result["open_questions"]:
        print("open questions: " + "; ".join(result["open_questions"][:3]))
    return 0


def _leader_name(d) -> str:
    c = d.leading()
    if c is None:
        return ""
    return c.label or ""


def _read_json(json_str: str | None, file: str | None):
    if file:
        return json.loads(Path(file).read_text())
    if json_str:
        if json_str == "-":
            return json.loads(sys.stdin.read())
        return json.loads(json_str)
    raise SystemExit("error: pass --json '<...>' (or '-' to read stdin) or --file <path>")
