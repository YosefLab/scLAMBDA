"""Annotation decisions shared by the CLI and Python pipeline adapters."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CriticDecision:
    status: str = "updated"
    candidate: str = ""
    reason: str = ""


def apply_critic_verdict(dossier, payload: dict) -> CriticDecision:
    """Apply ladder updates and promote only a candidate evidenced this turn.

    A resolved rung from an earlier turn cannot justify a new promotion. The
    dossier's own leader-change and single-supported-candidate rules still apply.
    """
    if not isinstance(payload, dict):
        raise ValueError("A critic verdict must be a JSON object")
    update = {}
    if isinstance(payload.get("axes"), dict):
        update["axes"] = payload["axes"]
    if isinstance(payload.get("needs_subcluster"), bool):
        update["needs_subcluster"] = payload["needs_subcluster"]
    if hasattr(dossier, "apply_resolution"):
        dossier.apply_resolution(update.get("axes"), update.get("needs_subcluster"), payload.get("traits"))
    elif update:
        dossier.apply_update(update, default_source="critic")
    name = str(payload.get("resolved_as") or "").strip()
    if not name or name.lower() in ("null", "none"):
        return CriticDecision()
    target = dossier.match(name)
    if target is None:
        return CriticDecision("refused", name, f"{name} is not a candidate in this dossier")
    if target is dossier.leading() and target.state == "supported":
        return CriticDecision("unchanged", name, "already the leader")
    notes = [str(axis.get("note", "")).strip()
             for key, axis in (payload.get("axes") or {}).items()
             if isinstance(axis, dict) and axis.get("resolved") is True
             and str(axis.get("note", "")).strip()
             and (dossier.axes.get(key) or {}).get("resolved")]
    if not notes:
        return CriticDecision("refused", name, "no axis was marked resolved with a note in this call")
    field = "cell_type" if hasattr(dossier, "best_cell_type") else "compartment"
    canonical = target.cell_type if field == "cell_type" else name
    dossier.apply_update({
        "candidates": [{field: canonical, "state": "supported",
                        "evidence_for": [f"critic resolved: {note}" for note in notes]}],
        f"best_{field}": canonical,
    }, default_source="critic")
    if target is dossier.leading() and target.state == "supported":
        return CriticDecision("promoted", name, "promoted to the supported candidate")
    return CriticDecision("refused", name, "the dossier leader-change contract rejected the promotion")


def supported_label(dossier) -> str | None:
    """Return critic-approved cell identity, or the supported spatial leader."""
    if hasattr(dossier, "apply_resolution"):
        return dossier.cell_type
    if getattr(dossier, "decision_mode", False):
        return dossier.critic_decision.get("label")
    leading = dossier.leading()
    if leading is None or leading.state != "supported":
        return None
    if hasattr(dossier, "best_cell_type"):
        return leading.label_at(dossier.committed_granularity)
    return leading.label


def decision_fingerprint(dossier) -> str | None:
    """Identify the compact decision and split flag a result actually commits."""
    import hashlib
    import json
    data = dossier if isinstance(dossier, dict) else dossier.to_dict()
    annotation = data.get("critic_decision") or data.get("annotation")
    if not isinstance(annotation, dict) or not annotation:
        return None
    payload = {"annotation": annotation,
               "needs_subcluster": bool(data.get("needs_subcluster", False))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def result_matches_dossier(result, dossier) -> bool:
    """Legacy results remain readable; compact decisions need a matching commit."""
    if not result or not result.get("label"):
        return False
    data = dossier if isinstance(dossier, dict) else (dossier.to_dict() if dossier else {})
    if data.get("decision_mode") and not data.get("critic_decision") and result.get("provenance") != "manual_override":
        return False
    current = decision_fingerprint(data)
    committed = result.get("decision_fingerprint")
    return current == committed


def build_commit(dossier, *, label: str | None = None, justification: str = "",
                 manual_override_reason: str = "") -> dict:
    """Build a validated result; manual labels carry separate provenance."""
    accepted = supported_label(dossier)
    manual_override_reason = manual_override_reason.strip()
    label = (label or "").strip()
    if manual_override_reason:
        if not label:
            raise ValueError("A manual override requires an explicit --label")
    elif accepted is None:
        raise ValueError("No supported candidate to commit; new dossiers require a critic decision")
    elif label and label != accepted:
        raise ValueError("--label differs from the supported candidate; use --manual-override-reason for a manual correction")
    chosen = label or accepted
    leading = dossier.leading()
    annotation = getattr(dossier, "critic_decision", None) or getattr(dossier, "annotation", {})
    if not isinstance(annotation, dict):
        annotation = {}
    if not justification and annotation and not manual_override_reason:
        justification = annotation["reason"]
    elif not justification and hasattr(dossier, "apply_resolution") and accepted and not manual_override_reason:
        justification = dossier.axes[dossier.committed_granularity]["note"]
    elif not justification and leading is not None and not manual_override_reason:
        justification = " ".join((leading.evidence_for or [])[:3])
    confidence = "unknown"
    if not manual_override_reason:
        if annotation:
            confidence = "high" if annotation["status"] == "resolved" else "uncertain"
        elif hasattr(dossier, "apply_resolution") and accepted:
            confidence = "high"
        elif leading:
            confidence = leading.confidence
    result = {
        "cluster_id": dossier.cluster_id,
        "label": chosen,
        "justification": justification or manual_override_reason,
        "confidence": confidence,
        "needs_subcluster": bool(getattr(dossier, "needs_subcluster", False)),
        "granularity": getattr(dossier, "committed_granularity", "") or "",
        "open_questions": list(dossier.open_questions or []),
        "n_evidence": len(dossier.evidence_log or []),
        "provenance": "manual_override" if manual_override_reason else "dossier",
        "supported_traits": list(getattr(dossier, "traits", [])),
    }
    fingerprint = decision_fingerprint(dossier)
    if annotation.get("major_region") and not manual_override_reason:
        result["major_region"] = annotation["major_region"]
    if fingerprint:
        result["decision_fingerprint"] = fingerprint
    if manual_override_reason:
        result["manual_override_reason"] = manual_override_reason
        result["dossier_label"] = accepted
    elif annotation:
        result.update(assessment_status=annotation["status"],
                      uncertainty=annotation.get("uncertainty", ""),
                      alternatives=annotation.get("alternatives", []))
        if annotation.get("output_origin"):
            result["output_origin"] = annotation["output_origin"]
            if annotation["output_origin"] in {"failure_record", "previous_critic"}:
                result["execution_status"] = "inference_failed"
    return result
