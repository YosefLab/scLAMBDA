"""Measurement provenance and mechanical checks, without cell-type marker rules.

Biological specificity remains the critic's responsibility. These checks only
establish that cited data exist and that a changed decision explains its revision.
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy


def number(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def record(genes, *, target, reference, method, ranked=False):
    rows = []
    for gene in genes:
        rows.append({
            "gene": gene.name, "lfc": number(gene.lfc),
            "proportion": number(gene.frac_non_zero if ranked else gene.frac_non_zero_target),
            "significance": number(gene.pval_adj if ranked else gene.pval),
        })
    value = {"target": str(target), "reference": reference, "method": method,
             "significance_kind": "adjusted_p" if ranked else "raw_p", "rows": rows}
    value["id"] = "m_" + hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]
    return value


def render(value):
    lines = [f"Measurement {value['id']}; target={value['target']}; reference={value['reference']}; "
             f"method={value['method']}; significance={value['significance_kind']}",
             "| Gene | LFC | Proportion expressing | Significance |",
             "|---|---|---|---|"]
    for row in value["rows"]:
        fmt = lambda v: "unavailable" if v is None else f"{v:.6g}"
        lines.append(f"| {row['gene']} | {fmt(row['lfc'])} | {fmt(row['proportion'])} | {fmt(row['significance'])} |")
    return "\n".join(lines)


class MeasurementText(str):
    """Keep the existing string tool interface, carrying its source record too."""
    def __new__(cls, value):
        obj = super().__new__(cls, render(value))
        obj.measurement = value
        return obj


def add_measurement(dossier, value):
    if not any(m["id"] == value["id"] for m in dossier.measurements):
        dossier.measurements.append(deepcopy(value))


def measured_genes(dossier):
    return {row["gene"] for m in dossier.measurements for row in m["rows"]}


def availability(dossier, panel):
    measured = measured_genes(dossier)
    # This partition comes from the actual input, never from model assertions.
    return {"measured": sorted(measured), "available_unmeasured": sorted(set(panel) - measured),
            "unavailable_requested": sorted(set(dossier.requested_genes) - set(panel))}


def availability_block(dossier, panel):
    # Do not send the entire transcriptome every turn. The tool checks every
    # requested symbol against the real panel and explicitly reports missing ones.
    value = availability(dossier, panel)
    return json.dumps({"measured": value["measured"],
                       "unavailable_requested": value["unavailable_requested"],
                       "available_unmeasured_count": len(value["available_unmeasured"]),
                       "query_policy": "Request any relevant genes; tools verify availability. Top-list omission is not absence."},
                      ensure_ascii=False)


def repair_annotation_refs(value, dossier):
    """Resolve misplaced genes only within an identical comparison context.

    Labels, explanations and measurements are never changed. Unknown IDs,
    unmeasured genes, different references or ambiguous sources are not repaired.
    The caller retains the original response and the returned repair log.
    """
    result = deepcopy(value)
    repairs = []
    if not isinstance(result, dict):
        return result, repairs
    tables = {m['id']: m for m in dossier.measurements}
    context = lambda m: tuple(m.get(k) for k in ('target', 'reference', 'method', 'significance_kind'))
    for field in ('support', 'counterevidence'):
        if not isinstance(result.get(field), list):
            continue
        updated = []
        for ref in result[field]:
            if not isinstance(ref, dict) or not isinstance(ref.get('measurement_id'), str):
                updated.append(ref); continue
            original = tables.get(ref['measurement_id'])
            genes = ref.get('genes')
            if original is None or not isinstance(genes, list) or not all(isinstance(g, str) for g in genes):
                updated.append(ref); continue
            groups, changes, valid = {}, [], True
            for gene in genes:
                mid = original['id']
                if gene not in {r['gene'] for r in original['rows']}:
                    matches = [m['id'] for m in tables.values() if context(m) == context(original)
                               and gene in {r['gene'] for r in m['rows']}]
                    if len(matches) != 1:
                        valid = False; break
                    mid = matches[0]
                    changes.append({'field': field, 'gene': gene, 'from': original['id'], 'to': mid})
                groups.setdefault(mid, []).append(gene)
            if valid and changes:
                updated.extend({**ref, 'measurement_id': mid, 'genes': genes} for mid, genes in groups.items())
                repairs.extend(changes)
            else:
                updated.append(ref)
        result[field] = updated
    return result, repairs


def validate_annotation(value, dossier):
    """Validate data references and output shape, without biological vetoes.

    A best-effort choice is an annotation with uncertainty, not a claim that
    identity has been established. Both statuses must cite actual measurements.
    Invalid references are returned for a bounded correction by the same critic.
    """
    if not isinstance(value, dict):
        return ["Return a complete JSON object"]
    issues = []
    label = value.get("label")
    if not isinstance(label, str) or label.strip().casefold() in {"", "unknown", "unresolved", "none", "null"}:
        issues.append("Choose a named best-supported cell identity, with status best_effort if uncertain")
    if value.get("status") not in {"resolved", "best_effort"}:
        issues.append("status must be resolved or best_effort")
    if not isinstance(value.get("reason"), str) or not value["reason"].strip():
        issues.append("Explain the evidence supporting the choice")
    for field in ("uncertainty", "next_question"):
        if not isinstance(value.get(field), str):
            issues.append(f"{field} must be a string (empty when none)")
    if value.get("status") == "best_effort" and not str(value.get("uncertainty") or "").strip():
        issues.append("Describe the uncertainty of the best-effort choice")
    # Alternatives are explanatory output, not an acceptance condition. Models
    # can provide names or structured label/reason entries without another call.
    if not isinstance(value.get("alternatives"), list):
        issues.append("alternatives must be a list")
    known = {m["id"]: {r["gene"] for r in m["rows"]} for m in dossier.measurements}
    for field in ("support", "counterevidence"):
        refs = value.get(field)
        if not isinstance(refs, list) or (field == "support" and not refs):
            issues.append(f"{field} must be a list of real measurement references" )
            continue
        for ref in refs:
            if not isinstance(ref, dict):
                issues.append(f"Invalid {field} reference")
                continue
            genes, mid = ref.get("genes"), ref.get("measurement_id")
            if (not isinstance(mid, str) or not isinstance(genes, list) or not genes
                    or not all(isinstance(g, str) for g in genes)
                    or not set(genes) <= known.get(mid, set())):
                issues.append(f"Invalid {field} reference: {ref!r}; copy the ID of a table containing those genes")
    return issues


def evidence_fingerprint(dossier):
    return tuple(m["id"] for m in dossier.measurements)


def validate_verdict(verdict, dossier, *, previous_measurements=None):
    """Reject unsupported citations/omitted reconciliation; never judge biology.

    Returns a copy and explicit problems. The controller retains these questions
    rather than replacing them with a generic request about a deeper rung.
    """
    verdict = deepcopy(verdict)
    known = {m["id"]: {r["gene"] for r in m["rows"]} for m in dossier.measurements}
    problems = []

    def check(value, name, previous):
        if not isinstance(value, dict) or value.get("resolved") is not True:
            return
        issues = []
        refs = value.get("measurement_refs")
        if not isinstance(refs, list) or not refs:
            issues.append("cite measured supporting genes using measurement_refs")
        else:
            for ref in refs:
                if (not isinstance(ref, dict) or not isinstance(ref.get("genes"), list)
                        or not ref["genes"] or not all(isinstance(g, str) for g in ref["genes"])
                        or not isinstance(ref.get("measurement_id"), str)
                        or not set(ref["genes"]) <= known.get(ref.get("measurement_id"), set())):
                    issues.append("support cites unknown measurements or unmeasured genes")
                    break
        counter = value.get("counterevidence")
        if not isinstance(counter, list):
            issues.append("explicitly assess the strongest measured counterevidence (empty list only if none)")
        else:
            for ref in counter:
                if (not isinstance(ref, dict) or not isinstance(ref.get("genes"), list)
                        or not ref["genes"] or not all(isinstance(g, str) for g in ref["genes"])
                        or not isinstance(ref.get("measurement_id"), str)
                        or not set(ref["genes"]) <= known.get(ref.get("measurement_id"), set())
                        or not str(ref.get("interpretation") or "").strip()):
                    issues.append("counterevidence needs real measurement references and interpretation")
                    break
        if not str(value.get("reconciliation") or "").strip():
            issues.append("explain why the counterevidence does or does not prevent resolution, or why none applies")
        changed = previous and (not previous.get("resolved") or previous.get("label") != value.get("label"))
        if changed and previous_measurements == evidence_fingerprint(dossier) and not str(value.get("revision_reason") or "").strip():
            issues.append("no new measurement: explain which previous objection/interpretation was wrong before changing the verdict")
        if issues:
            value["resolved"] = False
            value["evidence_status"] = "inconclusive"
            problems.append((name, "; ".join(issues)))

    for rung, value in (verdict.get("axes") or {}).items():
        check(value, rung, dossier.axes.get(rung))
    for value in verdict.get("traits") or []:
        if isinstance(value, dict):
            previous = next((t for t in dossier.traits if t.get("label") == value.get("label")), None)
            check(value, "trait " + str(value.get("label", "")), previous)
    return verdict, problems


def assessment_status(state):
    """Execution success is deliberately separate from annotation assessment."""
    reason = state.get("dossier_stop_reason", "incomplete")
    if "failed" in reason or "truncated" in reason or "malformed" in reason:
        return "inference_failed"
    annotation = (state.get("ct_dossier") or {}).get("annotation") or {}
    if annotation.get("status") in {"resolved", "best_effort"}:
        return annotation["status"]
    if not state.get("dossier_review_complete"):
        return "review_incomplete"
    axes = (state.get("ct_dossier") or {}).get("axes") or {}
    return "resolved" if axes.get("cell_type", {}).get("resolved") else "biologically_unresolved"
