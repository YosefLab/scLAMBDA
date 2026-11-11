"""
CellTypeDossier — persistent per-cluster context object for cell typing.

A cell-typing port of the spatial NicheDossier. One dossier is created per
cluster and threaded through the reasoning steps (generation → marker
refinement → reevaluation → cross-cluster), so the model accumulates a
characterization of the cluster instead of re-deriving it each prompt:

- what the cluster **IS** (confirmed traits, e.g. "expresses CD3E, CD8A")
- what it **IS NOT** (ruled-out identities with reasons, e.g. "not B cell —
  CD19/MS4A1 absent")
- a lifecycle-managed pool of candidate **cell types**
  (``candidate → supported | refuted | retired``), deduped by canonical name,
  with marker evidence for/against, prior/likelihood, and source.

Sub-states (activation/cycling/exhaustion) are kept as ``states`` on a single
cell-type candidate, NOT as separate competing candidates — this avoids the
over-splitting failure mode. At most one candidate is the provisional supported leader.

Candidates enter from: ``catalog`` (tissue cell-type ontology seed),
``data`` (DE/marker generation), ``signature`` (targeted marker test),
``subtype`` (finer label under a supported lineage), ``neighbor``
(cross-cluster comparison).

Plain-data and JSON-serializable so it lives in the LangGraph state dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
from typing import Any


_STATE_RANK = {"supported": 3, "candidate": 2, "refuted": 1, "retired": 0}
_LEVEL_RANK = {"very high": 5, "high": 4, "medium": 3, "low": 2, "very low": 1, "unknown": 0}
_VALID_STATES = ("candidate", "supported", "refuted", "retired")
_VALID_SOURCES = ("catalog", "data", "neighbor", "signature", "subtype", "agent")

# The identity-granularity ladder, coarse -> fine. Identity is resolved one rung
# at a time; the dossier commits at the finest rung the evidence supports.
#   lineage   — coarse identity (e.g. lymphoid -> T cell)
#   cell_type — fine identity (e.g. CD8+ T cell)
#   state     — subtype / activation / cycling (the Candidate.states field)
RUNGS = ("lineage", "cell_type", "state")


def _has_defining_evidence(axis: dict) -> bool:
    """Validate the critic's explicit assessment, not biological specificity."""
    evidence = axis.get("defining_evidence")
    return (
        axis.get("evidence_status") == "established"
        and isinstance(evidence, list) and bool(evidence)
        and all(isinstance(item, str) and item.strip() for item in evidence)
        and isinstance(axis.get("specificity"), str) and bool(axis["specificity"].strip())
    )


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _base(text: str) -> str:
    s = str(text or "")
    cut = s.find("(")
    if cut != -1:
        s = s[:cut]
    return _norm(s)


def _matches(key: str, claim: str) -> bool:
    a, b = _norm(key), _base(claim)
    if not a or not b:
        return False
    return a == b or a in b or b in a


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        k = _norm(it)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(str(it).strip())
    return out


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


@dataclass
class Candidate:
    """One candidate cell type. ``cell_type`` is the canonical identity (dedup
    key); activation/sub-states are ``states`` (modifiers), not new candidates."""

    cell_type: str
    states: list[str] = field(default_factory=list)
    state: str = "candidate"   # candidate | supported | refuted | retired
    source: str = "data"       # catalog | data | neighbor | signature | subtype | agent
    prior: str = "unknown"
    likelihood: str = "unknown"
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    lineage: str = ""          # coarse parent identity (the lineage rung), e.g. "T cell"

    candidate_id: str = ""

    def __post_init__(self):
        if not self.candidate_id:
            self.candidate_id = "ct_" + hashlib.sha256(self.key.encode()).hexdigest()[:16]

    MAX_STATES = 2

    @property
    def key(self) -> str:
        return _norm(self.cell_type)

    @property
    def label(self) -> str:
        if self.states:
            return f"{self.cell_type} ({', '.join(self.states[: self.MAX_STATES])})"
        return self.cell_type

    def label_at(self, granularity: str = "") -> str:
        """Project this candidate's name onto a rung of the granularity ladder.

        The supported candidate's specific ``cell_type`` name is ALWAYS written —
        granularity only controls whether confirmed sub-states are appended. We do
        NOT substitute the generic lineage parent for a named, marker-backed call
        (that dropped specificity, e.g. "plasmacytoid dendritic cell" -> generic
        "dendritic cell" when committed only at the lineage rung). A genuinely
        coarse commit happens by the agent choosing a coarse ``cell_type`` as the
        leader, not by demoting a specific one here. The rung still governs
        ``needs_subcluster`` (whether to refine further), just not the name.
        """
        if granularity == "state":
            return self.label  # cell_type + confirmed sub-states
        return self.cell_type  # specific name, sub-states withheld until confirmed

    @property
    def rank(self) -> tuple[int, int, int]:
        return (
            _STATE_RANK.get(self.state, 0),
            _LEVEL_RANK.get(str(self.likelihood).lower(), 0),
            _LEVEL_RANK.get(str(self.prior).lower(), 0),
        )

    @property
    def confidence(self) -> str:
        if self.state == "supported":
            return "high"
        if self.state in ("refuted", "retired"):
            return "low"
        lvl = _LEVEL_RANK.get(str(self.likelihood).lower(), 0)
        return "high" if lvl >= 4 else "medium" if lvl >= 3 else "low"

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate | None":
        name = str(
            d.get("cell_type") or d.get("label") or d.get("annotation") or d.get("compartment") or ""
        ).strip()
        if not name:
            return None
        st = str(d.get("state", "candidate")).strip().lower()
        if st not in _VALID_STATES:
            st = "candidate"
        src = str(d.get("source", "data")).strip().lower()
        if src not in _VALID_SOURCES:
            src = "data"
        # states must be short string qualifiers; models sometimes emit rival
        # sub-states as dict objects — those belong as candidates, not in the
        # label, so drop non-strings rather than stringifying them.
        raw_states = _as_list(d.get("states")) or _as_list(d.get("modifiers"))
        states = [s for s in raw_states if isinstance(s, str) and s.strip()]
        return cls(
            cell_type=name,
            candidate_id=str(d.get("candidate_id") or ""),
            states=_dedup_keep_order(states)[: cls.MAX_STATES],
            state=st,
            source=src,
            prior=str(d.get("prior", "unknown")).strip().lower() or "unknown",
            likelihood=str(d.get("likelihood", "unknown")).strip().lower() or "unknown",
            evidence_for=_dedup_keep_order(_as_list(d.get("evidence_for"))),
            evidence_against=_dedup_keep_order(_as_list(d.get("evidence_against"))),
            lineage=str(d.get("lineage", "")).strip(),
        )

    def merge(self, other: "Candidate") -> None:
        if other.states:  # snapshot, not cumulative
            self.states = _dedup_keep_order(other.states)[: self.MAX_STATES]
        if other.state != "candidate":
            self.state = other.state
        if other.prior not in ("", "unknown"):
            self.prior = other.prior
        if other.likelihood not in ("", "unknown"):
            self.likelihood = other.likelihood
        if other.lineage:  # keep the coarse parent once known
            self.lineage = other.lineage
        self.evidence_for = _dedup_keep_order([*self.evidence_for, *other.evidence_for])
        self.evidence_against = _dedup_keep_order([*self.evidence_against, *other.evidence_against])


@dataclass
class CellTypeDossier:
    cluster_id: str = ""
    iteration: int = 0
    description: str = ""
    confirmed: list[str] = field(default_factory=list)   # IS
    ruled_out: list[dict] = field(default_factory=list)   # IS NOT [{claim, reason}]
    candidates: list[Candidate] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    decision: str = "needs_refinement"   # confident | needs_refinement
    best_cell_type: str | None = None

    # ---------- hierarchical / agentic refinement state ----------
    # Critic-owned resolution snapshot. Each rung has resolved, label and note.
    # Belief updates cannot edit it. A deeper rung cannot remain resolved while
    # a shallower one is open (enforced in apply_resolution).
    axes: dict = field(default_factory=dict)
    # Finest rung the evidence supports == the longest resolved prefix of RUNGS.
    # The annotation is emitted at this granularity (see cell_type / to_hypotheses).
    committed_granularity: str = ""
    # Set when a finer rung is genuinely unresolvable from this cluster's data
    # (the critic's irreducible-ambiguity verdict) -> hand off to subclustering.
    needs_subcluster: bool = False
    # Append-only log of MEASURED facts (DE results, signature tests, etc.).
    # Never overwritten — belief is re-derived from it each agent step.
    evidence_log: list[dict] = field(default_factory=list)
    # Source data for tables and citation checks; model updates cannot write it.
    measurements: list[dict] = field(default_factory=list)
    requested_genes: list[str] = field(default_factory=list)
    # Critic-owned, independent observations (e.g. isotype or cycling). They do
    # not require resolving a finer identity and never establish that identity.
    traits: list[dict] = field(default_factory=list)
    # Compact critic-owned decision for the simplified loop. Older axes remain
    # readable for archived dossiers/CLI compatibility, but are not acceptance
    # gates for this decision. Best-effort labels retain explicit uncertainty.
    annotation: dict = field(default_factory=dict)

    def add_evidence(self, source: str, fact: str) -> None:
        """Append one measured fact to the immutable evidence log."""
        fact = str(fact).strip()
        if not fact:
            return
        self.evidence_log.append({"round": self.iteration, "source": str(source), "fact": fact})

    # ---------- pool management ----------

    def get(self, name: str) -> Candidate | None:
        k = _norm(name)
        for c in self.candidates:
            if c.key == k or c.candidate_id == name:
                return c
        return None

    def match(self, name: str) -> Candidate | None:
        """Resolve a (possibly qualifier-laden) name to a candidate.

        Models often return ``best_cell_type`` as the type + a qualifier
        concatenated (e.g. "CD8+ T cell, effector memory"), which exact
        ``get()`` misses — leaving the leader unpromoted, several candidates
        ``supported`` at once, and the label/justification drifting apart.
        Fall back to a base-name match (longest/most specific wins).
        """
        exact = self.get(name)
        if exact is not None:
            return exact
        cands = [c for c in self.candidates if _matches(c.key, name)]
        return max(cands, key=lambda c: len(c.key)) if cands else None

    def upsert(self, cand: Candidate) -> None:
        existing = self.get(cand.cell_type)
        if existing is None:
            self.candidates.append(cand)
        else:
            existing.merge(cand)

    def add_candidates(self, dicts: list[dict], default_source: str | None = None) -> int:
        added = 0
        for d in dicts or []:
            cand = Candidate.from_dict(d)
            if cand is None:
                continue
            if default_source and not (d or {}).get("source"):
                cand.source = default_source
            if self.get(cand.cell_type) is None:
                added += 1
            self.upsert(cand)
        return added

    def active(self) -> list[Candidate]:
        live = [c for c in self.candidates if c.state in ("candidate", "supported")]
        return sorted(live, key=lambda c: c.rank, reverse=True)

    def ranked(self) -> list[Candidate]:
        return sorted(self.candidates, key=lambda c: c.rank, reverse=True)

    def leading(self) -> Candidate | None:
        live = self.active()
        return live[0] if live else None

    def prune(self, max_active: int = 8) -> None:
        """Keep untested candidates, regardless of prior or catalogue size.

        Retain the method for old callers and recover candidates retired solely
        by the legacy pool cap. Evidence-based exclusions remain untouched.
        """
        cap_note = "retired: exceeded active-pool cap"
        for c in self.candidates:
            if c.state == "retired" and c.evidence_against == [cap_note]:
                c.state = "candidate"
                c.evidence_against = []

    # ---------- update contract ----------

    def apply_update(self, update: dict, default_source: str | None = None) -> "CellTypeDossier":
        """Patch provisional beliefs; supplied lists are replacement snapshots.

        Missing fields preserve state. Explicit null clears the preferred label.
        No provisional update can change the critic-owned accepted resolution.
        """
        if not isinstance(update, dict):
            return self
        if "description" in update:
            self.description = str(update["description"] or "").strip()
        for field_name in ("confirmed", "open_questions"):
            if field_name in update:
                setattr(self, field_name, _dedup_keep_order(_as_list(update[field_name])))
        if "ruled_out" in update:
            self.ruled_out = [dict(r) for r in _as_list(update["ruled_out"]) if isinstance(r, dict)]

        explicitly_supported = []
        revived = set()
        for item in _as_list(update.get("candidates")):
            if not isinstance(item, dict):
                continue
            cid = item.get("candidate_id")
            name = str(item.get("cell_type") or item.get("label") or "").strip()
            existing = self.get(str(cid)) if cid else self.get(name)
            if cid and existing is None:
                self.add_evidence("system", f"Unknown candidate_id {cid}; candidate update ignored.")
                continue
            if existing is None:
                existing = Candidate.from_dict(item)
                if existing is None:
                    continue
                if default_source and not item.get("source"):
                    existing.source = default_source
                self.candidates.append(existing)
            else:
                # IDs identify existing candidates even when the explanatory
                # label changes. Preserve the canonical name and ID.
                for field_name in ("states", "evidence_for", "evidence_against"):
                    if field_name in item:
                        values = _dedup_keep_order(_as_list(item[field_name]))
                        setattr(existing, field_name, values[:Candidate.MAX_STATES] if field_name == "states" else values)
                for field_name in ("prior", "likelihood", "lineage"):
                    if field_name in item:
                        setattr(existing, field_name, str(item[field_name] or "unknown"))
                if item.get("state") in _VALID_STATES:
                    existing.state = item["state"]
            if item.get("state") in ("candidate", "supported"):
                revived.add(existing.candidate_id)
            if item.get("state") == "supported":
                explicitly_supported.append(existing)

        # Exclusions target stable IDs (or exact legacy canonical names), never
        # a broad substring that could accidentally exclude an entire lineage.
        exclusions = []
        for exclusion in self.ruled_out:
            cid = exclusion.get("candidate_id")
            target = self.get(str(cid)) if cid else self.get(str(exclusion.get("claim") or ""))
            if target is not None:
                if target.candidate_id in revived:
                    continue
                exclusion["candidate_id"] = target.candidate_id
                target.state = "refuted"
            exclusions.append(exclusion)
        self.ruled_out = exclusions

        best_present = "best_cell_type" in update or "best_annotation" in update
        best = update.get("best_cell_type") if "best_cell_type" in update else update.get("best_annotation")
        chosen = None
        if best_present:
            if best:
                chosen = self.get(str(best))
                if chosen is None:
                    self.add_evidence("system", f"Unknown preferred candidate {best}; preference cleared.")
            # Explicit null wins over candidate-level supported flags.
        elif explicitly_supported:
            chosen = explicitly_supported[-1]
        else:
            chosen = self.get(self.best_cell_type) if self.best_cell_type else None
            if chosen is None:
                chosen = next((c for c in self.candidates if c.state == "supported"), None)
            if chosen and chosen.state != "supported":
                chosen = None
        if chosen and chosen.state in ("refuted", "retired"):
            chosen = None
        for candidate in self.candidates:
            if candidate is chosen:
                candidate.state = "supported"
            elif candidate.state == "supported":
                candidate.state = "candidate"
        self.best_cell_type = chosen.cell_type if chosen else None
        self._sync_granularity()
        self.prune()
        return self

    def reopen_hypotheses(self, rung: str = "lineage") -> None:
        """Discard invalidated interpretations, preserving raw evidence and axes."""
        self.description = ""
        self.confirmed = []
        self.open_questions = []
        if rung == "state":
            # A rejected qualifier does not invalidate the accepted identity.
            for candidate in self.candidates:
                candidate.states = []
            return
        self.ruled_out = []
        self.best_cell_type = None
        for candidate in self.candidates:
            candidate.state = "candidate"
            candidate.likelihood = "unknown"
            candidate.states = []
            candidate.evidence_for = []
            candidate.evidence_against = []

    def apply_resolution(self, axes: dict | None, needs_subcluster: bool | None = None,
                         traits: list[dict] | None = None) -> None:
        """Accept a full critic snapshot; never infer a label from actor beliefs.

        Every resolved rung needs an established defining-evidence assessment,
        cited measurements, specificity, label and justification. Reopening a
        rung clears all deeper resolutions, including those from earlier turns.
        Legacy axes without labels remain unconfirmed until reviewed.
        """
        if isinstance(needs_subcluster, bool):
            self.needs_subcluster = needs_subcluster
        if isinstance(traits, list):
            self.traits = [dict(t) for t in traits if isinstance(t, dict)
                           and t.get("resolved") is True and t.get("label") and t.get("note")
                           and _has_defining_evidence(t)]
        if axes is None:
            return  # The critic may update only the split flag through the CLI.
        self.axes = {}
        for rung in RUNGS:
            value = axes.get(rung, {})
            value = value if isinstance(value, dict) else {}
            label = str(value.get("label") or "").strip()
            note = str(value.get("note") or "").strip()
            self.axes[rung] = {
                "resolved": value.get("resolved") is True and bool(label and note) and _has_defining_evidence(value),
                "label": label,
                "note": note,
                "evidence_status": value.get("evidence_status") if value.get("evidence_status") in
                    {"established", "inconclusive", "untested", "unavailable"} else "untested",
                "defining_evidence": value.get("defining_evidence") if isinstance(value.get("defining_evidence"), list) else [],
                "specificity": value.get("specificity") if isinstance(value.get("specificity"), str) else "",
                **{k: value.get(k) for k in ("measurement_refs", "counterevidence", "reconciliation", "revision_reason")},
            }
        # Models may supply only the state qualifier. It modifies the accepted
        # cell type; it must never replace the identity with e.g. "activated".
        state_label = self.axes["state"]["label"]
        type_label = self.axes["cell_type"]["label"]
        if state_label and type_label and not _norm(state_label).startswith(_norm(type_label)):
            self.axes["state"]["label"] = f"{type_label} ({state_label})"
        self._sync_granularity()
        self.decision = "confident" if self.committed_granularity == "state" else "needs_refinement"

    def _sync_granularity(self) -> None:
        """Enforce top-down ladder ordering and recompute committed_granularity.

        A deeper rung cannot count as resolved while a shallower one is open, so
        the committed granularity is the longest *resolved prefix* of RUNGS.
        Demotes any deeper rung the model marked resolved out of order.
        """
        open_seen = False
        committed = ""
        for ax in RUNGS:
            a = self.axes.get(ax)
            resolved = isinstance(a, dict) and a.get("resolved") is True and bool(a.get("label") and a.get("note")) and _has_defining_evidence(a)
            if isinstance(a, dict):
                a["resolved"] = resolved
            if open_seen and resolved:
                a["resolved"] = False  # parent rung not closed -> demote
                resolved = False
            if resolved:
                committed = ax
            else:
                open_seen = True
        self.committed_granularity = committed

    # ---------- rendering ----------

    def to_markdown(self, *, include_evidence: bool = True) -> str:
        lines = [f"### Cluster dossier (carried forward — update {self.iteration})", ""]
        legacy = self.candidates or self.axes or self.confirmed or self.ruled_out or self.traits
        if self.annotation or not legacy:
            from .measurements import render
            a = self.annotation
            if a:
                lines += [f"**Critic decision:** {a['label']} ({a['status']})", "", a['reason'],
                          f"Uncertainty: {a.get('uncertainty') or '(none)'}",
                          f"Alternatives: {a.get('alternatives', [])}",
                          f"Next question: {a.get('next_question') or '(none)'}",
                          f"Needs subcluster: {self.needs_subcluster}"]
            else:
                lines.append("**Critic decision:** (none yet)")
            if include_evidence:
                lines.extend(render(m) for m in self.measurements)
                if self.evidence_log:
                    lines.append("Notes: " + "; ".join(
                        f"[{e.get('source', '?')}] {e.get('fact', '')}" for e in self.evidence_log))
            return "\n\n".join(lines)
        lines.append(f"**Working description:** {self.description or '(not yet characterized)'}")
        lines.append("")
        lines.append("**Working claims (generation/actor interpretations; not critic-confirmed):**")
        lines += [f"- {t}" for t in self.confirmed] or ["- (nothing confirmed yet)"]
        lines.append("")
        lines.append("**Proposed exclusions (reconsider when evidence is insufficient):**")
        if self.ruled_out:
            lines += [
                f"- {r.get('claim')}" + (f" — {r.get('reason')}" if r.get("reason") else "")
                for r in self.ruled_out
            ]
        else:
            lines.append("- (nothing ruled out yet)")
        lines.append("")
        lines.append("**Candidate cell types (supported = working preference, not resolution):**")
        ranked = self.ranked()
        if ranked:
            lines.append("| Cell type | Candidate ID | State | Prior/Lik | Source | Evidence for | Evidence against |")
            lines.append("|---|---|---|---|---|---|---|")
            for c in ranked:
                lines.append(
                    f"| {c.label} | {c.candidate_id} | {c.state} | {c.prior}/{c.likelihood} | {c.source} "
                    f"| {'; '.join(c.evidence_for) or '—'} "
                    f"| {'; '.join(c.evidence_against) or '—'} |"
                )
        else:
            lines.append("(no candidates yet)")
        lines.append("")

        if self.axes:
            committed = self.committed_granularity or "(none yet)"
            lines.append(f"**Resolution ladder (coarse→fine; committed: {committed}):**")
            for ax in RUNGS:
                if ax in self.axes:
                    a = self.axes[ax] or {}
                    st = "RESOLVED" if a.get("resolved") else "OPEN"
                    note = ": ".join(filter(None, [a.get("label"), a.get("note")]))
                    lines.append(f"- {ax}: {st}" + (f" — {note}" if note else ""))
                    lines.append(f"  Defining evidence: {a.get('evidence_status', 'untested')}; "
                                 f"measurements: {'; '.join(str(x) for x in a.get('defining_evidence', [])) or 'none'}; "
                                 f"specificity: {a.get('specificity') or 'not established'}")
                    if a.get("counterevidence"):
                        lines.append(f"  Previous counterevidence: {a['counterevidence']}")
                    if a.get("reconciliation"):
                        lines.append(f"  Previous reconciliation: {a['reconciliation']}")
            if self.needs_subcluster:
                lines.append("- → finer rung unresolvable from this cluster; hand off to subclustering")
            lines.append("")

        if self.traits:
            lines.append("**Independently supported traits (do not establish finer identity):**")
            lines += [f"- {t['label']}: {t['note']}" for t in self.traits]
            lines.append("")

        if self.measurements and include_evidence:
            from .measurements import render
            lines.extend(render(m) for m in self.measurements)

        if self.evidence_log and include_evidence:
            lines.append("**Measured evidence so far (immutable — facts, not interpretation):**")
            lines += [f"- [{e.get('source', '?')}] {e.get('fact', '')}" for e in self.evidence_log]
            lines.append("")

        if self.open_questions:
            lines.append("**Open questions to resolve:**")
            lines += [f"- {q}" for q in self.open_questions]
        return "\n".join(lines).rstrip()

    # ---------- backward-compatible hypotheses view ----------

    def to_hypotheses(self) -> list[dict]:
        if self.annotation:
            a = self.annotation
            return [{"cell_type": a["label"], "confidence": "high" if a["status"] == "resolved" else "uncertain",
                     "justification": a["reason"], "uncertainty": a.get("uncertainty", ""),
                     "supporting_genes": sorted({g for r in a.get("support", []) for g in r["genes"]}),
                     "state": "supported" if a["status"] == "resolved" else "best_effort",
                     "source": "critic", "assessment_status": a["status"]}]
        # Put the critic's accepted label first, followed by working hypotheses.
        # A preferred candidate is never implicitly a resolved annotation.
        g = self.committed_granularity
        out: list[dict] = []
        for c in self.ranked():
            label = c.label
            out.append(
                {
                    "cell_type": label,
                    "confidence": c.confidence,
                    "justification": _build_justification(c),
                    "supporting_genes": [],
                    "state": "candidate" if c.state == "supported" else c.state,
                    "source": c.source,
                }
            )
        if self.cell_type:
            axis = self.axes[g]
            out = [h for h in out if h["cell_type"] != self.cell_type]
            out.insert(0, {
                "cell_type": self.cell_type, "confidence": "high",
                "justification": axis["note"], "supporting_genes": [],
                "state": "supported", "source": "critic",
            })
        return out

    @property
    def cell_type(self) -> str | None:
        if self.annotation:
            return self.annotation.get("label")
        label = None
        for rung in RUNGS:
            axis = self.axes.get(rung) or {}
            if not (axis.get("resolved") is True and axis.get("label") and axis.get("note") and _has_defining_evidence(axis)):
                break
            label = axis["label"]
        if label and self.traits:
            label += " (" + ", ".join(str(t["label"]) for t in self.traits) + ")"
        return label

    # ---------- (de)serialization ----------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "CellTypeDossier":
        if not d:
            return cls()
        cands = [Candidate.from_dict(c) for c in d.get("candidates", [])]
        dossier = cls(
            cluster_id=str(d.get("cluster_id", "")),
            iteration=int(d.get("iteration", 0) or 0),
            description=str(d.get("description", "")),
            confirmed=list(d.get("confirmed", []) or []),
            ruled_out=list(d.get("ruled_out", []) or []),
            candidates=[c for c in cands if c is not None],
            open_questions=list(d.get("open_questions", []) or []),
            decision=str(d.get("decision", "needs_refinement")),
            best_cell_type=d.get("best_cell_type"),
            axes=dict(d.get("axes", {}) or {}),
            committed_granularity=str(d.get("committed_granularity", "") or ""),
            needs_subcluster=bool(d.get("needs_subcluster", False)),
            evidence_log=list(d.get("evidence_log", []) or []),
            measurements=list(d.get("measurements", []) or []),
            requested_genes=list(d.get("requested_genes", []) or []),
            annotation=dict(d.get("annotation", {}) or {}),
            traits=[dict(t) for t in (d.get("traits") or []) if isinstance(t, dict)
                    and t.get("resolved") is True and t.get("label") and t.get("note")
                    and _has_defining_evidence(t)],
        )
        dossier._sync_granularity()
        dossier.prune()
        return dossier


def _build_justification(c: Candidate) -> str:
    parts = (
        [f"This cluster is supported as {c.label} by {'; '.join(c.evidence_for)}."]
        if c.evidence_for
        else [f"This cluster is proposed as {c.label}."]
    )
    if c.evidence_against:
        parts.append(f"Ambiguities/contradictions: {'; '.join(c.evidence_against)}.")
    return " ".join(parts)


def seed_ct_dossier(
    cluster_id: str,
    hypotheses: list[dict] | None = None,
    catalog_cell_types: list[dict] | None = None,
) -> CellTypeDossier:
    """Initial dossier: seed the candidate universe from the tissue cell-type
    catalog (source=catalog), then merge data-driven hypotheses (source=data)."""
    d = CellTypeDossier(cluster_id=cluster_id, iteration=0)
    for entry in catalog_cell_types or []:
        name = str(entry.get("cell_type", "")).strip()
        if not name:
            continue
        d.upsert(
            Candidate(
                cell_type=name,
                state="candidate",
                source="catalog",
                prior=str(entry.get("prior", "unknown")).strip().lower() or "unknown",
                lineage=str(entry.get("lineage", "")).strip(),
            )
        )
    d.add_candidates(hypotheses or [], default_source="data")
    d.prune()
    return d
