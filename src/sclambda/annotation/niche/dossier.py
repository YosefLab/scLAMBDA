"""
NicheDossier — the single persistent context object for spatial annotation.

One dossier is created per niche and threaded through every LLM step
(generation → refinement → reevaluation → regeneration → cross-niche
correction).  Each step receives the dossier rendered as markdown and returns
a JSON *update*; the dossier merges that update and carries forward.  This is
what lets the model accumulate a characterization of the niche — explicitly
tracking what the niche **is** (confirmed traits) and **is not** (ruled-out
claims with reasons) — rather than re-deriving everything from scratch at each
prompt.

The candidate pool is a lifecycle-managed set of anatomical compartments
(``candidate → supported | refuted | retired``).  Candidates enter from five
sources (Stage A–E):

    catalog   — seeded from the tissue anatomical catalog (Stage A)
    data      — proposed from composition/DE at generation (Stage B)
    neighbor  — injected from adjacent-niche labels at reconciliation (Stage C)
    signature — surfaced by an unexpected targeted-signature result (Stage D)
    subzone   — finer label added beneath a supported compartment (Stage E)

The dossier is plain-data and JSON-serializable so it can live in the
LangGraph state dict and be written to the prompt-history output.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# Ordering used to rank candidates and to map to the legacy confidence field.
_STATE_RANK = {"supported": 3, "candidate": 2, "refuted": 1, "retired": 0}
_LEVEL_RANK = {"very high": 5, "high": 4, "medium": 3, "low": 2, "very low": 1, "unknown": 0}
_VALID_STATES = ("candidate", "supported", "refuted", "retired")
_VALID_SOURCES = ("catalog", "data", "neighbor", "signature", "subzone")


def _norm(text: str) -> str:
    """Normalize a compartment name for identity matching (dedup key)."""
    return " ".join(str(text or "").strip().lower().split())


def _base(text: str) -> str:
    """Compartment identity without parenthetical modifiers, normalized."""
    s = str(text or "")
    cut = s.find("(")
    if cut != -1:
        s = s[:cut]
    return _norm(s)


def _compartment_matches(compartment_key: str, claim: str) -> bool:
    """Whether a (normalized) compartment refers to the same thing as a claim.

    Claims often carry modifiers (e.g. "Cerebellum, white matter (folial, arbor
    vitae)") while the candidate key is the bare compartment ("cerebellum, white
    matter"). Match on the modifier-stripped base, in either direction.
    """
    a, b = _norm(compartment_key), _base(claim)
    if not a or not b:
        return False
    return a == b or a in b or b in a


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        key = _norm(it)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(str(it).strip())
    return out


@dataclass
class Candidate:
    """One anatomical-compartment hypothesis in the pool.

    ``compartment`` is the canonical anatomical identity and serves as the
    dedup key; cosmetic variants (e.g. dark-zone vs light-zone of the same GC)
    are kept as ``modifiers`` rather than as separate candidates.
    """

    compartment: str
    modifiers: list[str] = field(default_factory=list)
    state: str = "candidate"  # candidate | supported | refuted | retired
    source: str = "data"      # catalog | data | neighbor | signature | subzone
    prior: str = "unknown"        # prior probability (tissue context alone)
    likelihood: str = "unknown"   # likelihood given the observed data
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    tested_signatures: list[str] = field(default_factory=list)
    # The cell type(s) from the niche composition whose presence *defines* this
    # compartment — i.e. the lineage that must dominate for this label to hold.
    # Used by the deterministic dominance gate to reject minority-promoted labels.
    defining_cell_types: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return _norm(self.compartment)

    # Cap on modifiers carried/shown, to keep labels readable. Modifiers are
    # anatomical sub-zone refinements only (see dossier_update_parse.md).
    MAX_MODIFIERS = 2

    @property
    def label(self) -> str:
        """Human-readable annotation: compartment plus a few modifiers."""
        if self.modifiers:
            return f"{self.compartment} ({', '.join(self.modifiers[: self.MAX_MODIFIERS])})"
        return self.compartment

    @property
    def rank(self) -> tuple[int, int, int]:
        """Sort key: state, then prior, then likelihood (all descending)."""
        return (
            _STATE_RANK.get(self.state, 0),
            _LEVEL_RANK.get(str(self.prior).lower(), 0),
            _LEVEL_RANK.get(str(self.likelihood).lower(), 0),
        )

    @property
    def confidence(self) -> str:
        """Map (state, likelihood) onto the legacy high/medium/low scale."""
        if self.state == "supported":
            return "high"
        if self.state in ("refuted", "retired"):
            return "low"
        lvl = _LEVEL_RANK.get(str(self.likelihood).lower(), 0)
        return "high" if lvl >= 4 else "medium" if lvl >= 3 else "low"

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate | None":
        compartment = str(d.get("compartment") or d.get("annotation") or "").strip()
        if not compartment:
            return None
        state = str(d.get("state", "candidate")).strip().lower()
        if state not in _VALID_STATES:
            state = "candidate"
        source = str(d.get("source", "data")).strip().lower()
        if source not in _VALID_SOURCES:
            source = "data"
        return cls(
            compartment=compartment,
            modifiers=_dedup_keep_order(_as_list(d.get("modifiers")))[: cls.MAX_MODIFIERS],
            state=state,
            source=source,
            prior=str(d.get("prior", "unknown")).strip().lower() or "unknown",
            likelihood=str(d.get("likelihood", "unknown")).strip().lower() or "unknown",
            evidence_for=_dedup_keep_order(_as_list(d.get("evidence_for"))),
            evidence_against=_dedup_keep_order(_as_list(d.get("evidence_against"))),
            tested_signatures=_dedup_keep_order(_as_list(d.get("tested_signatures"))),
            defining_cell_types=_dedup_keep_order(_as_list(d.get("defining_cell_types"))),
        )

    def merge(self, other: "Candidate") -> None:
        """Fold an updated view of the same compartment into this candidate."""
        # Modifiers are a *snapshot* of the current best sub-characterization,
        # not a cumulative history: the latest non-empty set replaces the prior
        # one (otherwise rephrasings pile up into a run-on label).
        if other.modifiers:
            self.modifiers = _dedup_keep_order(other.modifiers)[: self.MAX_MODIFIERS]
        # Defining cell types: latest non-empty declaration replaces (snapshot).
        if other.defining_cell_types:
            self.defining_cell_types = _dedup_keep_order(other.defining_cell_types)
        # State transitions: a later confident verdict (supported/refuted/retired)
        # overrides the prior `candidate`; otherwise keep the most decided state.
        if other.state != "candidate":
            self.state = other.state
        if other.prior not in ("", "unknown"):
            self.prior = other.prior
        if other.likelihood not in ("", "unknown"):
            self.likelihood = other.likelihood
        # Source is sticky to the earliest origin; do not overwrite.
        self.evidence_for = _dedup_keep_order([*self.evidence_for, *other.evidence_for])
        self.evidence_against = _dedup_keep_order([*self.evidence_against, *other.evidence_against])
        self.tested_signatures = _dedup_keep_order([*self.tested_signatures, *other.tested_signatures])


@dataclass
class NicheDossier:
    """Accumulated, structured characterization of one niche."""

    cluster_id: str = ""
    iteration: int = 0
    description: str = ""                       # running free-text characterization
    spatial_position: str = ""                  # image-read location: where it sits, what that excludes, and any unresolved positional ambiguity
    composition: list[dict] = field(default_factory=list)  # [{cell_type, proportion}] — niche cell-type fractions, for the dominance gate
    confirmed: list[str] = field(default_factory=list)        # what the niche IS
    ruled_out: list[dict] = field(default_factory=list)       # [{claim, reason}] — IS NOT
    candidates: list[Candidate] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    decision: str = "needs_refinement"          # confident | needs_refinement
    best_compartment: str | None = None
    # ---------- agentic refinement state ----------
    # Three resolution axes the niche must close before it is final. Each is a
    # dict like {"resolved": bool, "note": "..."}; the agent edits them freely.
    #   identity — is the dominant lineage / compartment type settled?
    #   markers  — are the leading label's DEFINING markers present (absolute)?
    #   position — for a position-defined label, is its location read off the image?
    axes: dict = field(default_factory=dict)
    # Append-only log of MEASURED facts (DE results, image position reads, etc.).
    # Never overwritten — belief is re-derived from it each agent step.
    evidence_log: list[dict] = field(default_factory=list)

    critic_decision: dict = field(default_factory=dict)
    measurements: list[dict] = field(default_factory=list)
    requested_genes: list[str] = field(default_factory=list)
    decision_mode: bool = True

    def add_evidence(self, source: str, fact: str) -> None:
        """Append one measured fact to the immutable evidence log."""
        fact = str(fact).strip()
        if not fact:
            return
        self.evidence_log.append({"round": self.iteration, "source": str(source), "fact": fact})

    # ---------- candidate-pool management ----------

    def get(self, compartment: str) -> Candidate | None:
        k = _norm(compartment)
        for c in self.candidates:
            if c.key == k:
                return c
        return None

    def match(self, name: str) -> Candidate | None:
        """Resolve a (possibly modifier-laden) name to a candidate.

        Models often return ``best_compartment`` as compartment + modifiers
        concatenated (e.g. "Lymphoid follicle, mantle zone, activated"), which
        an exact ``get()`` misses. Fall back to a base-name match so the leader
        is correctly resolved, demoted/promoted, and used for the annotation +
        its justification.
        """
        exact = self.get(name)
        if exact is not None:
            return exact
        # Prefer the longest base-name match (most specific compartment).
        cands = [c for c in self.candidates if _compartment_matches(c.key, name)]
        if not cands:
            return None
        return max(cands, key=lambda c: len(c.key))

    def upsert(self, cand: Candidate) -> None:
        """Insert a new candidate or merge into the existing one (by key)."""
        existing = self.get(cand.compartment)
        if existing is None:
            self.candidates.append(cand)
        else:
            existing.merge(cand)

    def add_candidates(self, dicts: list[dict], default_source: str | None = None) -> int:
        """Add/merge a batch of candidate dicts; returns count of *new* ones."""
        added = 0
        for d in dicts or []:
            cand = Candidate.from_dict(d)
            if cand is None:
                continue
            if default_source and not d.get("source"):
                cand.source = default_source
            if self.get(cand.compartment) is None:
                added += 1
            self.upsert(cand)
        return added

    def active(self) -> list[Candidate]:
        """Candidates still in play (not refuted/retired), best first."""
        live = [c for c in self.candidates if c.state in ("candidate", "supported")]
        return sorted(live, key=lambda c: c.rank, reverse=True)

    def ranked(self) -> list[Candidate]:
        """All candidates, best first (supported → candidate → refuted → retired)."""
        return sorted(self.candidates, key=lambda c: c.rank, reverse=True)

    def leading(self) -> Candidate | None:
        live = self.active()
        return live[0] if live else None

    def prune(self, max_active: int = 8) -> None:
        """Retire the weakest *candidate*-state entries beyond ``max_active``.

        Supported and refuted entries are never auto-pruned (refuted entries are
        the exclusion record). Only low-ranked open candidates are retired so
        the pool cannot grow without bound during extension.
        """
        open_cands = [c for c in self.candidates if c.state == "candidate"]
        if len(open_cands) <= max_active:
            return
        for c in sorted(open_cands, key=lambda c: c.rank, reverse=True)[max_active:]:
            c.state = "retired"
            if "exceeded active-pool cap" not in c.evidence_against:
                c.evidence_against.append("retired: exceeded active-pool cap")

    # ---------- update contract (consumed from each LLM step) ----------

    def apply_update(self, update: dict, default_source: str | None = None) -> "NicheDossier":
        """Merge a parsed LLM update dict into this dossier (in place).

        Recognized keys (all optional): ``description``, ``confirmed`` (IS),
        ``ruled_out`` (IS NOT — list of {claim, reason} or strings),
        ``candidates``, ``open_questions``, ``decision``, ``best_compartment``.
        """
        if not isinstance(update, dict):
            return self

        desc = str(update.get("description", "") or "").strip()
        if desc:
            self.description = desc

        for trait in _as_list(update.get("confirmed")):
            t = str(trait).strip()
            if t and _norm(t) not in {_norm(x) for x in self.confirmed}:
                self.confirmed.append(t)

        # Candidates first, so we know which compartments this step is actively
        # proposing (those must not be simultaneously ruled out — reinstatement).
        update_cands = _as_list(update.get("candidates"))
        self.add_candidates(update_cands, default_source=default_source)

        # Compartments this update treats as live (candidate/supported) or names
        # as best — these override any prior exclusion of the same compartment.
        promoted: set[str] = set()
        for d in update_cands:
            comp = str((d or {}).get("compartment") or (d or {}).get("annotation") or "").strip()
            state = str((d or {}).get("state", "candidate")).strip().lower()
            if comp and state not in ("refuted", "retired"):
                promoted.add(_norm(comp))
        best = str(update.get("best_compartment", "") or update.get("best_annotation", "") or "").strip()
        if best:
            promoted.add(_norm(best))

        for item in _as_list(update.get("ruled_out")):
            if isinstance(item, dict):
                claim = str(item.get("claim", "")).strip()
                reason = str(item.get("reason", "")).strip()
            else:
                claim, reason = str(item).strip(), ""
            if not claim:
                continue
            # Don't rule out something this same update is actively proposing.
            if any(_compartment_matches(p, claim) for p in promoted):
                continue
            existing = next((r for r in self.ruled_out if _norm(r.get("claim")) == _norm(claim)), None)
            if existing is None:
                self.ruled_out.append({"claim": claim, "reason": reason})
            elif reason:
                existing["reason"] = reason

        # Reinstatement: drop any prior exclusion overturned by this update.
        if promoted:
            self.ruled_out = [
                r for r in self.ruled_out
                if not any(_compartment_matches(p, r.get("claim", "")) for p in promoted)
            ]

        oq = _as_list(update.get("open_questions"))
        if oq:
            self.open_questions = _dedup_keep_order([str(x) for x in oq])

        decision = str(update.get("decision", "") or "").strip().lower()
        if decision in ("confident", "needs_refinement"):
            self.decision = decision

        # Resolution axes (agent-edited belief state). Merge per-axis so a step
        # can update one axis without clobbering the others.
        axes = update.get("axes")
        if isinstance(axes, dict):
            for ax, val in axes.items():
                if isinstance(val, dict):
                    cur = self.axes.get(ax, {}) if isinstance(self.axes.get(ax), dict) else {}
                    cur.update(val)
                    self.axes[ax] = cur

        if best and not self.decision_mode:
            # Promote the chosen compartment to `supported`, and keep a single
            # supported leader: demote any other `supported` back to `candidate`
            # so the pool doesn't accumulate several "supported" identities.
            # Match fuzzily — `best` often carries the modifiers too.
            chosen = self.match(best)
            self.best_compartment = chosen.compartment if chosen is not None else best
            if chosen is not None:
                chosen.state = "supported"
                for c in self.candidates:
                    if c is not chosen and c.state == "supported":
                        c.state = "candidate"

        # Synchronize ruled_out -> candidate state: a compartment that is ruled
        # out cannot remain a live/leading candidate (this is what let a
        # "supported" label survive after the analysis had excluded it).
        for r in self.ruled_out:
            claim = r.get("claim", "")
            for c in self.candidates:
                if c.state in ("candidate", "supported") and _compartment_matches(c.key, claim):
                    c.state = "refuted"
                    note = f"ruled out: {r.get('reason', '')}".strip().rstrip(":")
                    if note and note not in c.evidence_against:
                        c.evidence_against.append(note)

        self.prune()
        return self

    # ---------- rendering ----------

    def to_markdown(self) -> str:
        """Render current measurements and decision, or an archived legacy dossier."""
        if self.decision_mode:
            import json
            from .measurements import render
            return "\n\n".join([f"# Niche {self.cluster_id}",
                "Current critic decision (revisable): " + json.dumps(self.critic_decision),
                *[render(m) for m in self.measurements],
                "Tool notes: " + json.dumps(self.evidence_log)])
        lines: list[str] = []
        lines.append(f"### Niche dossier (carried forward — update iteration {self.iteration})")
        lines.append("")
        lines.append(f"**Working description:** {self.description or '(not yet characterized)'}")
        lines.append("")

        if self.spatial_position:
            lines.append("**Spatial position (read from the image — constrains *where* the niche can be):**")
            lines.append(self.spatial_position)
            lines.append("")

        lines.append("**What this niche IS (confirmed so far):**")
        if self.confirmed:
            lines += [f"- {t}" for t in self.confirmed]
        else:
            lines.append("- (nothing confirmed yet)")
        lines.append("")

        lines.append("**What this niche IS NOT (ruled out — do not re-propose):**")
        if self.ruled_out:
            lines += [
                f"- {r.get('claim')}" + (f" — {r.get('reason')}" if r.get("reason") else "")
                for r in self.ruled_out
            ]
        else:
            lines.append("- (nothing ruled out yet)")
        lines.append("")

        lines.append("**Candidate compartments:**")
        ranked = self.ranked()
        if ranked:
            lines.append("| Compartment | State | Prior×Lik | Source | Evidence for | Evidence against |")
            lines.append("|---|---|---|---|---|---|")
            for c in ranked:
                lines.append(
                    f"| {c.label} | {c.state} | {c.prior}/{c.likelihood} | {c.source} "
                    f"| {'; '.join(c.evidence_for[:4]) or '—'} "
                    f"| {'; '.join(c.evidence_against[:4]) or '—'} |"
                )
        else:
            lines.append("(no candidates yet)")
        lines.append("")

        if self.axes:
            lines.append("**Resolution axes (close all to finalize):**")
            for ax in ("identity", "markers", "position"):
                if ax in self.axes:
                    a = self.axes[ax] or {}
                    state = "RESOLVED" if a.get("resolved") else "OPEN"
                    note = a.get("note") or ""
                    lines.append(f"- {ax}: {state}" + (f" — {note}" if note else ""))
            lines.append("")

        if self.evidence_log:
            lines.append("**Measured evidence so far (immutable — facts, not interpretation):**")
            for e in self.evidence_log:
                lines.append(f"- [{e.get('source', '?')}] {e.get('fact', '')}")
            lines.append("")

        if self.open_questions:
            lines.append("**Open questions to resolve:**")
            lines += [f"- {q}" for q in self.open_questions]
            lines.append("")

        return "\n".join(lines).rstrip()

    # ---------- backward-compatible views ----------

    def to_hypotheses(self) -> list[dict]:
        """Project the pool onto the legacy hypotheses list.

        Active candidates first (best ranked), then refuted/retired ones at low
        confidence so the exclusion record survives downstream consumers that
        still read ``hypotheses``/``updated_hypotheses``.
        """
        out: list[dict] = []
        for c in self.ranked():
            out.append(
                {
                    "annotation": c.label,
                    "confidence": c.confidence,
                    "justification": _build_justification(c),
                    "state": c.state,
                    "source": c.source,
                }
            )
        return out

    @property
    def annotation(self) -> str | None:
        if self.decision_mode:
            return self.critic_decision.get("label")
        if self.best_compartment:
            chosen = self.match(self.best_compartment)
            if chosen is not None:
                return chosen.label
            return self.best_compartment
        lead = self.leading()
        return lead.label if lead else None

    # ---------- (de)serialization for the state dict ----------

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> "NicheDossier":
        if not d:
            return cls()
        cands = [Candidate.from_dict(c) for c in d.get("candidates", [])]
        return cls(
            critic_decision=dict(d.get("critic_decision", {}) or {}),
            measurements=list(d.get("measurements", []) or []),
            requested_genes=list(d.get("requested_genes", []) or []),
            decision_mode=bool(d.get("decision_mode", False)),
            cluster_id=str(d.get("cluster_id", "")),
            iteration=int(d.get("iteration", 0) or 0),
            description=str(d.get("description", "")),
            spatial_position=str(d.get("spatial_position", "") or ""),
            composition=list(d.get("composition", []) or []),
            confirmed=list(d.get("confirmed", []) or []),
            ruled_out=list(d.get("ruled_out", []) or []),
            candidates=[c for c in cands if c is not None],
            open_questions=list(d.get("open_questions", []) or []),
            decision=str(d.get("decision", "needs_refinement")),
            best_compartment=d.get("best_compartment"),
            axes=dict(d.get("axes", {}) or {}),
            evidence_log=list(d.get("evidence_log", []) or []),
        )


# ---------- helpers ----------


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _build_justification(c: Candidate) -> str:
    """Compose a justification string from a candidate's evidence ledger."""
    parts = [f"This spatial region is supported by {'; '.join(c.evidence_for)}."] if c.evidence_for else [
        f"This spatial region is proposed as {c.label}."
    ]
    if c.evidence_against:
        parts.append(f"Ambiguities/contradictions: {'; '.join(c.evidence_against)}.")
    if c.tested_signatures:
        parts.append(f"Signatures tested: {', '.join(c.tested_signatures)}.")
    return " ".join(parts)


def seed_dossier(
    cluster_id: str,
    hypotheses: list[dict] | None = None,
    catalog_compartments: list[dict] | None = None,
    spatial_position: str = "",
    composition: list[dict] | None = None,
) -> NicheDossier:
    """Build the initial dossier at generation time (Stages A + B).

    ``catalog_compartments`` seed the candidate universe (source=``catalog``);
    data-driven ``hypotheses`` from the generation step are merged on top
    (source=``data``), overriding priors with observed likelihood.
    """
    dossier = NicheDossier(
        cluster_id=cluster_id,
        iteration=0,
        spatial_position=spatial_position or "",
        composition=[
            {"cell_type": str(p.get("cell_type", "")), "proportion": float(p.get("proportion", 0) or 0)}
            for p in (composition or [])
            if str(p.get("cell_type", "")).strip()
        ],
    )

    for entry in catalog_compartments or []:
        comp = str(entry.get("compartment", "")).strip()
        if not comp:
            continue
        dossier.upsert(
            Candidate(
                compartment=comp,
                state="candidate",
                source="catalog",
                prior=str(entry.get("prior", "unknown")).strip().lower() or "unknown",
                likelihood="unknown",
            )
        )

    dossier.add_candidates(hypotheses or [], default_source="data")
    dossier.prune()
    return dossier
