"""
Stage A — tissue anatomical catalog.

Before any niche is annotated, enumerate (once per run) the canonical spatial
compartments that *can* occur in this ``{organism} {tissue}``, each with its
prior, defining cell types, marker genes, and the partner compartments it is
normally adjacent to.

This catalog is the shared "universe" of candidate labels.  It is what lets
low-prior-but-real compartments (e.g. an interfollicular T-cell zone next to a
germinal center) be enumerated up front and compete on data, instead of
depending on whatever a single per-niche generation happened to think of.  The
``adjacent_partners`` field also drives neighbor-induced extension (Stage C):
when a neighbor is confidently a germinal center, its catalogued partners
(mantle zone, T-cell zone) are injected as candidates for the target niche.

The catalog is computed by one LLM call and cached on the pipeline instance for
the whole run.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")


@dataclass
class CatalogEntry:
    compartment: str
    prior: str = "medium"                       # very high / high / medium / low / very low
    defining_cell_types: list[str] = field(default_factory=list)
    markers: list[str] = field(default_factory=list)
    adjacent_partners: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "compartment": self.compartment,
            "prior": self.prior,
            "defining_cell_types": self.defining_cell_types,
            "markers": self.markers,
            "adjacent_partners": self.adjacent_partners,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CatalogEntry | None":
        comp = str(d.get("compartment", "")).strip()
        if not comp:
            return None
        return cls(
            compartment=comp,
            prior=str(d.get("prior", "medium")).strip().lower() or "medium",
            defining_cell_types=[str(x).strip() for x in d.get("defining_cell_types", []) if str(x).strip()],
            markers=[str(x).strip() for x in d.get("markers", []) if str(x).strip()],
            adjacent_partners=[str(x).strip() for x in d.get("adjacent_partners", []) if str(x).strip()],
        )


@dataclass
class AnatomicalCatalog:
    organism: str
    tissue: str
    entries: list[CatalogEntry] = field(default_factory=list)

    def to_dicts(self) -> list[dict]:
        return [e.to_dict() for e in self.entries]

    def compartment_names(self) -> list[str]:
        return [e.compartment for e in self.entries]

    def partners_of(self, compartment: str) -> list[str]:
        """Adjacent partner compartments for a (possibly fuzzy) compartment name."""
        target = _norm(compartment)
        partners: list[str] = []
        for e in self.entries:
            if _norm(e.compartment) in target or target in _norm(e.compartment):
                partners.extend(e.adjacent_partners)
        # dedup, preserve order
        seen, out = set(), []
        for p in partners:
            if _norm(p) not in seen:
                seen.add(_norm(p))
                out.append(p)
        return out

    def to_markdown(self) -> str:
        if not self.entries:
            return "(no catalog available)"
        lines = [
            f"### Anatomical catalog — canonical compartments of {self.organism} {self.tissue}",
            "",
            "| Compartment | Prior | Defining cell types | Marker genes | Usually adjacent to |",
            "|---|---|---|---|---|",
        ]
        for e in self.entries:
            lines.append(
                f"| {e.compartment} | {e.prior} | {', '.join(e.defining_cell_types) or '—'} "
                f"| {', '.join(e.markers) or '—'} | {', '.join(e.adjacent_partners) or '—'} |"
            )
        return "\n".join(lines)


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _parse_catalog(text: str) -> list[CatalogEntry]:
    match = re.search(r"\[.*\]", text or "", re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse catalog JSON: %s", e)
        return []
    if not isinstance(data, list):
        return []
    out = []
    for d in data:
        if isinstance(d, dict):
            entry = CatalogEntry.from_dict(d)
            if entry is not None:
                out.append(entry)
    return out
