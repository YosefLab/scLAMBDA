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
from sclambda.annotation.niche.catalog import AnatomicalCatalog, CatalogEntry, _norm, _parse_catalog


import json
import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(filename: str) -> str:
    with open(os.path.join(PROMPT_DIR, filename), "r") as f:
        return f.read().rstrip()


def build_catalog(llm_client, organism: str, tissue: str) -> AnatomicalCatalog:
    """Generate the anatomical catalog with one LLM call (Stage A).

    Failures degrade gracefully to an empty catalog — the pipeline then behaves
    like the data-only generation it had before, just without catalog seeding.
    """
    catalog = AnatomicalCatalog(organism=organism, tissue=tissue, entries=[])
    try:
        template = _load_prompt("A_catalog.md")
    except FileNotFoundError:
        logger.warning("A_catalog.md not found; proceeding without anatomical catalog")
        return catalog

    prompt = template.format(organism=organism, tissue=tissue)
    try:
        resp = llm_client.query(prompt, power="medium")
    except Exception as e:  # pragma: no cover - network/LLM errors
        logger.warning("Catalog generation failed: %s", e)
        return catalog

    if getattr(resp, "failed_after_retries", False):
        logger.warning("Catalog generation LLM failed after retries")
        return catalog

    entries = _parse_catalog(resp.content)
    catalog.entries = entries
    logger.info("Anatomical catalog: %d compartments for %s %s", len(entries), organism, tissue)
    return catalog


