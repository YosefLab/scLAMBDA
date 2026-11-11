"""
Pipeline nodes for cell-typing cluster processing.

All nodes are classes inheriting from CellTypingNode or CellTypingLLMNode.
They implement execute(state) -> dict and are callable via __call__ for LangGraph.

Each node declares:
- default_output: dict returned on failure (class attribute)
- self.output: initialized from default_output by __call__, updated incrementally
- Parsing uses ParseSchema + parse_llm_response() instead of per-node functions
"""

from __future__ import annotations
import math
from difflib import SequenceMatcher
from typing import TYPE_CHECKING
import logging
from sclambda.annotation.decisions import apply_critic_verdict, supported_label
import re
import os
import numpy as np

from sclambda.framework.base_node import BaseNode, LLMNode, ParseSchema, gene_symbol_convention
from sclambda.annotation.celltype.formatters import (
    format_gene_list,
    format_top_genes_de_table,
    format_hypotheses,
    SignatureFormatter,
)
from sclambda.analysis.rank_genes import rank_genes
from sclambda.analysis.targeted_de import targeted_de_signatures
from sclambda.analysis.subcluster import subcluster, SubclusterMethod
from sclambda.analysis.ontology import load_ontology, fuzzy_match, normalize
from sclambda.annotation.celltype.dossier import CellTypeDossier, seed_ct_dossier, RUNGS
from sclambda.annotation.celltype import substructure as _ss

if TYPE_CHECKING:
    from sclambda.core.accessor import Accessor
    from sclambda.core.task_runner import TaskRunner
    from sclambda.llm.api import LLMClient
    from sclambda.pipelines.celltype.pipeline import CellTypingState

logger = logging.getLogger(__name__)


# ============ CELL-TYPING CONSTANTS ============

PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")

CONFIDENCE_ORDER = {"high": 3, "medium": 2, "low": 1, "unknown": 0}

# Minimum cells on each side of a DE comparison for the dossier-agent tools.
MIN_CELLS = 10


# ============ CELL-TYPING BASE CLASSES ============


class _CellTypingHelpers:
    """Cell-typing specific accessor helpers (mixin)."""

    def update_annotation(self, cluster_id: str, annotation: str):
        self.runner.with_lock(self.accessor.set_annotation, cluster_id, annotation)
        self.log_info(cluster_id, f"Set annotation: '{annotation}'")

    def update_hypotheses(self, cluster_id: str, hypotheses: list[dict]):
        if hypotheses:
            self.runner.with_lock(self.accessor.set_hypotheses, cluster_id, hypotheses)
            self.log_info(cluster_id, f"Stored {len(hypotheses)} hypotheses")

    def update_ont_cell_type(self, cluster_id: str, ont_term: str):
        self.runner.with_lock(self.accessor.set_ont_cell_type, cluster_id, ont_term)
        self.log_info(cluster_id, f"Set Cell Ontology term: '{ont_term}'")

    def _get_parent_context(self, state: "CellTypingState") -> str:
        """Return a parent-context hint for LLM prompts when classification is 'subtype'."""
        parent_id = state.get("parent_cluster_id")
        parent_classification = state.get("parent_classification")

        if not (parent_id and parent_classification == "subtype"):
            return ""

        parent_depth = self.accessor._parse_cluster_depth(parent_id)
        parent_annotations = self.accessor.get_annotations(depth=parent_depth)
        parent_cell_type = parent_annotations.get(parent_id)

        if not parent_cell_type:
            return ""

        return (
            f"\n**Important Context:**\n"
            f"This cluster can be **{parent_cell_type}**.\n"
            f"Explore both the subtypes of {parent_cell_type} and alternative cell types.\n"
        )

    def _parent_label(self, state: "CellTypingState") -> str:
        """The parent's committed label, when this cluster is a subtype of it.

        Used as a hypothesis in the child; the child's critic must review it.
        """
        parent_id = state.get("parent_cluster_id")
        if not (parent_id and state.get("parent_classification") == "subtype"):
            return ""
        try:
            depth = self.accessor._parse_cluster_depth(parent_id)
            return (self.accessor.get_annotations(depth=depth) or {}).get(parent_id, "") or ""
        except Exception:  # noqa: BLE001
            return ""

    def apply_label_change(
        self,
        cluster_id: str,
        old_cell_type: str,
        new_cell_type: str,
        reasoning: str,
    ) -> None:
        """Apply a cell-type label change consistently across all accessor stores.

        Called by any node that overrides the final cell type after initial
        annotation (e.g. CrossClusterCorrectionNode).

        Updates:
        - Annotation overlay (via update_annotation)
        - Cell Ontology store: clears the stale term so OntCellTypeNode can
          re-map from the new label on the next run
        - Hypotheses: updates the best hypothesis cell_type and reasoning so
          that justification in the output Excel reflects the new label
        """
        self.update_annotation(cluster_id, new_cell_type)
        # Clear stale ontology term from accessor so OntCellTypeNode re-maps.
        self.runner.with_lock(self.accessor.set_ont_cell_type, cluster_id, None)
        # Update best hypothesis so justification reflects the new label.
        hypotheses = list(self.accessor.get_hypotheses(cluster_id) or [])
        if hypotheses:
            best = dict(hypotheses[0])
            best["cell_type"] = new_cell_type
            best["justification"] = reasoning
            best["reasoning"] = reasoning
            hypotheses[0] = best
        else:
            hypotheses = [
                {
                    "cell_type": new_cell_type,
                    "confidence": "medium",
                    "justification": reasoning,
                    "reasoning": reasoning,
                    "supporting_genes": [],
                }
            ]
        self.update_hypotheses(cluster_id, hypotheses)


class CellTypingNode(_CellTypingHelpers, BaseNode):
    """Base node for cell-typing pipeline (non-LLM)."""

    pass


class CellTypingLLMNode(_CellTypingHelpers, LLMNode):
    """LLM node for cell-typing pipeline."""

    prompt_dir = PROMPT_DIR


# ============ HELPER FUNCTIONS ============


def _load_prompt(filename: str) -> str:
    """Load prompt from prompts directory."""
    path = os.path.join(PROMPT_DIR, filename)
    with open(path, "r") as f:
        return f.read().rstrip()


# ============ DOSSIER MIXIN ============


class CTDossierMixin:
    """Load, update and commit the per-cluster CellTypeDossier (``use_dossier`` runs).

    ``commit_dossier`` also writes a ``hypotheses`` view so routing and output
    code shared with the non-dossier path keep working.
    """

    def load_dossier(self, state: "CellTypingState") -> CellTypeDossier:
        d = CellTypeDossier.from_dict(state.get("ct_dossier"))
        if not d.cluster_id:
            d.cluster_id = str(state.get("cluster_id", ""))
        return d

    def update_dossier_from_response(
        self,
        dossier: CellTypeDossier,
        reasoning_response: str,
        default_source: str,
        stage: str,
    ) -> CellTypeDossier:
        parse_tmpl = _load_prompt(f"dossier_update_parse_{stage}.md")
        formatted = parse_tmpl.format(
            dossier=dossier.to_markdown(), reasoning_response=reasoning_response
        )
        parse_resp = self.llm_client.query(formatted, power="weak")
        self.output[f"{stage}_dossier_parse_response"] = parse_resp.content
        if parse_resp.failed_after_retries:
            return dossier
        update = self.parse_llm_response(parse_resp.content, ParseSchema(format="dict"), fallback={})
        dossier.iteration += 1
        dossier.apply_update(update, default_source=default_source)
        return dossier

    def commit_dossier(self, dossier: CellTypeDossier, cluster_id: str) -> None:
        hypotheses = dossier.to_hypotheses()
        cell_type = supported_label(dossier)
        self.output["ct_dossier"] = dossier.to_dict()
        self.output["hypotheses"] = hypotheses
        self.output["updated_hypotheses"] = hypotheses
        # Explicitly clear a previous specific label when the critic reopens it.
        self.output["cell_type"] = cell_type or "Unresolved"
        self.output["cell_types"] = [cell_type] if cell_type else []
        if hypotheses:
            self.update_hypotheses(cluster_id, hypotheses)
        self.update_annotation(cluster_id, cell_type or "Unresolved")


# ============ NODE CLASSES ============

