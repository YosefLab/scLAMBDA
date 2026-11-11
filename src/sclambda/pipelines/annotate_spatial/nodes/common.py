"""
Pipeline nodes for spatial transcriptomics annotation.

Operates per-niche: each niche (from external spatial clustering) is a
contiguous spatial region containing multiple cell types. For each niche,
computes absolute cell type proportions with Moran's Z, DE per cell type
(niche vs rest of dataset) with gene spatial coherence, and uses LLM for
spatial region annotation with cross-cell-type evidence.

Target cells = cells in the niche.
Reference cells = cells of the same type OUTSIDE the niche.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import logging
from sclambda.annotation.decisions import supported_label
import os

import numpy as np

from sclambda.framework.base_node import BaseNode, LLMNode
from sclambda.annotation.niche.dossier import NicheDossier

if TYPE_CHECKING:
    from sclambda.pipelines.annotate_spatial.pipeline import AnnotateSpatialState

logger = logging.getLogger(__name__)


# ============ CONSTANTS ============

PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")

MIN_CELLS = 10


# ============ BASE CLASSES ============


class SpatialNode(BaseNode):
    """Base non-LLM node for spatial annotation pipeline."""

    pass


class SpatialLLMNode(LLMNode):
    """Base LLM node for spatial annotation pipeline."""

    prompt_dir = PROMPT_DIR


class DossierMixin:
    """Shared logic for nodes that read and update the persistent NicheDossier.

    The dossier is the single object threaded between the actor and critic
    steps; ``commit_dossier`` writes it and its ``hypotheses`` view to the node
    output and the accessor.
    """

    def load_dossier(self, state: "AnnotateSpatialState") -> NicheDossier:
        d = NicheDossier.from_dict(state.get("niche_dossier"))
        if not d.cluster_id:
            d.cluster_id = str(state.get("cluster_id", ""))
        return d

    def commit_dossier(self, dossier: NicheDossier, cluster_id: str) -> None:
        """Write dossier + backward-compatible views into the node output."""
        if dossier.decision_mode:
            decision = dossier.critic_decision
            hypotheses = ([{"annotation": decision["label"], "confidence": decision["status"],
                            "justification": decision["reason"]}] if decision else [])
        else:
            hypotheses = dossier.to_hypotheses()
        annotation = supported_label(dossier)

        self.output["niche_dossier"] = dossier.to_dict()
        self.output["niche_description"] = dossier.description
        self.output["hypotheses"] = hypotheses
        self.output["updated_hypotheses"] = hypotheses
        self.output["annotation"] = annotation

        if hypotheses:
            self.runner.with_lock(self.accessor.set_hypotheses, cluster_id, hypotheses)
        if annotation:
            self.runner.with_lock(self.accessor.set_annotation, cluster_id, annotation, column="region")


# ============ HELPER FUNCTIONS ============


def _load_prompt(filename: str) -> str:
    """Load prompt from prompts directory."""
    path = os.path.join(PROMPT_DIR, filename)
    with open(path, "r") as f:
        return f.read().rstrip()


def _slice_restricted_niche_indices(accessor, cluster_id, slice_key=None, slice_val=None):
    """Niche cell indices restricted to one slide (the largest-niche slide).

    Used for spatial (Moran's Z) computations and the niche image, which need
    cells from a single physical section — neighbours across slides are not
    comparable.
    """
    niche_indices = accessor.get_cell_indices(cluster_id, depth=1)
    if slice_key and slice_val:
        all_indices = np.arange(accessor.n_cells)
        slice_obs = accessor.get_obs([slice_key], all_indices)
        slice_mask = slice_obs[slice_key].values.astype(str) == str(slice_val)
        niche_indices = niche_indices[slice_mask[niche_indices]]
    return niche_indices
