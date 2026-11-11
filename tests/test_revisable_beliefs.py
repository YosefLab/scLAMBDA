"""Regressions for belief anchoring observed in saved annotation traces."""
import copy
import json
from types import SimpleNamespace

import pytest

from sclambda.annotation.celltype.dossier import CellTypeDossier
from sclambda.annotation.decisions import apply_critic_verdict
from sclambda.pipelines.celltype.nodes.agent import CTDossierAgentNode
from test_critic_resolution import agent_harness, axis, proposal


def test_explicit_null_with_new_supported_candidate_does_not_restore_incumbent():
    d = proposal()
    d.apply_update({"candidates": [{"cell_type": "Alternative", "state": "supported"}],
                    "best_cell_type": None})
    assert d.best_cell_type is None
    assert all(c.state == "candidate" for c in d.candidates)
    d.apply_update({"candidates": [{"cell_type": "Alternative", "state": "supported"}]})
    assert d.best_cell_type == "Alternative"
    d.apply_update({"description": "updated"})
    assert d.best_cell_type == "Alternative"
    d.apply_update({"candidates": [{"cell_type": "Alternative", "state": "candidate"}]})
    assert d.best_cell_type is None


def test_interpretation_snapshots_can_be_replaced_and_cleared_without_erasing_measurements():
    d = proposal()
    d.add_evidence("signature", "| GENE1 | 0.1 |")
    raw = copy.deepcopy(d.evidence_log)
    d.apply_update({"confirmed": ["revised trait"], "open_questions": ["new question"],
                    "candidates": [{"cell_type": "Specific type", "evidence_for": ["new evidence"]}]})
    assert d.confirmed == ["revised trait"]
    d.apply_update({"confirmed": [], "open_questions": [], "description": "",
                    "candidates": [{"cell_type": "Specific type", "evidence_for": [], "states": []}]})
    assert not d.confirmed and not d.open_questions and not d.description
    assert not d.get("Specific type").evidence_for and not d.get("Specific type").states
    assert d.evidence_log == raw


def test_stable_id_targets_exclusion_despite_different_description_and_roundtrip():
    d = proposal()
    cid = d.get("Specific type").candidate_id
    restored = CellTypeDossier.from_dict(d.to_dict())
    assert restored.get(cid).cell_type == "Specific type"
    d.apply_update({"ruled_out": [{"candidate_id": cid, "claim": "Descriptive qualifier of identity",
                                  "reason": "contradicted by measurements"}]})
    assert d.get(cid).state == "refuted" and d.best_cell_type is None
    d.apply_update({"candidates": [{"candidate_id": cid, "cell_type": "different spelling", "state": "candidate"}]})
    assert not d.ruled_out
    assert d.get(cid).cell_type == "Specific type" and d.get(cid).state == "candidate"
    legacy = d.to_dict()
    del legacy["candidates"][0]["candidate_id"]
    assert CellTypeDossier.from_dict(legacy).get("Specific type").candidate_id == cid


def test_exclusion_does_not_use_substring_or_unknown_id_fallback():
    d = proposal()
    d.apply_update({"ruled_out": [{"claim": "type", "reason": "not a precise identity"}],
                    "candidates": [{"candidate_id": "unknown", "cell_type": "Specific type", "state": "refuted"}]})
    assert d.get("Specific type").state == "supported"
    assert len(d.candidates) == 1


def test_reassessment_preserves_raw_tables_and_critic_axes_but_clears_interpretations():
    d = proposal()
    apply_critic_verdict(d, {"axes": {"lineage": axis("Broad lineage")}})
    d.add_evidence("signature", "| GENE1 | 0.1 |")
    raw, axes = copy.deepcopy(d.evidence_log), copy.deepcopy(d.axes)
    d.reopen_hypotheses()
    assert d.evidence_log == raw and d.axes == axes
    assert d.best_cell_type is None and not d.confirmed and not d.ruled_out
    assert all(c.state == "candidate" and c.likelihood == "unknown" and not c.evidence_for and not c.states for c in d.candidates)


def test_measured_likelihood_precedes_catalogue_prior():
    d = CellTypeDossier()
    d.add_candidates([{"cell_type": "Common", "prior": "high", "likelihood": "low"},
                      {"cell_type": "Rare", "prior": "low", "likelihood": "high"}])
    assert d.leading().cell_type == "Rare"


def audit(identity="Alternative", genes=None, status="tested"):
    return {"identity": identity, "genes": genes or ["GENE1"], "status": status,
            "reason": "Measured expression does not distinguish the competing identities"}
