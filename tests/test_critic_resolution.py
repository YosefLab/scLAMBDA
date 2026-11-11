"""Only reviewed measurements can become accepted cell-type annotations."""
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from sclambda.annotation.celltype.dossier import CellTypeDossier, seed_ct_dossier
from sclambda.annotation.decisions import apply_critic_verdict, build_commit, supported_label


def axis(label, note="measured signature supports this identity"):
    return {"resolved": True, "label": label, "note": note,
            "evidence_status": "established", "defining_evidence": ["GENE1 measured in signature table"],
            "measurement_refs": [{"measurement_id": "m_test", "genes": ["GENE1"]}],
            "counterevidence": [], "reconciliation": "No measured contradictory pattern in this fixture",
            "specificity": "Measured combination distinguishes the plausible alternative"}


def proposal():
    d = CellTypeDossier(cluster_id="0")
    d.measurements = [{"id": "m_test", "target": "0", "reference": "test reference", "method": "ttest",
                       "significance_kind": "raw_p", "rows": [{"gene": "GENE1", "lfc": 1., "proportion": .6, "significance": .01}]}]
    d.apply_update({"candidates": [{"cell_type": "Specific type", "state": "supported",
                                   "lineage": "Broad lineage", "states": ["activated"]}],
                    "best_cell_type": "Specific type", "confirmed": ["Specific identity is certain"]})
    return d


@pytest.mark.parametrize("source", [None, "data", "agent", "neighbor", "critic"])
def test_belief_update_cannot_resolve_or_request_subclustering(source):
    d = proposal()
    d.apply_update({"axes": {"lineage": axis("Specific type")},
                    "decision": "confident", "needs_subcluster": True,
                    "committed_granularity": "state"}, default_source=source)
    assert not d.axes and not d.needs_subcluster
    assert d.decision == "needs_refinement"
    assert supported_label(d) is None
    with pytest.raises(ValueError):
        build_commit(d)
    assert "not critic-confirmed" in d.to_markdown()
    assert all(h["state"] != "supported" for h in d.to_hypotheses())


def test_actor_cannot_overwrite_critic_label_through_candidate_or_lineage():
    d = proposal()
    apply_critic_verdict(d, {"axes": {"lineage": axis("Broad lineage")}})
    d.apply_update({"candidates": [{"cell_type": "Other type", "state": "supported",
                                   "lineage": "Wrong lineage", "evidence_for": ["new claim"]}],
                    "best_cell_type": "Other type", "axes": {"lineage": axis("Wrong lineage")}})
    assert supported_label(d) == "Broad lineage"
    assert build_commit(d)["label"] == "Broad lineage"
    assert d.to_hypotheses()[0]["cell_type"] == "Broad lineage"
    assert d.to_hypotheses()[0]["source"] == "critic"


@pytest.mark.parametrize("state_label", ["activated", "Specific type (activated)"])
def test_state_label_cannot_replace_identity(state_label):
    d = proposal()
    apply_critic_verdict(d, {"axes": {"lineage": axis("Broad lineage"),
                                      "cell_type": axis("Specific type"),
                                      "state": axis(state_label)}})
    assert build_commit(d)["label"] == "Specific type (activated)"
    assert CellTypeDossier.from_dict(d.to_dict()).cell_type == "Specific type (activated)"


def test_critic_can_update_split_flag_without_replacing_resolution():
    d = proposal()
    apply_critic_verdict(d, {"axes": {"lineage": axis("Broad lineage")}})
    apply_critic_verdict(d, {"needs_subcluster": True})
    assert d.needs_subcluster and d.cell_type == "Broad lineage"
    d.apply_update({"needs_subcluster": False}, default_source="agent")
    assert d.needs_subcluster


def test_critic_can_retract_subtype_then_lineage_and_roundtrip():
    d = proposal()
    apply_critic_verdict(d, {"axes": {"lineage": axis("Broad lineage"),
                                      "cell_type": axis("Specific type"),
                                      "state": axis("Specific type (activated)")}})
    assert supported_label(d) == "Specific type (activated)"
    apply_critic_verdict(d, {"axes": {"lineage": axis("Broad lineage"),
                                      "cell_type": {"resolved": False, "note": "test did not confirm subtype"}}})
    assert supported_label(d) == "Broad lineage"
    assert not d.axes["state"]["resolved"]
    d = CellTypeDossier.from_dict(d.to_dict())
    assert d.cell_type == "Broad lineage"
    apply_critic_verdict(d, {"axes": {"lineage": {"resolved": False},
                                      "cell_type": axis("Specific type")}})
    assert supported_label(d) is None
    assert not d.axes["cell_type"]["resolved"]


@pytest.mark.parametrize("value", [{"resolved": True}, {"resolved": True, "note": "certain"},
                                  {"resolved": True, "label": "Broad lineage"}])
def test_incomplete_or_legacy_resolution_cannot_commit(value):
    d = proposal()
    apply_critic_verdict(d, {"axes": {"lineage": value}})
    assert d.cell_type is None
    old = d.to_dict()
    old.update(axes={"lineage": value}, committed_granularity="lineage")
    restored = CellTypeDossier.from_dict(old)
    assert restored.cell_type is None and restored.committed_granularity == ""


def agent_harness(monkeypatch, verdicts, actions):
    from sclambda.pipelines.celltype.nodes.agent import CTDossierAgentNode
    node = object.__new__(CTDossierAgentNode)
    node._output_local = threading.local()
    node.output = {}
    node.accessor = SimpleNamespace(gene_names=np.array(["GENE1"]))
    node.llm_client = SimpleNamespace(query=lambda *a, **kw: SimpleNamespace(
        content=json.dumps(actions.pop(0)), failed_after_retries=False))
    seen = []
    def critic(state, dossier, *args):
        seen.append(dossier.to_dict())
        return verdicts.pop(0)
    monkeypatch.setattr(node, "_run_critic", critic)
    monkeypatch.setattr(node, "_peer_block", lambda state: "none")
    monkeypatch.setattr(node, "log_info", lambda *a: None)
    monkeypatch.setattr(node, "log_warning", lambda *a: None)
    monkeypatch.setattr(node, "update_annotation", lambda cid, label: node.output.update(written_label=label))
    monkeypatch.setattr(node, "update_hypotheses", lambda *a: None)
    state = {"cluster_id": "0", "ct_dossier": proposal().to_dict(),
             "top_genes": ["GENE1"], "dossier_agent_max_steps": 3}
    return node, state, seen








def test_dossier_child_cannot_skip_critic_via_inheritance():
    from sclambda.pipelines.celltype.nodes.preparation import InheritParentAnnotationNode
    node = object.__new__(InheritParentAnnotationNode)
    result = node.execute({"cluster_id": "0_1", "parent_cluster_id": "0",
                           "use_dossier": True, "can_inherit_parent": True})
    assert result == {"parent_classification": "subtype", "can_inherit_parent": False}


def test_low_prior_untested_catalogue_candidates_survive_updates_and_reload():
    catalog = [{"cell_type": f"Type {i}", "prior": "high"} for i in range(20)]
    catalog.append({"cell_type": "Rare alternative", "prior": "very low"})
    d = seed_ct_dossier("0", catalog_cell_types=catalog)
    d.apply_update({"candidates": [{"cell_type": "Type 0", "state": "supported"}]})
    restored = CellTypeDossier.from_dict(d.to_dict())
    assert len(restored.active()) == 21
    assert restored.get("Rare alternative").state == "candidate"
    assert not restored.get("Rare alternative").evidence_against


def test_legacy_pool_retirements_recover_without_erasing_measured_exclusions():
    d = CellTypeDossier.from_dict({"candidates": [
        {"cell_type": "Untested", "state": "retired", "evidence_against": ["retired: exceeded active-pool cap"]},
        {"cell_type": "Excluded", "state": "refuted", "evidence_against": ["measured competing identity"]},
    ]})
    assert d.get("Untested").state == "candidate"
    assert d.get("Excluded").state == "refuted"






@pytest.mark.parametrize("status", ["inconclusive", "untested", "unavailable", None, "certain"])
def test_insufficient_defining_evidence_reopens_identity_and_deeper_rungs(status):
    d = proposal()
    apply_critic_verdict(d, {"axes": {"lineage": axis("Broad lineage"), "cell_type": axis("Specific type")}})
    insufficient = {**axis("Broad lineage"), "evidence_status": status}
    apply_critic_verdict(d, {"axes": {"lineage": insufficient, "cell_type": axis("Specific type")}})
    assert d.cell_type is None
    assert not d.axes["lineage"]["resolved"] and not d.axes["cell_type"]["resolved"]
    assert CellTypeDossier.from_dict(d.to_dict()).cell_type is None
    with pytest.raises(ValueError):
        build_commit(d)


@pytest.mark.parametrize("field,value", [("defining_evidence", []), ("defining_evidence", "trust me"),
                                        ("defining_evidence", [""]), ("specificity", "")])
def test_established_without_citations_or_specificity_cannot_resolve(field, value):
    d = proposal()
    apply_critic_verdict(d, {"axes": {"lineage": {**axis("Broad lineage"), field: value}}})
    assert d.cell_type is None


def test_legacy_resolution_without_evidence_assessment_requires_review():
    old = proposal().to_dict()
    old["axes"] = {"lineage": {"resolved": True, "label": "Old identity", "note": "old assertion"}}
    assert CellTypeDossier.from_dict(old).cell_type is None






def test_multi_signatures_deduplicate_and_handle_unavailable_or_invalid_panels(monkeypatch):
    node, state, _ = agent_harness(monkeypatch, [], [])
    d = proposal()
    measured = []
    from sclambda.annotation.celltype.measurements import MeasurementText
    def measure(state, genes):
        measured.append(genes)
        return MeasurementText({"id": f"m_{len(measured)}", "target": "0", "reference": "rest",
                                "method": "ttest", "significance_kind": "raw_p",
                                "rows": [{"gene": g, "lfc": 1., "proportion": .5, "significance": .01} for g in genes]})
    monkeypatch.setattr(node, "_measure_signature", measure)
    seen = set()
    node._test_signatures(state, d, {"tool": "test_signatures", "signatures": [
        {"name": "A", "genes": ["GENE1", "GENE1", "GENE2"]},
        {"name": "B", "genes": ["GENE2", "GENE1", "MISSING"]},
        {"name": "C", "genes": ["MISSING"]}, {"genes": "GENE1"}, None,
    ]}, {"GENE1", "GENE2"}, seen)
    node._test_signatures(state, d, {"tool": "test_signature", "genes": ["GENE3"]}, {"GENE3"}, seen)
    assert measured == [["GENE1", "GENE2"], ["GENE3"]]
    facts = "\n".join(e["fact"] for e in d.evidence_log)
    assert "already measured" in facts and "unavailable evidence, not absent expression" in facts
    assert "Invalid signature" in facts


def test_markdown_without_candidates_shows_only_decision_and_tables():
    from sclambda.annotation.celltype.measurements import add_measurement
    d = CellTypeDossier(cluster_id="0")
    add_measurement(d, {"id": "m1", "target": "0", "reference": "all other cells", "method": "ttest",
                        "significance_kind": "fdr",
                        "rows": [{"gene": "CD3E", "lfc": 2.0, "proportion": 0.9, "significance": 1e-5}]})
    text = d.to_markdown()
    assert "Critic decision:** (none yet)" in text
    assert "Measurement m1" in text and "| CD3E |" in text
    assert "Candidate cell types" not in text
