"""Spatial annotation: measurement-seeking actor and one deciding critic."""
from copy import deepcopy
from sclambda.llm.usage import usage_context
import json
import numpy as np

from sclambda.annotation.niche.populations import niche_and_rest_indices
from .common import DossierMixin, SpatialLLMNode, _load_prompt, MIN_CELLS
from sclambda.analysis.targeted_de import compute_de_for_genes, compute_pathology_de_from_indices
from sclambda.annotation.niche.measurements import (
    add_measurement, seed_measurements, render, scope, expression_record, validate_annotation, normalize_annotation,
)
from sclambda.pipelines.celltype.nodes.agent import CTDossierAgentNode


class SpatialDossierAgentNode(DossierMixin, SpatialLLMNode):
    default_output = {"dossier_review_complete": False, "dossier_stop_reason": "incomplete"}
    _parse_turn = CTDossierAgentNode._parse_turn

    def execute(self, state):
        dossier = self.load_dossier(state)
        dossier.decision_mode = True
        seed_measurements(dossier, state)
        transcript = []
        rounds = max(1, int(state.get("dossier_agent_max_steps", 4)))
        completed, stop = False, "incomplete"
        candidate_history = []
        pending = []
        comparison_plans = []
        comparison_feedback = None
        comparison_mode = state.get("spatial_comparison_review", False) and not state.get("spatial_resolution_comparison", False)
        for step in range(1, rounds + 1):
            hypotheses = []
            actor_error = None
            try:
                if comparison_feedback:
                    # A rejected comparative judgment is reconsidered by the
                    # same critic within the existing round budget. It is not a
                    # formatting repair and does not become measured evidence.
                    pass
                elif pending:
                    # The critic has already specified the experiment. Execute
                    # it through the actor tools instead of asking another model
                    # to reinterpret or silently discard the request.
                    for check in pending:
                        transcript.append({"step": step, "tool": "actor", "source": "critic_request",
                                           "proposal": {"action": check["action"], "question": check["question"]}})
                        try:
                            self._act(state, dossier, check["action"])
                        except Exception as exc:
                            actor_error = str(exc)
                            dossier.add_evidence("tool_status", f"Requested check failed: {check['question']}: {exc}; no result was obtained.")
                            transcript.append({"step": step, "tool": "actor", "error": actor_error})
                else:
                    prompt = _load_prompt("19_comparison_actor.md" if comparison_mode else "18_resolution_actor.md" if state.get("spatial_resolution_comparison") else "17_dossier_agent.md").format(
                        organism=state.get("organism", "human"), tissue=state.get("tissue", "unknown"),
                        measurements=self._evidence(dossier), previous=json.dumps(dossier.critic_decision),
                        availability=self._availability(dossier),
                        catalog=state.get("anatomical_catalog_md") or "Optional catalog absent; propose identities from evidence.")
                    with usage_context(role="actor", iteration=step, cluster_id=state["cluster_id"]):
                        response = self.llm_client.query(prompt, power="medium", image_base64=self._images(dossier) or None)
                    turn = None if response.failed_after_retries else self._parse_turn(response)
                    if not isinstance(turn, dict) or not isinstance(turn.get("action"), dict):
                        raise ValueError("Actor response failed or malformed")
                    if comparison_mode:
                        plan = turn.get("comparison_plan")
                        if isinstance(plan, dict):
                            comparison_plans.append(deepcopy(plan))
                        else:
                            dossier.add_evidence("tool_status", "Actor omitted the comparison plan; critic must reconstruct the competing expectations from evidence, not assume a test verified an identity.")
                    hypotheses = turn.get("hypotheses", [])
                    if isinstance(hypotheses, list):
                        candidate_history.extend(hypotheses)
                    transcript.append({"step": step, "tool": "actor", "proposal": turn})
                    self._act(state, dossier, turn["action"])
            except Exception as exc:
                actor_error = str(exc)
                dossier.add_evidence("tool_status", f"Actor action failed: {exc}; no result was obtained.")
                transcript.append({"step": step, "tool": "actor", "error": actor_error})
            verdict, attempts = self.review(state, dossier, candidate_history, final=step == rounds, comparison_plans=comparison_plans, comparison_feedback=comparison_feedback)
            transcript.append({"step": step, "tool": "critic", "attempts": attempts})
            if verdict is None:
                last = attempts[-1] if attempts else {}
                errors = last.get("validation_errors", [])
                proposal = last.get("proposal")
                comparative_proposal = isinstance(proposal, dict) and isinstance(proposal.get("comparison"), dict)
                malformed_complete_response = proposal is None and last.get("response_complete", False)
                if (comparison_mode and step < rounds
                        and (comparative_proposal or malformed_complete_response)
                        and errors):
                    # Citation/scope errors need substantive reconsideration too.
                    # Never execute actions from an unvalidated proposal.
                    comparison_feedback = {"rejected_proposal": proposal, "errors": errors,
                                           "original_attempt": attempts[0]}
                    if malformed_complete_response:
                        # Ask the same critic for a fresh complete decision;
                        # never close braces or infer missing biological fields.
                        comparison_feedback["rejected_response"] = last["raw_response"]
                    continue
                stop = "critic_failed"
                break
            comparison_feedback = None
            dossier.critic_decision = deepcopy(verdict)
            pending = verdict.get("pending_checks", [])
            candidate_history.append({"label": verdict["label"]})
            candidate_history.extend({"label": a} if isinstance(a, str) else a
                                     for a in verdict.get("alternatives", []))
            dossier.description = verdict["reason"]
            dossier.open_questions = [verdict["next_question"]] if verdict["next_question"] else []
            dossier.iteration += 1
            if not verdict["next_question"] or step == rounds:
                completed = True
                stop = "critic_complete" if not verdict["next_question"] else "budget_exhausted"
                break
            # A tool error is evidence about availability, not a reason to lose
            # the annotation. The accepted critic can request a corrected action
            # or finalize from the measurements that are already available.
        if not completed:
            verdict, attempts = self.final_review(state, dossier, candidate_history, transcript)
            transcript.append({"step": step, "tool": "final_critic", "attempts": attempts})
            if verdict is not None:
                dossier.critic_decision = verdict
                completed, stop = True, "best_effort_finalized"
            else:
                self._retain_output(dossier)
                stop = "retained_previous" if dossier.critic_decision.get("output_origin") == "previous_critic" else "inference_failed"
        dossier.description = dossier.critic_decision.get("reason", dossier.description)
        question = dossier.critic_decision.get("next_question", "")
        dossier.open_questions = [question] if question else []
        self.commit_dossier(dossier, state["cluster_id"])
        self.output.update(dossier_agent_transcript=transcript, dossier_agent_steps=step,
            dossier_review_complete=completed, dossier_stop_reason=stop,
            annotation_assessment=(dossier.critic_decision.get("status", "review_incomplete")
                                   if completed else "inference_failed"),
            annotation_status=dossier.critic_decision.get("status"),
            annotation_uncertainty=dossier.critic_decision.get("uncertainty", ""),
            annotation_alternatives=dossier.critic_decision.get("alternatives", []))
        return self.output

    def final_review(self, state, dossier, hypotheses, transcript):
        """Two bounded, substantive reviews with a compact output contract.

        This is explicitly provisional. It retains citation/scope checks, but
        does not require the comparison bookkeeping that exhausted the main loop.
        A rejected comparison is never silently converted into an accepted label.
        """
        prompt = _load_prompt("23_final_critic.md").format(
            organism=state.get("organism", "human"), tissue=state.get("tissue", "unknown"),
            measurements=self._evidence(dossier), previous=json.dumps(dossier.critic_decision),
            candidates=json.dumps(sorted({h['label'] for h in hypotheses
                if isinstance(h, dict) and isinstance(h.get('label'), str)})),
            failures=json.dumps([{"step": t.get("step"), "tool": t.get("tool"),
                "errors": [a.get("validation_errors", []) for a in t.get("attempts", [])],
                "tool_error": t.get("error")} for t in transcript]))
        attempts = []
        for _ in range(2):
            raw = response = None
            try:
                with usage_context(role="final_critic", cluster_id=state["cluster_id"]):
                    response = self.llm_client.query(prompt, power="strong", image_base64=self._images(dossier) or None)
                raw = None if response.failed_after_retries else self._parse_turn(response)
                issues = validate_annotation(raw, dossier, require_checks=True)
                if isinstance(raw, dict):
                    if "comparison" in raw or "resolution" in raw:
                        issues.append("Final review uses the flat decision contract")
                    if raw.get("status") != "best_effort":
                        issues.append("Final recovery must remain best_effort")
            except Exception as exc:
                issues = [str(exc)]
            attempts.append({"proposal": raw, "raw_response": getattr(response, "content", None),
                             "validation_errors": issues})
            if not issues:
                value = normalize_annotation(raw, dossier)
                value["output_origin"] = "final_critic"
                return value, attempts
            prompt += ("\nYour previous final response was rejected. Reassess from actual measurements; "
                       "correct the claim or cite evidence in its actual scope. This is substantive revision, "
                       "not a formatting-only repair. Return the complete flat best_effort decision.\n"
                       + json.dumps({"rejected": raw, "errors": issues}))
        return None, attempts

    @staticmethod
    def _retain_output(dossier):
        """Keep a usable output even if no final model response can be validated."""
        from sclambda.annotation.niche.recovery import retain_failed_output
        retain_failed_output(dossier)

    def _evidence(self, dossier):
        return "\n\n".join(render(m) for m in dossier.measurements) + "\nTool notes: " + json.dumps(dossier.evidence_log)

    @staticmethod
    def _images(dossier):
        # Preserve view order and avoid paying twice for the same render.
        return list(dict.fromkeys(im for m in dossier.measurements for im in m.get("images", [])))

    def _availability(self, dossier):
        panel = set(self.accessor.gene_names.tolist())
        resolved = {r["item"]: r for m in dossier.measurements if m.get("kind") == "gene_resolution"
                    for r in m["rows"]}
        return json.dumps({"requested_name_resolution": resolved,
                          "unresolved_requested": sorted(set(dossier.requested_genes) - panel - set(resolved)),
                          "panel": sorted(panel) if len(panel) <= 2000 else "Query any symbol; tools resolve names and check availability."})

    def review(self, state, dossier, hypotheses, *, final=False, comparison_plans=None, comparison_feedback=None):
        # Retain candidate names only, never an actor's preferred ordering or
        # arguments. Earlier candidates are not implicitly refuted by omission.
        candidates = sorted({name.strip() for h in (hypotheses if isinstance(hypotheses, list) else []) if isinstance(h, dict)
                             for name in [h.get("label") or h.get("sub_region")]
                             if isinstance(name, str) and name.strip()})
        resolution_mode = bool(state.get("spatial_resolution_comparison", False))
        comparison_mode = state.get("spatial_comparison_review", False) and not resolution_mode
        prompt = _load_prompt("22_comparison_critic.md" if comparison_mode else "21_resolution_critic.md" if resolution_mode else "20_critic.md").format(
            organism=state.get("organism", "human"), tissue=state.get("tissue", "unknown"),
            # Names are an unordered search aid; actor arguments/ranking stay
            # in the trace. Global observations retain their source and views.
            measurements=self._evidence(dossier),
            candidates=json.dumps(candidates),
            comparison_plans=json.dumps(comparison_plans or []),
            comparison_feedback=json.dumps(comparison_feedback),
            previous=json.dumps(dossier.critic_decision),
            availability=self._availability(dossier),
            policy="FINAL REVIEW: no tool rounds remain. Select the best-supported label without untested qualifiers. "
                   "Retain unperformed pending_checks and uncertainty; do not manufacture resolution."
                   if final else "Tool rounds remain. Put concrete tests of decisive unresolved comparisons in pending_checks; they will execute before your next review.")
        attempts = []
        # Images remain inspectable by the deciding critic, not only prose from a reader.
        images = self._images(dossier)
        original = None
        for attempt in range(2):
            raw = None
            response = None
            try:
                with usage_context(role="critic_repair" if attempt else "critic",
                                   cluster_id=state["cluster_id"], final_review=final):
                    response = self.llm_client.query(prompt, power="strong", image_base64=(images or None) if attempt == 0 else None)
                raw = None if response.failed_after_retries else self._parse_turn(response)
                issues = validate_annotation(raw, dossier, require_resolution=resolution_mode, require_checks=True, require_comparison=comparison_mode)
                if attempt and isinstance(original, dict) and isinstance(raw, dict):
                    # Repair citation/JSON shape, not the biological assessment.
                    for key in ("label", "major_region", "status", "reason", "uncertainty", "alternatives", "next_question", "pending_checks"):
                        if key in original and raw.get(key) != original[key]:
                            issues.append(f"Format repair changed biological field {key}")
                    if "comparison" in original:
                        from sclambda.annotation.niche.comparison import without_measurement_ids
                        if without_measurement_ids(original["comparison"]) != without_measurement_ids(raw.get("comparison")):
                            issues.append("Format repair changed the biological comparison")
                    if isinstance(original.get("resolution"), dict):
                        for key, value in original["resolution"].items():
                            if key != "discriminating_support" and raw.get("resolution", {}).get(key) != value:
                                issues.append(f"Format repair changed resolution.{key}")
            except Exception as exc:
                issues = [str(exc)]
            attempts.append({"proposal": raw, "raw_response": getattr(response, "content", None),
                             "response_complete": bool(response is not None
                                 and not getattr(response, "failed_after_retries", False)
                                 and not CTDossierAgentNode._truncated(response)
                                 and isinstance(getattr(response, "content", None), str)
                                 and response.content.strip()),
                             "measurement_ids": [m["id"] for m in dossier.measurements],
                             "validation_errors": issues})
            if not issues:
                return normalize_annotation(raw, dossier), attempts
            if (isinstance(raw, dict) and "pending_checks" not in raw) or any(
                    issue.startswith(("Scope mismatch:", "Comparison:")) for issue in issues):
                # A missing execution plan or a population/reference mismatch
                # cannot be filled in as a harmless formatting repair.
                return None, attempts
            original = raw
            if not isinstance(raw, dict) or not (isinstance(raw.get("reason"), str) or isinstance(raw.get("comparison"), dict)):
                return None, attempts
            if getattr(response, "failed_after_retries", False) or CTDossierAgentNode._truncated(response):
                return None, attempts
            prompt = ("Repair JSON/citation formatting only. Preserve all biological claims, labels, status, "
                      "uncertainty and questions verbatim. Do not perform a new biological assessment. "
                      "For a comparison decision, correct measurement_id values only; preserve all other comparison fields exactly. "
                      "For a legacy decision, support and counterevidence must be flat arrays of reference objects with measurement_id and "
                      "genes/items (plus cell_type/reference for expression); never nest them under text/references. "
                      "If the decision is incomplete or ambiguous, return {}. Return ONLY JSON.\n"
                      + "Original response: " + json.dumps(getattr(response, "content", None))
                      + "\nMechanical errors: " + json.dumps(issues)
                      + "\nCitation vocabulary: " + json.dumps([{ "id": m["id"],
                          "cell_type": m.get("cell_type"), "reference": m.get("reference"),
                          "key": "genes" if m.get("kind") == "expression" else "items",
                          "names": [r.get("gene", r.get("item")) for r in m["rows"]]}
                          for m in dossier.measurements]))
        return None, attempts

    def _act(self, state, dossier, action):
        tool = str(action.get("tool", "")).lower()
        checks = action.get("reference_queries", action.get("queries", []))
        if "reference_queries" in action and "queries" in action and action["reference_queries"] != action["queries"]:
            raise ValueError("Conflicting reference_queries and queries; supply one unambiguous query list")
        if tool == "lookup_reference" and action.get("query"):
            checks = [{"query": action["query"], "terms": action.get("terms", [])}]
        if tool == "lookup_reference" and not checks:
            raise ValueError("lookup_reference requires query or reference_queries")
        if not isinstance(checks, list) or len(checks) > 2:
            raise ValueError("At most two focused reference queries per action")
        for check in checks:
            self._reference(state, dossier, check)
        if tool == "lookup_reference":
            return
        if tool in {"done", "finalize", "concede"}:
            return  # Always followed by critic review.
        if tool in {"test_signature", "test_signatures", "get_celltype_de", "get_de"}:
            panels = action.get("signatures") if tool == "test_signatures" else [action]
            if not isinstance(panels, list):
                raise ValueError("signatures must be a list of gene panels")
            for panel in panels:
                ct = panel.get("cell_type") or None
                genes = panel.get("genes") or []
                if not isinstance(genes, list) or not all(isinstance(g, str) for g in genes):
                    raise ValueError("genes must be a list of symbols")
                dossier.requested_genes = sorted(set(dossier.requested_genes) | set(genes))
                from sclambda.analysis.gene_names import resolve_genes
                from sclambda.annotation.niche.measurements import identify
                resolution = resolve_genes(genes, self.accessor.gene_names,
                    organism=state.get("organism", "human"), alias_path=state.get("gene_alias_path"),
                    cache_dir=state.get("evidence_cache_dir"), download=state.get("gene_alias_download", True))
                valid = resolution["genes"]
                add_measurement(dossier, identify({"kind": "gene_resolution", "target": dossier.cluster_id,
                    "scope": {**scope(state), "cell_type": ct}, "source": resolution["version"],
                    "source_error": resolution["source_error"],
                    "rows": [{"item": r["requested"], "text": json.dumps(r), **r} for r in resolution["rows"]]}))
                unresolved = [r for r in resolution["rows"] if r["feature"] is None]
                if unresolved:
                    note = f"Requested signature in {ct or 'whole niche'}: " + json.dumps(unresolved) + "; unresolved names are not expression zeros."
                    if not any(e.get("fact") == note for e in dossier.evidence_log):
                        dossier.add_evidence("tool_status", note)
                key = json.dumps([tool, ct, valid, scope(state)], sort_keys=True)
                if any(m.get("query") == key for m in dossier.measurements):
                    continue
                target, reference = self._indices(state, ct)
                if len(target) < MIN_CELLS or len(reference) < MIN_CELLS:
                    dossier.add_evidence("tool_status", f"Cannot compare {ct or 'whole niche'}: n_target={len(target)}, n_reference={len(reference)}; not evidence of absence.")
                    continue
                if tool.startswith("test_signature"):
                    if not valid:
                        dossier.add_evidence("tool_status", "No requested name resolved to a measured feature; consult the gene-resolution statuses before interpreting availability.")
                        continue
                    de = compute_de_for_genes(accessor=self.accessor, gene_names=valid,
                        target_indices=target, reference_indices=reference, de_method=state.get("de_method", "ttest"))
                    method = state.get("de_method", "ttest")
                else:
                    de = compute_pathology_de_from_indices(accessor=self.accessor, target_indices=target,
                        reference_indices=reference, cell_type_label=ct or "Niche overall", n_top_genes=state.get("n_top_genes", 20))
                    method = "ttest"
                value = expression_record(dossier.cluster_id, de, scope(state), ct, method)
                value["query"] = key
                add_measurement(dossier, value)
        elif tool == "consult_image":
            from .images import read_views
            slides = action.get("slides") or [action.get("slide", state.get("selected_slide"))]
            records = read_views(self.accessor, self.runner, self.llm_client, state, slides,
                                 action.get("question", "Describe the visible position and uncertainty."))
            for record in records:
                add_measurement(dossier, record)
            if not records:
                dossier.add_evidence("tool_status", "No usable image observation returned; position remains unmeasured.")
        else:
            raise ValueError(f"Unknown actor tool: {tool}")

    def _reference(self, state, dossier, check):
        if not state.get("reference_lookup", True):
            dossier.add_evidence("tool_status", "Reference lookup disabled; marker specificity has not been externally verified.")
            return
        try:
            from sclambda.annotation.niche.references import lookup_reference
            query = check["query"]
            if any(m.get("kind") == "literature" and m.get("query") == query for m in dossier.measurements):
                return
            value = lookup_reference(dossier.cluster_id, query,
                organism=state.get("organism", "human"), tissue=state.get("tissue", "unknown"),
                terms=check.get("terms", []), cache_dir=state.get("evidence_cache_dir"))
            add_measurement(dossier, value)
            if not value["rows"]:
                dossier.add_evidence("tool_status", "No reference passages retrieved for " + query + "; this does not refute a biological claim.")
        except Exception as exc:
            dossier.add_evidence("tool_status", f"Reference lookup failed: {type(exc).__name__}: {exc}; no claim was verified.")

    def _indices(self, state, cell_type=None):
        target, obs, rest = niche_and_rest_indices(self.accessor, state["cluster_id"], state["cell_type_col"],
            slice_key=state.get("slice_key"), slice_val=state.get("slice"), cross_slide=state.get("cross_slide", False))
        if cell_type:
            mask = obs[state["cell_type_col"]].values.astype(str) == cell_type
            target = target[mask[target]]
            rest = rest & mask
        return target, np.where(rest)[0]
