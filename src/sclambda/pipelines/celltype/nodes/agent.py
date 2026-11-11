"""Cell annotation: measurement-seeking actor and a single deciding critic."""
from __future__ import annotations
import json
import re
from copy import deepcopy
from sclambda.llm.usage import usage_context
import numpy as np
from sclambda.annotation.celltype.measurements import (
    MeasurementText, record, render, add_measurement, availability_block,
    validate_annotation, repair_annotation_refs, assessment_status,
)
from .common import (CTDossierMixin, CellTypingLLMNode, MIN_CELLS, _load_prompt,
                     gene_symbol_convention, rank_genes)


class CTDossierAgentNode(CTDossierMixin, CellTypingLLMNode):
    """One actor action (possibly several panels), then one critic decision.

    Only input/schema/reference checks can reject a response. Biological
    uncertainty is emitted beside the critic's best-supported label. There is
    no second reviewer, quotation audit, resolution ladder or stopping audit.
    """
    default_output = {"dossier_agent_transcript": [], "dossier_agent_steps": 0,
                      "dossier_review_complete": False, "dossier_stop_reason": "incomplete"}

    def execute(self, state):
        dossier = self.load_dossier(state)
        cid = state["cluster_id"]
        rg = state.get("rank_genes_result")
        if rg is not None:
            add_measurement(dossier, self._ranked_measurement(state, rg))
        panel = set(self.accessor.gene_names.tolist())
        seen_queries = set()
        if rg is not None:
            seen_queries.add(f"de::{state.get('n_top_genes', 50)}")
        # Reuse targeted measurements even across partially overlapping panels.
        for m in dossier.measurements:
            if (m.get("target") == cid and m.get("method") == state.get("de_method", "ttest")
                    and m.get("reference", "").startswith("all other dataset cells, target excluded;")):
                seen_queries.update(f"sig_gene::{row['gene']}" for row in m["rows"])
        transcript = []
        max_steps = max(1, int(state.get("dossier_agent_max_steps", 4)))
        hypotheses = []
        stop_reason = "incomplete"
        completed = False
        for step in range(1, max_steps + 1):
            actor_error = None
            try:
                prompt = _load_prompt("ct_dossier_agent.md").format(
                    organism=state.get("organism", "human"), tissue=state.get("tissue", "unknown"),
                    gene_symbol_convention=gene_symbol_convention(state.get("organism", "human")),
                    measurement_block=self._measurement_block(dossier),
                    gene_availability=availability_block(dossier, panel),
                    previous_decision=json.dumps(dossier.annotation, ensure_ascii=False),
                    peer_clusters=self._peer_block(state),
                )
                with usage_context(role="actor", iteration=step, cluster_id=cid):
                    response = self.llm_client.query(prompt, power="medium")
                turn = None if response.failed_after_retries else self._parse_turn(response)
                if not isinstance(turn, dict) or not isinstance(turn.get("action"), dict):
                    raise ValueError("actor response failed or was incomplete")
                hypotheses = [h for h in turn.get("hypotheses", []) if isinstance(h, dict)]
                action = turn["action"]
                transcript.append({"step": step, "tool": "actor", "reasoning": turn.get("reasoning", ""),
                                   "hypotheses": hypotheses, "action": action})
                self._execute_action(state, dossier, action, panel, seen_queries)
            except Exception as exc:
                actor_error = str(exc)
                transcript.append({"step": step, "tool": "actor", "status": "failed", "error": actor_error})
                # The critic can still choose from existing measurements.
                self.log_warning(cid, f"Actor failed; reviewing existing measurements: {exc}")
            dossier.iteration += 1
            verdict = self._run_critic(state, dossier, hypotheses, step, max_steps)
            attempts = self.output.pop("_critic_attempts", [])
            if verdict is None:
                transcript.append({"step": step, "tool": "critic", "status": "failed", "attempts": attempts})
                stop_reason = "critic_failed"
                break
            dossier.annotation = deepcopy(verdict)
            dossier.open_questions = [verdict["next_question"]] if verdict["next_question"] else []
            dossier.description = verdict["reason"]
            transcript.append({"step": step, "tool": "critic", "annotation": deepcopy(verdict),
                               "accepted_label": verdict["label"], "attempts": attempts})
            if not verdict["next_question"]:
                completed, stop_reason = True, "critic_complete"
                break
            if step == max_steps:
                completed, stop_reason = True, "budget_exhausted"
                break
            if actor_error:
                stop_reason = "actor_failed"
                break
        self.commit_dossier(dossier, cid)
        self.output.update(dossier_agent_transcript=transcript, dossier_agent_steps=step,
                           dossier_review_complete=completed, dossier_stop_reason=stop_reason,
                           annotation_status=dossier.annotation.get("status"),
                           annotation_uncertainty=dossier.annotation.get("uncertainty", ""),
                           annotation_alternatives=dossier.annotation.get("alternatives", []),
                           needs_subcluster=dossier.needs_subcluster)
        self.output["annotation_assessment"] = assessment_status(self.output)
        return self.output

    def _execute_action(self, state, dossier, action, panel, seen_queries):
        tool = str(action.get("tool", "")).lower()
        if tool in {"test_signature", "test_signatures"}:
            self._test_signatures(state, dossier, action, panel, seen_queries)
        elif tool == "get_de":
            n = max(1, int(action.get("n", 100)))
            key = f"de::{n}"
            if key not in seen_queries:
                self._append_measurement(dossier, "celltype_de", self._measure_de(state, n))
                seen_queries.add(key)
        elif tool in {"compare_to_parent", "compare_to_sibling"}:
            other = str(state.get("parent_cluster_id") or "") if tool == "compare_to_parent" else str(action.get("cluster_id") or "")
            requested = [g for g in action.get("genes", []) if isinstance(g, str)]
            dossier.requested_genes = sorted(set(dossier.requested_genes) | set(requested))
            genes = sorted(set(requested) & panel)
            key = f"comparison::{other}::{','.join(genes)}"
            if not other or not genes:
                dossier.add_evidence("neighbor", "Comparison needs a valid cluster and measurable genes.")
            elif key not in seen_queries:
                self._append_measurement(dossier, "neighbor", self._measure_sibling(state, other, genes))
                seen_queries.add(key)
        elif tool not in {"finalize", "done", "concede"}:
            dossier.add_evidence("signature", f"Unknown action {tool!r}; no measurement executed.")

    def _run_critic(self, state, dossier, hypotheses, step=1, max_steps=4):
        prompt = _load_prompt("ct_critic.md").format(
            organism=state.get("organism", "human"), tissue=state.get("tissue", "unknown"),
            measurement_block=self._measurement_block(dossier),
            gene_availability=availability_block(dossier, set(self.accessor.gene_names.tolist())),
            hypotheses=json.dumps(hypotheses, ensure_ascii=False),
            previous_decision=json.dumps(dossier.annotation, ensure_ascii=False),
            measurement_policy=("FINAL REVIEW: choose the best-supported option using existing measurements; record remaining uncertainty."
                                if step >= max_steps else "Further measurements are available if a concrete test could change the identity choice."),
        )
        attempts = []
        for attempt in range(2):
            raw, repairs = None, []
            try:
                with usage_context(role="critic_repair" if attempt else "critic", iteration=step,
                                   cluster_id=state["cluster_id"], final_review=step >= max_steps):
                    response = self.llm_client.query(prompt, power="strong")
                raw = None if response.failed_after_retries else self._parse_turn(response)
                value, repairs = repair_annotation_refs(raw, dossier)
                issues = validate_annotation(value, dossier)
            except Exception as exc:
                value, issues = None, [str(exc)]
            attempts.append({"proposal": raw, "reference_repairs": repairs, "validation_errors": issues})
            self.output["_critic_attempts"] = attempts
            if not issues:
                return value
            # One correction for transport/JSON/reference errors, by the SAME
            # critic. No change to measurements, no biological second opinion.
            if not attempt:
                prompt += "\nYour preceding response was invalid. Return the full corrected decision.\n" + json.dumps(
                    {"previous_response": value, "errors": issues}, ensure_ascii=False)
        return None

    @staticmethod
    def _measurement_block(dossier):
        if dossier.measurements:
            notices = [str(e.get("fact", "")) for e in dossier.evidence_log
                       if e.get("source") in {"signature", "celltype_de", "neighbor"}
                       and not e.get("measurement_id")]
            return "\n\n".join([*(render(m) for m in dossier.measurements), *notices])
        return "\n\n".join(
            f"[{e.get('source')}] {e.get('fact', '')}" for e in dossier.evidence_log
            if e.get("source") in {"signature", "celltype_de", "neighbor"}
        ) or "(No additional measurements executed yet.)"

    @staticmethod
    def _append_measurement(dossier, source, result):
        if isinstance(result, MeasurementText):
            add_measurement(dossier, result.measurement)
        dossier.add_evidence(source, str(result))
        if isinstance(result, MeasurementText):
            dossier.evidence_log[-1]["measurement_id"] = result.measurement["id"]

    def _test_signatures(self, state, dossier, action, all_gene_names, seen_queries):
        """Execute all requested panels before the next critic review.

        Names explain the query; they are not annotations or signature scores.
        Preserve the original per-gene measurement tables and legacy action.
        """
        signatures = action.get("signatures") if action.get("tool") == "test_signatures" else [action]
        if not isinstance(signatures, list) or not signatures:
            dossier.add_evidence("signature", "No signatures supplied; provide named gene panels.")
            return
        for signature in signatures:
            if not isinstance(signature, dict) or not isinstance(signature.get("genes"), list):
                dossier.add_evidence("signature", "Invalid signature: expected an object with a genes list; no measurement executed.")
                continue
            genes = list(dict.fromkeys(g for g in signature["genes"] if isinstance(g, str) and g))
            dossier.requested_genes = sorted(set(dossier.requested_genes) | set(genes))
            valid = [g for g in genes if g in all_gene_names]
            dropped = [g for g in genes if g not in all_gene_names]
            name = str(signature.get("name") or "unnamed")
            prefix = f"Requested panel '{name}' (hypothesis only, not a confirmed identity):\n"
            suffix = f" (unavailable, not in panel: {', '.join(dropped)})" if dropped else ""
            fresh = [g for g in valid if f"sig_gene::{g}" not in seen_queries]
            key = f"sig::{','.join(sorted(valid))}"
            if not valid:
                result = "No measurable genes; this is unavailable evidence, not absent expression."
            elif key in seen_queries or not fresh:
                result = f"Genes {valid} already measured — use the existing table."
            else:
                result = self._measure_signature(state, fresh)
                if isinstance(result, MeasurementText):
                    add_measurement(dossier, result.measurement)
                    seen_queries.update(f"sig_gene::{r['gene']}" for r in result.measurement["rows"])
                    seen_queries.add(key)
            dossier.add_evidence("signature", prefix + result + suffix)
            if isinstance(result, MeasurementText):
                dossier.evidence_log[-1]["measurement_id"] = result.measurement["id"]

    @staticmethod
    def _truncated(resp):
        return getattr(resp, "stop_reason", None) in {"max_tokens", "length", "max_output_tokens"}

    @classmethod
    def _parse_turn(cls, resp):
        # Do not repair a partial JSON response into an apparently valid verdict.
        if cls._truncated(resp) or not isinstance(resp.content, str):
            return None
        content = resp.content.strip()
        try:
            value = json.loads(content)
            return value if isinstance(value, dict) else None
        except (ValueError, TypeError):
            # Providers can prepend prose despite the JSON-only instruction.
            # Accept exactly one COMPLETE fenced JSON object, never repair a
            # partial object or choose between multiple competing verdicts.
            blocks = re.findall(r"```(?:json)?[ \t]*\n(.*?)\n[ \t]*```", content,
                                flags=re.DOTALL | re.IGNORECASE)
            if len(blocks) != 1 or content.count("```") != 2:
                return None
            try:
                value = json.loads(blocks[0])
                return value if isinstance(value, dict) else None
            except (ValueError, TypeError):
                return None

    def _peer_block(self, state) -> str:
        cid = str(state.get("cluster_id", ""))
        try:
            depth = self.accessor._parse_cluster_depth(cid)
            anns = self.accessor.get_annotations(depth=depth) or {}
        except Exception:  # noqa: BLE001
            return "(no peer clusters available)"
        peers = [f"- {k}: {v}" for k, v in anns.items() if k != cid and v]
        parent = state.get("parent_cluster_id")
        context = f"Parent {parent}: compare_to_parent measures target vs parent excluding target.\n" if parent else ""
        return context + ("\n".join(peers[:20]) or "(no annotated peer clusters yet)")

    # ---- deterministic tools ----

    def _target_reference(self, state):
        from sclambda.analysis.targeted_de import get_reference_indices
        cid = state["cluster_id"]
        target = self.accessor.get_cell_indices(cid)
        reference, _ = get_reference_indices(self.accessor, cid, target)
        return target, reference

    def _measure_signature(self, state, genes) -> str:
        from sclambda.analysis.targeted_de import compute_de_for_genes, format_genes_as_gene_table
        target, reference = self._target_reference(state)
        if len(target) < MIN_CELLS or len(reference) < MIN_CELLS:
            return f"signature: insufficient cells (cluster={len(target)}, rest={len(reference)})."
        de = compute_de_for_genes(
            self.accessor, list(genes), target, reference, de_method=state.get("de_method", "ttest"),
        )
        return MeasurementText(record(de.genes, target=state["cluster_id"],
            reference=f"all other dataset cells, target excluded; n={len(reference)}; target n={len(target)}",
            method=state.get("de_method", "ttest")))

    def _measure_de(self, state, n) -> str:
        res = rank_genes(self.accessor, state["cluster_id"], n_genes=int(n), de_method=state.get("de_method", "ttest"))
        return MeasurementText(self._ranked_measurement(state, res))

    def _ranked_measurement(self, state, res):
        method = state.get("de_method", "ttest")
        reference = ("all other dataset cells, target excluded; sampled up to 50000" if method == "ols"
                     else "whole dataset INCLUDING target (cached global statistics)")
        return record(res.genes, target=state["cluster_id"], reference=reference, method=method, ranked=True)

    def _measure_sibling(self, state, sibling_id, genes) -> str:
        from sclambda.analysis.targeted_de import compute_de_for_genes, format_genes_as_gene_table
        target = self.accessor.get_cell_indices(state["cluster_id"])
        try:
            # Explicit comparison can use a sibling or a parent; always disjoint.
            sib = np.setdiff1d(self.accessor.get_cell_indices(sibling_id), target)
        except Exception:  # noqa: BLE001
            return f"compare vs {sibling_id}: cluster not found."
        if len(target) < MIN_CELLS or len(sib) < MIN_CELLS:
            return f"compare vs {sibling_id}: insufficient cells (cluster={len(target)}, sibling={len(sib)})."
        de = compute_de_for_genes(
            self.accessor, list(genes), target, sib, de_method=state.get("de_method", "ttest"),
        )
        return MeasurementText(record(de.genes, target=state["cluster_id"],
            reference=f"cluster {sibling_id} excluding target; n={len(sib)}; target n={len(target)}",
            method=state.get("de_method", "ttest")))
