"""celltype pipeline: hypotheses."""
from __future__ import annotations
from .common import (
    CONFIDENCE_ORDER,
    CTDossierMixin,
    CellTypingLLMNode,
    CellTypingNode,
    ParseSchema,
    SequenceMatcher,
    SignatureFormatter,
    _load_prompt,
    format_gene_list,
    format_hypotheses,
    format_top_genes_de_table,
    gene_symbol_convention,
    normalize,
    seed_ct_dossier,
    targeted_de_signatures,
)

class CTGenerationNode(CTDossierMixin, CellTypingLLMNode):
    """Generate and parse cell-type hypotheses.

    With ``use_dossier`` it only initializes the per-cluster dossier; the
    dossier agent proposes hypotheses itself.
    """

    parse_schema = ParseSchema(
        format="list",
        defaults={
            "cell_type": "unknown",
            "confidence": "low",
            "justification": lambda item: (
                f"This cluster expresses genes associated with {item.get('cell_type', 'unknown')}. "
                f"However, further evidence is needed to confirm the assignment."
            ),
            "supporting_genes": [],
        },
        sort_by="confidence",
        sort_order=CONFIDENCE_ORDER,
    )

    fallback_hypothesis = {
        "cell_type": "unknown",
        "confidence": "low",
        "justification": "Failed to generate hypotheses.",
        "supporting_genes": [],
    }

    default_output = {
        "cell_type": "unknown",
        "cell_types": ["unknown"],
        "hypotheses": [fallback_hypothesis],
        "generation_prompt": "",
        "raw_llm_response": "",
        "parse_response": "",
        "ct_generation_failed": True,
    }

    _INVALID_CELL_TYPES = {"unknown", "none", "null", "n/a", "na", "unidentified", ""}

    def execute(self, state: "CellTypingState") -> dict:
        if state.get("use_dossier"):
            return self._execute_with_dossier(state)

        cluster_id = state["cluster_id"]
        top_genes = state["top_genes"]

        # Format DATA: prefer full DE table with LFCs when available
        rank_genes_result = state.get("rank_genes_result")
        if rank_genes_result is not None:
            data_str = format_top_genes_de_table(rank_genes_result)
        else:
            data_str = format_gene_list(top_genes, numbered=True)

        variables = {
            "organism": state.get("organism", "human"),
            "tissue": state.get("tissue", "unknown"),
            "parent_cell_type_info": self._get_parent_context(state),
            "DATA": data_str,
        }

        for attempt in range(2):
            result = self.chain.execute(
                prompt_file="1_ct_hypothesis_generation.md",
                parse_prompt_file="2_ct_parse_response.md",
                variables=variables,
                parse_as="list",
            )

            self.output["generation_prompt"] = result.formatted_prompt
            self.output["raw_llm_response"] = result.raw_response
            self.output["parse_response"] = result.parse_response
            self.output["ct_generation_failed"] = not result.success

            # Parse hypotheses - schema handles defaults, fallback covers total failure
            hypotheses = self.parse_llm_response(
                result.parse_response,
                self.parse_schema,
                fallback=[self.fallback_hypothesis],
            )
            if not hypotheses:
                hypotheses = [self.fallback_hypothesis]

            cell_type = hypotheses[0]["cell_type"]

            # Retry once if best hypothesis is invalid
            if (cell_type or "").strip().lower() not in self._INVALID_CELL_TYPES:
                break
            if attempt == 0:
                self.log_warning(cluster_id, f"Best cell type is '{cell_type}', retrying generation")

        self.output["cell_type"] = cell_type
        self.output["cell_types"] = [h["cell_type"] for h in hypotheses]
        self.output["hypotheses"] = hypotheses

        self.update_annotation(cluster_id, cell_type)
        self.update_hypotheses(cluster_id, hypotheses)
        self.log_info(cluster_id, f"Generated {len(hypotheses)} hypotheses, best={cell_type}")

        return self.output

    def _execute_with_dossier(self, state: "CellTypingState") -> dict:
        """Initialize storage; the actor handles hypotheses in one structured call.

        No separate prose-generation/parser pair or catalog lifecycle is needed.
        Existing measurements may be retained when resuming an experiment.
        """
        dossier = self.load_dossier(state)
        self.commit_dossier(dossier, state["cluster_id"])
        self.output["ct_generation_failed"] = False
        return self.output


class MarkerRefinementNode(CellTypingLLMNode):
    """Suggest additional genes to refine/validate cell type predictions."""

    default_output = {
        "refinement_prompt": "",
        "refinement_response": "",
        "refinement_signatures": {},
        "refinement_confidence_decision": "needs_refinement",
        "refinement_best_cell_type": None,
        "refinement_reasoning": None,
        "cell_type": None,
        "cell_types": None,
    }

    def execute(self, state: "CellTypingState") -> dict:
        cluster_id = state["cluster_id"]
        # Prefer updated_hypotheses from reevaluation loop; fall back to initial hypotheses
        hypotheses = state.get("updated_hypotheses") or state.get("hypotheses", [])
        top_genes = state.get("top_genes", [])

        # Preserve incoming hypotheses unless this node explicitly changes them.
        if hypotheses:
            self.output["hypotheses"] = hypotheses
            self.output["updated_hypotheses"] = hypotheses

        # Always pass at least 3 hypotheses: prefer medium/high, pad with low if needed
        high_med = [h for h in hypotheses if h.get("confidence", "").lower() in ("medium", "high")]
        low = [h for h in hypotheses if h.get("confidence", "").lower() == "low"]
        relevant_hypotheses = high_med + [h for h in low if h not in high_med]
        relevant_hypotheses = relevant_hypotheses[: max(3, len(high_med))]

        if not relevant_hypotheses:
            self.log_info(cluster_id, "No hypotheses available, skipping refinement")
            return self.output

        # Format names of signatures from previous refinement rounds
        prev_sigs = state.get("refinement_signatures", {})
        if prev_sigs:
            names = ", ".join(prev_sigs.keys())
            previously_tested = f"\n**Previously tested signatures:** {names}\n"
        else:
            previously_tested = ""

        organism = state.get("organism", "human")
        variables = {
            "organism": organism,
            "tissue": state.get("tissue", "unknown"),
            "hypotheses": format_hypotheses(relevant_hypotheses),
            "top_genes": format_gene_list(top_genes[: state.get("n_top_genes", 20)], numbered=False),
            "previously_tested_signatures": previously_tested,
            "gene_symbol_convention": gene_symbol_convention(organism),
        }

        result = self.chain.execute(
            prompt_file="3_marker_refinement.md",
            variables=variables,
            parse_as="dict",
        )

        self.output["refinement_prompt"] = result.formatted_prompt
        self.output["refinement_response"] = result.raw_response

        confidence_decision = result.data.get("confidence_decision", "needs_refinement")
        best_cell_type = result.data.get("best_cell_type")
        reasoning = result.data.get("reasoning")
        signatures = result.data.get("signatures", {})

        # Merge with previously tested signatures so the list grows across rounds
        merged_signatures = dict(prev_sigs)
        merged_signatures.update(signatures)
        self.output["refinement_signatures"] = merged_signatures
        self.output["refinement_confidence_decision"] = confidence_decision
        self.output["refinement_best_cell_type"] = best_cell_type
        self.output["refinement_reasoning"] = reasoning

        if confidence_decision == "confident" and best_cell_type:
            # Find the best-matching hypothesis and promote it to the top,
            # preserving its existing justification rather than overwriting.
            hypotheses = [dict(h) for h in (state.get("updated_hypotheses") or state.get("hypotheses", []))]
            best_norm = normalize(best_cell_type)

            # Score each hypothesis by similarity to the selected cell type
            scored = []
            for i, h in enumerate(hypotheses):
                h_norm = normalize(h.get("cell_type", ""))
                ratio = SequenceMatcher(None, best_norm, h_norm).ratio()
                scored.append((i, ratio))
            scored.sort(key=lambda x: x[1], reverse=True)

            matched_idx = None
            if scored and scored[0][1] >= 0.6:
                matched_idx = scored[0][0]

            if matched_idx is not None:
                # Promote matching hypothesis to top
                matched = dict(hypotheses.pop(matched_idx))
                matched["confidence"] = "high"
                if matched.get("supporting_genes") is None:
                    matched["supporting_genes"] = []
                hypotheses.insert(0, matched)
            elif hypotheses:
                # No exact match — update the top hypothesis cell type,
                # keep its existing justification
                best = dict(hypotheses[0])
                best["cell_type"] = best_cell_type
                best["confidence"] = "high"
                if best.get("supporting_genes") is None:
                    best["supporting_genes"] = []
                hypotheses[0] = best
            else:
                hypotheses = [
                    {
                        "cell_type": best_cell_type,
                        "confidence": "high",
                        "justification": f"This cluster expresses markers associated with {best_cell_type}.",
                        "reasoning": f"Cell type {best_cell_type} is supported by marker refinement evidence.",
                        "supporting_genes": [],
                    }
                ]

            self.update_annotation(cluster_id, best_cell_type)
            # Force remapping in OntCellTypeNode when label changed mid-pipeline.
            if best_cell_type != state.get("cell_type"):
                self.runner.with_lock(self.accessor.set_ont_cell_type, cluster_id, None)
            self.update_hypotheses(cluster_id, hypotheses)
            self.log_info(cluster_id, f"Confident annotation: {best_cell_type}")
            self.output["cell_type"] = best_cell_type
            self.output["cell_types"] = [best_cell_type]
            self.output["hypotheses"] = hypotheses
            self.output["updated_hypotheses"] = hypotheses
        else:
            self.log_info(cluster_id, f"Needs refinement - {len(signatures)} signatures")

        return self.output


class SignatureDENode(CellTypingNode):
    """Compute differential expression for refinement signatures."""

    default_output = {
        "signature_de_results": [],
        "signature_de_result_obj": None,
    }

    def execute(self, state: "CellTypingState") -> dict:
        cluster_id = state["cluster_id"]
        refinement_signatures = state.get("refinement_signatures", {})

        if not refinement_signatures:
            self.log_info(cluster_id, "No refinement signatures, skipping DE")
            from sclambda.analysis.targeted_de import SignaturesDEResult

            self.output["signature_de_result_obj"] = SignaturesDEResult(
                cluster_id=cluster_id,
                reference_type="no_signatures",
                n_target=0,
                n_reference=0,
                signatures=[],
            )
            return self.output

        de_result = targeted_de_signatures(
            accessor=self.accessor,
            cluster_id=cluster_id,
            signatures=refinement_signatures,
            de_method=state.get("de_method", "ttest"),
        )

        signature_results = de_result.to_serializable()
        total_genes = sum(len(s["genes"]) for s in signature_results)
        total_enriched = sum(s["n_enriched"] for s in signature_results)
        total_depleted = sum(s["n_depleted"] for s in signature_results)
        total_neutral = total_genes - total_enriched - total_depleted

        self.log_info(
            cluster_id,
            f"Signature DE complete - {len(signature_results)} signatures, "
            f"{total_genes} genes ({total_enriched} enriched, {total_depleted} depleted, {total_neutral} n.s.)",
        )

        self.output["signature_de_results"] = signature_results
        self.output["signature_de_result_obj"] = de_result
        return self.output


class CTReevaluationNode(CellTypingLLMNode):
    """Reevaluate cell type hypotheses based on signature DE results."""

    parse_schema = ParseSchema(
        format="list",
        defaults={
            "cell_type": "unknown",
            "confidence": "low",
            "justification": lambda item: (
                f"This cluster expresses genes associated with {item.get('cell_type', 'unknown')}. "
                f"However, further evidence is needed to confirm the assignment."
            ),
            "supporting_genes": [],
        },
        sort_by="confidence",
        sort_order=CONFIDENCE_ORDER,
    )

    _dict_parse_schema = ParseSchema(format="dict")

    default_output = {
        "reevaluation_prompt": "",
        "reevaluation_response": "",
        "reevaluation_parse_response": "",
        "hypotheses": [],
        "updated_hypotheses": [],
        "reevaluation_iteration": 0,
        "reevaluation_best_confidence": "unknown",
        "reevaluation_confidence_decision": "needs_refinement",
        "cell_type": None,
        "cell_types": [],
    }

    def execute(self, state: "CellTypingState") -> dict:
        cluster_id = state["cluster_id"]
        hypotheses = state.get("updated_hypotheses") or state.get("hypotheses", [])
        signature_de_result_obj = state.get("signature_de_result_obj")
        organism = state.get("organism", "human")
        tissue = state.get("tissue", "unknown")

        self.output["reevaluation_iteration"] = state.get("reevaluation_iteration", 0) + 1
        self.output["hypotheses"] = hypotheses
        self.output["updated_hypotheses"] = hypotheses  # fallback to current
        if hypotheses:
            self.output["reevaluation_best_confidence"] = str(hypotheses[0].get("confidence", "unknown")).lower()

        # Always pass at least 3 hypotheses: prefer medium/high, pad with low if needed
        high_med = [h for h in hypotheses if h.get("confidence", "").lower() in ("medium", "high")]
        low = [h for h in hypotheses if h.get("confidence", "").lower() == "low"]
        relevant_hypotheses = high_med + [h for h in low if h not in high_med]
        relevant_hypotheses = relevant_hypotheses[: max(3, len(high_med))]

        if not relevant_hypotheses:
            self.log_info(cluster_id, "No hypotheses available, skipping reevaluation")
            return self.output

        # Format hypotheses as plain list (DE evidence comes from signatures)
        hypotheses_str = self._format_hypotheses_plain(relevant_hypotheses)

        # Format signature results
        refinement_signatures = state.get("refinement_signatures", {})
        signature_de_results = state.get("signature_de_results", [])
        signature_formatter = SignatureFormatter(
            refinement_signatures,
            signature_de_result_obj,
            signature_de_results=signature_de_results,
            signature_de_results_present="signature_de_results" in state,
        )

        # Generate reevaluation
        reevaluation_prompt = _load_prompt("5_ct_reevaluation.md")
        formatted_prompt = reevaluation_prompt.format(
            organism=organism,
            tissue=tissue,
            parent_cell_type_info=self._get_parent_context(state),
            hypotheses=hypotheses_str,
            signature_results=signature_formatter.format(),
        )
        self.output["reevaluation_prompt"] = formatted_prompt

        gen_response = self.llm_client.query(formatted_prompt, power="medium")
        self.output["reevaluation_response"] = gen_response.content

        if gen_response.failed_after_retries:
            return self.output

        # Parse with second LLM call
        parse_prompt = _load_prompt("6_ct_reevaluation_parse.md")
        formatted_parse = parse_prompt.format(reevaluation_response=gen_response.content)
        parse_resp = self.llm_client.query(formatted_parse, power="weak")
        self.output["reevaluation_parse_response"] = parse_resp.content

        if not parse_resp.failed_after_retries:
            parsed = self.parse_llm_response(
                parse_resp.content,
                self._dict_parse_schema,
                fallback={},
            )
            decision = parsed.get("confidence_decision", "needs_refinement")
            self.output["reevaluation_confidence_decision"] = (
                "confident" if decision == "confident" else "needs_refinement"
            )
            hyps = parsed.get("hypotheses", [])
            if hyps:
                self.output["updated_hypotheses"] = self.parse_schema.apply_defaults(hyps)

        # Extract best cell type
        updated = self.output["updated_hypotheses"]
        if updated:
            self.output["cell_type"] = updated[0].get("cell_type", state.get("cell_type"))
            self.output["cell_types"] = [h.get("cell_type") for h in updated if h.get("cell_type")]
            self.output["reevaluation_best_confidence"] = str(updated[0].get("confidence", "unknown")).lower()
            self.output["hypotheses"] = updated

        cell_type = self.output["cell_type"]
        confidence = self.output["reevaluation_best_confidence"]
        decision = self.output["reevaluation_confidence_decision"]
        iteration = self.output["reevaluation_iteration"]
        self.log_info(
            cluster_id,
            f"CT reevaluation complete (iter {iteration}) - {len(updated)} hypotheses, "
            f"best: {cell_type}, confidence: {confidence}, decision: {decision}",
        )

        self.update_hypotheses(cluster_id, updated)
        if cell_type and cell_type != state.get("cell_type"):
            self.update_annotation(cluster_id, cell_type)

        return self.output

    @staticmethod
    def _format_hypotheses_plain(hypotheses: list[dict]) -> str:
        """Format hypotheses with justification (or fallback to supporting genes)."""
        lines = []
        for h in hypotheses:
            ct = h.get("cell_type", "unknown")
            confidence = h.get("confidence", "unknown")
            justification = h.get("justification", "")
            if justification:
                lines.append(f"- **{ct}** (confidence: {confidence})\n  Justification: {justification}")
            else:
                genes = h.get("supporting_genes", [])
                gene_str = ", ".join(genes[:10]) if genes else "none"
                lines.append(f"- **{ct}** (confidence: {confidence}) — supporting genes: {gene_str}")
        return "\n".join(lines) if lines else "No hypotheses available."
