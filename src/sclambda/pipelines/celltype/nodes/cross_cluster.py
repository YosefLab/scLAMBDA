"""celltype pipeline: cross cluster."""
from __future__ import annotations
from .common import (
    CTDossierMixin,
    CellTypingLLMNode,
    CellTypingNode,
    ParseSchema,
    _load_prompt,
    normalize,
)

class CrossClusterSuggestionNode(CellTypingLLMNode):
    """Suggest which clusters to compare via DE for annotation correction.

    Runs during post-processing. Given this cluster's annotation and all
    other same-depth annotations, the LLM selects clusters with potentially
    confusable cell types and proposes discriminating genes to check.
    """

    parse_schema = ParseSchema(
        format="dict",
        defaults={
            "target_genes": [],
            "comparisons": [],
        },
    )

    default_output = {
        "cross_cluster_suggestion_prompt": "",
        "cross_cluster_suggestion_response": "",
        "cross_cluster_suggestion_parse_response": "",
        "cross_cluster_comparisons": [],
        "cross_cluster_target_genes": [],
    }

    def execute(self, state: "CellTypingState") -> dict:
        cluster_id = state["cluster_id"]
        cross_cluster_context = state.get("cross_cluster_context")

        if not cross_cluster_context:
            self.log_info(cluster_id, "No cross-cluster context, skipping comparison")
            return self.output

        hypotheses = state.get("updated_hypotheses") or state.get("hypotheses", [])
        if not hypotheses:
            self.log_info(cluster_id, "No hypotheses, skipping cross-cluster comparison")
            return self.output

        best = hypotheses[0]
        cell_type = best.get("cell_type", state.get("cell_type", "unknown"))
        confidence = best.get("confidence", "unknown")
        justification = best.get("justification", "") or best.get("reasoning", "")

        # Format alternative hypotheses (top 5, sorted high→medium→low, excluding best)
        _CONF_ORDER = {"high": 0, "medium": 1, "low": 2}
        alt_hyps = sorted(hypotheses[1:], key=lambda h: _CONF_ORDER.get(h.get("confidence", "low").lower(), 3))
        alternative_hypotheses = (
            "\n".join(
                f"- {h.get('cell_type', 'unknown')} (confidence: {h.get('confidence', 'unknown')})"
                for h in alt_hyps[:5]
            )
            if alt_hyps
            else "None"
        )

        organism = state.get("organism", "human")
        tissue = state.get("tissue", "unknown")

        prompt_template = _load_prompt("10_cross_cluster_suggestion.md")
        formatted_prompt = prompt_template.format(
            organism=organism,
            tissue=tissue,
            cluster_id=cluster_id,
            cell_type=cell_type,
            confidence=confidence,
            justification=justification or "none",
            alternative_hypotheses=alternative_hypotheses,
            other_clusters=cross_cluster_context,
        )
        self.output["cross_cluster_suggestion_prompt"] = formatted_prompt

        gen_response = self.llm_client.query(formatted_prompt, power="medium")
        self.output["cross_cluster_suggestion_response"] = gen_response.content

        if gen_response.failed_after_retries:
            self.log_warning(cluster_id, "Cross-cluster suggestion LLM call failed")
            return self.output

        parse_prompt = _load_prompt("11_cross_cluster_suggestion_parse.md")
        formatted_parse = parse_prompt.format(
            suggestion_response=gen_response.content,
        )
        parse_resp = self.llm_client.query(formatted_parse, power="weak")
        self.output["cross_cluster_suggestion_parse_response"] = parse_resp.content

        if not parse_resp.failed_after_retries:
            parsed = self.parse_llm_response(
                parse_resp.content,
                self.parse_schema,
                fallback={},
            )
            if parsed:
                target_genes = parsed.get("target_genes", [])
                comparisons = parsed.get("comparisons", [])
                # Filter to valid cluster IDs (must appear in context)
                valid = [c for c in comparisons if c.get("cluster_id")]
                self.output["cross_cluster_target_genes"] = target_genes
                self.output["cross_cluster_comparisons"] = valid
                self.log_info(
                    cluster_id,
                    f"Suggested {len(valid)} cross-cluster comparison(s) with {len(target_genes)} genes",
                )

        return self.output


class CrossClusterDENode(CellTypingNode):
    """Compute DE for a shared gene panel across all comparator clusters.

    Non-LLM node. Uses the single diagnostic gene panel from the suggestion
    step and computes DE against each comparator cluster and the global
    reference, formatted as a unified table with one LFC column per comparator.
    """

    default_output = {
        "cross_cluster_de_summary": "",
    }

    def execute(self, state: "CellTypingState") -> dict:
        from sclambda.analysis.targeted_de import compute_de_for_genes, get_reference_indices

        cluster_id = state["cluster_id"]
        target_genes = state.get("cross_cluster_target_genes", [])
        comparisons = state.get("cross_cluster_comparisons", [])

        if not comparisons or not target_genes:
            self.log_info(cluster_id, "No comparisons or target genes, skipping cross-cluster DE")
            return self.output

        cell_type = state.get("cell_type", "unknown")

        target_indices = self.accessor.get_cell_indices(cluster_id)
        n_target = len(target_indices)
        if n_target == 0:
            self.log_warning(cluster_id, "No cells in target cluster")
            return self.output

        # Reference = all cells except target
        reference_indices, _ = get_reference_indices(self.accessor, cluster_id, target_indices)

        # DE vs global reference (shared across all comparisons)
        de_vs_ref = compute_de_for_genes(
            self.accessor,
            target_genes,
            target_indices,
            reference_indices,
            de_method=state.get("de_method", "ttest"),
        )

        # Compute DE vs each comparator cluster
        valid_comps = []  # (comp_cell_type, comp_cluster_id, n_comp, de_result)
        all_not_found = set(de_vs_ref.genes_not_found)
        for comp in comparisons:
            comp_cluster_id = comp.get("cluster_id", "")
            comp_cell_type = comp.get("cell_type", "unknown")

            if not comp_cluster_id:
                continue

            comp_indices = self.accessor.get_cell_indices(comp_cluster_id)
            n_comp = len(comp_indices)
            if n_comp == 0:
                self.log_warning(
                    cluster_id,
                    f"No cells in comparison cluster {comp_cluster_id}, skipping",
                )
                continue

            de_vs_comp = compute_de_for_genes(
                self.accessor,
                target_genes,
                target_indices,
                comp_indices,
                de_method=state.get("de_method", "ttest"),
            )
            all_not_found.update(de_vs_comp.genes_not_found)
            valid_comps.append((comp_cell_type, comp_cluster_id, n_comp, de_vs_comp))

        if not valid_comps:
            self.log_info(cluster_id, "No valid comparator clusters found")
            return self.output

        # Build unified table header
        header_parts = [
            f"### {cell_type} (cluster {cluster_id}, {n_target} cells) — cross-cluster comparison\n",
            "Comparators: ",
            ", ".join(f"Cluster {comp_cid} ({nc} cells)" for _, comp_cid, nc, _ in valid_comps),
            "\n",
        ]

        # Column headers: Gene | Expression | Frac | LFC vs ref | LFC vs <cluster_id> | ...
        col_headers = ["Gene", "Expression", "Frac Expressed", "LFC vs reference"]
        for _, comp_cid, _, _ in valid_comps:
            col_headers.append(f"LFC vs {comp_cid}")
        header_line = "| " + " | ".join(col_headers) + " |"
        sep_line = "|" + "|".join(["---"] * len(col_headers)) + "|"

        table_lines = [header_line, sep_line]

        for gene_name in target_genes:
            gde_ref = de_vs_ref.get_gene(gene_name)
            if gde_ref is None:
                continue
            # Skip genes absent in the target cluster
            if gde_ref.mean_target < 0.005 and gde_ref.frac_non_zero_target < 0.005:
                continue

            row = [
                gene_name,
                f"{gde_ref.mean_target:.2f}",
                f"{gde_ref.frac_non_zero_target:.2f}",
                f"{gde_ref.lfc:+.2f}",
            ]
            for _, _, _, de_vs_comp in valid_comps:
                gde_comp = de_vs_comp.get_gene(gene_name)
                if gde_comp:
                    row.append(f"{gde_comp.lfc:+.2f}")
                else:
                    row.append("N/A")
            table_lines.append("| " + " | ".join(row) + " |")

        not_found = list(all_not_found)
        summary = "".join(header_parts) + "\n".join(table_lines)
        if not_found:
            summary += f"\n\nGenes not found: {', '.join(not_found)}"

        self.output["cross_cluster_de_summary"] = summary
        self.log_info(
            cluster_id,
            f"Computed cross-cluster DE for {len(valid_comps)} comparison(s) with {len(target_genes)} genes",
        )

        return self.output


class CrossClusterCorrectionNode(CTDossierMixin, CellTypingLLMNode):
    """Correct cell type annotation based on cross-cluster DE evidence.

    Runs during post-processing. Given the DE comparison results, the LLM
    decides whether the original annotation is correct or needs correction.

    On the dossier path (``use_dossier``) the correction is folded THROUGH the
    dossier — sibling labels are seeded as ``neighbor`` candidates and the
    reasoning is merged via ``apply_update`` — so the reinstatement /
    single-leader / sticky-leader invariants and the evidence record apply here
    too, instead of a direct label overwrite (§3).
    """

    parse_schema = ParseSchema(
        format="dict",
        defaults={
            "cell_type": "unknown",
            "confidence": "low",
            "justification": lambda d: d.get("reasoning", "No updated justification provided."),
            "reasoning": "No correction reasoning available.",
            "changed": False,
        },
    )

    default_output = {
        "cross_cluster_correction_prompt": "",
        "cross_cluster_correction_response": "",
        "cross_cluster_correction_parse_response": "",
        "cross_cluster_updated_justification": "",
        "cross_cluster_changed": False,
    }

    def execute(self, state: "CellTypingState") -> dict:
        cluster_id = state["cluster_id"]
        de_summary = state.get("cross_cluster_de_summary", "")
        incoming_hypotheses = state.get("updated_hypotheses") or state.get("hypotheses", [])
        if incoming_hypotheses:
            self.output["hypotheses"] = incoming_hypotheses
            self.output["updated_hypotheses"] = incoming_hypotheses

        if not de_summary:
            self.log_info(cluster_id, "No DE summary, skipping cross-cluster correction")
            return self.output

        cell_type = state.get("cell_type", "unknown")
        hypotheses = state.get("updated_hypotheses") or state.get("hypotheses", [])
        best = hypotheses[0] if hypotheses else {}
        confidence = best.get("confidence", "unknown")
        # Use justification specifically tied to the currently assigned annotation
        # whenever possible (not always the top hypothesis if they diverge).
        assigned_hyp = None
        target_norm = normalize(str(cell_type)) if cell_type else ""
        for h in hypotheses:
            h_cell_type = h.get("cell_type")
            if h_cell_type and normalize(str(h_cell_type)) == target_norm:
                assigned_hyp = h
                break
        assigned_hyp_justification = ""
        if assigned_hyp:
            assigned_hyp_justification = assigned_hyp.get("justification", "") or assigned_hyp.get("reasoning", "")
        assigned_justification = (
            state.get("justification_summary")
            or assigned_hyp_justification
            or best.get("justification", "")
            or best.get("reasoning", "")
            or ""
        )
        organism = state.get("organism", "human")
        tissue = state.get("tissue", "unknown")

        # Format top 5 alternative hypotheses (excluding assigned), sorted high→medium→low
        _CONF_ORDER = {"high": 0, "medium": 1, "low": 2}
        alt_hyps = [h for h in hypotheses if normalize(str(h.get("cell_type", ""))) != target_norm]
        alt_hyps = sorted(alt_hyps, key=lambda h: _CONF_ORDER.get(h.get("confidence", "low").lower(), 3))
        alternative_hypotheses = (
            "\n".join(
                f"- {h.get('cell_type', 'unknown')} (confidence: {h.get('confidence', 'unknown')})"
                for h in alt_hyps[:5]
            )
            if alt_hyps
            else "None"
        )

        # On the dossier path, give the correction step the carried-forward
        # dossier (confirmed facts, exclusions, candidate pool, resolution ladder)
        # so promotions are made WITHIN the hard constraints, not as free relabels.
        if state.get("use_dossier") and state.get("ct_dossier"):
            dossier_block = self.load_dossier(state).to_markdown()
        else:
            dossier_block = "(no dossier — reason from the evidence below)"

        prompt_template = _load_prompt("12_cross_cluster_correction.md")
        formatted_prompt = prompt_template.format(
            organism=organism,
            tissue=tissue,
            cluster_id=cluster_id,
            cell_type=cell_type,
            confidence=confidence,
            assigned_cell_type_justification=assigned_justification or "none",
            alternative_hypotheses=alternative_hypotheses,
            de_comparisons=de_summary,
            dossier_block=dossier_block,
        )
        self.output["cross_cluster_correction_prompt"] = formatted_prompt

        gen_response = self.llm_client.query(formatted_prompt, power="medium")
        self.output["cross_cluster_correction_response"] = gen_response.content

        if gen_response.failed_after_retries:
            self.log_warning(cluster_id, "Cross-cluster correction LLM call failed")
            return self.output

        # Dossier path: fold the correction through the dossier instead of a
        # direct label overwrite (§3).
        if state.get("use_dossier") and state.get("ct_dossier"):
            return self._commit_via_dossier(state, gen_response.content, cell_type)

        parse_prompt = _load_prompt("13_cross_cluster_correction_parse.md")
        formatted_parse = parse_prompt.format(
            correction_response=gen_response.content,
        )
        parse_resp = self.llm_client.query(formatted_parse, power="weak")
        self.output["cross_cluster_correction_parse_response"] = parse_resp.content

        if not parse_resp.failed_after_retries:
            parsed = self.parse_llm_response(
                parse_resp.content,
                self.parse_schema,
                fallback={},
            )
            updated_justification = ""
            if parsed:
                updated_justification = parsed.get("justification") or parsed.get("reasoning") or ""
            if updated_justification:
                self.output["cross_cluster_updated_justification"] = updated_justification
                self.output["justification_summary"] = updated_justification

            if parsed and parsed.get("changed") and parsed.get("cell_type"):
                new_cell_type = parsed["cell_type"]
                reasoning = (
                    parsed.get("justification")
                    or parsed.get("reasoning")
                    or f"Corrected from '{cell_type}' to '{new_cell_type}' based on cross-cluster DE evidence."
                )
                self.output["cell_type"] = new_cell_type
                self.output["cell_type_ont"] = None
                self.apply_label_change(cluster_id, cell_type, new_cell_type, reasoning)
                self.output["cross_cluster_changed"] = True
                corrected_hypotheses = list(self.accessor.get_hypotheses(cluster_id) or [])
                if corrected_hypotheses:
                    self.output["hypotheses"] = corrected_hypotheses
                    self.output["updated_hypotheses"] = corrected_hypotheses
                self.log_info(
                    cluster_id,
                    f"Cross-cluster correction: {cell_type} -> {new_cell_type} "
                    f"(confidence: {parsed.get('confidence', 'unknown')})",
                )
            else:
                # Even when the label is unchanged, persist the updated
                # justification so downstream outputs reflect the latest reasoning.
                if updated_justification:
                    current_hypotheses = list(self.accessor.get_hypotheses(cluster_id) or incoming_hypotheses or [])
                    if current_hypotheses:
                        best_h = dict(current_hypotheses[0])
                        best_h["cell_type"] = cell_type
                        best_h["justification"] = updated_justification
                        best_h["reasoning"] = updated_justification
                        current_hypotheses[0] = best_h
                    else:
                        current_hypotheses = [
                            {
                                "cell_type": cell_type,
                                "confidence": str(parsed.get("confidence", "unknown")).lower() if parsed else "unknown",
                                "justification": updated_justification,
                                "reasoning": updated_justification,
                                "supporting_genes": [],
                            }
                        ]
                    self.update_hypotheses(cluster_id, current_hypotheses)
                    self.output["hypotheses"] = current_hypotheses
                    self.output["updated_hypotheses"] = current_hypotheses
                self.log_info(
                    cluster_id,
                    f"Cross-cluster check: annotation confirmed ({cell_type})",
                )

        return self.output

    def _commit_via_dossier(self, state: "CellTypingState", reasoning_response: str, prev_cell_type: str) -> dict:
        """Fold the cross-cluster correction through the dossier (§3).

        Seeds the comparator (sibling) labels as ``neighbor`` candidates, merges
        the correction reasoning via the dossier update contract, and commits —
        so reinstatement, single-leader and sticky-leader rules apply and the
        change is recorded, rather than a blind label overwrite.
        """
        cluster_id = state["cluster_id"]
        dossier = self.load_dossier(state)
        neighbor = [
            {"cell_type": c.get("cell_type"), "source": "neighbor", "prior": "medium"}
            for c in (state.get("cross_cluster_comparisons") or [])
            if c.get("cell_type")
        ]
        if neighbor:
            dossier.add_candidates(neighbor, default_source="neighbor")
        dossier = self.update_dossier_from_response(
            dossier, reasoning_response, default_source="neighbor", stage="cross_cluster_correction",
        )
        self.commit_dossier(dossier, cluster_id)

        new_cell_type = dossier.cell_type or prev_cell_type
        changed = normalize(str(new_cell_type)) != normalize(str(prev_cell_type))
        self.output["cross_cluster_changed"] = changed
        if changed:
            self.output["cell_type_ont"] = None
            self.runner.with_lock(self.accessor.set_ont_cell_type, cluster_id, None)
        if dossier.description:
            self.output["cross_cluster_updated_justification"] = dossier.description
            self.output["justification_summary"] = dossier.description
        self.log_info(
            cluster_id,
            f"Cross-cluster (dossier): {prev_cell_type} -> {new_cell_type} (changed={changed})",
        )
        return self.output

