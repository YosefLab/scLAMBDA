"""celltype pipeline: resolution."""
from __future__ import annotations
from .common import (
    CTDossierMixin,
    CellTypingLLMNode,
    CellTypingNode,
    ParseSchema,
    SubclusterMethod,
    _load_prompt,
    _ss,
    fuzzy_match,
    load_ontology,
    normalize,
    rank_genes,
    re,
    subcluster,
)

class SubstructureGateNode(CTDossierMixin, CellTypingLLMNode):
    """Decide ``needs_subcluster`` from deterministic graph substructure + an LLM verdict.

    Runs after the dossier agent (dossier path). Two contaminant-agnostic probes on the
    cluster's kNN graph (``substructure.py``): (1) a connected pocket where the committed
    self-signature is ABSENT → off-lineage contamination; (2) nuisance-light co-expression
    modules → candidate subtype axes. The probe surfaces the module signatures; the **LLM**
    (``subcluster_decision.md``) judges whether each is a meaningful subdivision or a
    confound (clonality/donor/technical/activation/ambient), and sets ``needs_subcluster``.
    """

    default_output = {
        "substructure_probe": None,
        "subcluster_decision": None,
        "needs_subcluster": False,
    }

    def execute(self, state: "CellTypingState") -> dict:
        cluster_id = state["cluster_id"]
        if not state.get("use_dossier") or not state.get("ct_dossier"):
            return self.output
        # No point probing/deciding a split when subclustering isn't possible.
        if int(state.get("depth", 1)) >= int(state.get("max_depth", 1)):
            return self.output
        dossier = self.load_dossier(state)
        self_genes = _ss.signature_genes_from_dossier(dossier, self.accessor.gene_names)
        try:
            probe = _ss.probe_substructure(self.accessor, cluster_id, self_genes)
        except Exception as e:  # noqa: BLE001
            self.log_warning(cluster_id, f"substructure probe failed: {e}")
            return self.output
        self.output["substructure_probe"] = probe

        dep = probe.get("depletion") or {}
        st = probe.get("structure") or {}
        if not dep.get("present") and not st.get("modules"):
            self.output["needs_subcluster"] = False
            self.log_info(cluster_id, "substructure: no contamination pocket, no modules -> leaf")
            return self.output

        decision = self._decide(state, dossier, probe)
        if decision is not None:
            ns = bool(decision.get("needs_subcluster"))
        else:
            ns = bool(dep.get("present"))  # fall back to the deterministic contamination flag
        self.output["subcluster_decision"] = decision
        self.output["needs_subcluster"] = ns

        dossier.needs_subcluster = ns
        reason = (decision or {}).get("reason", "")
        dossier.add_evidence("substructure", f"needs_subcluster={ns}" + (f": {reason}" if reason else ""))
        self.commit_dossier(dossier, cluster_id)
        self.log_info(cluster_id, f"substructure gate: needs_subcluster={ns} | {str(reason)[:160]}")
        return self.output

    def _decide(self, state, dossier, probe):
        dep = probe.get("depletion") or {}
        mods = (probe.get("structure") or {}).get("modules") or []
        if dep.get("present"):
            pm = ", ".join(dep.get("pocket_markers", [])) or "(none)"
            depl_block = (
                f"PRESENT: {dep['frac_absent']*100:.0f}% of cells form a connected pocket "
                f"(graph autocorrelation {dep['morans_i']:.2f}) that expresses none of the "
                f"identity markers. Top genes enriched in that pocket: {pm}."
            )
        else:
            depl_block = "Not detected — the identity signature is present across all cells."
        if mods:
            mods_block = "\n".join(
                f"- Module {i+1}: {', '.join(m['genes'])}  "
                f"(a discrete subpopulation of {m.get('frac_high', 0)*100:.0f}% of cells, "
                f"autocorrelation {m['mean_i']:.2f}, {m['size']} genes)"
                for i, m in enumerate(mods)
            )
        else:
            mods_block = "(no structured modules above chance)"
        try:
            prompt = _load_prompt("subcluster_decision.md").format(
                cell_type=dossier.cell_type or state.get("cell_type", "unknown"),
                organism=state.get("organism", "human"),
                tissue=state.get("tissue", "unknown"),
                depletion_block=depl_block,
                modules_block=mods_block,
            )
            resp = self.llm_client.query(prompt, power="medium")
            self.output["subcluster_decision_prompt"] = prompt
            self.output["subcluster_decision_response"] = resp.content
            if resp.failed_after_retries:
                return None
            v = self.parse_llm_response(resp.content, ParseSchema(format="dict"), fallback=None)
            return v if isinstance(v, dict) else None
        except Exception as e:  # noqa: BLE001
            self.log_warning(state.get("cluster_id", ""), f"subcluster decision failed: {e}")
            return None


class JustificationSynthesisNode(CellTypingNode):
    """Pass through the existing justification from hypothesis/reevaluation without an extra LLM call."""

    default_output = {
        "justification_summary": "",
    }

    def execute(self, state: "CellTypingState") -> dict:
        cluster_id = state["cluster_id"]
        if state.get("cross_cluster_changed", False):
            source_hypotheses = (
                self.accessor.get_hypotheses(cluster_id)
                or state.get("updated_hypotheses")
                or state.get("hypotheses")
                or []
            )
        else:
            source_hypotheses = (
                state.get("updated_hypotheses")
                or state.get("hypotheses")
                or self.accessor.get_hypotheses(cluster_id)
                or []
            )
        hypotheses = [dict(h) for h in source_hypotheses]

        if not hypotheses:
            self.log_info(cluster_id, "No hypotheses available, skipping justification pass-through")
            return self.output

        best = dict(hypotheses[0])
        cell_type = state.get("cell_type") or best.get("cell_type", "unknown")
        confidence = str(best.get("confidence", "unknown")).lower()

        # Use existing justification from hypothesis generation / reevaluation
        justification = best.get("justification", "") or best.get("reasoning", "")
        if not justification:
            justification = state.get("refinement_reasoning") or ""
        if not justification:
            justification = f"This cluster expresses markers associated with {cell_type}."

        best["cell_type"] = cell_type
        best["justification"] = justification
        best["reasoning"] = justification  # backward compat
        if not best.get("confidence"):
            best["confidence"] = confidence if confidence else "unknown"
        if best.get("supporting_genes") is None:
            best["supporting_genes"] = []

        hypotheses[0] = best
        self.update_hypotheses(cluster_id, hypotheses)

        if cell_type and cell_type != state.get("cell_type"):
            self.update_annotation(cluster_id, cell_type)

        self.output["justification_summary"] = justification
        self.output["cell_type"] = cell_type
        self.output["cell_types"] = [h.get("cell_type") for h in hypotheses if h.get("cell_type")]
        self.output["hypotheses"] = hypotheses
        self.output["updated_hypotheses"] = hypotheses
        self.log_info(cluster_id, "Using existing justification from hypothesis/reevaluation")
        return self.output


class SubclusterNode(CellTypingNode):
    """Subcluster if not at max depth."""

    default_output = {"children": []}

    def execute(self, state: "CellTypingState") -> dict:
        cluster_id = state["cluster_id"]
        depth = state["depth"]
        max_depth = state["max_depth"]

        if depth >= max_depth:
            return self.output

        # Dossier path: the substructure gate is authoritative. Only split when it
        # found off-lineage contamination or a real subtype module (needs_subcluster);
        # otherwise treat the cluster as a leaf and don't over-split.
        if state.get("use_dossier") and not state.get("needs_subcluster", False):
            self.log_info(cluster_id, "Substructure gate: no split-worthy structure; leaf")
            return self.output

        method = SubclusterMethod.LEIDEN if state.get("use_leiden", True) else SubclusterMethod.FROM_OBS

        result = subcluster(
            self.accessor,
            cluster_id,
            method=method,
            resolution=state.get("resolution", 1.0),
            min_clusters=2,
            n_neighbors=state.get("subcluster_n_neighbors", 30),
            obs_columns=state.get("obs_columns"),
            synchronize=self.runner.with_lock,
        )

        clusters = result.clusters
        if clusters:
            min_prop = state.get("min_subcluster_proportion", 0.01)
            def filter_and_validate():
                kept = self._filter_by_proportion(cluster_id, clusters, min_prop)
                actual = set(self.accessor.get_children(cluster_id))
                if len(set(kept)) != len(kept) or set(kept) != actual:
                    raise RuntimeError(f"Subcluster IDs do not match overlay membership for {cluster_id}")
                return kept
            clusters = self.runner.with_lock(filter_and_validate)

        self.output["children"] = clusters
        self.output["subcluster_method"] = result.method
        return self.output

    def _filter_by_proportion(
        self,
        parent_id: str,
        child_ids: list[str],
        min_prop: float,
        high_prop: float = 0.05,
        max_clusters: int = 20,
    ) -> list[str]:
        """Filter subclusters by minimum proportion of parent cells.

        If applying min_prop still yields >max_clusters, escalates to high_prop.
        Dissolved clusters are removed from the accessor overlay.
        """
        n_parent = self.accessor.get_n_cells_in_cluster(parent_id)
        if n_parent == 0:
            return child_ids

        sizes = {cid: self.accessor.get_n_cells_in_cluster(cid) for cid in child_ids}

        def _apply_threshold(threshold: float) -> list[str]:
            return [cid for cid, n in sizes.items() if n / n_parent >= threshold]

        # If there are few subclusters, keep all regardless of proportion
        if len(child_ids) <= 10:
            return child_ids

        kept = _apply_threshold(min_prop)

        if len(kept) > max_clusters:
            kept = _apply_threshold(high_prop)
            self.log_info(
                parent_id,
                f">{max_clusters} subclusters after {min_prop*100:.0f}% filter "
                f"— escalated to {high_prop*100:.0f}% ({len(kept)} kept)",
            )
        else:
            self.log_info(
                parent_id,
                f"Subcluster proportion filter ({min_prop*100:.0f}%): " f"{len(child_ids)} → {len(kept)} clusters",
            )

        dissolved = [cid for cid in child_ids if cid not in set(kept)]
        if dissolved:
            self.runner.with_lock(self.accessor.dissolve_clusters, dissolved)
            rename_map = self.runner.with_lock(
                self.accessor.reindex_clusters,
                parent_id,
                kept,
            )
            if rename_map:
                kept = [rename_map.get(cid, cid) for cid in kept]

        return kept


class OntCellTypeNode(CellTypingLLMNode):
    """Map cell type to Cell Ontology format using data-driven selection."""

    default_output = {"ontology_canonical_names": []}

    def execute(self, state: "CellTypingState") -> dict:
        cluster_id = state["cluster_id"]
        cell_type = state.get("cell_type")
        use_cell_ont = state.get("use_cell_ont", False)
        ontology_file = state.get("ontology_file")
        organism = state.get("organism", "human")
        tissue = state.get("tissue", "unknown")

        if not use_cell_ont or not ontology_file or not cell_type:
            # Return empty dict so LangGraph does not overwrite cell_type_ont
            # that may already be set (e.g. from the main pipeline wave).
            return {}

        # Skip if term already set (e.g. inherited from parent)
        if self.accessor.get_ont_cell_type(cluster_id):
            return {}

        # Get justification from best hypothesis
        hypotheses = self.accessor.get_hypotheses(cluster_id)
        justification = "No detailed reasoning available."
        if hypotheses:
            justification = hypotheses[0].get("reasoning", justification)

        # Generate canonical names via LLM
        canonical_names = self._get_canonical_names(cell_type, organism, tissue, justification)
        self.output["ontology_canonical_names"] = canonical_names

        # Fuzzy match against ontology
        ontology_terms = load_ontology(ontology_file)
        candidate_terms = self._fuzzy_match_candidates(canonical_names, ontology_terms)

        if not candidate_terms:
            return self.output

        # Get top genes for context
        top_genes = state.get("top_genes", [])
        if not top_genes:
            rank_result = rank_genes(
                self.accessor, cluster_id,
                n_genes=state.get("n_top_genes", 20),
                de_method=state.get("de_method", "ttest"),
            )
            top_genes = rank_result.gene_names

        # LLM-based selection from candidates
        ont_term = self._select_ontology_term(
            cell_type,
            canonical_names,
            candidate_terms,
            top_genes,
            justification,
            organism,
            tissue,
        )

        self.log_info(cluster_id, f"Mapped '{cell_type}' -> '{ont_term}'")

        if ont_term:
            self.output["cell_type_ont"] = ont_term
            self.update_ont_cell_type(cluster_id, ont_term)

        return self.output

    def _get_canonical_names(self, cell_type: str, organism: str, tissue: str, justification: str) -> list[str]:
        canonical_prompt = (
            f'What is the canonical name for the cell type "{cell_type}"? '
            f"Context: {organism} {tissue}. "
            f"My goal is to find the exact match from Cell Ontology using string matching.\n\n"
            f"## Cell Type Justification\n{justification}\n\n"
            f"Start with analysis and then output multiple candidate names in double square brackets like: [[name1]] [[name2]]"
        )
        self.output["ontology_canonical_prompt"] = canonical_prompt
        response = self.llm_client.query(canonical_prompt, power="weak")
        self.output["ontology_canonical_response"] = response.content

        canonical_names = [cell_type]
        if not response.failed_after_retries:
            matches = re.findall(r"\[\[([^\]]+)\]\]", response.content)
            canonical_names.extend(m.strip() for m in matches if m.strip())

        # Deduplicate preserving order
        seen = set()
        unique = []
        for name in canonical_names:
            key = name.lower()
            if key not in seen:
                seen.add(key)
                unique.append(name)
        return unique

    def _fuzzy_match_candidates(self, canonical_names: list[str], ontology_terms) -> list[str]:
        candidates_dict = {}
        for name in canonical_names:
            for term, score in fuzzy_match(name, ontology_terms, top=15):
                candidates_dict[term] = max(candidates_dict.get(term, 0), score)

        sorted_candidates = sorted(candidates_dict.items(), key=lambda x: x[1], reverse=True)[:20]
        return [term for term, _ in sorted_candidates]

    def _select_ontology_term(
        self,
        cell_type,
        canonical_names,
        candidate_terms,
        top_genes,
        justification,
        organism,
        tissue,
    ) -> str | None:
        _SPECIAL_TERMS = ["unknown", "doublet", "cancer cell"]
        all_terms = candidate_terms + _SPECIAL_TERMS

        selection_prompt = _load_prompt("9_ontology_selection.md")
        formatted_prompt = selection_prompt.format(
            organism=organism,
            tissue=tissue,
            cell_type=cell_type,
            canonical_name=canonical_names[0] if canonical_names else cell_type,
            top_genes=", ".join(top_genes) if top_genes else "No marker genes",
            best_hypothesis_cell_type=cell_type,
            best_hypothesis_reasoning=justification,
            candidate_terms="\n".join(f"{i+1}. {term}" for i, term in enumerate(all_terms)),
        )
        self.output["ontology_selection_prompt"] = formatted_prompt

        for attempt in range(2):
            response = self.llm_client.query(formatted_prompt, power="medium")
            self.output[f"ontology_selection_attempt_{attempt + 1}_prompt"] = formatted_prompt
            self.output[f"ontology_selection_attempt_{attempt + 1}_response"] = response.content
            self.output["ontology_selection_response"] = response.content
            if response.failed_after_retries:
                if attempt == 0:
                    continue
                return None

            selected = re.findall(r"\[\[([^\]]+)\]\]", response.content)
            if not selected:
                if attempt == 0:
                    continue
                return None

            selected = selected[0]

            # Try numeric index
            if selected.isdigit():
                idx = int(selected) - 1
                if 0 <= idx < len(all_terms):
                    return all_terms[idx]

            # Try string match
            selected_normalized = normalize(selected)
            for term in all_terms:
                if normalize(term) == selected_normalized:
                    return term

            return None

        return None

