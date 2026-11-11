"""celltype pipeline: preparation."""
from __future__ import annotations
from .common import (
    CellTypingLLMNode,
    CellTypingNode,
    _load_prompt,
    _ss,
    format_gene_list,
    math,
    rank_genes,
    re,
)

class PrepareClusterNode(CellTypingNode):
    """Prepare cluster information before processing."""

    default_output = {
        "depth": 1,
        "parent_cluster_id": None,
        "can_inherit_parent": False,
        "top_genes": None,
    }

    def execute(self, state: "CellTypingState") -> dict:
        cluster_id = state["cluster_id"]

        self.output["depth"] = self.accessor._parse_cluster_depth(cluster_id) if cluster_id else 1

        parent_id = self.accessor.get_parent_cluster_id(cluster_id)
        self.output["parent_cluster_id"] = parent_id

        self.output["can_inherit_parent"] = self._can_inherit_annotation(
            cluster_id=cluster_id,
            parent_id=parent_id,
            min_size=state.get("min_cluster_size", 50),
        )
        if state.get("use_dossier"):
            self.output["can_inherit_parent"] = False

        # Compute per-cluster tissue from obs column if provided
        tissue_col = state.get("tissue_col")
        if tissue_col:
            tissue_str = self._compute_cluster_tissue(cluster_id, tissue_col)
            if tissue_str:
                self.output["tissue"] = tissue_str

        self.log_info(
            cluster_id,
            f"depth={self.output['depth']}, parent={parent_id or 'none'}, "
            f"can_inherit={self.output['can_inherit_parent']}",
        )

        return self.output

    def _compute_cluster_tissue(self, cluster_id: str, tissue_col: str) -> str | None:
        """Compute tissue composition for a cluster from obs column.

        Returns top 3 tissues. Single tissue: "Brain".
        Multiple: "Brain (60%), Lung (30%), Heart (10%)".
        """
        try:
            cell_indices = self.accessor.get_cell_indices(cluster_id)
        except (KeyError, ValueError):
            return None

        if len(cell_indices) == 0:
            return None

        obs_subset = self.accessor.get_obs([tissue_col], cell_indices)
        values = obs_subset[tissue_col].dropna().astype(str)
        if values.empty:
            return None

        counts = values.value_counts()
        total = counts.sum()

        if len(counts) == 1:
            return str(counts.index[0])

        top = counts.head(3)
        parts = []
        for tissue_name, count in top.items():
            pct = round(100 * count / total)
            parts.append(f"{tissue_name} ({pct}%)")
        return ", ".join(parts)

    def _can_inherit_annotation(self, cluster_id, parent_id, min_size) -> bool:
        if not parent_id:
            return False
        n_cells = self.accessor.get_n_cells_in_cluster(cluster_id)
        if n_cells < min_size:
            self.log_info(cluster_id, f"Too small ({n_cells} < {min_size}), will inherit parent")
            return True
        return False


class RankGenesNode(CellTypingNode):
    """Compute top marker genes for cluster."""

    default_output = {"top_genes": [], "rank_genes_result": None}

    def execute(self, state: "CellTypingState") -> dict:
        cluster_id = state["cluster_id"]
        n_top_genes = state.get("n_top_genes", 20)
        result = rank_genes(
            self.accessor,
            cluster_id,
            n_genes=n_top_genes,
            de_method=state.get("de_method", "ttest"),
        )
        self.output["top_genes"] = result.gene_names
        self.output["rank_genes_result"] = result
        return self.output


class NaiveCTGenerationNode(CellTypingLLMNode):
    """Simplified single-question cell type annotation.

    Sends a single prompt with top enriched genes and expects the cell type
    in double square brackets.  Used when ``annotate(naive=True)``.
    """

    default_output = {
        "cell_type": "unknown",
        "cell_types": ["unknown"],
        "hypotheses": [],
        "naive_ct_prompt": "",
        "naive_ct_response": "",
        "ct_generation_failed": True,
    }

    _PROMPT_TEMPLATE = (
        "Provide cell type annotation for a cluster of cells from "
        "{organism} {tissue} tissue based on a list of top enriched genes. "
        "The cell type should be provided in double square brackets. "
        "Input: {gene_list}"
    )

    def execute(self, state: "CellTypingState") -> dict:
        cluster_id = state["cluster_id"]
        top_genes = state["top_genes"]

        gene_list = format_gene_list(top_genes, numbered=True)
        prompt = self._PROMPT_TEMPLATE.format(
            organism=state.get("organism", "human"),
            tissue=state.get("tissue", "unknown"),
            gene_list=gene_list,
        )
        self.output["naive_ct_prompt"] = prompt

        response = self.llm_client.query(prompt, power="medium")
        self.output["naive_ct_response"] = response.content

        if response.failed_after_retries:
            self.log_warning(cluster_id, "Naive CT LLM call failed")
            return self.output

        # Extract cell type from [[...]]
        match = re.search(r"\[\[(.+?)\]\]", response.content)
        if match:
            cell_type = match.group(1).strip()
        else:
            cell_type = "unknown"
            self.log_warning(cluster_id, "Could not parse [[cell_type]] from response")

        self.output["cell_type"] = cell_type
        self.output["cell_types"] = [cell_type]
        self.output["hypotheses"] = [
            {
                "cell_type": cell_type,
                "confidence": "medium",
                "justification": response.content,
                "supporting_genes": list(top_genes),
            }
        ]
        self.output["ct_generation_failed"] = False

        self.update_annotation(cluster_id, cell_type)
        self.update_hypotheses(cluster_id, self.output["hypotheses"])
        self.log_info(cluster_id, f"Naive annotation: {cell_type}")

        return self.output


class InheritParentAnnotationNode(CellTypingLLMNode):
    """Evaluate a subcluster against its parent annotation.

    Small clusters (can_inherit_parent=True): inherit parent label directly,
    no further subclustering.

    Large clusters: LLM evaluates same_type / subtype / unrelated.
    - same_type: inherit parent label, no further subclustering.
    - subtype / unrelated: proceed to full ct_generation pipeline,
      passing parent_classification through state so CTGenerationNode
      and CTReevaluationNode can inject parent context into their prompts.
    """

    default_output = {
        "cell_type": None,
        "cell_types": [],
        "parent_classification": None,
    }

    def execute(self, state: "CellTypingState") -> dict:
        cluster_id = state["cluster_id"]
        parent_id = state.get("parent_cluster_id")

        if not parent_id:
            self.log_warning(cluster_id, "No parent to inherit from")
            return {}

        if state.get("use_dossier"):
            # Even small/same-type children need a critic review. "subtype"
            # routes to generation with parent context; it confirms no identity.
            return {"parent_classification": "subtype", "can_inherit_parent": False}

        # Small cluster: inherit directly, skip full annotation
        if state.get("can_inherit_parent"):
            return self._do_inherit(cluster_id, parent_id, reason="cluster too small: inherit parent annotation")

        # Large cluster: ask LLM to classify
        classification = self._evaluate_parent_match(state, parent_id)
        self.log_info(cluster_id, f"Parent match evaluation: {classification}")

        if classification == "same_type":
            return self._do_inherit(cluster_id, parent_id, reason="same cell type as parent: inherit annotation")

        # subtype or unrelated: full annotation pipeline, pass classification for context
        return {"parent_classification": classification}

    def _do_inherit(self, cluster_id: str, parent_id: str, reason: str) -> dict:
        parent_depth = self.accessor._parse_cluster_depth(parent_id)
        parent_annotations = self.accessor.get_annotations(depth=parent_depth)
        parent_cell_type = parent_annotations.get(parent_id, "unknown")

        if parent_cell_type == "unknown":
            self.log_warning(cluster_id, f"Parent {parent_id} has no annotation to inherit")
            return {}

        self.update_annotation(cluster_id, parent_cell_type)
        self.update_hypotheses(
            cluster_id,
            [
                {
                    "cell_type": parent_cell_type,
                    "confidence": "medium",
                    "reasoning": reason,
                    "supporting_genes": [],
                }
            ],
        )

        # Inherit parent's ontology term if available
        parent_ont_term = self.accessor.get_ont_cell_type(parent_id)
        if parent_ont_term:
            self.update_ont_cell_type(cluster_id, parent_ont_term)
            self.output["cell_type_ont"] = parent_ont_term

        self.log_info(cluster_id, f"Inherited '{parent_cell_type}' from {parent_id}")

        self.output["cell_type"] = parent_cell_type
        self.output["cell_types"] = [parent_cell_type]
        self.output["parent_classification"] = "same_type"
        return self.output

    def _evaluate_parent_match(self, state: "CellTypingState", parent_id: str) -> str:
        """LLM evaluation: classify cluster vs. parent as same_type/subtype/unrelated."""
        cluster_id = state["cluster_id"]

        parent_depth = self.accessor._parse_cluster_depth(parent_id)
        parent_annotations = self.accessor.get_annotations(depth=parent_depth)
        parent_cell_type = parent_annotations.get(parent_id, "unknown")

        if parent_cell_type == "unknown":
            return "unrelated"

        # Formalized grounding: how much of the PARENT's own identity signature does this
        # child retain? (parent markers detected in the child, relative to the parent
        # itself). A child depleted of the parent identity is unrelated; a child that
        # strongly retains it is at most a subtype — never "unrelated".
        retention = None
        try:
            phyp = self.accessor.get_hypotheses(parent_id) or []
            ptexts = []
            for h in phyp[:3]:
                ptexts += [str(h.get("justification", "")), str(h.get("reasoning", "")), str(h.get("cell_type", ""))]
            parent_genes = _ss.extract_signature_genes(ptexts, self.accessor.gene_names)
            if parent_genes:
                p_self = _ss.signature_score(self.accessor, self.accessor.get_cell_indices(parent_id), parent_genes)
                c_ret = _ss.signature_score(self.accessor, self.accessor.get_cell_indices(cluster_id), parent_genes)
                if p_self and not math.isnan(p_self) and p_self > 0 and not math.isnan(c_ret):
                    retention = float(c_ret / p_self)
                    self.output["parent_signature_retention"] = retention
                    self.log_info(cluster_id, f"parent-signature retention={retention:.2f} (child {c_ret:.2f}/parent {p_self:.2f})")
                    if retention <= 0.30:
                        self.log_info(cluster_id, "child depleted of parent identity -> unrelated (grounded)")
                        return "unrelated"
        except Exception as e:  # noqa: BLE001
            self.log_warning(cluster_id, f"parent-signature retention failed: {e}")

        top_genes = state.get("top_genes", [])
        eval_prompt = _load_prompt("16_parent_match_evaluation.md")
        formatted_eval = eval_prompt.format(
            organism=state.get("organism", "human"),
            tissue=state.get("tissue", "unknown"),
            parent_cell_type=parent_cell_type,
            top_genes=", ".join(top_genes) if top_genes else "No marker genes",
        )
        self.output["parent_match_evaluation_prompt"] = formatted_eval

        eval_response = self.llm_client.query(formatted_eval, power="medium")
        self.output["parent_match_evaluation_response"] = eval_response.content
        if eval_response.failed_after_retries:
            self.log_warning(cluster_id, "Parent match LLM call failed, defaulting to unrelated")
            return "unrelated"

        json_template = '{"classification": "same_type|subtype|unrelated", "annotation": "cell_type_name"}'
        parse_prompt = _load_prompt("17_parent_match_evaluation_parse.md")
        formatted_parse = parse_prompt.format(
            json_template=json_template,
            evaluation_response=eval_response.content,
        )

        parse_response = self.llm_client.query(formatted_parse, power="weak")
        if parse_response.failed_after_retries:
            self.log_warning(cluster_id, "Parent match parse failed, defaulting to unrelated")
            return "unrelated"

        match = re.search(
            r'"classification"\s*:\s*"(same_type|subtype|unrelated)"',
            parse_response.content,
        )
        if match:
            cls = match.group(1)
            # Strong retention of the parent identity rules out "unrelated": the child
            # is the parent lineage, so at most a subtype. Grounds the LLM call.
            if retention is not None and retention >= 0.70 and cls == "unrelated":
                self.log_info(cluster_id, "strong parent retention overrides 'unrelated' -> subtype")
                cls = "subtype"
            return cls

        self.log_warning(cluster_id, f"Could not parse classification from: {parse_response.content[:100]}")
        return "unrelated"
