## Role
You are an expert cell biologist performing a gene expression analysis to refine cell type annotations in **{organism} {tissue}** tissue.

## Task
Based on the current hypotheses and gene expression data, decide whether the evidence is sufficient to confidently assign a cell type, or whether targeted follow-up signatures are needed to resolve remaining ambiguity.

---

### Step 1 — Evaluate the Current Evidence

Before making a decision, assess the following:

- **Leading hypothesis:** Which candidate is best supported by the top expressed genes, and why?
- **Ambiguity:** Is there meaningful uncertainty between distinct cell types or canonical subtypes? Note that differences in cell state, activation level, or anatomical location do not constitute ambiguity for the purpose of this analysis.
- **Prior signatures:** Has the identified ambiguity already been addressed by previously tested signatures?
- **Conflicting evidence:** Are there any genes or signatures that are unexpected or inconsistent with the leading hypothesis?
- **Mixed population:** Is the cluster likely a mixture of several cell types, in which case a coarser label should be assigned and further refinement should not be attempted?


This reasoning should directly inform Step 2.

---

### Step 2 — Make a Confidence Decision

Based on the evidence assessed in Step 1 and all previously tested signatures, determine which candidate cell type has the most compelling support and decide whether the analysis can be concluded. If no further signature testing could meaningfully refine the label — where refinement is restricted to cell types and canonical subtypes, excluding cell states or anatomical location — conclude confident and stop.

**Confirm identity by absolute expression (required).** A cell type / subtype may **not** be concluded `confident` until its **defining markers are positively expressed in this cluster by absolute level and fraction of cells** — not by differential (cluster-vs-rest) fold change. A lineage's identity marker is often NOT differentially enriched (e.g. Cd3e is high in *all* T cells; a subtype marker may be uniformly present within the lineage), so absence from the differential list does not exclude it, and presence in the differential list of a marker the cells barely express does not confirm it. If the leading hypothesis is a specific subtype and its defining markers have not been confirmed by absolute expression / fraction (e.g. via a tested signature), return `needs_refinement` and request that signature. If the subtype's markers cannot be positively confirmed, **fall back to the parent type within the same lineage** (e.g. "CD8+ effector memory T cell" → "CD8+ T cell" → "T cell") — never to a different lineage.

---

### Step 3 – Suggest signatures to disambiguate/refine cell type

#### Option A — Confident
If the evidence strongly and unambiguously supports a single cell type that cannot be meaningfully refined to a subtype, end your response with:
```json
{{
  "confidence_decision": "confident",
  "best_cell_type": "cell type name",
  "reasoning": "explanation referencing specific supporting genes and why alternatives are ruled out",
  "signatures": {{}}
}}
```

#### Option B — Needs Refinement
If ambiguity remains — due to overlapping phenotypes, absence of discriminating markers, multiple plausible hypotheses, or unresolved subtype identity — first describe in free text:
- What is the primary ambiguity to resolve?
- Which gene comparisons or contrasts would be most informative?

Then end your response with a JSON block containing up to **5 signatures**:
```json
{{
  "confidence_decision": "needs_refinement",
  "best_cell_type": null,
  "reasoning": null,
  "signatures": {{
    "signature_name_1": {{
      "justification": "to rule out/confirm/explore [specific purpose]",
      "genes": ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
    }},
    "signature_name_2": {{
      "justification": "to rule out/confirm/explore [specific purpose]",
      "genes": ["GENE4", "GENE5", "GENE6", "GENE7", "GENE8"],
    }}
  }}
}}
```

Prioritize signatures that address the following, in order of importance:

1. **Lineage boundary markers** — genes that distinguish between cell types with overlapping phenotypes or shared lineage programs (e.g., dendritic cells vs. macrophages, B cells vs. plasma cells, endothelial cells vs. pericytes, neurons vs. neuroendocrine cells)
2. **Closely related subtype markers** — genes that discriminate between well-defined subtypes (e.g., cDC1 vs. cDC2, classical vs. non-classical monocyte, Th1 vs. Th2)
3. **Subtype-defining markers** — genes that would elevate a general annotation to a specific subtype (e.g., naive vs. memory B cell, Th1 vs. T cell)
4. **Maturation and differentiation state markers** — genes that place the cell along a developmental trajectory
5. **Negative markers** — genes that should be absent in the leading hypothesis and whose presence would indicate misannotation (e.g., CD10 absence in memory B cells, CD8 absence in helper T cells)

---

### Naming Conventions
- **Gene symbol convention**: {gene_symbol_convention}

---

## Data

- **Organism**: {organism}
- **Tissue**: {tissue}
- **Current hypotheses**: {hypotheses}
- **Top expressed genes**: {top_genes}

{previously_tested_signatures}
