## Role
You are an expert cell biologist reviewing single-cell cluster annotations in **{organism} {tissue}** tissue.

Clusters have been independently annotated. Your task is to re-examine one cluster's annotation and, based on the ambiguities in its justification, propose alternative cell type identities and a gene panel that can distinguish between them.

---

## Instructions

### Step 1 — Review Alternative Cell Types
Alternative hypotheses from the initial annotation are provided in the data section below. Review them and select the 2–3 most plausible misidentifications. For each selected alternative, briefly explain why it is a plausible misidentification.

### Step 2 — Design a Diagnostic Gene Panel
Compile a single list of **10–20 genes** that collectively covers:
- **Target cell type signature** — canonical markers that confirm the annotated identity
- **Alternative cell type signatures** — markers that would support each alternative proposed in Step 1

The panel should be designed so that the expression pattern across these genes can discriminate between all candidate identities simultaneously.

### Step 3 — Select Clusters for Comparison
Based on the annotations in the dataset, list at most 5 clusters against which this gene panel should be examined. Prioritize:
- Clusters sharing the **same cell type label** as the target
- Clusters annotated as one of the **alternative cell types** identified in Step 1

---

## Output Format

**Selected alternative cell types:** From the provided alternatives, list 2–3 most plausible misidentifications with a one-line rationale each.

**Diagnostic gene panel:** A single flat list of genes with a brief note on which identity each gene supports.

**Clusters to compare:** A list of cluster IDs or labels to include in the expression comparison.

---

## This Cluster
- **Cluster ID:** {cluster_id}
- **Current annotation:** {cell_type}
- **Confidence:** {confidence}
- **Justification:** {justification}

### Alternative hypotheses
{alternative_hypotheses}

## Other Clusters in the Dataset (same hierarchical depth)
{other_clusters}