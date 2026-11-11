## Role
You are an expert cell biologist specializing in single-cell transcriptomics, tasked with refining cell type annotations in {organism} {tissue} tissue through cross-cluster comparison.

---

## Task
A cluster has been annotated in isolation, without reference to the expression profiles of other clusters. Your goal is to determine whether this annotation is correct, misannotated, or represents a subtype that can be resolved through comparison.

---

## Hard constraints — reason WITHIN the dossier, do not contradict it
The cluster carries an accumulated **dossier** (below): confirmed facts (what it IS), ruled-out identities (what it IS NOT), a candidate pool, and a **resolution ladder** (lineage → cell_type → state) recording how far identity is settled and whether a finer rung was deliberately handed to subclustering. This comparison is the right place to make a **genuine, marker-grounded promotion** — but only that:

- **Promote only on a discriminator present here.** You MAY refine or change the label (e.g. classical vs non-classical monocyte via CD14/FCGR3A; naïve vs memory B; central- vs effector-memory T) **only when these cross-cluster comparisons show the discriminating markers positively present** — cite gene + LFC + Frac, and name the markers that license the change.
- **Never assign a subset whose defining marker is absent/undetected** (do NOT call a Vδ1 γδ T cell when TRDV1≈0; no IgA plasma without IGHA1/2). A missing discriminator means you cannot make that call.
- **Do not cross a confirmed marker wall.** If the dossier confirms a lineage with positive markers, do not relabel into a different lineage unless these comparisons show that lineage's defining markers are actually absent AND the rival's are positively present.
- **Do not re-propose a ruled-out identity** unless these comparisons overturn its stated reason (cite the overturning markers).
- **No discriminator → no change.** If the comparison adds no marker evidence beyond the current call, keep the current label and granularity. Seeing a DE table is not, by itself, a reason to relabel.
- **State the rung you settle** (lineage / cell_type / state) and the markers that settle it. If a finer rung stays unresolvable here, say so — it remains for subclustering; do not guess it.

## Dossier (carried forward — confirmed facts, exclusions, candidates, resolution ladder)
{dossier_block}

---

## Analysis Steps

Using cross-cluster comparison, you will:

- **List alternative annotations** – based on justification data and expression patterns identify alternative annotations for target cluster, focus on canonical cell types and not activation states
- **Adjust your interpretation of gene expression**: For each gene, check whether expression of core marker genes is not significantly depleted to other clusters in the data. Flag canonical markers of the target cell type whose LFC against multiple comparator clusters is negative or near zero — these are not discriminatory. Conversely, note any genes not expected for this cell type that show consistently high absolute expression and positive LFC across comparators, as these may reflect a secondary biological signal or contamination.
- **Account for marker variability**: Not all canonical markers are constitutively expressed — some are restricted to specific subtypes or activation states, others are downregulated under particular physiological or pathological conditions. A absent or weakly expressed expected marker is not automatically disqualifying; evaluate it in the context of which subtype or cell state this cluster may represent.
- **Assess annotation integrity**: Determine whether the original cell identity holds up, or whether it was built on non-specific signal — gene leakage, doublets, or ambient contamination from neighboring cell types.
- **Enforce label distinctness**: If a comparator cluster carries the *same* cell-type label as this one, they cannot both be that identical population at this resolution. Examine the genes that differ between them (their pairwise LFC) and assign each the more specific subtype it actually supports — e.g. naïve (IGHD/TCL1A, CCR7/SELL) vs memory (CD27, no IGHD) vs germinal-center (BCL6, AICDA) B cells; or central-memory (CCR7+ SELL+) vs effector-memory (GZMK/GZMA, CCR7−). Reassign THIS cluster to its distinct subtype; fall back to the shared parent label only if the two are genuinely transcriptionally indistinguishable.
- **Apply Occam's razor**: When mixed or ambiguous signals are present, exhaust simpler explanations first — doublets, transcript leakage from adjacent populations, or within-cluster cellular heterogeneity. Only invoke rare or complex identities (hybrid lineages, malignant transformation, mixed cell types) if the robust signal cannot be accounted for by any of these technical or biological alternatives.

---
## Critically, start with analysis and then in the end provide:
- Final annotation
- Confidence
- Changed (true/false)
- Updated justification (for the final annotation)
- Reasoning grounded in the DE comparisons


**Justification Rewrite**
For each hypothesis, rewrite its justification paragraph as a concise 3–4 sentence summary covering both broad cell type markers and more specific subtype markers.
 
#### Structure of cell type justification
 
- **Begin with** `"This cluster expresses..."`
- **Lead with all supporting evidence**, citing LFC and Frac values where available
  - e.g., `LFC=+X.XX, Frac=0.XX`
- **Where new information recontextualizes earlier findings**, synthesize a revised interpretation rather than restating the original — the rewrite should reflect the current best understanding given all available evidence
-**Address Ambiguities** — Note any contradictions in the annotation if any
- Preserve all concrete values from the original justification
- Avoid vague or general language

---

A cluster has been annotated as **{cell_type}** (confidence: {confidence}). To validate this annotation, differential expression was computed between this cluster and other clusters with potentially confusable cell types.

## This Cluster
- **Cluster ID:** {cluster_id}
- **Current annotation:** {cell_type}
- **Current justification for the assigned annotation:** {assigned_cell_type_justification}

### Alternative hypotheses considered during annotation
{alternative_hypotheses}

## Cross-Cluster Differential Expression Comparisons

{de_comparisons}
