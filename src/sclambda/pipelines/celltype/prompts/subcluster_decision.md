## Role
You decide whether a cluster annotated as **{cell_type}** in **{organism} {tissue}** should be **subclustered**. You are given deterministic graph evidence computed on the cluster's cells; your job is the **biological judgment** the statistics cannot make: is each structured program a *meaningful subdivision worth pursuing*, or a confound to ignore?

## Evidence

### A. Off-lineage contamination (self-signature depletion)
{depletion_block}
A structured pocket where this cell type's OWN identity markers are ABSENT means the cluster contains cells that are **not {cell_type}** (a different lineage admixed). If present, those cells should be split off.

### B. Structured expression programs (co-expression modules)
Each module is a set of genes that co-vary in a graph-structured way across the cluster's cells (a candidate axis of heterogeneity):
{modules_block}

## How to judge each module
Classify every module as **subtype** (a real, biologically distinct subpopulation/state worth resolving) or **confound** (structure that does NOT warrant splitting the cell type). Common confounds — decide from the genes, do not assume:
- **clonality / repertoire**: immunoglobulin or TCR variable/light-chain genes (IGKV/IGLV/IGHV/IGKC/IGLC, TRBV/TRAV…) — antibody/TCR clones, not subtypes.
- **donor / sex / batch**: Y-chromosome genes (DDX3Y, RPS4Y1, UTY), XIST.
- **technical**: ribosomal, mitochondrial, housekeeping/translation (EEF1A1, GAPDH, ACTB).
- **ambient vs. real contamination** — a module of a *different* lineage's markers (e.g. epithelial PIGR/FABP1 inside an immune cluster) is only a separable population if section A reports a **self-depletion pocket** (those cells have LOST the {cell_type} identity). If there is **no depletion** (the cells still express {cell_type} markers), a foreign-lineage module is **ambient RNA / co-expression**, NOT a separate cell type — do **not** split on it; subclustering would just yield the same cell type again.
- **transient activation / stress**: immediate-early genes (FOS/JUN/EGR1/NR4A1), heat-shock (HSPA*) — usually a state, not a subdivision (use judgment: a stable activated subset can still be real).
A module is a **subtype** when its genes are lineage-/function-defining for a recognized subpopulation of {cell_type} (e.g. for CD4 T: Th1 TBX21/CXCR3, Tfh CXCR5/BCL6, Treg FOXP3; for fibroblasts: distinct matrix/inflammatory programs; for macrophages: a stable polarization).

## Decide
- **needs_subcluster = true** if contamination is present, OR at least one module is a real subtype/contaminant worth separating.
- **needs_subcluster = false** if every module is a confound (clonality/donor/technical/transient) — the cluster is one cell type with no meaningful subdivisions to pursue.

## Output ONLY
```json
{{
  "needs_subcluster": false,
  "reason": "one sentence",
  "modules": [{{"genes": ["G1","G2","G3"], "verdict": "subtype | confound", "label": "what it is (e.g. 'Tfh', 'clonality', 'epithelial ambient')"}}],
  "expected_subtypes": ["named subpopulations to expect if split, else empty"]
}}
```
