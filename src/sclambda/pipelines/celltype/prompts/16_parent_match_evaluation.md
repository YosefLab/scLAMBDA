## Role
You are an expert cell biologist performing a gene expression analysis to refine cell type annotations in {organism} {tissue} tissue.

## Task
Evaluate whether this cluster matches the parent cell type annotation.

### Step 1 — Define the Parent Cell Type Signature

List the key lineage-defining markers expected for **{parent_cell_type}**. Distinguish between:
- **Core lineage markers** — genes whose absence would rule out the parent cell type entirely
- **Supporting markers** — genes that are characteristic but not strictly required
- Identify other cell types that express same genes

---

### Step 2 — Match Cluster Markers Against the Parent Signature

For each core lineage marker, assess whether it is:
- **Present** among the top cluster genes with meaningful expression
- **Absent** — not detected or below meaningful expression levels
- **Contradicted** — a marker of a mutually exclusive cell type is present instead

Apply fraction-based filtering: expression in fewer than 15% of cells should not be used as primary evidence for or against a match.

---

### Step 3 — Red Flag Check

Before classifying, explicitly check for mismatches that would indicate the cluster does not belong under this parent:

- Are any **core lineage markers absent** that should be present in any member of the **{parent_cell_type}** lineage?
- Are any **markers of a mutually exclusive cell type** present at high fraction and high expression?
- Is the overall marker profile more consistent with a **completely different lineage**?

If any red flags are triggered, classification must be **unrelated** regardless of any superficial overlap with the parent label. State the red flag explicitly in your response.

---

### Step 4 — Classify the Cluster

Assign exactly one of the following categories:

- **same_type** — The cluster belongs to the same cell type as the parent. All core lineage markers for **{parent_cell_type}** are present and no contradicting markers are detected.

- **subtype** — The cluster is a canonical, well-established subtype of **{parent_cell_type}**. All core lineage markers are present, and additional markers define a recognized subpopulation with a distinct identity (e.g., CD4+ vs. CD8+ T cells, classical vs. non-classical monocytes, naive vs. memory B cells). Do not assign this category based solely on activation state (e.g., *MKI67*, *HSPA1A*), stress response, or non-canonical markers that do not define a recognized subtype.

- **unrelated** — The cluster belongs to a different cell type. Core lineage markers are absent, contradicted, or a red flag was triggered in Step 3.

---

### Required Output Structure

Provide a response with the following sections:

1. **Parent Cell Type Signature** — Core and supporting markers expected for {parent_cell_type}
2. **Marker Match Assessment** — Which core markers are present, absent, or contradicted in this cluster
3. **Red Flag Check** — Explicit statement of any mismatches or contradictions identified in Step 3, or confirmation that none were found
4. **Classification** — One of: `same_type`, `subtype`, or `unrelated`, with reasoning
5. **Cell Type Annotation** — Your specific cell type label for this cluster

## Data
**Context:**
- Organism: {organism}
- Tissue: {tissue}
- Parent cell type: {parent_cell_type}

**Top marker genes for this cluster:**
{top_genes}
