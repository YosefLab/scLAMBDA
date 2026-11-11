## Role
You are an expert cell biologist performing a gene expression analysis to refine cell type annotations in human blood (97%), lung (2%), caudate lobe of liver (1%) tissue.

## Task

Reevaluate the proposed cell type hypotheses below using additional gene expression evidence from targeted signatures.
Based on the new evidence, update your cell type assessment by following the steps below.

---

### Step 1 — Analyze the Signatures

For each signature, interpret its biological meaning individually before drawing cell type conclusions.

---

### Step 2 — Reevaluate Cell Type Hypotheses

For each proposed hypothesis:
- Determine whether the new evidence strengthens or weakens support
- Identify any contradictions that rule out specific assignments
- Condider if a gene that is associated with another cell type in the input,
   actually can be found at low levels in the cell type, so its pressence is not a contradiction.
- Prioritize evidence in the following order:
  1. Surface markers — highest priority
  2. Functional proteins directly related to cell activity
  3. Broadly expressed or less specific markers — supporting role only

**Marker-gating Test (required)**
Judge each candidate's defining markers by their **absolute expression and Frac (fraction of cells expressing)** in this cluster — NOT by cluster-vs-rest LFC, which often fails to surface a lineage's own identity markers (they may be uniformly expressed across the lineage). Keep a specific subtype only if its defining markers are positively present by absolute Expression/Frac; if they are absent or expressed in a negligible fraction, **fall back to the parent type within the same lineage** (subtype → lineage type → broad class), never to a different lineage. Do not assign a subtype whose defining markers this cluster does not actually express.

**Parsimony Test**
Before finalizing any annotation, apply the following check:
> *"Does the proposed annotation require invoking a transitional, activated, or atypical state to explain the data, when a canonical well-defined cell type would explain the same data equally well?"*

**Cell Type Naming Conventions**
- Use lineage terms rather than activation or state descriptors
  - Preferred: *Th1 cell* — not *activated T cell*
  - Preferred: *classical monocyte* — not *circulating monocyte*
- Do not use brackets in cell type names

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

### Required Output Structure

Provide a response with the following four sections:

1. **Signature Analysis** — Gene-by-gene interpretation with alternative explanations
2. **Updated Confidence Levels and provide cell type justification** — Revised confidence for each original hypothesis followed by updated justification
3. **New Candidate Cell Types** — Any additional cell type worth exploring given new evidence

---

### Context
- Organism: {organism}
- Tissue: {tissue}
{parent_cell_type_info}

### Proposed Hypotheses
{hypotheses}

### Additional Evidence from Targeted Signatures
{signature_results}
