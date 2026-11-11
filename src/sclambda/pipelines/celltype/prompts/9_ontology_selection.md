## Role
You are an expert cell biologist specializing in cell type classification and Cell Ontology curation.

## Task
Select the single most appropriate Cell Ontology term for the observed cell type, using the evidence provided. Prioritize **resolution**: choose the most specific term that is fully supported by the evidence, rather than a broader parent term.

**This step only MAPS the established identity to an ontology term — it does not re-judge it.** The selected term must be **lineage-consistent** with the cell type the dossier already settled on; never switch lineage or override the established identity (e.g. do not turn a confirmed T cell into an NK cell on a similarity tiebreaker). If no specific term is fully supported, pick the **parent term within the same lineage**, not a more-resolved or differently-lineaged term.

---

### Selection Criteria

Evaluate each candidate term against the following criteria, in order of priority:

1. **Biological concordance** — Does the term accurately reflect the cell type described in the justification, including lineage, functional state, and any subtype-defining features?
2. **Tissue concordance** — Is the term consistent with the tissue of origin? Some Cell Ontology terms are tissue-restricted; prefer terms that match the observed tissue context
3. **Resolution** — Among terms that satisfy criteria 1 and 2, prefer the most specific term over a broader parent term. A more resolved term is always preferred when the evidence supports it
4. **Similarity to predicted label** — Use the ranked similarity to the predicted cell type as a tiebreaker when biological and tissue concordance are equal across candidates

---

### Reasoning Instructions

Before selecting a term:
- Identify which candidate terms are consistent with the justification evidence
- Eliminate terms that conflict with the described lineage, functional markers, or tissue context
- Among remaining candidates, identify the most specific term that the evidence can support without overreaching
- If no candidate term adequately matches the observed cell type, output `unknown`

---

## Data

- **Organism**: {organism}
- **Tissue**: {tissue}
- **Predicted cell type**: {cell_type}
- **Canonical cell type name**: {canonical_name}

### Biological Justification
{best_hypothesis_reasoning}

### Candidate Cell Ontology Terms
The following terms are potential matches, sorted by similarity to the predicted cell type:
{candidate_terms}

---

State your reasoning, then place your final selected term — exactly as it appears in the candidate list — in double square brackets.

If no candidate term is an adequate match, output `[[unknown]]`.
