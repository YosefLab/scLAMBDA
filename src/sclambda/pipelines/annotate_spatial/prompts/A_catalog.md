## Role
You are an expert anatomist and tissue biologist. Before any spatial niche is annotated, you will enumerate the **canonical spatial compartments** that can occur in **{organism} {tissue}**.

## Task
List the spatial regions / microenvironments a pathologist or spatial-transcriptomics analyst could legitimately encounter in this tissue. This is the *universe of possible labels* — it must be broad enough that almost any real niche maps onto one or more entries.

Reason from **this** tissue's own architecture and biology — do not import compartments from other tissues. Be comprehensive across these *kinds* of compartments, wherever they genuinely apply here:
- **Common (high-prior)** compartments expected in most sections.
- **Rare / specialized but real** compartments — actively include these; do not collapse to only the obvious regions.
- **Distinct sub-compartments of larger structures** — when a structure has functionally or architecturally distinct sub-zones, layers, or zones, list them **separately** rather than merging them into one coarse entry. (Decide which structures have such sub-zones from this tissue's biology.)
- **Altered / pathological compartments**, *if this tissue can be diseased or injured*: regions where the normal resident architecture is lost or replaced and a damage, repair, fibrotic, necrotic, neoplastic, or dense-infiltrate program dominates. Name each by **what is actually present there** (its own pathological identity), not by the intact compartment it replaced. Derive which of these apply from this tissue and disease context — do not assume a fixed template.

For each compartment provide:
1. **compartment** — canonical anatomical name for this tissue (lead with the anatomical structure, then any sub-zone)
2. **prior** — how commonly it appears in this tissue: *very high / high / medium / low / very low*
3. **defining_cell_types** — the cell types whose presence/abundance defines it
4. **markers** — defining marker genes (use the gene-symbol convention standard for {organism})
5. **adjacent_partners** — the compartments this one is normally spatially adjacent to (used to reason about neighbors). Use names from your own list.

## Output
Brief reasoning is allowed first, then output a single JSON array:

```json
[
  {{
    "compartment": "string",
    "prior": "very high|high|medium|low|very low",
    "defining_cell_types": ["...", "..."],
    "markers": ["GENE1", "GENE2"],
    "adjacent_partners": ["another compartment from this list", "..."]
  }}
]
```

Aim for 12–25 compartments. Be comprehensive but non-redundant.
