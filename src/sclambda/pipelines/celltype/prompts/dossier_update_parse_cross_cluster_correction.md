## Role
You maintain a structured **dossier** for one cell cluster. Convert the cross-cluster correction analysis below into a single JSON update. Extract only what the analysis states or clearly implies, and **honor the hard constraints** — this step may PROMOTE the label, but only on marker evidence the analysis actually cites.

## Rules
- **description**: one revised paragraph stating what this cluster most likely IS, given the cross-cluster comparison.
- **confirmed** (what it IS): factual traits the analysis newly establishes, each with its key evidence (gene, LFC, Frac). Carry prior confirmed items forward.
- **ruled_out** (what it IS NOT): `{{"claim": "...", "reason": "..."}}` for each excluded identity. Carry prior exclusions forward unless the analysis explicitly overturns one with cited markers.
- **candidates**: each cell-type hypothesis discussed, as objects with:
  - `cell_type` (canonical name)
  - `lineage` (the immediate coarse parent, e.g. "T cell", "plasma cell", "monocyte/macrophage", "epithelial cell" — REQUIRED so a coarse fallback label exists)
  - `states` (≤2 marker-backed sub-states: memory/naïve/effector-memory, tissue-resident vs circulating, activated, cycling, isotype — never embedded in the base name)
  - `state`: `supported` | `candidate` | `refuted` | `retired`
  - `prior`, `likelihood`: very high|high|medium|low|very low
  - `evidence_for`, `evidence_against`: concrete bullets citing genes, LFC, **Frac**
- **best_cell_type**: the leading cell type after the comparison (else null).
- **Marker-backed promotion only.** Mark a candidate `supported` / set `best_cell_type` to a FINER or DIFFERENT identity ONLY if the analysis cites the discriminating markers as positively present. If the analysis says a discriminator is absent/undetected (e.g. TRDV1≈0), DO NOT assign that subset — drop the qualifier and keep the plainer label.
- **axes** — the resolution ladder `{{"lineage": {{"resolved": bool, "note": "markers"}}, "cell_type": {{...}}, "state": {{...}}}}`. Mark a rung `resolved: true` ONLY when the analysis confirms that rung with cited positive markers. A residency/memory/activation distinction is a **state**-rung resolution, not cell_type. Omit a rung the analysis does not settle. Never resolve a deeper rung while a shallower one is unresolved.
- **needs_subcluster**: `true` if the analysis concludes a finer rung cannot be resolved from these data and should be split; else `false`.
- Every qualifier (states / isotype / subset) must cite a gene, else use the plainer canonical name.

## Prior dossier (state before this step)
{dossier}

## Cross-cluster analysis to convert
{reasoning_response}

## Output
Return ONLY this JSON object:
```json
{{
  "description": "string",
  "confirmed": ["trait — evidence"],
  "ruled_out": [{{"claim": "string", "reason": "string"}}],
  "candidates": [
    {{"cell_type": "string", "lineage": "string", "states": [], "state": "candidate",
      "prior": "medium", "likelihood": "high",
      "evidence_for": ["gene LFC Frac"], "evidence_against": []}}
  ],
  "axes": {{"lineage": {{"resolved": false, "note": ""}}, "cell_type": {{"resolved": false, "note": ""}}, "state": {{"resolved": false, "note": ""}}}},
  "needs_subcluster": false,
  "best_cell_type": null
}}
```
