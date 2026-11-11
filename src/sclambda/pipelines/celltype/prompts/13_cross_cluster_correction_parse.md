Extract the corrected cell type annotation from the cross-cluster comparison review below.

## Response to parse
{correction_response}

---

Return a single JSON object with these fields:

```json
{{
  "cell_type": "corrected or confirmed cell type label",
  "confidence": "high|medium|low",
  "justification": "updated justification for the FINAL annotation",
  "reasoning": "brief explanation referencing the DE comparisons that support this decision",
  "changed": true or false
}}
```

- Set `"changed": true` only if the cell type was actually modified from the original.
- If the reviewer confirmed the original annotation unchanged, set `"changed": false` and copy the original cell type exactly.
- The `"confidence"` should reflect how strongly the DE evidence supports the final annotation.
- `"justification"` must be the final updated justification text for the final annotation (not the old one).
