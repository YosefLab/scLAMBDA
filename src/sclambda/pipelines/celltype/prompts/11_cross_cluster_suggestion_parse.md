Extract the cross-cluster comparison suggestions from the response below.

## Response to parse
{suggestion_response}

---

Return a JSON object with two keys:

```json
{{
  "target_genes": ["GENE1", "GENE2", "GENE3"],
  "comparisons": [
    {{
      "cluster_id": "the cluster ID to compare against",
      "cell_type": "the cell type annotation of that cluster",
      "reason": "brief reason for this comparison"
    }}
  ]
}}
```

Rules:
- `target_genes` is the single shared diagnostic gene panel — include all genes suggested, using standard gene symbols (uppercase for human, title case for mouse)
- `comparisons` lists the clusters to compare against
- `cluster_id` must exactly match one of the cluster IDs from the input list
- If no comparisons were suggested, return `{{"target_genes": [], "comparisons": []}}`
