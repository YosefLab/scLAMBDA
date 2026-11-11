# Cross-cluster review (optional)

Use when the user asks for it or a specific ambiguity needs a direct comparison
between clusters. Several clusters can correctly share a label.

```bash
lambdakit de peers --run RUN --cluster ID
lambdakit de genes --run RUN --cluster ID --genes GENE_A,GENE_B --vs PEER_ID --record
```

Shared cells are excluded. A near-zero pairwise LFC means the gene is shared,
not absent. Expression differences alone do not show contamination or a
distinct subtype.

Write a new critic decision citing the comparison tables, then commit. Keeping
the same label is a valid outcome. When the review is finished:

```bash
lambdakit phase done --run RUN --name cross-cluster
```
