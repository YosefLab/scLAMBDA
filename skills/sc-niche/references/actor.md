# Actor

Read the tables, the images and the previous critic decision, including its
`next_question`.

## Comparison plan

Before testing, write a plan: the candidate compartments, what they share, and
for each test the question, what each candidate predicts, and any uncertain
premise. Store it so the critic sees it as hypotheses, not evidence:

```bash
lambdakit dossier evidence --run RUN --cluster ID --source actor_plan --fact 'PLAN_JSON'
```

- When the defining expression of a broad compartment conflicts, compare a
  different broad identity or a mixture before refining within it.
- For a fine distinction, choose a measurement that would favour one candidate
  over the other. If none is available, say so and keep the label broad; do not
  repeat queries for unavailable genes.
- When the critic proposes a new winner, test it against the strongest
  remaining rival. Earlier candidates stay as alternatives until tested.
- Within-label differences under supplied cell types are not automatically
  specialisation.

## Tools

```bash
lambdakit spatial genes --run RUN --niche ID --genes GENE1,GENE2
lambdakit spatial genes --run RUN --niche ID --cell-type "TYPE" --genes GENE3,GENE4
lambdakit spatial de --run RUN --niche ID --cell-type "TYPE"
lambdakit spatial reference --run RUN --niche ID --query "what distinguishes A from B" --terms "GENE,compartment"
lambdakit spatial image --run RUN --niche ID --slide SLIDE --number
lambdakit spatial observe --run RUN --niche ID --image PATH --text "what is visible"
```

- Without `--cell-type`, the comparison is the niche against every spot
  outside it. With `--cell-type`, it is that type inside the niche against the
  same type outside. Neither is a comparison with one neighbouring niche.
- Several panels, including rival signatures in different cell types, can be
  measured in one turn. Reuse existing tables and views.
- Gene names are matched exactly first, then through species-specific NCBI
  symbols and aliases; ambiguous matches are not chosen. `--alias-file` pins a
  `gene_info` snapshot, `--evidence-cache` sets the cache directory, and
  `SCLAMBDA_GENE_ALIAS_DOWNLOAD=0` disables downloads.
- `spatial reference` returns Europe PMC passages with IDs and URLs. Run it
  once per query, for consequential uncertain claims about marker specificity.

## Pending checks

When the critic leaves `pending_checks`, run every one with the tools above,
keeping the requested population and reference:

| Check | Command |
|---|---|
| `test_signatures`, `test_signature` | `spatial genes` per panel |
| `get_de`, `get_celltype_de` | `spatial de` |
| `lookup_reference` | `spatial reference` |
| `consult_image` | render, read and `spatial observe` each slide |

Record a check that produced no table:

```bash
lambdakit dossier evidence --run RUN --cluster ID --source tool_status --fact "check, population, error"
```

Then return to the critic. If you conclude nothing useful is left to test, the
critic still reviews.
