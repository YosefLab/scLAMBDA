---
name: sc-celltype
description: Annotate cell types in single-cell RNA-seq clusters using an actor-first measurement and critic loop. Test competing gene signatures, assign the best-supported identity with explicit uncertainty, and assess whether to subcluster. Use for cell-type annotation or identification in h5ad, AnnData, or scRNA-seq datasets.
---

# Cell-type annotation

You do the reasoning; `lambdakit` computes differential expression and stores
everything in a run directory. Each cluster has a dossier: an append-only list
of measurement tables plus the current critic decision. You alternate two
roles: the **actor** proposes identities and chooses measurements; the
**critic** reads the tables and decides the label.

Use `lambdakit` if it is on the PATH; otherwise run `echo "$CLAUDE_PLUGIN_ROOT"`
once and write `<that path>/bin/lambdakit` literally in every command. Commands
containing `$`, pipes or redirects are not auto-approved.

On large datasets `init`, `de` and `spatial` commands can take minutes. Give
every `lambdakit` call the Bash tool's maximum timeout (600000 ms). If a call is
still moved to the background, wait for it to finish and read its output; never
continue without the measurement or treat it as a negative result.

## Start or resume

```bash
lambdakit init DATA.h5ad --run RUN --organism human --tissue "human peripheral blood" \
  --embedding-key X_scVI --gene-col feature_name --obs-column cluster_labels --depth 2
lambdakit status --run RUN
lambdakit clusters --run RUN
```

Omit `--gene-col` if `var.index` holds symbols. Omit `--obs-column` and pass
`--use-leiden` to cluster de novo. Existing cluster names are replaced with
anonymous IDs. On resume, `status` names the next action; keep existing
measurements and decisions.

A catalog is optional; see [catalog](references/catalog.md).

## Each cluster

```bash
lambdakit dossier init --run RUN --cluster ID
lambdakit de rank --run RUN --cluster ID --n 50 --record
lambdakit dossier show --run RUN --cluster ID
```

`dossier init` keeps an existing dossier. `dossier show` prints the current
decision and every recorded table with its ID, target, reference and method.
Read it before measuring again.

Then repeat **actor → measurements → critic**, up to four rounds:

1. **Actor.** Propose plausible identities and the measurement most likely to
   change the choice. Test alternatives, not only subtypes of the favourite;
   several signatures can be tested in one round.
2. **Measure.** Every command with `--record` stores a table in the dossier:

   ```bash
   lambdakit de genes --run RUN --cluster ID --genes GENE_A,GENE_B --record
   lambdakit de rank  --run RUN --cluster ID --n 100 --record
   lambdakit de peers --run RUN --cluster ID
   lambdakit de genes --run RUN --cluster ID --genes GENE_A,GENE_B --vs PEER_OR_PARENT_ID --record
   ```

   The default reference is every other cell; `--vs` compares against one peer
   or the parent, excluding shared cells. Ask for any relevant gene, including
   genes outside the top 50; the engine reports genes the dataset lacks.
   Reuse a table if one already has the same target, reference and method.
3. **Critic.** Follow [critic](references/critic.md) and save the decision with
   `lambdakit dossier decision --run RUN --cluster ID --file decision.json`.
4. **Continue or stop.** A non-empty `next_question` starts another round. An
   empty one finishes the cluster, including with `best_effort` status. In the
   fourth round, leave it empty.

On resume, count rounds from the accepted critic decisions in the transcript.

## Reading the evidence

These apply to both roles.

- Read LFC, proportion expressing and significance together, against the
  table's reference. Negative LFC is not absence. Significance is not
  specificity. A gene missing from the top-N list was not measured, not zero.
- Identity comes from a coherent program that separates the candidate from its
  alternatives. Shared activation, signalling, cytotoxic or effector genes do
  not establish identity.
- Weigh the dominant program against the strongest competing one. One
  discordant gene does not veto a coherent program; a coherent competing
  program means real uncertainty.
- A negative result for one subtype does not exclude its whole lineage.
- Do not invent thresholds (detection rates, LFC ranges, maturation cutoffs)
  or explanations (dropout, contamination, developmental history) to dismiss
  a conflict.
- RNA is not surface protein. Cluster-level proportions do not show same-cell
  co-expression, purity or a mixture.
- Parent, sibling and earlier labels are provisional. Different clusters can
  share a label; do not assign a cluster by elimination from its siblings.

## Subclustering and committing

When the requested depth allows another level, probe for structure after the
label is decided; see [subclustering](references/subcluster-decision.md). Then:

```bash
lambdakit dossier axes --run RUN --cluster ID --json '{"needs_subcluster":false}'
lambdakit dossier commit --run RUN --cluster ID
```

`commit` writes the critic's label, status and uncertainty. A later change to
the decision or the flag needs another commit; export refuses stale results.

Between waves, the orchestrator splits flagged parents and the children are
annotated like any other cluster:

```bash
lambdakit subcluster --run RUN --cluster ID
lambdakit clusters --run RUN --depth 2
lambdakit dossier init --run RUN --cluster ID_0 --parent ID --parent-label "PARENT_LABEL"
```

A child needs its own evidence and may disagree with its parent. Running
`subcluster` again on a split parent keeps the existing children.

For many clusters, give each subagent one cluster. Only the orchestrator runs
`subcluster`.

## Cell Ontology terms (optional)

Only when the user asks. This names the committed labels in Cell Ontology; it
never changes a label. Use the user's `.obo` file (`cl.obo` or `cl-basic.obo`).
If they have none, ask before downloading
`http://purl.obolibrary.org/obo/cl/cl-basic.obo`.

```bash
lambdakit ontology candidates --run RUN --obo PATH/cl.obo
lambdakit ontology candidates --run RUN --cluster ID --query "alternative name"
lambdakit ontology set --run RUN --cluster ID --term CL:0000000 --note "why"
lambdakit ontology set --run RUN --cluster ID --none --note "why no term fits"
lambdakit ontology show --run RUN
lambdakit phase done --run RUN --name ontology
```

The first call lists every label without a term; the run remembers the file.
For each cluster:

- The score ranks string similarity, not biology. Read the definition and
  `is_a` before choosing.
- Choose the most specific term that states what the label states. A term for
  this tissue is fine when the identity is the same (`colon goblet cell` for a
  goblet cell in colon). Do not pick a term more specific than the evidence,
  e.g. a subtype the label leaves open.
- When the label has a qualifier the ontology lacks, such as an activation
  state, take the closest broader term and name the dropped qualifier in the
  note.
- If no listed term fits, search synonyms or the broader identity with
  `--query`. Use `--none` only when nothing fits.

Relabelling a cluster later makes its term stale; export leaves stale terms
out and names those clusters.

## Export

Cross-cluster review is optional; see [cross-cluster](references/cross-cluster.md).

```bash
lambdakit export --run RUN --csv annotations.csv --xlsx annotations.xlsx --md report.md
lambdakit phase done --run RUN --name export
lambdakit usage --run RUN
```

Report resolved labels, `best_effort` labels and failures separately. Mapped
terms appear as `cell_ontology_id` and `cell_ontology` columns. `lambdakit
clean --run RUN` only reports disk use; delete nothing unless asked.
