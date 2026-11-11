# LAMBDA

LAMBDA uses an LLM to name the clusters in a single-cell or spatial `.h5ad`
file: cell types for scRNA-seq clusters, anatomical regions for spatial niches.
The LLM proposes an identity and has to support it with differential
expression it asks for; a second LLM pass reviews the evidence before a label
is recorded.

You can run it in Claude Code or from Python.

## Using it from Claude Code

Open Claude Code next to your data and ask:

> Annotate the cell types in `pbmc.h5ad`. Human blood, clusters are in `leiden`.

> Name the spatial niches in `brain.h5ad`. Coordinates are in `X_spatial`, cell
> types in `cell_type`, niches in `niche`. Mouse brain, coronal section.

> Map the cell-type labels to Cell Ontology terms using `~/data/cl.obo`.

Claude checks the file and then annotates the clusters, using the bundled
`lambdakit` tool for the statistics. You don't need a provider API
key; your Claude Code login is enough.

Results go to a run directory, `.lambda/<run>` by default:

```text
overlay.parquet         per-cell cluster and annotation columns
results/<id>.json       label, justification, confidence and open questions
ontology.json           Cell Ontology term per label, if you asked for them
dossiers/<id>.json      the measurements behind each decision
transcripts/<id>.jsonl  the reasoning, step by step
```

To pick up an interrupted run or export the results:

```bash
lambdakit status --run RUN                      # where the run stopped and what is next
lambdakit export --run RUN --csv out.csv --xlsx out.xlsx --md report.md
lambdakit usage  --run RUN                      # tokens spent, per model
```

### Install

```bash
pip install -e '.[excel]'   # excel is optional, for .xlsx export
lambdakit doctor            # check the environment
lambdakit plugin-path       # print the plugin directory
```

Then, in Claude Code:

```text
/plugin marketplace add <directory printed by lambdakit plugin-path>
/plugin install lambda-annotate@lambda-annotate
```

The plugin includes a hook that approves its own `lambdakit` calls, so a run
does not stop for permission prompts. Commands that delete files still ask. To
turn the hook off, delete `hooks/hooks.json`.

More on the run directory, the cache and resuming:
[docs/skill-workflow.md](docs/skill-workflow.md).

## Using it from Python

```bash
pip install -e '.[anthropic,excel]'   # or openai / google
export ANTHROPIC_API_KEY=...          # or OPENAI_API_KEY / GEMINI_API_KEY
```

### Cell types

```python
from sclambda.main import Lambda

lam = Lambda("pbmc.h5ad", provider="anthropic", embedding_key="X_scVI")
lam.annotate(organism="human", tissue="peripheral blood", use_dossier=True)
lam.write(output_path="annotations.csv", excel_path="annotations.xlsx")

lam.get_annotations()   # {cluster_id: label}
lam.get_overlay()       # per-cell DataFrame with clusters and labels
```

This clusters the cells with Leiden on `embedding_key` and labels each
cluster. Always pass `use_dossier=True`: without it you get the older
single-pass workflow, which is still the default.

To label clusters you already have, name the `.obs` column, and optionally the
clusters to process:

```python
lam.annotate(obs_columns="leiden", organism="human", tissue="peripheral blood",
             use_dossier=True)
lam.annotate(obs_columns="leiden", clusters=["3", "7"], organism="human",
             tissue="peripheral blood", use_dossier=True)
```

With `depth=2` or more, LAMBDA also decides which clusters contain distinct
subpopulations, subclusters them and labels the children. A fuller example:

```python
lam = Lambda(
    "pbmc.h5ad",
    provider="anthropic",
    gene_col="feature_name",   # var column with gene symbols, if not var.index
    embedding_key="X_scVI",
    n_top_genes=50,
)
lam.accessor.ref_mean          # compute dataset-wide statistics once, before workers start

lam.annotate(
    organism="human",
    tissue="peripheral blood",
    depth=2,
    use_dossier=True,
    use_cross_cluster=False,   # True adds a pass comparing similar clusters
    de_method="ttest",         # "ols" corrects for dropout but is slower
    max_workers=8,
)
```

`use_cell_ont=True` with an `ontology_file` maps labels to Cell Ontology terms.

### Spatial niches

```python
lam = Lambda("brain.h5ad", provider="anthropic", gene_col="feature_name")

lam.annotate(
    pipeline="annotate_spatial",
    obs_columns="niche",                 # column with niche / region labels
    cell_type_col="cell_type",           # optional
    spatial_key="X_spatial",             # obsm key with 2-D coordinates
    organism="mouse",
    tissue="brain, coronal section",
    slice_key="brain_section_label",     # omit for a single section
    spatial_comparison_review=True,
    flip_spatial_y=False,
    max_workers=8,
)
lam.write(output_path="annotations.csv", excel_path="annotations.xlsx")
```

For each niche, LAMBDA looks at which cell types it contains, which genes it
expresses compared with the rest of the tissue, and rendered images of where
it sits in the section. Two settings change the result:

- `spatial_comparison_review=True` makes the reviewer weigh each label against
  its closest alternative. Claude Code runs always do this; in Python it is off
  unless you set it.
- `flip_spatial_y` defaults to `True`, which mirrors the images vertically.
  Set it to `False` to keep the orientation stored in the file.

Expression is pooled across sections unless you set `slice`; images and spatial
statistics are always per section. Without `cell_type_col`, every spot gets the
same placeholder type, so composition is unavailable and the comparison uses
all spots; the input file is not modified.

## What your data needs

- Gene symbols (`CD8A`, not Ensembl IDs) in `var.index` or the column given by
  `gene_col`.
- For cell types: a cell embedding in `obsm`, ideally batch-corrected
  (`X_scVI`, `X_pca`, ...).
- For niches: 2-D coordinates in `obsm` and a niche column in `obs`.
- Raw counts are best. Normalised data is detected and used as is.

Be specific in `organism` and `tissue`; the model uses them as context. Lower
`max_workers` if you hit provider rate limits.

## Reading the results

Each label has a status. `resolved` means the evidence singled out one
identity; `best_effort` means it is the best-supported candidate but a close
alternative remains, which is listed with the label. In spatial runs, a niche
where the review failed is kept and marked `execution_status=inference_failed`
rather than dropped. These fields are included in the CSV, Excel and Markdown
exports.

## Models and cost

The provider defaults to Anthropic; OpenAI and Google also work. Each has
`weak`, `medium` and `strong` models. The model proposing labels uses
`medium` and the reviewer uses `strong`. To choose them yourself:

```python
Lambda("pbmc.h5ad", provider="anthropic", model_overrides={
    "medium": "PROPOSER_MODEL_ID",
    "strong": "REVIEWER_MODEL_ID",
})
```

Token usage is reported in `results["llm_usage"]` and written next to the
annotations by `lam.write()`. Pass `usage_log_path="requests.jsonl"` to
`Lambda` to log every request as it happens, which is useful if a run is
interrupted.

## Reference

- All parameters: [PARAMETERS.md](PARAMETERS.md)
- Upgrading from an older version: [docs/MIGRATION.md](docs/MIGRATION.md)
- Requires Python 3.10–3.12.
- Licence: MIT, see [LICENSE](LICENSE).
