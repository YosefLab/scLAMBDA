---
name: sc-annotate
description: Entry point for annotating single-cell or spatial transcriptomics data with LAMBDA — inspects an .h5ad file, works out whether the job is cell-type annotation or spatial niche annotation, checks the dataset is usable (gene symbols, embedding, coordinates), and hands off to the right pipeline. Use when the user points at an h5ad / AnnData / scRNA-seq / spatial dataset and wants it annotated but has not said which kind.
---

# Annotating a dataset

Pick the pipeline and check the dataset before starting. Most failed runs come
from something visible in the first minute: Ensembl IDs instead of symbols, no
embedding, coordinates under an unexpected key.

## 1. Find the engine

Run `lambdakit doctor`. If `lambdakit` is not on the PATH, run
`echo "$CLAUDE_PLUGIN_ROOT"` once and use `<that path>/bin/lambdakit doctor`.
`doctor` reports missing packages and the command that installs them.

From then on, write the engine path literally in every command, and pass the
same form to the downstream skill and to any subagents. The plugin auto-approves
`lambdakit` calls only when the command contains no `$`, pipe or redirect, so a
command with `${CLAUDE_PLUGIN_ROOT}` in it stops for a permission prompt.

## 2. Look at the file

```bash
python3 -c "
import anndata as ad
a = ad.read_h5ad('DATA.h5ad', backed='r')
print(a)
print('var index sample:', list(a.var.index[:5]))
"
```

Decide:

- **Which pipeline.** Coordinates in `obsm` (`spatial`, `X_spatial`) and a
  request to label regions means **spatial niches**. Otherwise, or when the
  goal is per-cluster identity, it is **cell types**. If the request is
  ambiguous, ask whether the user wants cell identities or region identities.
- **Which clustering column.** Look for `leiden`, `cluster`, or a niche/region
  column. If there is none, cluster de novo with `--use-leiden`.
- **Gene symbols.** `var.index` or a `var` column must hold symbols (`CD8A`,
  `Slc17a7`). Pass the column as `--gene-col` if it is not the index.
- **Embedding** (cell types only). Needed for subclustering; prefer a
  batch-corrected one (`X_scVI`) over `X_pca`.

Niche annotation does not need cell-type labels. Without `--cell-type-col` the
engine treats all spots as one population, so composition is unavailable.

## 3. Ask what the file cannot tell you

- **Organism**: sets the gene-symbol convention.
- **Tissue**, specifically: "human colon", not "gut"; "mouse brain, coronal
  section", not "brain".
- **Depth**: `--depth 2` allows impure clusters to be subclustered.
- **Benchmarking**: if an existing annotation column is ground truth, cluster on
  it; its names are encoded and hidden from the reasoning.
- **Cell Ontology terms** (cell types only, optional): whether they want each
  label mapped to a Cell Ontology term, and where their `.obo` file is.

## 4. Hand off

Cell types → `sc-celltype`. Spatial niches → `sc-niche`.

## Before a large run

- **Speed.** The first `lambdakit init` on a dataset computes reference
  statistics and writes two memory-mapped copies of the matrix to
  `~/.cache/lambdakit`, one by cell and one by gene (`--cache-dir` to move
  them; each is about the size of the matrix, and building the gene copy needs
  about twice that in RAM). This can take many minutes; run it with the Bash
  tool's maximum timeout and wait for it. Later runs on the same file reuse
  both. Without the gene copy (`--no-column-cache`), each gene query on a large
  dataset reads the whole matrix. A run created before this copy existed gets
  it with `lambdakit cache columns --run RUN`.
- **Resuming.** Each step is written to disk as it completes. Rerun with the
  same `--run` and start with `lambdakit status`, which names the next action.
  Warn the user up front if quota may run out partway.
- **Parallelism.** Beyond a handful of clusters, use one subagent per cluster.
  Per-cluster state lives in separate files. Keep subclustering and the final
  passes in the orchestrator, because they change shared state.
