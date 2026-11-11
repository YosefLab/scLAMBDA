# LAMBDA 2.0 -- Parameter Reference

## `Lambda()` constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| Path` | *required* | Path to `.h5ad` file |
| `usage_log_path` | `str \| Path \| None` | `None` | Append-only JSONL request/usage log, including actor/critic calls and retries. Contains prompt text; useful for interrupted runs. |
| `provider` | `str` | `"anthropic"` | LLM provider: `"openai"`, `"anthropic"`, `"google"` |
| `power` | `str` | `"medium"` | LLM power level: `"weak"`, `"medium"`, `"strong"` |
| `embedding_key` | `str` | `"X_scVI"` | Key in `adata.obsm` used for clustering |
| `gene_col` | `str \| None` | `None` | Column in `adata.var` with gene names. `None` = use `var.index` |
| `gene_filter` | `Callable \| None` | filters RPL/RPS/MT | Function to filter gene names. `None` disables filtering |
| `chunk_size` | `int` | `50_000` | Chunk size for computing reference stats |
| `target_sum` | `float` | `1e4` | Target sum for normalization |
| `n_top_genes` | `int` | `20` | Number of top DE genes per cluster |
| `use_mmap_csr` | `bool` | `True` | Use memory-mapped CSR arrays for faster access |
| `column_cache` | `bool` | `True` | Before `annotate()`, build or reuse a gene-major copy of a sparse matrix in the dataset cache so targeted DE reads only the requested genes. About the matrix's size on disk; twice that in RAM while building. |
| `max_retries` | `int` | `3` | Max retry attempts for LLM errors |
| `subcluster_n_neighbors` | `int` | `30` | Number of neighbors for Leiden subclustering |
| `model_overrides` | `dict \| None` | `None` | Override model IDs per power level, e.g. `{"medium": "claude-sonnet-4-5-20250929"}` |

## `annotate()`

### General parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pipeline` | `str` | `"celltype"` | Pipeline to run. Options: `"celltype"`, `"annotate_spatial"`, `"narrativize"` |
| `depth` | `int` | `1` | Max hierarchy depth for recursive subclustering |
| `resolution` | `float` | `0.3` | Leiden clustering resolution |
| `max_workers` | `int` | `8` | Max parallel workers for concurrent cluster annotation |
| `obs_columns` | `str \| list[str] \| None` | `None` | Use pre-existing clustering from these `.obs` columns. Single string or list for hierarchical levels |
| `clusters` | `str \| list[str] \| None` | `None` | Only annotate these specific clusters (requires `obs_columns`) |
| `use_leiden` | `bool` | `True` | Use Leiden clustering when `obs_columns` is not set |
| `organism` | `str` | `"human"` | Organism name (used in LLM prompts) |
| `tissue` | `str` | `"unknown"` | Tissue type (used in LLM prompts) |
| `tissue_col` | `str \| None` | `None` | `.obs` column with per-cell tissue labels (overrides `tissue`) |
| `de_method` | `str` | `"ttest"` | Differential expression method |

### Cell type pipeline parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `full_pipeline` | `bool` | `True` | Run full pipeline with marker refinement. `False` = use best initial prediction |
| `naive` | `bool` | `False` | Simplified single-question annotation (skips refinement, cross-cluster, ontology) |
| `min_cluster_size` | `int` | `50` | Minimum cells per cluster |
| `min_subcluster_proportion` | `float` | `0.01` | Minimum subcluster proportion to keep |
| `min_reevaluation_iterations` | `int` | `1` | Minimum reevaluation iterations |
| `max_reevaluation_iterations` | `int` | `2` | Maximum reevaluation iterations |
| `use_cell_ont` | `bool` | `False` | Map annotations to Cell Ontology terms |
| `ontology_file` | `str \| Path \| None` | `None` | Path to `.obo` ontology file (required if `use_cell_ont=True`) |
| `use_dossier` | `bool` | `False` | Select the actor–measurements–critic workflow; explicitly enable for the current cell-type skill equivalent |
| `dossier_agent_max_steps` | `int` | `4` | Maximum actor–critic rounds |
| `use_cross_cluster` | `bool` | mode-dependent | Cross-cluster review: defaults to `False` in dossier mode, `True` in the legacy workflow |

The dossier critic always decides the label. Substructure assessment separately decides whether
to split a parent when the requested depth permits it. Uncertain valid decisions
retain a best-supported label with `best_effort` status.

### Spatial pipeline parameters

Used when `pipeline="annotate_spatial"`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cell_type_col` | `str \| None` | `None` | Optional `.obs` column with cell type annotations. If omitted, creates uniform `spot` labels in memory for pooled spatial analysis; source H5AD is unchanged. |
| `spatial_key` | `str` | `"spatial"` | Key in `adata.obsm` for spatial coordinates |
| `k_neighbors` | `int` | `10` | Number of neighbors for spatial weight matrix |
| `k_top_cell_types` | `int` | `5` | Number of top cell types per niche for DE |
| `slice_key` | `str \| None` | `None` | `.obs` column with slice/section labels |
| `slice` | `str \| None` | `None` | Restrict to this single slice. `None` = multi-slide mode |
| `control_slide` | `str \| None` | `None` | Control/reference slide identifier |
| `min_niche_count` | `int` | `30` | Min cells on control slide to use it for a niche |
| `include_spatial_image` | `bool` | `True` | Attach spatial niche images to LLM prompts |
| `spatial_image_px` | `int` | `700` | Target pixel size for spatial niche images |
| `flip_spatial_y` | `bool` | `True` in public API | Negate Y coordinates. Pass `False` to preserve stored orientation, as in the README. Direct pipeline calls default to `False` |
| `standardize_niches` | `bool` | `True` in public API | Legacy compatibility argument; the current spatial graph has no separate standardization pass |
| `spatial_background_col` | `str \| None` | `None` | `.obs` column for background cell colouring in spatial images |
| `spatial_comparison_review` | `bool` | `False` | Enable comparative critic decisions; set `True` to match new spatial skill runs |
| `spatial_resolution_comparison` | `bool` | `False` | Older resolution-comparison experiment; keep `False` when using comparative review |
| `dossier_agent_max_steps` | `int` | `4` | Main review budget; up to two compact final critic attempts follow if no valid final decision is obtained |
| `spatial_consistency_review` | `bool` | `False` | Optional whole-section contradiction review by the deciding critic |
| `use_anatomical_catalog` | `bool` | `False` | Optional compartment vocabulary; not a restriction on candidate identities |
| `reference_lookup` | `bool` | `True` | Allow focused literature retrieval for uncertain biological claims |
| `gene_alias_download` | `bool` | `True` | Allow NCBI gene-name source downloads for alias resolution |
| `gene_alias_path` | `str \| None` | `None` | Pinned local NCBI gene_info source for gene-name resolution |
| `evidence_cache_dir` | `str \| None` | `None` | Evidence cache override; otherwise uses `SCLAMBDA_EVIDENCE_CACHE` or `~/.cache/sclambda/evidence` |

The current spatial graph uses direct neutral whole-section/crop images and one
deciding critic. Overall and within-type expression contrasts retain their own
reference populations; expression pools slides when `slice` is omitted, while
images and spatial statistics are slide-specific. See the
[spatial skill](skills/sc-niche/SKILL.md) for decision and recovery semantics.

## CPU clustering runtime

CPU Leiden jobs run in fresh Python processes with atomic stage checkpoints.
Failed or stalled workers are terminated and retried once from the available
checkpoints. Shared annotation state is locked only when reading or publishing
cluster membership, not during clustering. GPU execution remains in-process.

| Environment variable | Default | Description |
|---|---|---|
| `SCLAMBDA_CLUSTER_PROCESSES` | `2` | Maximum simultaneous CPU clustering workers per calling process; set before importing the package |
| `SCLAMBDA_CLUSTER_THREADS` | `1` | Native threads per clustering worker |
| `SCLAMBDA_CLUSTER_STAGE_TIMEOUT` | `1800` | Seconds without a new stage event before terminating a worker |
| `SCLAMBDA_CLUSTER_CACHE` | `<cache root>/clustering-v1` | Checkpoint directory; jobs are keyed by embeddings, parameters, implementation and dependency versions |
| `SCLAMBDA_SILHOUETTE_CACHE_BYTES` | `536870912` | Maximum pairwise-distance cache size for the resolution sweep; larger problems use the uncached calculation |
| `SCLAMBDA_DISABLE_HNSWLIB` | unset | Set to `1` to use exact nearest-neighbor propagation; this is also the fallback when `hnswlib` is unavailable |

Each job directory contains `worker.log`, `events.jsonl`, and `monitor.json` for
diagnosing failures. Checkpoints do not contain or mutate the annotation accessor.

## `set_reference()`

Provide external reference statistics for improved DE. Either pass arrays directly or load from a `.npz` file.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mean` | `np.ndarray \| None` | `None` | Reference mean array (n_genes,) |
| `var` | `np.ndarray \| None` | `None` | Reference variance array (n_genes,) |
| `n` | `int \| None` | `None` | Number of cells used to compute reference |
| `path` | `str \| Path \| None` | `None` | Path to `.npz` file with saved reference stats |

## `compute_reference()`

Compute reference statistics from external h5ad files.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `files` | `list[str \| Path]` | *required* | Paths to h5ad files |
| `output_path` | `str \| Path \| None` | `None` | Path to save reference stats as `.npz` |

Returns a dict with `mean`, `var`, `n`.

## `write()`

Export annotations to files.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `output_path` | `str \| Path \| None` | `None` | Path for CSV output |
| `excel_path` | `str \| Path \| None` | `None` | Path for Excel output |

Returns a list of written file paths.

## `get_annotations()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `depth` | `int \| None` | `None` | Specific hierarchy depth, or `None` for all |

Returns `dict[str, str]` mapping cluster IDs to annotation labels.

## `get_overlay()`

No parameters. Returns the full overlay `DataFrame` with all clustering columns and annotations.

## `write_narrative()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `results` | `dict` | *required* | Return value from `annotate(pipeline="narrativize")` |
| `output_dir` | `str \| Path` | `"narrativize_output"` | Output directory |

Returns the resolved output directory `Path`.
