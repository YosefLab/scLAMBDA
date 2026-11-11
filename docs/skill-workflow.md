# lambda-annotate

Cell-type and spatial-niche annotation for single-cell and spatial
transcriptomics, as a Claude Code plugin. **No API key.**

This is the Claude Code skill frontend of [sclambda](../README.md). Reasoning runs
in the agent session; the same local statistical engine also powers the Python/API frontend.

## What it does

Cell-type clusters use an **actor → measurements → single critic** loop, with
up to four rounds. The actor can test several competing signatures. The critic
chooses a best-supported label with `resolved` or `best_effort` status, evidence
references, alternatives, uncertainty and a next question. An empty next question
ends the loop. Tables contain LFC, expressing proportion and significance.

`lambdakit de rank --record` and `de genes --record` persist authoritative tables;
`lambdakit dossier decision --file decision.json` validates and stores the compact
critic decision. There is no additional claim reviewer, required catalog,
resolution ladder or mandatory cross-cluster correction in cell typing.
Substructure assessment is a separate decision before optional subclustering.

**Spatial niches** use one measurement-seeking actor and one deciding critic.
Population labels are optional: omit `--cell-type-col` (or `cell_type_col` in
Python) to use a uniform `spot` placeholder. Both frontends mark this provenance
in measurement scopes: expression is pooled across spatial observations and
cell-type composition is unavailable. CLI resume recreates the column in memory;
the source H5AD is unchanged.

New skill runs use comparative review: the actor records a comparison plan before measurement; the critic evaluates
observations as shared, favoring candidates, uninformative or conflicting. The
accepted comparison determines the exported label and explanation. Unverified
premises cannot be necessary support, and opposing observations require explicit
tradeoffs using existing evidence. This validates declared consistency, not
biological truth. Favoring interpretations explicitly state comparative expectation,
applicability to the actual measurement, and limitations. Expression facts in the
derived explanation come from cited records, not model transcription. The selection
review cites decisive observations against its strongest alternative; an explicitly
choice-changing remaining issue preserves the label with best-effort status even
if no useful tool is available. Ordinary limitations need not prevent resolution.
Matching Python runs must explicitly set `spatial_comparison_review=True` and
`spatial_resolution_comparison=False`; the Python API still defaults to compact
mode. Preserve explicitly selected legacy modes on resume.
Composition, expression tables and slide-scoped image observations remain
separate evidence records. Global review proposes contradictions for the same
critic; there is no mandatory majority override or separate biological
standardizer. Labels carry resolved/best-effort status and uncertainty.
The critic records executable `pending_checks` for useful remaining comparisons.
Python executes them before its next review; the skill runs the matching CLI
measurement commands. A nonempty list keeps the decision best-effort and survives
budget exhaustion. New spatial CLI decisions require this field, while older
saved dossiers remain readable. Expression citations declare their population
and reference; mismatches are rejected rather than repaired into another claim.
These checks verify execution and provenance, not biological correctness.
If the main review fails, at most two flat final critic attempts select a
best-effort label with the same evidence and scope validation. If both fail,
`lambdakit dossier retain-failure --cluster ID --reason "failure details" --run RUN`
uses the same recovery helper as Python: retain the last accepted label or an
explicit anatomy-unassigned record, flagged `execution_status=inference_failed`.
Commit this output before export. The CLI does not run the final model attempts;
the skill performs and records them. See the [spatial recovery procedure](../skills/sc-niche/references/recovery.md).

An opt-in experiment (`spatial_resolution_comparison=True`) separates a shared
supported identity from a finer candidate and its closest alternative. The
runtime derives the label from the critic's `resolution.fine_supported` decision.
This enforces internal consistency, not biological specificity. Real-model tests
showed regressions, so that older experiment remains opt-in. The skill's
decision contract documents the optional format; CLI decisions without it remain
valid.

## Install

The skills and their engine are bundled with `sclambda`. From the sc-lambda checkout:

```bash
python -m pip install -e .
lambdakit doctor
lambdakit plugin-path
```

In Claude Code, add the directory printed by `plugin-path` as a marketplace,
then install `lambda-annotate@lambda-annotate`. For an editable checkout this is
the repository root; for a wheel it is the installed, bundled plugin directory.
Use `pip install -e '.[excel]'` for Excel export.

The skills invoke `<plugin>/bin/lambdakit`. It works with the source tree or an
installed wheel; `LAMBDAKIT_PYTHON` selects the scientific Python interpreter.

## Permission prompts

An annotation run issues dozens of distinct `lambdakit` commands. Permission
rules match the *command string*, not the binary, so approving them one by one
means approving each subcommand separately and the run stops constantly.

The plugin ships a `PreToolUse` hook that auto-approves its own engine calls, so
after you install it a run completes without prompting. The approval moment is
installing the plugin, which is where Claude Code already asks you to trust it.

The grant is deliberately narrow. Three things still prompt every time:

- **commands that delete data** — `lambdakit clean`, `lambdakit cache rm`
- **anything with shell metacharacters**, so an approved prefix can never carry
  an unapproved payload behind it (`lambdakit status; rm -rf ~`)
- **everything that is not lambdakit**

If you would rather not have a plugin auto-approve anything, delete
`hooks/hooks.json` from your installed copy, or set
`"disableAllHooks": true`. You will then be prompted per command, or you can add
`"Bash(lambdakit:*)"` to your own `permissions.allow` — note that the blanket
rule *does* cover the destructive subcommands, which the hook does not.

## Use

Point Claude at a dataset and say what you want, in your own words:

> Annotate the cell types in `pbmc.h5ad` — human blood, clusters are in `leiden`.

> Name the spatial niches in `brain.h5ad`. Coordinates are in `X_spatial`, cell
> types in `cell_type`, niches in `niche`, mouse brain coronal section.

The skills are picked up automatically from what you ask — `sc-annotate` inspects
the file, checks it is usable, and routes to `sc-celltype` or `sc-niche`. Claude
then does the reasoning and calls the engine for every measurement. You do not
type any of the commands below yourself; they are what Claude runs.

## Interrupted runs

A long run will be interrupted — quota, a closed session, a reclaimed node. The
design assumes it:

- every state file is written to a temp name and **renamed into place**, so a
  process killed mid-write leaves the previous version, never truncated JSON;
- per-cluster work is checkpointed as it completes, so at most the step in
  flight is lost;
- one-shot whole-run phases (the cross-cluster pass, the spatial final passes,
  the export) are recorded in a ledger, because they leave no per-cluster
  artefact and would otherwise be silently repeated or skipped.

On restart, one command reconstructs everything:

```bash
lambdakit status --run RUN
```

It reports what is annotated per depth, which phases have run, which clusters
were opened but never committed, and the single next action. Interrupted
clusters resume from their dossier — saved decision, uncertainty and every
measurement already taken are all still there.

## What you get

```
<run>/manifest.json          dataset and run parameters
     /progress.json          which whole-run phases have completed
     /overlay.parquet        per-cell cluster and annotation columns
     /genes.txt              the measured gene panel
     /catalog.json           the tissue catalog
     /dossiers/<id>.json     per-cluster belief state
     /results/<id>.json      committed label, justification, confidence
     /transcripts/<id>.jsonl reasoning transcript
     /images/*.png           rendered niche maps
```

Export with `lambdakit export --run RUN --csv out.csv --xlsx out.xlsx --md report.md`.

## What the run cost

```bash
lambdakit usage --run RUN            # tokens, split by orchestrator vs subagents, and by model
lambdakit usage --run RUN --json     # the same, machine-readable
lambdakit usage --run RUN --price claude-opus-5=5,25 --price claude-fable-5=10,50
```

A run can span models — orchestrator on one, subagents on another, a mid-run
`/model` switch — and their prices differ by 2x or more, so usage is always split
per model and there is no single blended rate. Prices are never hard-coded, since
a stale table would quietly misreport every run made after the next price change.

The engine cannot see token usage — it lives in the harness — so this reads the
counts back out of Claude Code's own transcripts under `~/.claude/projects/`.
Subagent cost is exact, because their transcripts name the run directory they
were given; the orchestrator's share is bounded by the run's time window and
reported as an upper bound, since one session interleaves annotation with
whatever else was going on.

Cached tokens dominate the raw counts and are priced in three tiers, so they are
never summed into one number. Each tier is converted to a common
**input-token equivalent** (5-minute write ×1.25, 1-hour write ×2.0, read ×0.1)
— ratios against base input, not dollars, so they survive price changes. Pass
`--price MODEL=IN,OUT` per model, with current rates, for a dollar figure.

If the run was not done in a Claude Code session, or the transcripts have
been pruned, this reports `Unavailable` with the reason rather than failing.

## Reclaiming disk

```bash
lambdakit clean --run RUN            # report only — never deletes
lambdakit clean --run RUN --images   # rendered maps, regenerable
lambdakit clean --run RUN --cache    # the shared dataset cache, rebuilt on next use
```

On a real atlas the cache is roughly the size of the expression matrix and is
usually what is worth clearing — but it is shared, so only clear it when no other
run needs that dataset. Rendered images dominate a spatial run and scale with
`--px` rather than with the data.

`--run-dir` deletes the run itself and **refuses** when annotations have not been
exported: the labels and the reasoning behind them cannot be recomputed, and the
same run again would not reproduce them.

## The dataset cache

Anything derived purely from the `.h5ad` — the memory-mapped copy of the matrix
and the reference statistics — is **not** kept in the run directory. It lives in
a cache keyed on the file itself, so a second run over the same dataset reuses it
instead of spending minutes rebuilding a duplicate:

```
~/.cache/lambdakit/<stem>-<fingerprint>/
    meta.json           the file identity this was built from
    csr_*.npy           memory-mapped matrix
    stats-<params>.npz  reference mean/var, per-cell totals, dropout rates
```

The fingerprint covers path, size and mtime, so editing the `.h5ad` gets a new
directory rather than silently reusing stale arrays; the superseded entry is
reported as stale.

```bash
lambdakit cache ls                 # what is cached, how big, still current?
lambdakit cache path --run RUN     # where this run's dataset cache is
lambdakit cache rm --stale         # drop entries whose source changed
lambdakit cache rm --run RUN       # drop this dataset's cache
```

Relocate it with `--cache-dir` on `init` or `$LAMBDA_CACHE_DIR` — worth doing if
`$HOME` is small or on a slow filesystem, since the cache is roughly the size of
the expression matrix. The resolved location is recorded in the run's manifest,
so a run keeps finding its cache even if the environment changes later.

## The engine

You never invoke this directly when using the skills — Claude does. It is
documented because it is also a normal CLI, usable on its own:

```bash
lambdakit init data.h5ad --run .lambda/run1 --organism human \
    --tissue "human colon (intestinal tract)" --gene-col feature_name \
    --embedding-key X_scVI --obs-column cluster --depth 2

lambdakit clusters      --run .lambda/run1
lambdakit de rank       --run .lambda/run1 --cluster 3
lambdakit de genes      --run .lambda/run1 --cluster 3 --genes CD8A,CD8B --vs 7
lambdakit substructure  --run .lambda/run1 --cluster 3
lambdakit spatial composition --run .lambda/run1 --niche 3
lambdakit spatial image --run .lambda/run1 --niche 3 --number
```

Every command prints Markdown, because its reader is a language model.

## Shared implementation

The Python/API frontend obtains actor and critic responses from configured model
providers. The skill performs those roles in the agent session and invokes the
local CLI. Both use the same statistical implementation and compact cell-type
measurement/reference validation. The CLI persists decisions but does not call
models or automatically enforce the four-round reasoning budget.

The Python pipeline also provides optional Cell Ontology mapping. That mapping
is not part of the skill's current CLI workflow. Spatial niches share comparison normalization, evidence/scope validation and
failure retention across both frontends, with composition and image references
in addition to genes; legacy dossiers remain readable.

## Tests

```bash
pip install -e '.[dev]' && pytest tests/
```

No API key and no network needed.

## Licence

MIT; see `LICENSE`.

## Not included

Ontology mapping and the opt-in hierarchical coarse-domain pass of the spatial
pipeline have no skill equivalent.

### Spatial evidence handling

The actor and critic inspect identical neutral red-target/gray-context views;
the Python default does not ask a separate model to describe images. Automatic
`spatial_consistency_review` is now false. The critic can request additional
slide views within its existing loop; explicit global review remains available.

Spatial gene queries resolve species-aware NCBI aliases and record source hashes,
ambiguity and actual panel features. Configure `gene_alias_path`,
`gene_alias_download`, and `evidence_cache_dir` in Python, or `--alias-file` /
`--evidence-cache` on `spatial genes`. `SCLAMBDA_GENE_ALIAS_DOWNLOAD=0` disables
downloads. External-source failure is not gene absence.

The actor's `lookup_reference` action (or `reference_queries` alongside panels)
and `lambdakit spatial reference` retrieve attributed Europe PMC passages for a
specific distinction. Set `reference_lookup=False` for offline Python runs.
The existing critic assesses their relevance; no additional biological reviewer
or cell-specific marker gates are introduced. Cache location defaults to
`~/.cache/sclambda/evidence`, overridable with `SCLAMBDA_EVIDENCE_CACHE`.

Spatial plots preserve stored y orientation by default (`flip_spatial_y=False`); explicit reflection remains supported. Paired pipeline views share a section coordinate frame, have view titles, and mark the crop on the whole section. Critic prompts compare alternatives before choosing, and the runtime retains candidate names across rounds without treating earlier suggestions as evidence. No extra decision fields or reviewer were added.
