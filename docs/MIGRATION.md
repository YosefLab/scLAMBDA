# Consolidation in sclambda 0.3

## Computing fixes in 0.3.1

The maintained `main` branch combines the current annotation and skill workflows
with the runtime fixes previously developed on the blood-clustering debug branch.
It preserves the newer sparse column-cache implementation.

Subset annotation now keeps unselected clusters separate from encoded selected
clusters. Subcluster renumbering applies all replacements simultaneously, including
metadata, so renaming one child cannot accidentally merge it into another. These
changes prevent future corruption; they do not repair previously saved memberships.

CPU Leiden runs in isolated processes with stage checkpoints, bounded native
threads, timeout diagnostics and one recovery attempt. Shutdown waits outside the
shared-state lock, and long clustering work no longer holds that lock. Linux
native-library discovery uses a compatibility fix for affected threadpoolctl
versions. See [runtime settings](../PARAMETERS.md#cpu-clustering-runtime).

Usage totals now include actor and critic calls, retries and final recovery calls.
The optional request ledger can retain usage through interrupted runs. Historical
totals cannot be repaired without the corresponding request records. Spatial runs
without `cell_type_col` use a uniform in-memory `spot` population; supplied cell
type labels still enable within-population comparisons.

The study loader in `examples/census/load_data.py` applies `is_primary_data` to
every requested ontology term. It requires the optional `cellxgene-census` package.
Existing Census downloads must be filtered or regenerated separately.

Annotation modes are unchanged: explicitly set `use_dossier=True` for the current
cell-type actor–critic workflow and `spatial_comparison_review=True` for comparative
spatial review. This release does not make either mode the API default.

## Installation and compatibility

`sc-lambda` is the maintained repository. Its `sclambda` distribution includes
the engine, `lambdakit` command, skills, plugin, and prompt files. Install it once;
use a provider extra (`openai`, `anthropic`, or `google`) for Python/API annotation.
`llm` installs the pipeline runtime; `providers` installs all supported providers.

The previous `lambda-skill` repository is a compatibility bridge. Its Python
files delegate to sclambda, and its package metadata installs only a dependency
on sclambda. It owns no engine files or console script. Old plugin assets are
retained there for existing checkouts; install the consolidated plugin to receive
current skills. No plugin installation is changed automatically by this migration.

For existing editable installs, use the same interpreter as before:

```bash
python -m pip uninstall lambdakit
python -m pip install -e /path/to/sc-lambda
lambdakit doctor
lambdakit plugin-path
```

Uninstall the old distribution before installing the combined one to avoid two
distributions owning files under the same `lambdakit` namespace. The `lambdakit`
command is unchanged, but the old Python import paths have been removed; import
from the canonical modules instead:

| Old import | Canonical module |
|---|---|
| `lambdakit.tools.*`, `sclambda.tools.*` | `sclambda.analysis.*` |
| `lambdakit.core.*`, `lambdakit.cache` | `sclambda.core.*` |
| `lambdakit.session` | `sclambda.storage.session` |
| `lambdakit.commands.*` | `sclambda.cli.commands.*` |
| `lambdakit.dossier.celltype`, `sclambda.pipelines.celltype.dossier` | `sclambda.annotation.celltype.dossier` |
| `lambdakit.dossier.niche`, `sclambda.pipelines.annotate_spatial.dossier` | `sclambda.annotation.niche.dossier` |
| `lambdakit.fmt_celltype`, `sclambda.pipelines.celltype.formatters` | `sclambda.annotation.celltype.formatters` |
| `lambdakit.fmt_niche`, `sclambda.pipelines.annotate_spatial.formatters` | `sclambda.annotation.niche.formatters` |
| `lambdakit.substructure`, `sclambda.pipelines.celltype.substructure` | `sclambda.annotation.celltype.substructure` |
| `lambdakit.catalog_celltype` / `lambdakit.catalog_niche` | `sclambda.annotation.celltype.catalog` / `sclambda.annotation.niche.catalog` |

The `pathology` and `multi_pathology` pipelines have been removed, together with
`sclambda.analysis.cell_type_proportions`, `compute_pathology_de`, and the
`annotate()` arguments `reference_col`, `healthy_value`, `dataset_id_col`,
`prediction_type`, `sample_col` and `question`.

`sclambda.analysis.ontology_mapper` is now `sclambda.analysis.ontology`. Its
unused `OntologyMapper` class has been removed, and OBO files are parsed
directly, so the `ontology` extra (`obonet`, `networkx`, `thefuzz`) no longer
exists. The skill can map committed labels to Cell Ontology terms with
`lambdakit ontology`.

## Existing runs and caches

Legacy manifests without a schema version remain readable. New manifests include
a schema version and dataset fingerprint. New runs reject changed source datasets;
overlay restoration checks cell identities and order instead of assigning indices
positionally. Existing legacy runs cannot retrospectively prove source provenance.

Repeating an identical run initialization resumes it without replacing state.
Changing parameters requires a new run directory. Reopening an existing dossier
preserves its evidence and committed result. Overlay, dossier, composition, and
probe checkpoints use atomic replacement.

The API and CLI now use the dataset cache selected by `LAMBDA_CACHE_DIR`,
`XDG_CACHE_HOME`, or `~/.cache/lambdakit`. Its fingerprint includes the resolved
source path, size, modification time, and matrix key. Readers never delete shared
arrays on close. The old `sort_rows_by_labels` optimization is a compatibility
no-op: shared arrays retain the original cell order. Existing `.lambda_tmp`
files are not deleted automatically.

## Annotation contracts

Both frontends apply the same critic promotion rule: the target must be an
existing candidate, and a new promotion needs a resolved-axis note in the current
verdict. A previous note alone cannot promote a new candidate. The dossier's
leader-change gate still decides whether the promotion is accepted.

Normal commits require the supported candidate at the committed granularity.
For a user-requested manual correction, `dossier commit --label NAME
--manual-override-reason REASON` stores explicit override provenance and does not
borrow the previous candidate's confidence.

Numerical DE and clustering behavior is preserved. In particular, the existing
`get_reference_indices` behavior is global at every depth. Use explicit reference
indices with `compute_de_for_genes` for parent-relative or peer comparisons.
`zero_rate_diff` retains its serialized field name but means target minus reference
fraction expressing (DFE), as in the pre-merge implementation. Automatic Leiden
resolution selection still ignores its legacy `resolution` argument.

The test suite now checks these current contracts. Live provider tests are opt-in;
no real annotation run or biological benchmark is part of the consolidation tests.
