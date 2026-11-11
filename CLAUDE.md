# sclambda — development guide

sclambda annotates single-cell clusters (cell types) and spatial niches from
`.h5ad` files. It has two frontends over one statistical engine:

- **`lambdakit` CLI + Claude Code skills** (`skills/`, `src/sclambda/cli/`):
  Claude does the reasoning; the CLI computes DE, composition and images and
  persists decisions in a run directory.
- **Python API** (`sclambda.Lambda`): LangGraph pipelines call an LLM provider
  directly.

## Layout

```
src/sclambda/
├── main.py             # Lambda: annotate(), write(), get_annotations()
├── core/               # Accessor (AnnData + overlay), H5adMatrixReader, TaskRunner, dataset cache
├── analysis/           # rank_genes, targeted_de, subcluster, spatial_metrics, spatial_visualization,
│                       # ontology, clustering worker processes
├── annotation/         # Dossiers, catalogs, formatters and decision rules shared by both frontends
│   ├── celltype/
│   └── niche/
├── framework/          # BaseNode, LLMNode, ParseSchema, Pipeline, LLMPromptChain
├── pipelines/          # LangGraph pipelines + registry
│   ├── celltype/       # pipeline.py, nodes/ (package), prompts/
│   └── annotate_spatial/
├── llm/                # LLMClient, providers, usage ledger
├── cli/                # lambdakit entry point (main.py) and commands/
├── storage/            # Run-directory Session, atomic writes
├── reporting/          # Annotation files and prompt history
└── plugin/             # Locates the bundled Claude Code plugin
skills/                 # Skill definitions shipped in the plugin
```

Built-in pipelines are listed in `pipelines/registry.py` (`_BUILTINS`):
`celltype`, `annotate_spatial`. Third-party pipelines
register through `register_pipeline()` or the `sclambda.pipelines` entry point.

## Writing a pipeline

1. Define a `TypedDict` state (`total=False`) with `cluster_id`, `depth`,
   `max_depth`, `children`, plus your fields.
2. Write nodes: subclass `BaseNode` (computation) or `LLMNode` (LLM calls).
   Declare `default_output` as a class attribute; `__call__` copies it to
   `self.output` before `execute(state)`, so a failing node still returns valid
   state.
3. Subclass `Pipeline`: set `name`, `required_params`, `max_depth_override`
   (`1` for flat), implement `build_graph(runner)`, and call
   `super().make_cluster_state()` in `make_cluster_state()` — it sets the
   reevaluation counters, `organism`, `tissue` and `n_top_genes` that routing
   relies on.
4. Register it with `register_pipeline(name, cls)` or add it to `_BUILTINS`.

```python
from langgraph.graph import StateGraph, END
from sclambda.framework import Pipeline

class TrajectoryPipeline(Pipeline):
    name = "trajectory"
    required_params = ("root_cell_type",)
    max_depth_override = 1

    def make_cluster_state(self, cluster_id, depth, max_depth, **kwargs):
        state = super().make_cluster_state(cluster_id, depth, max_depth, **kwargs)
        state["root_cell_type"] = kwargs["root_cell_type"]
        state["children"] = []
        return state

    def build_graph(self, runner):
        graph = StateGraph(TrajectoryState)
        graph.add_node("rank", RankNode(self.accessor, runner))
        graph.add_node("infer", InferTrajectoryNode(self.accessor, runner, self.llm_client))
        graph.set_entry_point("rank")
        graph.add_edge("rank", "infer")
        graph.add_edge("infer", END)
        return graph.compile()
```

## Conventions

- **Prompts** live in `pipelines/<name>/prompts/` as Markdown with `{variable}`
  placeholders. Format data with `format_*` functions from `annotation/*/formatters.py`
  rather than string concatenation. Delete a prompt when its last caller goes.
- **Structured LLM output**: prefer `self.chain.execute(prompt_file, parse_prompt_file=...)`
  plus `ParseSchema` over hand-written query/parse sequences.
- **Accessor writes** from nodes go through `self.runner.with_lock(self.accessor.set_annotation, ...)`.
  Reads need no lock. Never call `with_lock()` inside a `runner.map_parallel()` task.
- **Overlay**: use `set_cluster_labels()`, `set_annotation()`, `get_cell_indices()`;
  do not write `accessor._overlay` or `accessor._obs` directly.
- **Errors**: let unexpected exceptions propagate. Catch only the specific
  exception you can handle, and do not turn a failure into a silent default.
- **No compatibility layers**: when a module moves, update its importers. Do not
  add `sys.modules` aliases, "legacy" fallback branches, or flags nobody reads.
- **Naming**: `XyzPipeline`, `XyzNode`, `XyzState`, `format_xyz`, verb-named
  analysis functions; prompt files `{number}_{description}.md`.

## Commands

```bash
pip install -e '.[dev,providers,excel]'
pytest tests/                       # no API keys or network required
ruff check src tests
lambdakit doctor                    # environment check for the CLI
```

Provider keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`.
Power levels (`weak`, `medium`, `strong`) map to models in `llm/api.py`.

## Tests

Tests build small synthetic AnnData files and never call an LLM; graph
compilation is checked with `Mock()` clients:

```python
def test_compiles(accessor):
    pipe = MyPipeline(accessor, Mock())
    with TaskRunner(max_workers=1) as runner:
        assert "my_node" in pipe.build_graph(runner).get_graph().nodes
```
