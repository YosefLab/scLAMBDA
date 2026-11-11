"""`lambdakit` — the statistical tool surface for the sc-lambda annotation skills.

Every command prints Markdown for a reader that is a language model: tables to
reason over, and, after any mutating call, the new state so the caller never has
to re-read a file it just wrote.

    lambdakit status      where the run is and what to do next
    lambdakit init        load a dataset and cache what is expensive
    lambdakit clusters    roster with per-cluster status
    lambdakit de          ranked markers / targeted signatures / peer comparison
    lambdakit dossier     the belief state and its update contract
    lambdakit substructure  deterministic split probe
    lambdakit subcluster  split a flagged cluster
    lambdakit spatial     niche composition, spatial DE, rendered maps
    lambdakit catalog     store the tissue catalog
    lambdakit ontology    map committed labels to Cell Ontology terms (optional)
    lambdakit cache       inspect the dataset cache shared by all runs
    lambdakit clean       report disk use and reclaim it
    lambdakit usage       what the annotation cost in tokens
    lambdakit export      annotations to CSV / Excel / Markdown
"""

from __future__ import annotations

import argparse
import logging
import sys

from sclambda._version import __version__


def _add_run(p, required=True):
    p.add_argument("--run", required=required, help="run directory created by `lambdakit init`")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lambdakit", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"lambdakit {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="log progress to stderr")
    sub = p.add_subparsers(dest="group", required=True)

    q = sub.add_parser("plugin-path", help="print the bundled plugin directory for installation")
    q.set_defaults(func=_plugin_path)

    # ---------------- init ----------------
    q = sub.add_parser("init", help="load a dataset, validate it, cache reference statistics")
    q.add_argument("h5ad")
    q.add_argument("--run", help="run directory (default .lambda/<dataset stem>)")
    q.add_argument("--organism", default="human")
    q.add_argument("--tissue", default="unknown", help="be specific: 'human colon (intestinal tract)', not 'gut'")
    q.add_argument("--gene-col", help="var column holding gene SYMBOLS (omit if var.index has them)")
    q.add_argument("--embedding-key", default="X_pca", help="obsm key used for subclustering and the substructure probe")
    q.add_argument("--obs-column", help="obs column holding the clustering to annotate")
    q.add_argument("--use-leiden", action="store_true", help="cluster de novo instead")
    q.add_argument("--n-neighbors", type=int, default=30)
    q.add_argument("--cell-type-col", help="optional obs column with cell types; spatial runs use uniform 'spot' labels when omitted")
    q.add_argument("--spatial-key", help="obsm key with 2-D coordinates; presence selects the spatial mode")
    q.add_argument("--slice-key", help="obs column identifying the slide/section")
    q.add_argument("--slice", help="annotate one slide only; omit for cross-slide mode")
    q.add_argument("--n-top-genes", type=int, default=20)
    q.add_argument("--de-method", default="ttest", choices=["ttest", "ols"])
    q.add_argument("--depth", type=int, default=1, help="1 = flat; 2+ allows subclustering")
    q.add_argument("--min-cells", type=int, help="run-wide DE floor; set below 10 for pseudobulk input")
    q.add_argument("--no-column-cache", action="store_true",
                   help="skip the gene-column copy of a sparse matrix (about the matrix's size "
                        "on disk, and twice that in RAM while building); gene queries on large "
                        "datasets are then much slower")
    q.add_argument("--cache-dir", help="where the shared dataset cache lives "
                                       "(default $LAMBDA_CACHE_DIR, else ~/.cache/lambdakit)")
    q.set_defaults(func=lambda a: _setup().cmd_init(a))

    # ---------------- status ----------------
    q = sub.add_parser("status", help="where the run is and what to do next (start here after a restart)")
    _add_run(q)
    q.add_argument("--stale-minutes", type=int, default=30,
                   help="how long an uncommitted cluster must be idle to count as interrupted")
    q.set_defaults(func=lambda a: _status().cmd_status(a))

    q = sub.add_parser("phase", help="record one-shot whole-run phases so a restart skips them")
    _add_run(q)
    q.add_argument("state", nargs="?", default="done", choices=["done", "in-progress", "skipped"])
    q.add_argument("--name", help="cross-cluster | consistency-review | ontology | export")
    q.add_argument("--note", default="")
    q.add_argument("--list", action="store_true", help="show the recorded phases")
    q.set_defaults(func=lambda a: _status().cmd_phase(a))

    # ---------------- clusters ----------------
    q = sub.add_parser("clusters", help="cluster roster with status")
    _add_run(q)
    q.add_argument("--depth", type=int, default=1)
    q.add_argument("--show-original", action="store_true",
                   help="reveal original cluster names (export/debugging only — never while annotating)")
    q.set_defaults(func=lambda a: _setup().cmd_clusters(a))

    # ---------------- de ----------------
    q = sub.add_parser("de", help="differential expression")
    ds = q.add_subparsers(dest="cmd", required=True)

    r = ds.add_parser("rank", help="ranked markers, cluster vs rest")
    _add_run(r); r.add_argument("--cluster", required=True)
    r.add_argument("--n", type=int, help="number of genes (default from the run manifest)")
    r.add_argument("--method", choices=["ttest", "ols"])
    r.add_argument("--record", action="store_true", help="log this measurement to the transcript")
    r.add_argument("--min-cells", type=int, help="override the 10-cell DE floor (pseudobulk rows are pools, not cells)")
    r.set_defaults(func=lambda a: _de().cmd_rank(a))

    r = ds.add_parser("genes", help="measure named genes in a cluster")
    _add_run(r); r.add_argument("--cluster", required=True)
    r.add_argument("--genes", required=True, help="comma-separated symbols")
    r.add_argument("--vs", help="compare against this peer cluster instead of the rest")
    r.add_argument("--method", choices=["ttest", "ols"])
    r.add_argument("--record", action="store_true")
    r.add_argument("--min-cells", type=int, help="override the 10-cell DE floor (pseudobulk rows are pools, not cells)")
    r.set_defaults(func=lambda a: _de().cmd_genes(a))

    r = ds.add_parser("peers", help="annotated clusters at the same depth")
    _add_run(r); r.add_argument("--cluster", required=True)
    r.set_defaults(func=lambda a: _de().cmd_peers(a))

    # ---------------- dossier ----------------
    q = sub.add_parser("dossier", help="belief state")
    ds = q.add_subparsers(dest="cmd", required=True)

    r = ds.add_parser("init", help="open a dossier, seeded from the catalog")
    _add_run(r); r.add_argument("--cluster", required=True)
    r.add_argument("--parent", default="", help="parent cluster id")
    r.add_argument("--parent-label", help="parent's provisional label — child requires its own critic review")
    r.add_argument("--position", default="", help="spatial position read (niche runs)")
    r.set_defaults(func=lambda a: _dossier().cmd_init(a))

    r = ds.add_parser("show", help="print the dossier")
    _add_run(r); r.add_argument("--cluster", required=True)
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=lambda a: _dossier().cmd_show(a))

    r = ds.add_parser("update", help="apply a belief_delta (legacy candidate dossiers)")
    _add_run(r); r.add_argument("--cluster", required=True)
    r.add_argument("--json", help="the belief_delta object ('-' reads stdin)")
    r.add_argument("--file")
    r.add_argument("--source", default="agent")
    r.set_defaults(func=lambda a: _dossier().cmd_update(a))

    r = ds.add_parser("evidence", help="append a measured fact")
    _add_run(r); r.add_argument("--cluster", required=True)
    r.add_argument("--source", required=True, help="e.g. signature, de, sibling, image, critic")
    r.add_argument("--fact", required=True)
    r.set_defaults(func=lambda a: _dossier().cmd_evidence(a))

    r = ds.add_parser("decision", help="validate and record a critic decision")
    _add_run(r); r.add_argument("--cluster", required=True)
    r.add_argument("--json"); r.add_argument("--file")
    r.set_defaults(func=lambda a: _dossier().cmd_decision(a))

    r = ds.add_parser("retain-failure", help="retain a flagged spatial output after final critic review fails")
    _add_run(r); r.add_argument("--cluster", required=True)
    r.add_argument("--reason", required=True, help="execution/validation failures after bounded final review")
    r.set_defaults(func=lambda a: _dossier().cmd_retain_failure(a))

    r = ds.add_parser("axes", help="set the needs_subcluster flag")
    _add_run(r); r.add_argument("--cluster", required=True)
    r.add_argument("--json"); r.add_argument("--file")
    r.set_defaults(func=lambda a: _dossier().cmd_axes(a))

    r = ds.add_parser("gap", help="record critic gaps (legacy candidate dossiers)")
    _add_run(r); r.add_argument("--cluster", required=True)
    r.add_argument("--issue", action="append")
    r.add_argument("--discriminator", action="append")
    r.set_defaults(func=lambda a: _dossier().cmd_gap(a))

    r = ds.add_parser("commit", help="commit the leading candidate as the annotation")
    _add_run(r); r.add_argument("--cluster", required=True)
    r.add_argument("--label")
    r.add_argument("--manual-override-reason", help="record an explicit manual correction outside the supported dossier label")
    r.add_argument("--justification")
    r.set_defaults(func=lambda a: _dossier().cmd_commit(a))

    # ---------------- split ----------------
    q = sub.add_parser("substructure", help="deterministic probe: is there anything to split?")
    _add_run(q); q.add_argument("--cluster", required=True)
    q.add_argument("--genes", help="override the identity signature genes")
    q.set_defaults(func=lambda a: _split().cmd_substructure(a))

    q = sub.add_parser("subcluster", help="split a flagged cluster (orchestrator only)")
    _add_run(q); q.add_argument("--cluster", required=True)
    q.add_argument("--force", action="store_true", help="split even without the needs_subcluster flag")
    q.add_argument("--min-cells", type=int, default=50)
    q.add_argument("--n-neighbors", type=int, default=30)
    q.set_defaults(func=lambda a: _split().cmd_subcluster(a))

    # ---------------- spatial ----------------
    q = sub.add_parser("spatial", help="niche composition, spatial DE, rendered maps")
    ds = q.add_subparsers(dest="cmd", required=True)

    r = ds.add_parser("composition", help="cell-type fractions + Moran's Z")
    _add_run(r); r.add_argument("--niche", required=True)
    r.add_argument("--k", type=int, default=10, help="kNN size for Moran's I")
    r.set_defaults(func=lambda a: _spatial().cmd_composition(a))

    r = ds.add_parser("de", help="region-level and per-cell-type DE")
    _add_run(r); r.add_argument("--niche", required=True)
    r.add_argument("--cell-type", help="restrict to one cell type")
    r.add_argument("--k", type=int, default=5, help="how many cell types to test")
    r.add_argument("--k-neighbors", type=int, default=10)
    r.add_argument("--n", type=int)
    r.add_argument("--enriched-only", action="store_true")
    r.set_defaults(func=lambda a: _spatial().cmd_de(a))

    r = ds.add_parser("genes", help="measure a spatial signature and record its comparison")
    _add_run(r); r.add_argument("--niche", required=True)
    r.add_argument("--genes", required=True); r.add_argument("--cell-type")
    r.add_argument("--alias-file", help="species-specific NCBI gene_info snapshot, plain or gzip")
    r.add_argument("--evidence-cache", help="gene-name/reference cache directory")
    r.set_defaults(func=lambda a: _spatial().cmd_genes(a))

    r = ds.add_parser("reference", help="retrieve source passages for an anatomical discriminator")
    _add_run(r); r.add_argument("--niche", required=True)
    r.add_argument("--query", required=True); r.add_argument("--terms", help="comma-separated terms for full-text passages")
    r.add_argument("--evidence-cache")
    r.set_defaults(func=lambda a: _spatial().cmd_reference(a))

    r = ds.add_parser("observe", help="record an observation of a tool-generated image")
    _add_run(r); r.add_argument("--niche", required=True)
    r.add_argument("--image", required=True); r.add_argument("--text", required=True)
    r.set_defaults(func=lambda a: _spatial().cmd_observe(a))

    r = ds.add_parser("image", help="render a niche map to PNG for reading")
    _add_run(r); r.add_argument("--niche")
    r.add_argument("--no-target", action="store_true",
                   help="whole-section overview with nothing highlighted (consistency review)")
    r.add_argument("--mode", default="whole", choices=["whole", "centered"])
    r.add_argument("--slide", help="which slide to draw (default: the one with most niche cells)")
    r.add_argument("--color-by", help="obs column colouring the background")
    r.add_argument("--number", action="store_true", help="draw each niche's id on the map")
    r.add_argument("--px", type=int, default=1500, help="long-edge pixels; above ~1600 is wasted")
    r.add_argument("--out", help="output path")
    r.set_defaults(func=lambda a: _spatial().cmd_image(a))

    r = ds.add_parser("distribution", help="where the niche sits across slides (no image)")
    _add_run(r); r.add_argument("--niche", required=True)
    r.add_argument("--condition-col")
    r.set_defaults(func=lambda a: _spatial().cmd_distribution(a))

    r = ds.add_parser("adjacency", help="which niches border which (per slide, never pooled)")
    _add_run(r); r.add_argument("--slide"); r.add_argument("--k", type=int, default=10)
    r.set_defaults(func=lambda a: _spatial().cmd_adjacency(a))

    r = ds.add_parser("ledger", help="composition table for every niche")
    _add_run(r)
    r.set_defaults(func=lambda a: _spatial().cmd_ledger(a))

    # ---------------- catalog ----------------
    q = sub.add_parser("catalog", help="tissue cell-type / anatomical-compartment catalog")
    ds = q.add_subparsers(dest="cmd", required=True)
    r = ds.add_parser("set"); _add_run(r)
    r.add_argument("--json", help="JSON array ('-' reads stdin)"); r.add_argument("--file")
    r.set_defaults(func=lambda a: _setup().cmd_catalog_set(a))
    r = ds.add_parser("show"); _add_run(r)
    r.set_defaults(func=lambda a: _setup().cmd_catalog_show(a))

    # ---------------- ontology ----------------
    q = sub.add_parser("ontology", help="map committed labels to Cell Ontology terms (optional)")
    ds = q.add_subparsers(dest="cmd", required=True)
    r = ds.add_parser("candidates", help="rank ontology terms for committed labels")
    _add_run(r)
    r.add_argument("--obo", help="OBO file, e.g. cl.obo; remembered for the run after first use")
    r.add_argument("--cluster", action="append", help="limit to this cluster (repeatable); "
                                                      "default: every label without a current term")
    r.add_argument("--query", help="search this name instead of the label (needs one --cluster)")
    r.add_argument("--top", type=int, default=8)
    r.set_defaults(func=lambda a: _ontology().cmd_candidates(a))
    r = ds.add_parser("set", help="record the term chosen for one cluster")
    _add_run(r)
    r.add_argument("--cluster", required=True)
    r.add_argument("--term", help="term ID, e.g. CL:0000160")
    r.add_argument("--none", action="store_true", help="no term fits; give --note")
    r.add_argument("--note", help="why this term, or why none")
    r.add_argument("--obo")
    r.set_defaults(func=lambda a: _ontology().cmd_set(a))
    r = ds.add_parser("show", help="recorded terms and which labels still lack one")
    _add_run(r)
    r.set_defaults(func=lambda a: _ontology().cmd_show(a))

    # ---------------- export / clean ----------------
    q = sub.add_parser("export", help="write annotations out")
    _add_run(q)
    q.add_argument("--csv", help="per-cell table")
    q.add_argument("--xlsx", help="workbook, one sheet per depth")
    q.add_argument("--md", help="report with reasoning transcripts")
    q.set_defaults(func=lambda a: _export().cmd_export(a))

    q = sub.add_parser("doctor", help="check that the engine and its dependencies work")
    q.set_defaults(func=lambda a: _doctor(a))

    q = sub.add_parser("clean", help="report disk use and reclaim it (reports unless told what to remove)")
    _add_run(q)
    q.add_argument("--images", action="store_true", help="delete rendered maps (regenerable)")
    q.add_argument("--cache", action="store_true", help="delete this dataset's shared cache")
    q.add_argument("--run-dir", action="store_true", help="delete the whole run, annotations included")
    q.add_argument("--force", action="store_true", help="allow deleting a run whose results were never exported")
    q.set_defaults(func=lambda a: _clean().cmd_clean(a))

    q = sub.add_parser("usage", help="what the annotation cost in tokens (read back from the "
                                     "harness transcripts)")
    _add_run(q)
    q.add_argument("--json", action="store_true", help="machine-readable counts")
    q.add_argument("--price", action="append", metavar="MODEL=IN,OUT",
                   help="per-model price in dollars per million tokens, repeatable, e.g. "
                        "--price claude-opus-5=5,25 --price claude-fable-5=10,50 . A run can "
                        "span models whose prices differ 2x, so there is no single blended "
                        "rate; none is assumed because prices change")
    q.set_defaults(func=lambda a: _usage().cmd_usage(a))

    q = sub.add_parser("cache", help="the dataset-scoped cache shared by all runs")
    ds = q.add_subparsers(dest="cmd", required=True)

    r = ds.add_parser("ls", help="list cached datasets and their size")
    r.add_argument("--cache-dir")
    r.set_defaults(func=lambda a: _setup().cmd_cache_ls(a))

    r = ds.add_parser("path", help="print this run's dataset cache directory")
    _add_run(r)
    r.set_defaults(func=lambda a: _setup().cmd_cache_path(a))

    r = ds.add_parser("columns", help="build the gene-column copy that makes gene queries fast")
    _add_run(r)
    r.set_defaults(func=lambda a: _setup().cmd_cache_columns(a))

    r = ds.add_parser("rm", help="delete cached datasets")
    r.add_argument("--run", help="remove the cache for this run's dataset")
    r.add_argument("--stale", action="store_true", help="remove entries whose source file changed")
    r.add_argument("--all", action="store_true", help="remove everything")
    r.add_argument("--cache-dir")
    r.set_defaults(func=lambda a: _setup().cmd_cache_rm(a))

    return p


# Lazy module loads: `lambdakit --help` and `clusters` should not pay for scanpy.
def _setup():
    from sclambda.cli.commands import setup
    return setup


def _de():
    from sclambda.cli.commands import de
    return de


def _dossier():
    from sclambda.cli.commands import dossier
    return dossier


def _split():
    from sclambda.cli.commands import split
    return split


def _spatial():
    from sclambda.cli.commands import spatial
    return spatial


def _clean():
    from sclambda.cli.commands import clean
    return clean


def _status():
    from sclambda.cli.commands import status
    return status


def _ontology():
    from sclambda.cli.commands import ontology
    return ontology


def _export():
    from sclambda.cli.commands import export
    return export


def _usage():
    from sclambda.cli.commands import usage
    return usage


def _plugin_path(args) -> int:
    from sclambda.plugin import plugin_directory
    print(plugin_directory())
    return 0


def _doctor(args) -> int:
    """Report whether everything the skills need is actually available."""
    import importlib.util

    required = ["numpy", "pandas", "scipy", "anndata", "scanpy", "statsmodels",
                "sklearn", "h5py", "matplotlib"]
    optional = {"pyarrow": "fast overlay storage (falls back to gzipped CSV)",
                "openpyxl": "Excel export (--xlsx)",
                "colorcet": "nicer niche colours"}

    print(f"# lambdakit {__version__}\n")
    print(f"- python: `{sys.executable}`")
    print(f"- engine: `{__file__}`\n")

    missing = []
    print("| dependency | status |")
    print("|---|---|")
    for mod in required:
        ok = importlib.util.find_spec(mod) is not None
        if not ok:
            missing.append(mod)
        print(f"| {mod} | {'ok' if ok else '**MISSING**'} |")
    for mod, why in optional.items():
        ok = importlib.util.find_spec(mod) is not None
        print(f"| {mod} _(optional: {why})_ | {'ok' if ok else 'not installed'} |")

    if missing:
        print(f"\n**{len(missing)} required package(s) missing.** Install with:")
        print(f"\n    {sys.executable} -m pip install {' '.join(m if m != 'sklearn' else 'scikit-learn' for m in missing)}\n")
        return 1

    from sclambda.core import cache
    print(f"\n- dataset cache root: `{cache.resolve_root()}`")
    print("\nEverything the annotation skills need is available.")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )
    try:
        return args.func(args) or 0
    except SystemExit:
        raise
    except ModuleNotFoundError as exc:
        # The scientific stack is the one thing this cannot bundle. Say exactly
        # what to run rather than surfacing a bare import error.
        print(f"error: missing dependency '{exc.name}'.", file=sys.stderr)
        print("Install the engine's requirements with:", file=sys.stderr)
        print(f"    {sys.executable} -m pip install 'numpy pandas scipy anndata scanpy "
              f"statsmodels scikit-learn h5py matplotlib pyarrow'", file=sys.stderr)
        print("or `pip install -e <plugin directory>` to get them all at once.", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        if args.verbose:
            raise
        print(f"error: {exc}", file=sys.stderr)
        print("(re-run with -v for the traceback)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
