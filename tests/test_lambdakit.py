"""End-to-end tests for the lambdakit CLI.

These exercise the whole surface a skill actually drives: init a dataset, measure
genes, run the belief contract, probe for substructure, render an image, export.
No API key and no network — which is the point of the conversion.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scipy import sparse


CACHE_ROOT = None  # set by the `dataset` fixture; keeps tests off the real cache


def run(*args, cwd=None):
    """Invoke the CLI the way a skill does, and fail loudly on a non-zero exit."""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    if CACHE_ROOT:
        env["LAMBDA_CACHE_DIR"] = str(CACHE_ROOT)
    proc = subprocess.run(
        [sys.executable, "-m", "sclambda.cli", *args],
        capture_output=True, text=True, cwd=cwd, env=env,
    )
    assert proc.returncode == 0, f"`lambdakit {' '.join(args)}` failed:\n{proc.stderr}"
    return proc.stdout


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """Three well-separated clusters with recognisable marker blocks."""
    global CACHE_ROOT
    d = tmp_path_factory.mktemp("data")
    CACHE_ROOT = d / "cache"
    rng = np.random.default_rng(0)
    n, g = 600, 200
    genes = [f"GENE{i}" for i in range(g)]
    for i, sym in enumerate(["CD3E", "CD8A", "CD8B", "CD4", "MS4A1", "CD19", "LYZ", "CD14"]):
        genes[i] = sym

    X = rng.poisson(1.0, (n, g)).astype(np.float32)
    labels = np.array(["a"] * 200 + ["b"] * 200 + ["c"] * 200)
    X[:200, 0:4] += 25      # T
    X[200:400, 4:6] += 25   # B
    X[400:, 6:8] += 25      # myeloid

    obs = pd.DataFrame(
        {"cluster": labels,
         "cell_type": np.where(labels == "a", "T", np.where(labels == "b", "B", "Mono"))},
        index=[f"c{i}" for i in range(n)],
    )
    adata = ad.AnnData(X=sparse.csr_matrix(X), obs=obs,
                       var=pd.DataFrame({"feature_name": genes}, index=genes))
    emb = rng.normal(size=(n, 10)).astype(np.float32)
    emb[:200] += 5
    emb[200:400] -= 5
    adata.obsm["X_pca"] = emb
    coords = rng.normal(size=(n, 2)).astype(np.float32) * 10
    coords[:200, 0] += 60
    coords[200:400, 1] += 60
    adata.obsm["X_spatial"] = coords

    path = d / "toy.h5ad"
    adata.write_h5ad(path)
    return d, path


@pytest.fixture(scope="module")
def run_dir(dataset):
    d, path = dataset
    out = run("init", str(path), "--run", str(d / "run"),
              "--organism", "human", "--tissue", "human blood",
              "--gene-col", "feature_name", "--embedding-key", "X_pca",
              "--obs-column", "cluster", "--depth", "2", cwd=d)
    assert "Run initialised" in out
    return d / "run"


# ---------------- init ----------------

def test_init_reports_dataset_health(dataset, run_dir):
    """init must surface the things that silently ruin a run."""
    out = run("clusters", "--run", str(run_dir))
    assert "3 total" in out
    for cid in ("0", "1", "2"):
        assert f"| {cid} |" in out


def test_cluster_names_are_encoded(run_dir):
    """Original names must never reach the reasoning steps."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["cluster_name_map"] == {"a": "0", "b": "1", "c": "2"}
    out = run("clusters", "--run", str(run_dir))
    for original in ("a", "b", "c"):
        assert f"| {original} |" not in out


def test_reference_stats_are_cached(run_dir):
    from sclambda.storage.session import Session

    assert Session(run_dir).refstats_cached()
    assert (run_dir / "genes.txt").exists()   # the gene panel is run state, not cache


def test_matrix_cache_lives_with_the_dataset_not_the_run(run_dir):
    """Keyed on the file, so every command and every run shares one copy."""
    from sclambda.core.cache import entries

    rows = entries(CACHE_ROOT)
    assert len(rows) == 1, f"expected one cached dataset, got {rows}"
    assert rows[0]["has_matrix"], "no memory-mapped matrix cache was written"
    assert rows[0]["status"] == "current"
    # Nothing dataset-derived should be duplicated inside the run.
    assert not (run_dir / "tmp").exists()
    assert not (run_dir / "refstats.npz").exists()

    cached = list(Path(rows[0]["dir"]).glob("*.npy"))
    assert all(str(os.getpid()) not in f.name for f in cached), \
        "cache filename contains a pid; it would not be reused by the next command"


def test_second_run_reuses_the_cache(dataset, run_dir):
    """A new run over the same .h5ad must not rebuild anything."""
    d, path = dataset
    out = run("init", str(path), "--run", str(d / "run2"),
              "--organism", "human", "--tissue", "human blood",
              "--gene-col", "feature_name", "--obs-column", "cluster", cwd=d)
    assert "reused" in out, out
    assert "built" not in out
    from sclambda.core.cache import entries
    assert len(entries(CACHE_ROOT)) == 1, "a second run created a second cache"


def test_modified_source_is_not_reused(dataset, tmp_path):
    """A changed file must get a fresh cache, never stale arrays."""
    import shutil

    from sclambda.core.cache import entries

    d, path = dataset
    copy = tmp_path / "copy.h5ad"
    shutil.copy(path, copy)
    run("init", str(copy), "--run", str(tmp_path / "r1"), "--organism", "human",
        "--tissue", "t", "--gene-col", "feature_name", "--obs-column", "cluster")
    before = {str(r["dir"]) for r in entries(CACHE_ROOT)}

    adata = ad.read_h5ad(copy)
    adata.X = adata.X * 2
    adata.write_h5ad(copy)

    out = run("init", str(copy), "--run", str(tmp_path / "r2"), "--organism", "human",
              "--tissue", "t", "--gene-col", "feature_name", "--obs-column", "cluster")
    assert "built" in out, "a modified file reused its old cache"
    after = {str(r["dir"]) for r in entries(CACHE_ROOT)}
    assert after - before, "no new cache directory for the modified file"
    stale = [r for r in entries(CACHE_ROOT) if r["status"] != "current"]
    assert stale, "the superseded entry should be reported stale"


def test_dropout_rate_is_cached_for_ols(run_dir):
    """Without this, --de-method ols rescans the whole matrix every command."""
    from sclambda.storage.session import Session

    sess = Session(run_dir)
    with np.load(sess.refstats_path()) as z:
        assert "dropout" in z.files
        assert z["dropout"].shape == (600,)
    out = run("de", "rank", "--run", str(run_dir), "--cluster", "0", "--method", "ols")
    assert "CD3E" in out


# ---------------- differential expression ----------------

def test_rank_finds_the_planted_markers(run_dir):
    out = run("de", "rank", "--run", str(run_dir), "--cluster", "0")
    assert "CD3E" in out and "CD8A" in out
    assert "MS4A1" not in out.split("| # |")[-1].split("\n\n")[0]


def test_targeted_de_reports_measurement_metrics(run_dir):
    """The skill receives proportions and significance alongside LFC."""
    out = run("de", "genes", "--run", str(run_dir), "--cluster", "0",
              "--genes", "CD3E,MS4A1")
    assert "Proportion expressing" in out and "Significance" in out and "LFC" in out


def test_unmeasured_genes_are_reported_not_silently_dropped(run_dir):
    """A gene that was never measured must not read as absent."""
    out = run("de", "genes", "--run", str(run_dir), "--cluster", "0",
              "--genes", "CD3E,TOTALLY_FAKE_GENE")
    assert "TOTALLY_FAKE_GENE" in out
    assert "do not treat as absent" in out


def test_peer_comparison_orients_lfc(run_dir):
    out = run("de", "genes", "--run", str(run_dir), "--cluster", "0",
              "--genes", "CD3E,MS4A1", "--vs", "1")
    assert "higher in 0" in out


# ---------------- the belief contract ----------------

@pytest.fixture(scope="module")
def seeded(run_dir):
    catalog = [
        {"cell_type": "CD8 αβ T cell", "prior": "high", "lineage": "T cell",
         "markers": ["CD3E", "CD8A"]},
        {"cell_type": "Naive B cell", "prior": "high", "lineage": "B cell",
         "markers": ["MS4A1", "CD19"]},
    ]
    run("catalog", "set", "--run", str(run_dir), "--json", json.dumps(catalog))
    run("dossier", "init", "--run", str(run_dir), "--cluster", "0")
    return run_dir


def test_catalog_seeds_every_entry_as_a_candidate(seeded):
    out = run("dossier", "show", "--run", str(seeded), "--cluster", "0")
    assert "CD8 αβ T cell" in out and "Naive B cell" in out


def test_ruling_out_a_type_refutes_its_candidate(seeded):
    """ruled_out -> refuted sync: an excluded type cannot stay in the running."""
    run("dossier", "update", "--run", str(seeded), "--cluster", "0", "--json", json.dumps({
        "ruled_out": [{"claim": "Naive B cell", "reason": "MS4A1 absent"}],
        "candidates": [{"cell_type": "CD8 αβ T cell", "state": "supported",
                        "evidence_for": ["CD8A frac 1.00"]}],
        "best_cell_type": "CD8 αβ T cell",
    }))
    data = json.loads((seeded / "dossiers" / "0.json").read_text())
    states = {c["cell_type"]: c["state"] for c in data["candidates"]}
    assert states["Naive B cell"] == "refuted"
    assert states["CD8 αβ T cell"] == "supported"


def test_only_one_candidate_stays_supported(seeded):
    """Promoting a second candidate must demote the first."""
    run("dossier", "update", "--run", str(seeded), "--cluster", "0", "--json", json.dumps({
        "candidates": [{"cell_type": "CD4 T cell", "state": "supported",
                        "evidence_for": ["CD4 frac 0.9"]}],
        "best_cell_type": "CD4 T cell",
    }))
    data = json.loads((seeded / "dossiers" / "0.json").read_text())
    supported = [c["cell_type"] for c in data["candidates"] if c["state"] == "supported"]
    assert len(supported) == 1, f"expected exactly one leader, got {supported}"


def test_evidence_log_accumulates(seeded):
    run("dossier", "evidence", "--run", str(seeded), "--cluster", "0",
        "--source", "signature", "--fact", "CD8A expr 6.5 frac 1.00")
    data = json.loads((seeded / "dossiers" / "0.json").read_text())
    assert any("CD8A" in e["fact"] for e in data["evidence_log"])


def test_commit_writes_a_result(seeded):
    run("dossier", "axes", "--run", str(seeded), "--cluster", "0", "--json", json.dumps({
        "axes": {"lineage": {"resolved": True, "label": "T cell", "note": "CD3E measured", "evidence_status": "established", "defining_evidence": ["CD3E measured"], "specificity": "Fixture measured combination distinguishes the alternative"}}}))
    out = run("dossier", "commit", "--run", str(seeded), "--cluster", "0")
    assert "Committed cluster 0" in out
    assert (seeded / "results" / "0.json").exists()


# ---------------- the split gate ----------------

def test_homogeneous_cluster_is_declared_a_leaf_without_reasoning(seeded):
    """The probe must short-circuit, so no model call is spent on a leaf."""
    out = run("substructure", "--run", str(seeded), "--cluster", "0")
    assert "leaf" in out.lower()
    assert "do not subcluster" in out.lower()


def test_subcluster_refuses_without_the_flag(seeded):
    out = run("subcluster", "--run", str(seeded), "--cluster", "0")
    assert "not flagged" in out


# ---------------- spatial ----------------

@pytest.fixture(scope="module")
def spatial_run(dataset):
    d, path = dataset
    run("init", str(path), "--run", str(d / "sp"),
        "--organism", "human", "--tissue", "lymph node section",
        "--gene-col", "feature_name", "--obs-column", "cluster",
        "--cell-type-col", "cell_type", "--spatial-key", "X_spatial", cwd=d)
    return d / "sp"


def test_composition_reports_fractions_and_morans(spatial_run):
    out = run("spatial", "composition", "--run", str(spatial_run), "--niche", "0")
    assert "Moran's Z" in out
    assert (spatial_run / "composition" / "0.json").exists()


def test_spatial_de_leads_with_the_region_contrast(spatial_run):
    """Composition-defined identity must not be buried under per-cell-type state."""
    run("spatial", "composition", "--run", str(spatial_run), "--niche", "0")
    out = run("spatial", "de", "--run", str(spatial_run), "--niche", "0")
    assert "all cell types pooled" in out
    assert out.index("all cell types pooled") < len(out) // 2


def test_image_is_written_and_readable(spatial_run):
    out = run("spatial", "image", "--run", str(spatial_run), "--niche", "0", "--number")
    png = spatial_run / "images" / "0_whole.png"
    assert png.exists() and png.stat().st_size > 1000
    assert "red" in out


def test_overview_image_has_no_target(spatial_run):
    run("spatial", "image", "--run", str(spatial_run), "--no-target")
    assert (spatial_run / "images" / "overview.png").exists()


def test_adjacency_is_reported(spatial_run):
    out = run("spatial", "adjacency", "--run", str(spatial_run))
    assert "borders" in out


# ---------------- export ----------------

def test_export_forward_fills_and_hides_nothing(seeded, tmp_path):
    csv = tmp_path / "out.csv"
    md = tmp_path / "out.md"
    out = run("export", "--run", str(seeded), "--csv", str(csv), "--md", str(md))
    assert "Exported" in out
    df = pd.read_csv(csv, index_col=0)
    assert "cell_type_L1" in df.columns
    assert df["cell_type_L1"].notna().sum() == 200  # only cluster 0 was committed
    # The original cluster names reappear only here, at export.
    assert "original_cluster" in df.columns


# ---------------- restart / robustness ----------------

def test_status_reports_progress_and_next_action(seeded):
    out = run("status", "--run", str(seeded))
    assert "Annotation progress" in out
    assert "Whole-run phases" in out
    assert "## Next" in out


def test_phase_ledger_survives_a_restart(seeded):
    """Phases with no per-cluster artefact must be recorded, or they repeat."""
    run("phase", "--run", str(seeded), "--name", "cross-cluster", "--note", "no changes")
    out = run("phase", "--run", str(seeded), "--list")
    assert "cross-cluster" in out and "done" in out
    # a fresh process (as after a restart) sees it
    assert "cross-cluster" in run("status", "--run", str(seeded))
    from sclambda.storage.session import Session
    assert Session(seeded).phase_state("cross-cluster") == "done"


def test_phase_records_the_positional_state(seeded):
    run("phase", "in-progress", "--run", str(seeded), "--name", "export")
    from sclambda.storage.session import Session
    assert Session(seeded).phase_state("export") == "in-progress"


def test_interrupted_clusters_are_surfaced(dataset, tmp_path):
    """A dossier with no result, untouched, is work to resume — not to redo."""
    import os
    import time

    d, path = dataset
    rd = tmp_path / "halted"
    run("init", str(path), "--run", str(rd), "--organism", "human", "--tissue", "t",
        "--gene-col", "feature_name", "--obs-column", "cluster")
    run("dossier", "init", "--run", str(rd), "--cluster", "1")

    old = time.time() - 3600
    os.utime(rd / "dossiers" / "1.json", (old, old))

    out = run("status", "--run", str(rd))
    assert "Interrupted" in out
    assert "continue the loop rather than starting it over" in out

    from sclambda.storage.session import Session
    assert Session(rd).stale_clusters(minutes=30) == ["1"]


def test_state_writes_leave_no_partial_files(seeded):
    """Every state write renames into place, so a kill cannot truncate JSON."""
    leftovers = [p for p in Path(seeded).rglob(".*tmp*") if p.is_file()]
    assert not leftovers, f"temp files left behind: {leftovers}"


def test_damaged_progress_ledger_does_not_block_restart(seeded):
    """A half-written ledger must degrade to 'nothing recorded', not crash."""
    from sclambda.storage.session import Session

    ledger = Path(seeded) / "progress.json"
    ledger.write_text('{"phases": {"export"')      # truncated mid-write
    assert Session(seeded).read_progress() == {"phases": {}}
    assert "## Next" in run("status", "--run", str(seeded))
    ledger.unlink()


# ---------------- cleanup ----------------

def test_clean_reports_without_deleting(seeded):
    """The default must never destroy anything."""
    out = run("clean", "--run", str(seeded))
    assert "Nothing deleted" in out
    assert (Path(seeded) / "dossiers").exists()


def test_clean_refuses_to_delete_unexported_annotations(dataset, tmp_path):
    """Annotations cannot be recomputed; losing them to a cleanup is the worst case."""
    import subprocess

    d, path = dataset
    rd = tmp_path / "unexported"
    run("init", str(path), "--run", str(rd), "--organism", "human", "--tissue", "t",
        "--gene-col", "feature_name", "--obs-column", "cluster")
    run("dossier", "init", "--run", str(rd), "--cluster", "0")
    run("dossier", "update", "--run", str(rd), "--cluster", "0", "--json", json.dumps(
        {"candidates": [{"cell_type": "T cell", "state": "supported"}], "best_cell_type": "T cell"}))
    run("dossier", "axes", "--run", str(rd), "--cluster", "0", "--json", json.dumps({
        "axes": {"lineage": {"resolved": True, "label": "T cell", "note": "CD3E measured", "evidence_status": "established", "defining_evidence": ["CD3E measured"], "specificity": "Fixture measured combination distinguishes the alternative"}}}))
    run("dossier", "commit", "--run", str(rd), "--cluster", "0")

    env = dict(os.environ, LAMBDA_CACHE_DIR=str(CACHE_ROOT))
    proc = subprocess.run([sys.executable, "-m", "sclambda.cli", "clean",
                           "--run", str(rd), "--run-dir"],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 2, "should refuse, not proceed"
    assert "Refusing to delete" in proc.stdout
    assert rd.exists(), "the run was deleted despite the refusal"

    # recording the export releases the guard
    run("phase", "--run", str(rd), "--name", "export")
    run("clean", "--run", str(rd), "--run-dir")
    assert not rd.exists()


def test_clean_images_leaves_annotations_intact(spatial_run):
    """Images are regenerable; removing them must not touch the run's work."""
    run("spatial", "image", "--run", str(spatial_run), "--niche", "0")
    assert (Path(spatial_run) / "images").exists()
    out = run("clean", "--run", str(spatial_run), "--images")
    assert "Removed: images" in out
    assert not (Path(spatial_run) / "images").exists()
    assert (Path(spatial_run) / "manifest.json").exists()
    assert (Path(spatial_run) / "composition").exists()


# ---------------------------------------------------------------- token usage


def _usage_module():
    from sclambda.cli.commands import usage
    return usage


def _msg(ts, out=0, inp=0, read=0, write_5m=0, write_1h=0):
    """One assistant record in the shape Claude Code writes to a transcript."""
    return json.dumps({
        "type": "assistant",
        "timestamp": ts,
        "message": {"model": "claude-opus-5", "usage": {
            "input_tokens": inp,
            "output_tokens": out,
            "cache_read_input_tokens": read,
            "cache_creation_input_tokens": write_5m + write_1h,
            "cache_creation": {
                "ephemeral_5m_input_tokens": write_5m,
                "ephemeral_1h_input_tokens": write_1h,
            },
        }},
    })


def test_cache_tiers_are_weighted_not_summed():
    """The three cached-token tiers are priced differently and must not be added.

    A million cache reads is a tenth of a million input tokens' worth; summing
    the raw counters would report it as ten times that.
    """
    usage = _usage_module()
    tally = {"input_tokens": 1_000, "cache_write_5m": 1_000,
             "cache_write_1h": 1_000, "cache_read": 1_000, "output_tokens": 9_999}
    # 1000*1.0 + 1000*1.25 + 1000*2.0 + 1000*0.1 — output excluded entirely.
    assert usage.input_equivalent(tally) == pytest.approx(4_350)


def test_cache_write_ttls_are_split():
    usage = _usage_module()
    rec = json.loads(_msg("2026-01-01T00:00:00Z", write_5m=10, write_1h=90))
    assert usage._split_cache_writes(rec["message"]["usage"]) == (10, 90)


def test_cache_writes_fall_back_to_the_cheaper_tier_when_undetailed():
    """Without a TTL breakdown, attribute writes to 5m — under-state, never inflate."""
    usage = _usage_module()
    assert usage._split_cache_writes({"cache_creation_input_tokens": 500}) == (500, 0)
    assert usage._split_cache_writes({}) == (0, 0)


def _fake_project(tmp_path, monkeypatch, cwd, session_id, rows, agents=()):
    """Build a Claude Code project directory the scanner can read."""
    usage = _usage_module()
    projects = tmp_path / "projects"
    encoded = usage.encode_cwd(Path(cwd))[0]
    proj = projects / encoded
    proj.mkdir(parents=True)
    (proj / f"{session_id}.jsonl").write_text("\n".join(rows) + "\n")
    if agents:
        sub = proj / session_id / "subagents"
        sub.mkdir(parents=True)
        for name, text in agents:
            (sub / f"agent-{name}.jsonl").write_text(text)
    monkeypatch.setattr(usage, "PROJECTS", projects)
    return proj


def test_usage_attributes_subagents_by_run_path(dataset, tmp_path, monkeypatch):
    """A subagent counts only when its transcript names this run's full path."""
    usage = _usage_module()
    from sclambda.storage.session import Session

    run_dir = tmp_path / "runs" / "alpha"
    _, h5ad = dataset
    run(*["init", str(h5ad), "--run", str(run_dir), "--obs-column", "cluster",
          "--embedding-key", "X_pca"])

    mine = _msg("2020-01-01T00:00:10Z", out=100) + "\n" + json.dumps(
        {"cmd": f"lambdakit de rank --run {run_dir} --cluster 0"})
    theirs = _msg("2020-01-01T00:00:10Z", out=100) + "\n" + json.dumps(
        {"cmd": "lambdakit de rank --run /somewhere/else --cluster 0"})
    _fake_project(tmp_path, monkeypatch, Path.cwd(), "sess1",
                  [_msg("2020-01-01T00:00:05Z", out=7)],
                  agents=[("aaa", mine), ("bbb", theirs)])

    sess = Session(run_dir)
    sess.set("created_at", "2020-01-01T00:00:00+00:00")
    data = usage.collect(sess)
    assert [a["id"] for a in data["agents"]] == ["aaa"]
    assert data["subagents"]["output_tokens"] == 100


def test_usage_ignores_a_concurrent_unrelated_session(dataset, tmp_path, monkeypatch):
    """Another session in the same project must not be billed to this run.

    Two terminals in one repo overlap in time, so the time window alone cannot
    separate them — ownership of a matching subagent is what decides.
    """
    usage = _usage_module()
    from sclambda.storage.session import Session

    run_dir = tmp_path / "runs" / "beta"
    _, h5ad = dataset
    run(*["init", str(h5ad), "--run", str(run_dir), "--obs-column", "cluster",
          "--embedding-key", "X_pca"])

    proj = _fake_project(
        tmp_path, monkeypatch, Path.cwd(), "sess1",
        [_msg("2020-01-01T00:00:05Z", out=7)],
        agents=[("aaa", json.dumps({"cmd": f"--run {run_dir} --cluster 0"}))],
    )
    # A second, busy session that never touched this run.
    (proj / "sess2.jsonl").write_text(_msg("2020-01-01T00:00:06Z", out=500_000) + "\n")

    sess = Session(run_dir)
    sess.set("created_at", "2020-01-01T00:00:00+00:00")
    data = usage.collect(sess)
    assert [s["id"] for s in data["sessions"]] == ["sess1"]
    assert data["session_whole"]["output_tokens"] == 7


def test_usage_does_not_match_on_a_generic_run_name(dataset, tmp_path, monkeypatch):
    """A bare directory name like `probe` collides with ordinary prose."""
    usage = _usage_module()
    from sclambda.storage.session import Session

    run_dir = tmp_path / "runs" / "probe"
    _, h5ad = dataset
    run(*["init", str(h5ad), "--run", str(run_dir), "--obs-column", "cluster",
          "--embedding-key", "X_pca"])

    _fake_project(tmp_path, monkeypatch, Path.cwd(), "sess1",
                  [_msg("2020-01-01T00:00:05Z", out=7)],
                  agents=[("aaa", "ran the substructure probe on cluster 3\n")])

    sess = Session(run_dir)
    sess.set("created_at", "2020-01-01T00:00:00+00:00")
    assert usage.collect(sess)["agents"] == []


def test_usage_is_unavailable_rather_than_raising(dataset, tmp_path, monkeypatch):
    """No transcripts must degrade to a message, never break a finished run."""
    usage = _usage_module()
    run_dir = tmp_path / "runs" / "gamma"
    _, h5ad = dataset
    run(*["init", str(h5ad), "--run", str(run_dir), "--obs-column", "cluster",
          "--embedding-key", "X_pca"])
    monkeypatch.setattr(usage, "PROJECTS", tmp_path / "nothing-here")
    out = run("usage", "--run", str(run_dir))
    assert "Unavailable" in out


def test_init_stamps_cwd_and_created_at(dataset, tmp_path):
    """Both are needed by `usage` and neither is recoverable afterwards."""
    run_dir = tmp_path / "runs" / "delta"
    _, h5ad = dataset
    run(*["init", str(h5ad), "--run", str(run_dir), "--obs-column", "cluster",
          "--embedding-key", "X_pca"])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["cwd"] and manifest["created_at"]


def test_streamed_fragments_of_one_response_are_counted_once(tmp_path, monkeypatch):
    """One API response is logged as several records, each copying the same usage.

    Summing records inflated a real run 2.30x and reported 1,638 API calls where
    665 were made. Deduplicate on the message id, not the record.
    """
    usage = _usage_module()
    t = tmp_path / "s.jsonl"
    one = json.dumps({
        "type": "assistant", "timestamp": "2020-01-01T00:00:05Z",
        "message": {"id": "msg_same", "usage": {
            "input_tokens": 7, "output_tokens": 11, "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 40,
            "cache_creation": {"ephemeral_5m_input_tokens": 40, "ephemeral_1h_input_tokens": 0}}},
    })
    other = one.replace("msg_same", "msg_two")
    t.write_text("\n".join([one, one, one, other]) + "\n")

    rows = list(usage._iter_usage(t))
    assert len(rows) == 2, "three fragments of one response must collapse to one"
    tally = usage._tally(rows)
    assert tally["messages"] == 2
    assert tally["output_tokens"] == 22
    assert tally["cache_read"] == 200


def test_records_without_an_id_are_not_merged(tmp_path):
    """An id-less record can't be deduplicated — keep it rather than collapse it."""
    usage = _usage_module()
    t = tmp_path / "s.jsonl"
    rec = json.dumps({"type": "assistant", "timestamp": "2020-01-01T00:00:05Z",
                      "message": {"usage": {"output_tokens": 5}}})
    t.write_text(rec + "\n" + rec + "\n")
    assert len(list(usage._iter_usage(t))) == 2


def test_usage_splits_by_model(tmp_path):
    """A run can span models whose prices differ 2x; one blended rate is wrong.

    Pricing a Fable-5-dominated run at Opus rates understated a real run by 1.92x.
    """
    usage = _usage_module()
    t = tmp_path / "s.jsonl"
    def rec(mid, model, out):
        return json.dumps({"type": "assistant", "timestamp": "2020-01-01T00:00:05Z",
                           "message": {"id": mid, "model": model,
                                       "usage": {"input_tokens": 10, "output_tokens": out,
                                                 "cache_read_input_tokens": 0,
                                                 "cache_creation_input_tokens": 0}}})
    t.write_text("\n".join([rec("a", "claude-fable-5", 100), rec("b", "claude-fable-5", 100),
                            rec("c", "claude-opus-5", 50)]) + "\n")
    tally = usage._tally(list(usage._iter_usage(t)))
    by = tally["by_model"]
    assert set(by) == {"claude-fable-5", "claude-opus-5"}
    assert by["claude-fable-5"]["messages"] == 2 and by["claude-fable-5"]["output_tokens"] == 200
    assert by["claude-opus-5"]["messages"] == 1 and by["claude-opus-5"]["output_tokens"] == 50


def test_price_flag_parses_and_rejects_junk():
    usage = _usage_module()
    from sclambda.storage.session import SessionError
    assert usage._parse_rates(["claude-opus-5=5,25"]) == {"claude-opus-5": (5.0, 25.0)}
    assert usage._parse_rates([]) == {}
    with pytest.raises(SessionError):
        usage._parse_rates(["claude-opus-5:5:25"])


# ---------------- resolved_as: a resolved rung names its winner ----------------

def test_resolved_as_promotes_the_refinement(dataset, tmp_path):
    """Without this, 'B cell' is emitted while the ladder says naive B cell."""
    d, path = dataset
    rd = tmp_path / "resolved"
    run("init", str(path), "--run", str(rd), "--organism", "human", "--tissue", "blood",
        "--gene-col", "feature_name", "--obs-column", "cluster")
    run("dossier", "init", "--run", str(rd), "--cluster", "1")
    run("dossier", "update", "--run", str(rd), "--cluster", "1", "--json", json.dumps({
        "candidates": [{"cell_type": "B cell", "state": "supported", "evidence_for": ["MS4A1 frac 0.9"]},
                       {"cell_type": "Naive B cell", "state": "candidate"}],
        "best_cell_type": "B cell"}))
    out = run("dossier", "axes", "--run", str(rd), "--cluster", "1", "--json", json.dumps({
        "axes": {"lineage": {"resolved": True, "label": "B cell", "note": "MS4A1 frac 0.9", "evidence_status": "established", "defining_evidence": ["MS4A1 frac 0.9"], "specificity": "Fixture measured combination distinguishes the alternative"},
                 "cell_type": {"resolved": True, "label": "Naive B cell", "note": "IGHD present, CD27 low", "evidence_status": "established", "defining_evidence": ["IGHD present, CD27 low"], "specificity": "Fixture measured combination distinguishes the alternative"}},
        "resolved_as": "Naive B cell"}))
    assert "promoted" in out
    data = json.loads((rd / "dossiers" / "1.json").read_text())
    states = {c["cell_type"]: c["state"] for c in data["candidates"]}
    assert states["Naive B cell"] == "supported" and states["B cell"] == "candidate"
    out = run("dossier", "commit", "--run", str(rd), "--cluster", "1")
    assert "Naive B cell" in out


def test_resolved_as_noop_when_already_leader(dataset, tmp_path):
    d, path = dataset
    rd = tmp_path / "resolved_noop"
    run("init", str(path), "--run", str(rd), "--organism", "human", "--tissue", "blood",
        "--gene-col", "feature_name", "--obs-column", "cluster")
    run("dossier", "init", "--run", str(rd), "--cluster", "0")
    run("dossier", "update", "--run", str(rd), "--cluster", "0", "--json", json.dumps({
        "candidates": [{"cell_type": "T cell", "state": "supported"}], "best_cell_type": "T cell"}))
    out = run("dossier", "axes", "--run", str(rd), "--cluster", "0",
              "--json", json.dumps({"resolved_as": "T cell"}))
    assert "already the leader" in out


def test_resolved_as_works_for_legacy_niches(spatial_run):
    run("spatial", "composition", "--run", str(spatial_run), "--niche", "1")
    run("dossier", "init", "--run", str(spatial_run), "--cluster", "1")
    # An archived dossier without the compact-decision mode still supports axes.
    saved = Path(spatial_run) / "dossiers" / "1.json"
    legacy = json.loads(saved.read_text())
    legacy.pop("decision_mode", None)
    saved.write_text(json.dumps(legacy))
    run("dossier", "update", "--run", str(spatial_run), "--cluster", "1", "--json", json.dumps({
        "candidates": [{"compartment": "Cerebral cortex", "state": "supported",
                        "defining_cell_types": ["B"], "evidence_for": ["neural markers"]},
                       {"compartment": "Cerebral cortex, layer 4", "state": "candidate",
                        "defining_cell_types": ["B"]}],
        "best_compartment": "Cerebral cortex"}))
    out = run("dossier", "axes", "--run", str(spatial_run), "--cluster", "1", "--json", json.dumps({
        "axes": {"position": {"resolved": True, "note": "band inside the cortical ribbon"}},
        "resolved_as": "Cerebral cortex, layer 4"}))
    assert "promoted" in out
    data = json.loads((Path(spatial_run) / "dossiers" / "1.json").read_text())
    states = {c["compartment"]: c["state"] for c in data["candidates"]}
    assert states["Cerebral cortex, layer 4"] == "supported"
    assert states["Cerebral cortex"] != "supported"


def test_resolved_as_refuses_unknown_and_unevidenced(dataset, tmp_path):
    """The critic may not invent labels, and may not promote without a resolved note."""
    d, path = dataset
    rd = tmp_path / "resolved_refuse"
    run("init", str(path), "--run", str(rd), "--organism", "human", "--tissue", "blood",
        "--gene-col", "feature_name", "--obs-column", "cluster")
    run("dossier", "init", "--run", str(rd), "--cluster", "1")
    run("dossier", "update", "--run", str(rd), "--cluster", "1", "--json", json.dumps({
        "candidates": [{"cell_type": "B cell", "state": "supported", "evidence_for": ["MS4A1"]},
                       {"cell_type": "Naive B cell", "state": "candidate"}],
        "best_cell_type": "B cell"}))
    out = run("dossier", "axes", "--run", str(rd), "--cluster", "1", "--json", json.dumps({
        "axes": {"cell_type": {"resolved": True, "note": "x"}}, "resolved_as": "Classical monocyte"}))
    assert "refused" in out and "not a candidate" in out
    out = run("dossier", "axes", "--run", str(rd), "--cluster", "1", "--json", json.dumps({
        "resolved_as": "Naive B cell"}))
    assert "refused" in out and "no axis was marked resolved" in out
    data = json.loads((rd / "dossiers" / "1.json").read_text())
    states = {c["cell_type"]: c["state"] for c in data["candidates"]}
    assert states["B cell"] == "supported", "a refused promotion must leave the leader untouched"
    assert "Classical monocyte" not in states
