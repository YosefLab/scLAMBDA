"""Cell Ontology parsing, matching, and the `lambdakit ontology` commands."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from sclambda.analysis.ontology import (
    OntologySearch, fuzzy_match, load_ontology, normalize, read_obo,
)

OBO = r'''format-version: 1.2
data-version: releases/2024-01-01

[Term]
id: CL:0000160
name: goblet cell
def: "A simple columnar epithelial cell that secretes mucin." [GOC:tfm]
synonym: "mucous cell" RELATED []
is_a: CL:0000066 ! epithelial cell

[Term]
id: CL:0009039
name: colon goblet cell
def: "A goblet cell located in the epithelium of the colon." []
is_a: CL:0000160 {is_inferred="true"} ! goblet cell

[Term]
id: CL:0000066
name: epithelial cell
def: "A cell that is usually found in a two-dimensional sheet with a \"free\" surface." []

[Term]
id: CL:1000347
name: colonocyte
synonym: "colonic epithelial cell" EXACT []
is_a: CL:0000066 ! epithelial cell

[Term]
id: CL:0000999
name: retired cell
is_obsolete: true

[Term]
id: UBERON:0001155
name: colon

[Typedef]
id: part_of
name: part of
'''


@pytest.fixture
def obo(tmp_path):
    path = tmp_path / "cl.obo"
    path.write_text(OBO)
    return path


def test_normalize():
    assert normalize("B-cell") == "b cell"
    assert normalize("\tNK cell\n") == "nk cell"


def test_fuzzy_match_ranks_exact_first():
    assert fuzzy_match("T-cell", ["T cell", "B cell", "monocyte"], top=3)[0][0] == "T cell"
    assert fuzzy_match("zzz", ["T cell"], top=3) == []


def test_read_obo_keeps_current_cell_terms_only(obo):
    terms = read_obo(obo)
    assert set(terms) == {"CL:0000160", "CL:0009039", "CL:0000066", "CL:1000347"}
    assert terms["CL:0000160"].synonyms == ["mucous cell"]
    assert terms["CL:0009039"].parents == ["CL:0000160"]
    assert '"free" surface' in terms["CL:0000066"].definition
    assert load_ontology(obo) == ["colon goblet cell", "colonocyte", "epithelial cell", "goblet cell"]


def test_search_matches_names_and_synonyms(obo):
    search = OntologySearch(read_obo(obo))
    hits = search.search("Goblet cell", top=3)
    assert hits[0]["id"] == "CL:0000160" and hits[0]["score"] == 1.0
    assert len({h["id"] for h in hits}) == len(hits)
    synonym = search.search("colonic epithelial cell", top=1)[0]
    assert synonym["id"] == "CL:1000347" and synonym["via"] == "synonym"


def lambdakit(*args, ok=True):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    proc = subprocess.run([sys.executable, "-m", "sclambda.cli", *args],
                          capture_output=True, text=True, env=env)
    assert (proc.returncode == 0) == ok, proc.stderr
    return proc.stdout + proc.stderr


@pytest.fixture
def run(tmp_path):
    """A run with two committed clusters, built without a dataset."""
    run = tmp_path / "run"
    (run / "results").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps(
        {"schema_version": 1, "organism": "human", "tissue": "colon", "h5ad": "x.h5ad"}))
    for cid, label in (("0", "Goblet cell"), ("1", "Colonocyte")):
        (run / "results" / f"{cid}.json").write_text(json.dumps({"cluster_id": cid, "label": label}))
    pd.DataFrame({"L1": ["0", "0", "1"]}, index=["a", "b", "c"]).to_csv(
        run / "overlay.csv.gz", compression="gzip")
    return run


def test_map_and_export(run, obo, tmp_path):
    out = lambdakit("ontology", "candidates", "--run", str(run), "--obo", str(obo))
    assert "Cluster 0: Goblet cell" in out and "CL:0000160" in out
    assert "Cluster 1: Colonocyte" in out

    lambdakit("ontology", "set", "--run", str(run), "--cluster", "0", "--term", "CL:0009039",
              "--note", "goblet cell in colon")
    lambdakit("ontology", "set", "--run", str(run), "--cluster", "1", "--term", "CL:1000347")
    assert "Every committed label" in lambdakit("ontology", "candidates", "--run", str(run))

    csv = tmp_path / "out.csv"
    lambdakit("export", "--run", str(run), "--csv", str(csv))
    df = pd.read_csv(csv, index_col=0)
    assert list(df["cell_ontology_id_L1"]) == ["CL:0009039", "CL:0009039", "CL:1000347"]
    assert df.loc["c", "cell_ontology_L1"] == "colonocyte"


def test_relabelled_cluster_makes_its_term_stale(run, obo, tmp_path):
    lambdakit("ontology", "set", "--run", str(run), "--cluster", "0", "--term", "CL:0000160",
              "--obo", str(obo))
    (run / "results" / "0.json").write_text(json.dumps({"cluster_id": "0", "label": "Enterocyte"}))
    assert "stale" in lambdakit("ontology", "show", "--run", str(run))
    out = lambdakit("export", "--run", str(run), "--csv", str(tmp_path / "out.csv"))
    assert "label changed after mapping: 0" in out
    assert "cell_ontology_id_L1" not in pd.read_csv(tmp_path / "out.csv").columns


def test_set_rejects_unknown_terms_and_bare_none(run, obo):
    assert "not a current term" in lambdakit("ontology", "set", "--run", str(run), "--cluster", "0",
                                             "--term", "CL:0000999", "--obo", str(obo), ok=False)
    assert "--none needs --note" in lambdakit("ontology", "set", "--run", str(run), "--cluster", "0",
                                              "--none", "--obo", str(obo), ok=False)
    lambdakit("ontology", "set", "--run", str(run), "--cluster", "0", "--none", "--note", "no fit")
    assert "no term" in lambdakit("ontology", "show", "--run", str(run))


def test_ontology_is_an_optional_phase(run):
    from sclambda.cli.commands.status import CELLTYPE_PHASES, OPTIONAL_PHASES
    assert "ontology" in dict(CELLTYPE_PHASES) and "ontology" in OPTIONAL_PHASES
