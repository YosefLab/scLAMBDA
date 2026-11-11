"""The primary-data restriction must cover every requested ontology term."""
from contextlib import nullcontext
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest


def test_all_requested_types_exclude_nonprimary_records(monkeypatch):
    requests = []
    marker = object()

    def get_anndata(**kwargs):
        requests.append(kwargs)
        return marker

    monkeypatch.setitem(sys.modules, "cellxgene_census", SimpleNamespace(
        open_soma=lambda **kwargs: nullcontext(marker), get_anndata=get_anndata))
    loader = runpy.run_path(str(Path(__file__).parents[1] / "examples/census/load_data.py"))["load_census_chunk"]
    assert loader(["CL:A", "CL:B"]) is marker
    query = requests[0]["obs_value_filter"]
    for term in ["CL:A", "CL:B", "CL:unselected"]:
        for primary in [False, True]:
            # Evaluate the generated Boolean query on the complete small truth table.
            actual = eval(query, {"__builtins__": {}}, {
                "cell_type_ontology_term_id": term, "is_primary_data": primary})
            assert actual == (primary and term in {"CL:A", "CL:B"})
    with pytest.raises(ValueError, match="ontology ID"):
        loader([])
    assert len(requests) == 1
