"""Isolate caches and require explicit opt-in for paid/network integration tests."""
import pytest


def pytest_addoption(parser):
    parser.addoption("--run-integration", action="store_true", default=False)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(reason="pass --run-integration to allow live provider calls")
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolate_dataset_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("LAMBDA_CACHE_DIR", str(tmp_path / "matrix-cache"))
