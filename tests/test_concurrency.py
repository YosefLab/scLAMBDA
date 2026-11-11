"""Regression tests for worker shutdown and subcluster lock scope."""
import importlib
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

from sclambda.core.task_runner import TaskRunner


def test_shutdown_allows_worker_to_finish_locked_update():
    # A subprocess bounds failures: the old implementation deadlocks at exit.
    source = Path(importlib.import_module('sclambda.core.task_runner').__file__)
    code = '''
import faulthandler, runpy, threading
TaskRunner = runpy.run_path(SOURCE)['TaskRunner']
runner = TaskRunner(max_workers=1)
ready = threading.Event()
joining = threading.Event()
def worker():
    ready.set()
    assert joining.wait(3)
    return runner.with_lock(lambda: 42)
future = runner.submit(worker)
assert ready.wait(3)
original = runner._executor.shutdown
def shutdown(*args, **kwargs):
    joining.set()
    return original(*args, **kwargs)
runner._executor.shutdown = shutdown
faulthandler.dump_traceback_later(10, exit=True)
runner.shutdown()
faulthandler.cancel_dump_traceback_later()
assert future.result() == 42
try:
    runner.submit(lambda: None)
except RuntimeError:
    pass
else:
    raise AssertionError('Submission after shutdown accepted')
runner.shutdown()  # repeated shutdown is safe
'''.replace('SOURCE', repr(str(source)))
    # Cold scientific imports on network storage can exceed ten seconds.
    # The watchdog above separately bounds the actual shutdown operation.
    subprocess.run([sys.executable, '-c', code], check=True, timeout=90)


class FakeAccessor:
    _embedding_key = 'embedding'

    def __init__(self):
        self.members = np.arange(4)
        self.committed = False

    def get_cell_indices(self, cluster):
        return self.members

    def get_obsm(self, key, indices):
        return np.zeros((len(indices), 2))

    def set_cluster_labels(self, cluster, labels):
        self.committed = True
        return {'clusters': ['0_0', '0_1'], 'n_clusters': 2}


def test_subclustering_does_not_block_annotation_updates(monkeypatch):
    module = importlib.import_module('sclambda.analysis.subcluster')
    computing = threading.Event()
    updated = threading.Event()
    accessor = FakeAccessor()
    def compute(**kwargs):
        computing.set()
        assert updated.wait(3), 'Annotation update blocked behind computation'
        return np.array([0, 0, 1, 1]), 0.15
    monkeypatch.setattr(module, '_subcluster_leiden', compute)
    with TaskRunner(2) as runner:
        future = runner.submit(module.subcluster, accessor, '0', synchronize=runner.with_lock)
        assert computing.wait(3)
        update = runner.submit(runner.with_lock, updated.set)
        update.result(timeout=5)
        assert future.result(timeout=5).n_clusters == 2
    assert accessor.committed


def test_subclustering_rejects_stale_membership(monkeypatch):
    module = importlib.import_module('sclambda.analysis.subcluster')
    accessor = FakeAccessor()
    def compute(**kwargs):
        accessor.members = np.array([1, 2, 3, 4])
        return np.array([0, 0, 1, 1]), 0.15
    monkeypatch.setattr(module, '_subcluster_leiden', compute)
    with TaskRunner(1) as runner:
        with pytest.raises(RuntimeError, match='membership changed'):
            module.subcluster(accessor, '0', synchronize=runner.with_lock)
    assert not accessor.committed
