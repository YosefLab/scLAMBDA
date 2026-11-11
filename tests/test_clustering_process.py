"""Real worker isolation, determinism, crash recovery and bounded termination."""
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from sclambda.analysis.clustering_process import cluster_in_process, _launch
from sclambda.analysis.subcluster import _run_leiden_local
from sclambda.core.native_threads import prepare_native_threads


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv('SCLAMBDA_CLUSTER_CACHE', str(tmp_path))
    monkeypatch.setenv('SCLAMBDA_CLUSTER_THREADS', '1')
    monkeypatch.setenv('NUMBA_CACHE_DIR', '/tmp/sclambda-process-test-numba')
    monkeypatch.setenv('MPLCONFIGDIR', '/tmp/sclambda-process-test-mpl')
    return tmp_path


def data(n=500):
    x = np.random.default_rng(42).normal(size=(n, 8)).astype('float32')
    x[:n//2] += 3
    return x


def test_concurrent_workers_match_serial_and_cache(cache):
    prepare_native_threads()
    x = data()
    settings = [dict(random_state=s) for s in [42, 43]]
    expected = [_run_leiden_local(x, **k) for k in settings]
    with ThreadPoolExecutor(2) as pool:
        jobs = [pool.submit(cluster_in_process, x, **k) for k in settings]
        actual = [j.result(timeout=120) for j in jobs]
    for (a, ar), (b, br) in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
        assert ar == br
    launches = list(cache.glob('*/launch.json'))
    assert len(launches) == 2
    pids = {json.loads(p.read_text())['pid'] for p in launches}
    assert len(pids) == 2 and os.getpid() not in pids
    before = {p: p.stat().st_mtime_ns for p in launches}
    np.testing.assert_array_equal(cluster_in_process(x, **settings[0])[0], actual[0][0])
    assert before == {p: p.stat().st_mtime_ns for p in launches}
    # Simulate loss of the overall sweep/result after individual resolutions
    # were committed. Resume must reconstruct the same selection from them.
    job = next(p.parent for p in launches
               if json.loads((p.parent/'request.json').read_text())['params']['random_state'] == 42)
    (job/'result.json').unlink()
    (job/'_resolution_sweep.pickle').unlink()
    resumed = cluster_in_process(x, **settings[0])
    np.testing.assert_array_equal(resumed[0], expected[0][0])
    assert resumed[1] == expected[0][1]
    events = [json.loads(l) for l in (job/'events.jsonl').read_text().splitlines()]
    assert any(e['stage'].startswith('leiden-_sweep-') and e['event'] == 'restored' for e in events)


def test_killed_worker_restores_neighbors(cache):
    x = data(6000)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(cluster_in_process, x, force_resolution=.2)
        deadline = time.monotonic()+120
        killed = None
        while time.monotonic() < deadline and not future.done():
            for graph in cache.glob('*/neighbors.h5ad'):
                pid = json.loads((graph.parent/'launch.json').read_text())['pid']
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    continue
                killed = graph.parent
                break
            if killed:
                break
            time.sleep(.01)
        assert killed is not None, 'Did not interrupt a live worker after graph checkpoint'
        labels, resolution = future.result(timeout=120)
    events = [json.loads(l) for l in (killed/'events.jsonl').read_text().splitlines()]
    assert any(e['stage'] == 'neighbors' and e['event'] == 'restored' for e in events)
    baseline, br = _run_leiden_local(x, force_resolution=.2)
    np.testing.assert_array_equal(labels, baseline)
    assert resolution == br


def test_stage_timeout_reaps_worker(cache, monkeypatch):
    module = importlib.import_module('sclambda.analysis.clustering_process')
    real_popen = subprocess.Popen
    processes = []
    def launch(args, **kwargs):
        proc = real_popen([sys.executable, '-c',
            'import faulthandler,signal,time; faulthandler.register(signal.SIGUSR1); time.sleep(60)'], **kwargs)
        processes.append(proc)
        return proc
    monkeypatch.setattr(module.subprocess, 'Popen', launch)
    with pytest.raises(TimeoutError, match='exceeded'):
        _launch(cache, .3)
    assert processes[0].poll() is not None
    assert (cache/'worker.log').stat().st_size > 0


def test_missing_hnsw_uses_exact_fallback(monkeypatch):
    module = importlib.import_module('sclambda.analysis.subcluster')
    monkeypatch.setitem(sys.modules, 'hnswlib', None)
    monkeypatch.delenv('SCLAMBDA_DISABLE_HNSWLIB', raising=False)
    labels = module._knn_propagate(np.array([[0., 0.], [10., 10.]]),
        np.array([[9., 9.]]), np.array(['a', 'b']), 900000)
    assert labels.tolist() == ['b']


def test_pynndescent_accepts_worker_embeddings(cache):
    # Above Scanpy's small-data exact-neighbor cutoff, below the 300k
    # subsampling cutoff: PyNNDescent sees the worker mmap directly.
    x = data(10000)
    labels, resolution = cluster_in_process(x, force_resolution=.2)
    assert labels.shape == (len(x),)
    assert len(set(labels)) >= 2
    job = next(cache.glob('*/embeddings.npy'))
    np.testing.assert_array_equal(np.load(job), x)
