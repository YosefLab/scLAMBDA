"""Bounded fresh-interpreter CPU workers with content-keyed stage checkpoints."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import numpy as np
from sclambda.core.cache import resolve_root

logger = logging.getLogger(__name__)
_slots = threading.BoundedSemaphore(max(1, int(os.getenv('SCLAMBDA_CLUSTER_PROCESSES', '2'))))


@contextmanager
def _job_lock(job):
    # Fail closed: concurrent writers must never share checkpoint temporary files.
    with (job/'.worker.lock').open('a') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _identity(embeddings, params):
    h = hashlib.sha256()
    h.update(str((embeddings.shape, embeddings.dtype.str)).encode())
    h.update(memoryview(embeddings).cast('B'))
    versions = {}
    for name in ['numpy', 'scipy', 'scanpy', 'igraph', 'scikit-learn', 'pynndescent', 'hnswlib']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    # Implementation changes invalidate checkpoints as well as data/settings.
    for name in ['subcluster.py', 'clustering_worker.py', 'clustering_process.py']:
        h.update(Path(__file__).with_name(name).read_bytes())
    settings = dict(params=params, versions=versions,
                    threads=os.getenv('SCLAMBDA_CLUSTER_THREADS', '1'),
                    disable_hnsw=os.getenv('SCLAMBDA_DISABLE_HNSWLIB', ''),
                    silhouette_cache=os.getenv('SCLAMBDA_SILHOUETTE_CACHE_BYTES', ''))
    h.update(json.dumps(settings, sort_keys=True).encode())
    return h.hexdigest(), settings


def _atomic_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, default=str))
    tmp.replace(path)


def _stop(proc, dump=False):
    if proc.poll() is not None:
        return
    if dump:
        proc.send_signal(signal.SIGUSR1)
        time.sleep(.2)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _cpu_seconds(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')
    except (OSError, ValueError):
        return None


def _launch(job, timeout):
    env = os.environ.copy()
    threads = env.get('SCLAMBDA_CLUSTER_THREADS', '1')
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        env[key] = threads
    env.pop('SCLAMBDA_DIAGNOSTICS_DIR', None)
    env.setdefault('NUMBA_CACHE_DIR', str(job.parent / 'numba'))
    env.setdefault('MPLCONFIGDIR', str(job.parent / 'matplotlib'))
    # Preserve source-checkout imports without depending on caller cwd.
    env['PYTHONPATH'] = os.pathsep.join(str(p) for p in sys.path if p)
    env['PYTHONUNBUFFERED'] = '1'
    with (job/'worker.log').open('ab', buffering=0) as log:
        proc = subprocess.Popen([sys.executable, '-m', 'sclambda.analysis.clustering_worker', str(job)],
                                env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        _atomic_json(job/'launch.json', dict(pid=proc.pid, started=time.time()))
        logger.info('Clustering worker started pid=%d job=%s', proc.pid, job)
        started = last_change = time.monotonic()
        last_sequence = None
        last_report = 0
        stage = 'importing worker'
        try:
            while proc.poll() is None:
                now = time.monotonic()
                try:
                    status = json.loads((job/'status.json').read_text())
                    if status.get('pid') == proc.pid and status['sequence'] != last_sequence:
                        last_sequence = status['sequence']
                        last_change = now
                        stage = status['stage']
                        logger.info('Clustering checkpoint pid=%d stage=%s event=%s', proc.pid, stage, status['event'])
                except (OSError, ValueError, KeyError):
                    pass
                if now-last_report >= 30:
                    report = dict(pid=proc.pid, stage=stage, elapsed=now-started,
                                  stage_seconds=now-last_change, cpu_seconds=_cpu_seconds(proc.pid), time=time.time())
                    _atomic_json(job/'monitor.json', report)
                    logger.info('Clustering monitor %s', report)
                    last_report = now
                if now-last_change > timeout:
                    _stop(proc, dump=True)
                    raise TimeoutError(f'Clustering stage {stage!r} exceeded {timeout}s; diagnostics: {job}')
                time.sleep(.5)
            if proc.returncode:
                raise RuntimeError(f'Clustering worker exited {proc.returncode}; checkpoints and log: {job}')
        finally:
            _stop(proc)


def cluster_in_process(embeddings, **params):
    """Only memberships/resolution cross back; workers never receive an accessor."""
    embeddings = np.ascontiguousarray(embeddings)
    key, identity = _identity(embeddings, params)
    root = Path(os.environ.get('SCLAMBDA_CLUSTER_CACHE', resolve_root()/'clustering-v1'))
    root.mkdir(parents=True, exist_ok=True)
    job = root/key
    job.mkdir(exist_ok=True)
    with _slots, _job_lock(job):
        if not (job/'result.json').exists():
            _atomic_json(job/'request.json', dict(params=params, identity=identity))
            if not (job/'embeddings.npy').exists():
                with (job/'embeddings.tmp').open('wb') as f:
                    np.save(f, embeddings, allow_pickle=False)
                (job/'embeddings.tmp').replace(job/'embeddings.npy')
            for attempt in range(2):
                try:
                    _launch(job, float(os.getenv('SCLAMBDA_CLUSTER_STAGE_TIMEOUT', '1800')))
                    break
                except (RuntimeError, TimeoutError):
                    if attempt:
                        raise
                    logger.warning('Retrying isolated clustering from stage checkpoints: %s', job, exc_info=True)
        meta = json.loads((job/'result.json').read_text())
        labels = np.load(job/'labels.npy', allow_pickle=False)
        if labels.shape != (len(embeddings),) or not np.issubdtype(labels.dtype, np.integer):
            raise RuntimeError(f'Invalid clustering output: {job}')
        return labels, float(meta['resolution'])
