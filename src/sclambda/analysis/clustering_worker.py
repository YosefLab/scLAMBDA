"""One CPU clustering job per interpreter; atomic checkpoints survive a crash."""
from __future__ import annotations
import faulthandler
import json
import os
from pathlib import Path
import pickle
import signal
import sys
import time


class Checkpoints:
    def __init__(self, root):
        self.root = Path(root)
        self.sequence = 0
        self.resolution = None

    def event(self, stage, event):
        self.sequence += 1
        value = dict(pid=os.getpid(), sequence=self.sequence, stage=stage, event=event, time=time.time())
        tmp = self.root/'status.tmp'
        tmp.write_text(json.dumps(value)); tmp.replace(self.root/'status.json')
        with (self.root/'events.jsonl').open('a') as f:
            f.write(json.dumps(value)+'\n')

    def cached(self, name, compute):
        path = self.root/(name+'.pickle')
        if path.exists():
            with path.open('rb') as f:
                value = pickle.load(f)
            self.event(name, 'restored')
            return value
        self.event(name, 'started')
        value = compute()
        tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        self.event(name, 'completed')
        return value

    def install(self, module):
        import anndata as ad
        import scanpy as sc
        original_neighbors = sc.pp.neighbors
        def neighbors(adata, **kwargs):
            path = self.root/'neighbors.h5ad'
            if path.exists():
                saved = ad.read_h5ad(path)
                adata.obsp = saved.obsp.copy(); adata.uns = saved.uns.copy()
                self.event('neighbors', 'restored')
                return
            self.event('neighbors', 'started')
            original_neighbors(adata, **kwargs)
            tmp = self.root/'neighbors.tmp.h5ad'
            snapshot = ad.AnnData(obs=adata.obs.copy(), obsp=adata.obsp.copy(), uns=adata.uns.copy())
            snapshot.write_h5ad(tmp); tmp.replace(path)
            self.event('neighbors', 'completed')
        sc.pp.neighbors = neighbors
        original_leiden = sc.tl.leiden
        def leiden(adata, **kwargs):
            self.resolution = kwargs.get('resolution', 1)
            key = kwargs.get('key_added', 'leiden')
            name = f'leiden-{key}-{self.resolution}'
            def compute():
                original_leiden(adata, **kwargs)
                return adata.obs[key].copy(), adata.uns[key].copy()
            series, meta = self.cached(name, compute)
            adata.obs[key] = series; adata.uns[key] = meta
        sc.tl.leiden = leiden
        for name in ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']:
            original = getattr(module, name)
            def metric(*args, _name=name, _original=original, **kwargs):
                return self.cached(f'{_name}-{self.resolution}', lambda: _original(*args, **kwargs))
            setattr(module, name, metric)
        for name in ['_resolution_sweep', '_knn_propagate']:
            original = getattr(module, name)
            def wrapped(*args, _name=name, _original=original, **kwargs):
                return self.cached(_name, lambda: _original(*args, **kwargs))
            setattr(module, name, wrapped)


def main():
    root = Path(sys.argv[1])
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    faulthandler.dump_traceback_later(600, repeat=True)
    checkpoints = Checkpoints(root)
    checkpoints.event('imports', 'started')
    import importlib
    import numpy as np
    from sclambda.core.native_threads import prepare_native_threads
    prepare_native_threads()
    module = importlib.import_module('sclambda.analysis.subcluster')
    checkpoints.install(module)
    params = json.loads((root/'request.json').read_text())['params']
    # PyNNDescent has writable-array Numba signatures. Copy-on-write
    # satisfies those without allowing workers to modify the saved input.
    x = np.load(root/'embeddings.npy', mmap_mode='c', allow_pickle=False)
    checkpoints.event('imports', 'completed')
    labels, resolution = module._run_leiden_local(x, **params)
    with (root/'labels.tmp').open('wb') as f:
        np.save(f, labels, allow_pickle=False)
    (root/'labels.tmp').replace(root/'labels.npy')
    tmp = root/'result.tmp'
    tmp.write_text(json.dumps(dict(resolution=resolution, cells=len(labels))))
    tmp.replace(root/'result.json')
    checkpoints.event('result', 'completed')
    faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
