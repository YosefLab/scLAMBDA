import importlib
import numpy as np
import pandas as pd
import anndata as ad


def test_cached_sweep_matches_uncached_scores_and_resolution(monkeypatch):
    m=importlib.import_module('sclambda.analysis.subcluster')
    rng=np.random.default_rng(9)
    x=rng.normal(size=(120,8)).astype(np.float32)
    def leiden(a,res):
        a.obs['_sweep']=pd.Categorical(np.arange(len(a))%int(round(res*10)))
    kw=dict(embeddings=x,resolutions=[.2,.3,.4,.5],subsample_n=100,leiden_fn=leiden)
    monkeypatch.setenv('SCLAMBDA_SILHOUETTE_CACHE_BYTES','0')
    old=m._resolution_sweep(ad.AnnData(x),**kw)
    monkeypatch.setenv('SCLAMBDA_SILHOUETTE_CACHE_BYTES','1048576')
    new=m._resolution_sweep(ad.AnnData(x),**kw)
    pd.testing.assert_frame_equal(old,new,rtol=1e-6,atol=1e-7)
    assert m._find_optimal_resolution(old)==m._find_optimal_resolution(new)
