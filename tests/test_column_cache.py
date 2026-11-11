import h5py
import numpy as np
import pytest
import scipy.sparse as sp
from sclambda.core.matrix_reader import H5adMatrixReader


def write_sparse(path, x):
    with h5py.File(path,'w') as f:
        g=f.create_group('X');g.attrs['encoding-type']='csr_matrix';g.attrs['shape']=x.shape
        for k in ['data','indices','indptr']:g[k]=getattr(x,k)


@pytest.mark.parametrize('rows,cols', [([4,0,4,2],[3,1,3]),([], [1]),([0,1],[]),([-1,1],[0,2])])
def test_column_reads_match_csr_and_reopen(tmp_path,rows,cols):
    # Includes unsorted indices, duplicate entries, empty rows, and an all-zero column.
    x=sp.csr_matrix((np.array([2,3,5,7,11,13],dtype=np.float32),np.array([3,1,1,0,3,1]),np.array([0,3,3,4,4,6])),shape=(5,5))
    path=tmp_path/'x.h5ad';write_sparse(path,x)
    r=H5adMatrixReader(str(path),cache_dir=str(tmp_path/'cache'))
    expected=x[np.array(rows,dtype=int)][:,np.array(cols,dtype=int)].toarray()
    assert r.prepare_column_cache()
    np.testing.assert_array_equal(r.fetch_rows_cols_dense(rows,cols),expected)
    reopened=H5adMatrixReader(str(path),cache_dir=str(tmp_path/'cache'));reopened.prepare_column_cache()
    np.testing.assert_array_equal(reopened.fetch_rows_cols_dense(rows,cols),expected)


def test_dense_unchanged(tmp_path):
    path=tmp_path/'x.h5ad'
    with h5py.File(path,'w') as f:f['X']=np.arange(12).reshape(4,3)
    r=H5adMatrixReader(str(path),cache_dir=str(tmp_path/'cache'))
    assert not r.prepare_column_cache()
    np.testing.assert_array_equal(r.fetch_rows_cols_dense([2,0],[1]),[[7],[1]])


def test_source_change_does_not_reuse_column_cache(tmp_path):
    path=tmp_path/'x.h5ad';write_sparse(path,sp.eye(3,format='csr'))
    r=H5adMatrixReader(str(path),cache_dir=str(tmp_path/'cache'));r.prepare_column_cache();r.close()
    write_sparse(path,sp.eye(4,format='csr')*7)
    r=H5adMatrixReader(str(path),cache_dir=str(tmp_path/'cache'));r.prepare_column_cache()
    np.testing.assert_array_equal(r.fetch_rows_cols_dense([0],[0]),[[7]])


def test_small_target_keeps_row_path_large_reference_uses_columns(tmp_path,monkeypatch):
    x=sp.eye(1000,format='csr',dtype=np.float32)
    path=tmp_path/'x.h5ad';write_sparse(path,x)
    r=H5adMatrixReader(str(path),cache_dir=str(tmp_path/'cache'));r.prepare_column_cache()
    original=r.fetch_rows_sparse;calls=[]
    def tracked(rows):
        calls.append(len(rows));return original(rows)
    monkeypatch.setattr(r,'fetch_rows_sparse',tracked)
    np.testing.assert_array_equal(r.fetch_rows_cols_dense([1],[1]),[[1]])
    assert calls==[1]
    r.fetch_rows_cols_dense(np.arange(1000),[1])
    assert calls==[1]


def test_attach_uses_an_existing_cache_and_never_builds(tmp_path):
    x = sp.eye(50, format='csr', dtype=np.float32)
    path = tmp_path / 'x.h5ad'
    write_sparse(path, x)
    r = H5adMatrixReader(str(path), cache_dir=str(tmp_path / 'cache'))
    assert not r.attach_column_cache()
    assert not list((tmp_path / 'cache').rglob('columns-v1'))
    assert r.prepare_column_cache()
    reopened = H5adMatrixReader(str(path), cache_dir=str(tmp_path / 'cache'))
    assert reopened.attach_column_cache()
    np.testing.assert_array_equal(reopened.fetch_rows_cols_dense(list(range(50)), [3]), x[:, [3]].toarray())


def test_accessor_prepares_once_and_a_new_accessor_attaches(tmp_path):
    import anndata as ad
    import pandas as pd
    from sclambda.core.accessor import Accessor
    x = sp.random(300, 20, density=0.2, format='csr', dtype=np.float32, random_state=0)
    path = tmp_path / 'x.h5ad'
    ad.AnnData(X=x, var=pd.DataFrame(index=[f'G{i}' for i in range(20)])).write_h5ad(path)
    first = Accessor(path=str(path), cache_dir=str(tmp_path / 'cache'), gene_filter=None)
    assert not first.attach_column_cache()
    assert first.prepare_column_cache()
    rows = np.arange(300)
    expected, _ = first.get_expression_for_genes(rows, ['G3', 'G7'])
    first._reader.close()
    second = Accessor(path=str(path), cache_dir=str(tmp_path / 'cache'), gene_filter=None)
    assert second.attach_column_cache()
    got, _ = second.get_expression_for_genes(rows, ['G3', 'G7'])
    np.testing.assert_array_equal(got, expected)
