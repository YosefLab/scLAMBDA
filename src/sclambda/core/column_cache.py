"""Dataset-scoped CSC copy of a sparse matrix, for fast small gene-panel reads.

Reading a few genes for most cells from row-major (CSR) storage scans the whole
matrix. A column-major copy reads only those genes' columns. It is built once
per dataset and memory-mapped afterwards.
"""

import json
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import cache

NAME = "columns-v1"
log = logging.getLogger(__name__)


def _destination(reader) -> Path:
    root = cache.dataset_dir(reader.path, root=reader._cache_dir, x_key=reader.x_key)
    return root / NAME


def is_built(reader) -> bool:
    reader._ensure_open()
    return reader.is_sparse and (_destination(reader) / "complete.json").exists()


def build(reader) -> bool:
    """Write the column copy if it is missing. Returns False for dense data."""
    reader._ensure_open()
    if not reader.is_sparse:
        return False
    destination = _destination(reader)
    root = destination.parent
    with cache.build_lock(root, NAME):
        if (destination / "complete.json").exists():
            return True
        if destination.exists():
            raise RuntimeError(f"Incomplete column cache at {destination}; refusing to reuse")
        tmp = Path(tempfile.mkdtemp(prefix=".columns-", dir=root))
        try:
            arrays = [getattr(reader, "_mmap_" + k) for k in ("data", "indices", "indptr")]
            if arrays[0] is None:
                arrays = [np.asarray(getattr(reader, "_" + k)) for k in ("data", "indices", "indptr")]
            # Same float32 representation as fetch_rows_sparse, before summing duplicates.
            arrays[0] = np.asarray(arrays[0], dtype=np.float32)
            matrix = sp.csr_matrix(tuple(arrays), shape=(reader.n_obs, reader.n_vars), copy=False)
            log.info("Building gene-column cache: %d nonzeros", matrix.nnz)
            columns = matrix.tocsc()
            for name in ("data", "indices", "indptr"):
                np.save(tmp / f"{name}.npy", getattr(columns, name), allow_pickle=False)
            (tmp / "complete.json").write_text(json.dumps(
                {"shape": list(columns.shape), "nnz": columns.nnz, "version": 1}))
            tmp.rename(destination)
            log.info("Gene-column cache complete: %s", destination)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
    return True


def attach(reader) -> bool:
    """Memory-map an existing column copy onto the reader; never builds one."""
    if reader._column_matrix is not None:
        return True
    if not is_built(reader):
        return False
    destination = _destination(reader)
    meta = json.loads((destination / "complete.json").read_text())
    if meta["shape"] != [reader.n_obs, reader.n_vars] or meta["version"] != 1:
        raise ValueError("Column cache shape/version mismatch")
    arrays = [np.load(destination / f"{k}.npy", mmap_mode="r") for k in ("data", "indices", "indptr")]
    if (len(arrays[0]) != meta["nnz"] or len(arrays[1]) != meta["nnz"]
            or len(arrays[2]) != reader.n_vars + 1 or arrays[2][-1] != meta["nnz"]):
        raise ValueError("Column cache arrays are incomplete")
    # The triple-array constructor scans int64 indices to consider downcasting,
    # which on a large atlas reads the whole index file before any query.
    columns = sp.csc_matrix(tuple(meta["shape"]), dtype=arrays[0].dtype)
    columns.data, columns.indices, columns.indptr = arrays
    reader._column_matrix = columns
    return True


def prepare(reader) -> bool:
    """Build if needed, then attach. Call before starting worker threads."""
    return build(reader) and attach(reader)
