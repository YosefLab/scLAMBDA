from __future__ import annotations
from typing import Iterable, Optional
import os
import atexit
import threading

import numpy as np
import h5py
import scipy.sparse as sp


class H5adMatrixReader:
    """
    Lazy X reader for .h5ad files.
    - Dense X: direct HDF5 slicing
    - CSR X: batch slicing via (data, indices, indptr)

    Opens file lazily on first read.

    OPTIMIZED VERSION - Key improvements:
    1. Single contiguous HDF5 read instead of chunked reads
    2. Vectorized numpy operations instead of Python loops
    3. Smarter density-based read strategy selection
    """

    def __init__(self, path: str, x_key: str = "X", use_mmap_csr: bool = True,
                 cache_dir: str | None = None):
        self.path = path
        self.x_key = x_key
        # Where the memory-mapped CSR copy lives. Dataset-scoped, so every run
        # over this file shares one copy — see sclambda.core.cache.
        self._cache_dir = cache_dir
        self._f: Optional[h5py.File] = None
        self._storage: Optional[str] = None
        self._on_disk_format: Optional[str] = None  # "csr", "csc", or None (dense)
        self._X: h5py.Dataset | h5py.Group | None = None
        self._data: h5py.Dataset | np.ndarray | None = None
        self._indices: h5py.Dataset | np.ndarray | None = None
        self._indptr: h5py.Dataset | np.ndarray | None = None
        self.n_obs: int = -1
        self.n_vars: int = -1
        self._use_mmap_csr = use_mmap_csr
        self._open_lock = threading.RLock()

        # Memory-mapped CSR arrays (if enabled)
        self._mmap_data: Optional[np.memmap] = None
        self._mmap_indices: Optional[np.memmap] = None
        self._mmap_indptr: Optional[np.memmap] = None
        self._mmap_temp_files: list[str] = []
        self._mmap_initialized = False
        self._column_matrix = None

        # Probe structure without keeping file open
        with h5py.File(self.path, "r", locking=False) as f:
            X = f[self.x_key]
            if isinstance(X, h5py.Dataset):
                self._storage = "dense"
                self._on_disk_format = None
                self.n_obs, self.n_vars = X.shape
            else:
                self.n_obs, self.n_vars = X.attrs["shape"]
                # Detect CSR vs CSC format
                encoding = X.attrs.get("encoding-type", "")
                if "csc" in encoding:
                    self._on_disk_format = "csc"
                elif "csr" in encoding:
                    self._on_disk_format = "csr"
                else:
                    # Fallback: infer from indptr length
                    indptr_len = X["indptr"].shape[0]
                    if indptr_len == self.n_vars + 1:
                        self._on_disk_format = "csc"
                    else:
                        self._on_disk_format = "csr"
                # After potential conversion, we always work in CSR
                self._storage = "csr"

        # Only release the maps on exit — the files themselves are a cross-process
        # cache now (see _init_mmap_csr) and must outlive this process. Call
        # ``purge_mmap()`` explicitly to delete them.
        atexit.register(self._close_mmap)

    @property
    def is_sparse(self) -> bool:
        return self._storage == "csr"

    def _ensure_open(self):
        """Open file if not already open (thread-safe)."""
        if self._f is not None:
            return

        with self._open_lock:
            if self._f is not None:
                return

            file_handle = h5py.File(self.path, "r", swmr=True, libver="latest", locking=False)
            x_obj = file_handle[self.x_key]

            # Re-check storage type when opening (may differ from initial probe)
            if isinstance(x_obj, h5py.Dataset):
                self._X = x_obj
                self._storage = "dense"
                self._data = None
                self._indices = None
                self._indptr = None
                # Set _f LAST — it is the lock-free "ready" flag (see line 83)
                self._f = file_handle
                return

            try:
                data = x_obj["data"]
                indices = x_obj["indices"]
                indptr = x_obj["indptr"]
            except KeyError as e:
                file_handle.close()
                raise ValueError(
                    f"Expected sparse matrix structure (CSR/CSC) but missing key: {e}. "
                    f"File may be corrupted or in an unexpected format."
                ) from e

            self._X = x_obj
            self._storage = "csr"

            if self._on_disk_format == "csc":
                # Convert CSC to CSR so all downstream code works with row-based access
                csc = sp.csc_matrix(
                    (np.asarray(data[:]), np.asarray(indices[:]), np.asarray(indptr[:])),
                    shape=(self.n_obs, self.n_vars),
                )
                csr = csc.tocsr()
                self._data = csr.data
                self._indices = csr.indices
                self._indptr = csr.indptr
                # Already in-memory numpy arrays — skip mmap initialization
                self._mmap_initialized = True
            else:
                self._data = data
                self._indices = indices
                self._indptr = indptr

            # Initialize memory-mapped arrays if enabled
            if self._use_mmap_csr and not self._mmap_initialized:
                self._init_mmap_csr()

            # Set _f LAST — it serves as the "ready" flag for the
            # lock-free fast path in line 83.  If set earlier, another
            # thread may see _f != None and return before _indptr etc.
            # are populated (race condition on CSC→CSR datasets).
            self._f = file_handle

    def _init_mmap_csr(self):
        """Create temporary memory-mapped CSR arrays for faster access."""
        if self._storage != "csr" or self._mmap_initialized:
            return

        try:
            # Ensure file is open
            if self._f is None:
                self._ensure_open()

            # The cache is keyed on the dataset's identity (not the pid, not the
            # run), so every lambdakit command and every run over this file
            # shares one copy. Each command is a separate short-lived process;
            # anything narrower would rebuild the matrix on every call.
            from sclambda.core import cache as _cache

            cache_dir = _cache.dataset_dir(self.path, root=self._cache_dir, x_key=self.x_key)

            data_path    = cache_dir / "csr_data.npy"
            indices_path = cache_dir / "csr_indices.npy"
            indptr_path  = cache_dir / "csr_indptr.npy"

            self._mmap_temp_files = [str(data_path), str(indices_path), str(indptr_path)]

            if not (data_path.exists() and indices_path.exists() and indptr_path.exists()):
                # Cold cache. Hold the build lock so parallel workers don't all
                # pay this, and re-check inside it — another process may have
                # finished while we waited.
                with _cache.build_lock(cache_dir, "csr"):
                    if not (data_path.exists() and indices_path.exists() and indptr_path.exists()):
                        self._build_csr_cache(data_path, indices_path, indptr_path, _cache)

            # Create memory-mapped arrays
            self._mmap_data    = np.load(data_path,    mmap_mode="r")
            self._mmap_indices = np.load(indices_path, mmap_mode="r")
            self._mmap_indptr  = np.load(indptr_path,  mmap_mode="r")

            self._mmap_initialized = True

        except Exception:
            # If memory mapping fails, fall back to HDF5 access
            self._close_mmap()

    def _build_csr_cache(self, data_path, indices_path, indptr_path, _cache):
        """Materialise the CSR triple to disk, downcast, written atomically.

        The rename-into-place is what makes a half-finished build harmless: a
        concurrent reader either sees no file or a complete one, never a
        truncated array.
        """
        data_array    = np.asarray(self._data[:])
        indices_array = np.asarray(self._indices[:])
        indptr_array  = np.asarray(self._indptr[:])

        # Choose optimal dtype for data array
        if np.issubdtype(data_array.dtype, np.floating):
            data_array = data_array.astype(np.float32)
        else:
            # Count data — pick smallest uint type that fits the max value
            max_val = int(data_array.max()) if data_array.size > 0 else 0
            if max_val <= np.iinfo(np.uint8).max:
                data_array = data_array.astype(np.uint8)
            elif max_val <= np.iinfo(np.uint16).max:
                data_array = data_array.astype(np.uint16)
            elif max_val <= np.iinfo(np.uint32).max:
                data_array = data_array.astype(np.uint32)
            else:
                data_array = data_array.astype(np.uint64)

        _cache.atomic_save_npy(data_path,    data_array)
        _cache.atomic_save_npy(indices_path, indices_array)
        _cache.atomic_save_npy(indptr_path,  indptr_array)

    def _close_mmap(self):
        """Release the memory maps but leave the cache files on disk."""
        self._mmap_data = None
        self._mmap_indices = None
        self._mmap_indptr = None
        self._mmap_initialized = False
        self._column_matrix = None

    def purge_mmap(self):
        """Delete the on-disk CSR cache (explicit; `lambdakit cache rm`)."""
        self._cleanup_mmap()

    def _cleanup_mmap(self):
        """Clean up temporary memory-mapped files."""
        # Close memory maps
        self._mmap_data = None
        self._mmap_indices = None
        self._mmap_indptr = None

        # Remove temporary files
        for temp_file in self._mmap_temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                pass  # Ignore errors during cleanup

        self._mmap_temp_files = []
        self._mmap_initialized = False
        self._column_matrix = None

    def fetch_rows_sparse(self, rows: Iterable[int]) -> sp.csr_matrix | None:
        """
        Fetch rows as a sparse CSR matrix.

        OPTIMIZED: Uses single contiguous HDF5 read + vectorized scatter.

        Returns None if underlying storage is dense.

        Args:
            rows: Cell indices to fetch

        Returns:
            CSR matrix of shape (len(rows), n_vars) or None if dense storage
        """
        self._ensure_open()

        if self._storage == "dense" or self._indptr is None:
            return None

        rows = np.asarray(list(rows), dtype=np.int64)
        n_rows = len(rows)

        if n_rows == 0:
            return sp.csr_matrix((0, self.n_vars))

        # === OPTIMIZATION 1: Single contiguous indptr read ===
        min_row, max_row = int(rows.min()), int(rows.max())

        # Use memory-mapped array if available, otherwise HDF5
        if self._mmap_indptr is not None:
            indptr_slice = np.asarray(self._mmap_indptr[min_row : max_row + 2], dtype=np.int64)
        else:
            # Read indptr as numpy array immediately (avoid repeated HDF5 access)
            indptr_slice = np.asarray(self._indptr[min_row : max_row + 2], dtype=np.int64)

        # Map to requested rows
        row_offsets = rows - min_row
        starts = indptr_slice[row_offsets]
        ends = indptr_slice[row_offsets + 1]

        # Build output indptr
        lengths = ends - starts
        new_indptr = np.zeros(n_rows + 1, dtype=np.int64)
        np.cumsum(lengths, out=new_indptr[1:])
        total_nnz = int(new_indptr[-1])

        if total_nnz == 0:
            return sp.csr_matrix((n_rows, self.n_vars))

        # === OPTIMIZATION 2: Single contiguous data/indices read ===
        data_min, data_max = int(starts.min()), int(ends.max())

        # Use memory-mapped arrays if available, otherwise HDF5
        if self._mmap_data is not None and self._mmap_indices is not None:
            # Memory-mapped access is faster for random access patterns
            data_chunk = np.asarray(self._mmap_data[data_min:data_max])
            indices_chunk = np.asarray(self._mmap_indices[data_min:data_max])
        else:
            # Read entire range at once - HDF5 is much faster with one big read
            data_chunk = np.asarray(self._data[data_min:data_max])
            indices_chunk = np.asarray(self._indices[data_min:data_max])

        # Adjust starts to be relative to chunk
        starts_rel = starts - data_min

        # === OPTIMIZATION 3: Vectorized scatter using np.repeat ===
        # Create output arrays — always float32 for scipy arithmetic compatibility
        # (mmap may store compact uint types for memory efficiency)
        new_data = np.empty(total_nnz, dtype=np.float32)
        new_indices = np.empty(total_nnz, dtype=indices_chunk.dtype)

        # For each row, we need to copy from [starts_rel[i]:ends_rel[i]]
        # to [new_indptr[i]:new_indptr[i+1]]

        # Build source indices for all elements at once
        # We need: for row i, source indices are range(starts_rel[i], ends_rel[i])

        # Create the source index array using vectorized operations
        # This is the key optimization - avoid Python loops entirely

        # Method: Use np.repeat to create row assignments, then add offsets
        row_ids = np.repeat(np.arange(n_rows), lengths)  # Which row each element belongs to

        # For each element, its position within its row
        within_row_pos = np.arange(total_nnz) - new_indptr[row_ids]

        # Source index in chunk = starts_rel[row] + position_within_row
        src_indices = starts_rel[row_ids] + within_row_pos

        # Vectorized copy
        new_data[:] = data_chunk[src_indices]
        new_indices[:] = indices_chunk[src_indices]

        return sp.csr_matrix(
            (new_data, new_indices, new_indptr),
            shape=(n_rows, self.n_vars),
        )

    def fetch_rows_dense(self, rows: Iterable[int], dtype=np.float32) -> np.ndarray:
        """
        Fetch rows as dense numpy array.

        Args:
            rows: Cell indices to fetch
            dtype: Output dtype

        Returns:
            Dense array of shape (len(rows), n_vars)
        """
        self._ensure_open()
        rows = np.asarray(list(rows), dtype=np.int64)

        if self._storage == "dense":
            uniq_sorted, inv = np.unique(rows, return_inverse=True)
            arr_uniq = self._X[uniq_sorted, :]
            return np.asarray(arr_uniq[inv], dtype=dtype)

        # CSR path - use sparse fetch then convert
        sparse_mat = self.fetch_rows_sparse(rows)
        return np.asarray(sparse_mat.toarray(), dtype=dtype)

    def prepare_column_cache(self):
        """Opt in to persistent column reads before launching worker threads."""
        from .column_cache import prepare
        return prepare(self)

    def attach_column_cache(self) -> bool:
        """Use an already-built column cache; never builds one."""
        from .column_cache import attach
        return attach(self)

    def fetch_rows_cols_dense(
        self,
        rows: Iterable[int],
        cols: Iterable[int],
        dtype=np.float32,
    ) -> np.ndarray:
        """
        Fetch specific rows and columns as dense numpy array.

        More efficient than fetching all columns when only a few genes are needed.

        Args:
            rows: Cell indices to fetch
            cols: Gene (column) indices to fetch
            dtype: Output dtype

        Returns:
            Dense array of shape (len(rows), len(cols))
        """
        self._ensure_open()
        rows = np.asarray(list(rows), dtype=np.int64)
        cols = np.asarray(list(cols), dtype=np.int64)

        if self._column_matrix is not None and (len(rows) == 0 or len(rows) >= self.n_obs / 100):
            # Identical values/order; avoid scanning every gene for large
            # reference populations when only a small panel was requested.
            selected = self._column_matrix[:, cols].tocsr()
            return np.asarray(selected[rows].toarray(), dtype=dtype)

        if self._storage == "dense":
            uniq_sorted, inv = np.unique(rows, return_inverse=True)
            arr_uniq = self._X[uniq_sorted, :][:, cols]
            return np.asarray(arr_uniq[inv], dtype=dtype)

        # CSR path – process rows in sorted chunks so that each call to
        # fetch_rows_sparse covers a small contiguous data range.
        # Without this, a large row span (e.g. all-other-cells reference)
        # forces fetch_rows_sparse to materialise the entire CSR
        # data/indices arrays (~18 GiB for large datasets).
        n_rows = len(rows)
        n_cols = len(cols)
        _CHUNK = 50_000

        if n_rows <= _CHUNK:
            sparse_mat = self.fetch_rows_sparse(rows)
            return np.asarray(sparse_mat[:, cols].toarray(), dtype=dtype)

        result = np.zeros((n_rows, n_cols), dtype=dtype)
        order = np.argsort(rows)
        sorted_rows = rows[order]

        for i in range(0, n_rows, _CHUNK):
            end = min(i + _CHUNK, n_rows)
            chunk_sparse = self.fetch_rows_sparse(sorted_rows[i:end])
            result[order[i:end]] = np.asarray(
                chunk_sparse[:, cols].toarray(), dtype=dtype
            )

        return result

    def compute_col_stats_chunked(
        self,
        chunk_size: int = 50000,
        normalize: bool = False,
        target_sum: float = 1e4,
        col_filter_mask: "np.ndarray | None" = None,
    ) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]":
        """
        Compute column (gene) mean and variance over all rows in chunks.

        For sparse data, avoids densifying by computing running statistics.

        Args:
            chunk_size: Number of rows per chunk
            normalize: Whether to normalize (total count -> target_sum -> log1p)
            target_sum: Target sum for normalization
            col_filter_mask: Optional boolean array of shape (n_vars,). When
                provided, also counts non-zero entries per row restricted to
                columns where the mask is True, and returns the result as the
                fourth element. The sparsity pattern used is pre-normalization
                (zeros stay zero after log1p so the result is identical).

        Returns:
            Tuple of (mean, variance, row_sums, nnz_filtered) arrays.
            nnz_filtered is None when col_filter_mask is not provided.
        """
        self._ensure_open()

        n_obs = self.n_obs
        n_vars = self.n_vars

        col_sum = np.zeros(n_vars, dtype=np.float32)
        col_sumsq = np.zeros(n_vars, dtype=np.float32)
        row_sums = np.zeros(n_obs, dtype=np.float32)
        nnz_filtered = np.zeros(n_obs, dtype=np.int32) if col_filter_mask is not None else None

        for start in range(0, n_obs, chunk_size):
            end = min(start + chunk_size, n_obs)

            if self._storage == "dense":
                X_chunk = np.asarray(self._X[start:end, :], dtype=np.float32)
                chunk_row_sums = X_chunk.sum(axis=1)
                row_sums[start:end] = chunk_row_sums

                if col_filter_mask is not None:
                    nnz_filtered[start:end] = np.sum(
                        (X_chunk[:, col_filter_mask]) != 0, axis=1
                    )

                if normalize:
                    scale = np.where(chunk_row_sums == 0, 1, chunk_row_sums)
                    X_chunk = X_chunk / scale[:, np.newaxis] * target_sum
                    X_chunk = np.log1p(X_chunk)

                col_sum += X_chunk.sum(axis=0)
                col_sumsq += (X_chunk**2).sum(axis=0)

            else:
                sparse_chunk = self.fetch_rows_sparse(np.arange(start, end)).tocsr()
                chunk_row_sums = np.asarray(sparse_chunk.sum(axis=1)).ravel()
                row_sums[start:end] = chunk_row_sums

                if col_filter_mask is not None and len(sparse_chunk.indices) > 0:
                    is_filtered = col_filter_mask[sparse_chunk.indices]
                    row_starts = sparse_chunk.indptr[:-1].astype(np.intp)
                    nnz_filtered[start:end] = np.add.reduceat(
                        is_filtered.astype(np.int32), row_starts
                    )

                if normalize:
                    scale = np.where(chunk_row_sums == 0, 1, chunk_row_sums)
                    sparse_chunk = sparse_chunk.multiply(1.0 / scale[:, np.newaxis] * target_sum)
                    sparse_chunk = sparse_chunk.tocsr()
                    sparse_chunk.data = np.log1p(sparse_chunk.data).astype(np.float32)

                col_sum += np.asarray(sparse_chunk.sum(axis=0)).ravel()

                sparse_sq = sparse_chunk.copy()
                sparse_sq.data = sparse_sq.data**2
                col_sumsq += np.asarray(sparse_sq.sum(axis=0)).ravel()

        mean = (col_sum / n_obs).astype(np.float32)
        var = ((col_sumsq / n_obs) - (mean.astype(np.float32) ** 2)).astype(np.float32)

        return mean, var, row_sums, nnz_filtered

    def close(self):
        """Close the file handle and release the memory maps.

        Releases but does **not** delete the on-disk CSR cache: it is shared
        across processes now, and each lambdakit command opens and closes the
        reader several times. Deleting here would rebuild the matrix on every
        call. Use ``purge_mmap()`` (or `lambdakit cache rm`) to remove it.
        """
        with self._open_lock:
            if self._f is not None:
                self._f.close()
                self._f = None
            self._X = None
            self._data = None
            self._indices = None
            self._indptr = None
            self._close_mmap()

    def __del__(self):
        self.close()
