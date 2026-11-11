from __future__ import annotations
import re
import warnings
from pathlib import Path
from typing import Sequence, Callable

import threading

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

from .matrix_reader import H5adMatrixReader


DEFAULT_GENE_FILTER: Callable[[str], bool] = lambda gene: not gene.upper().startswith(
    ("RPL", "RPS", "MT-", "HSP", "ENSG")
)

# Regex pattern to match gene symbols (e.g., TP53, CD8A, HLA-DRB1)
# Matches: uppercase letter followed by 1-11 alphanumeric chars, optionally with hyphenated suffix
GENE_SYMBOL_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:[A-Z][A-Za-z0-9]{1,11}(?:-[A-Za-z0-9]{1,11})?)(?![A-Za-z0-9])")


class Accessor:
    """
    Read-only AnnData access + mutable overlay for clustering.
    """

    def __init__(
        self,
        path: str | Path,
        gene_filter: Callable[[str], bool] | None = DEFAULT_GENE_FILTER,
        gene_col: str | None = None,
        chunk_size: int = 50_000,
        target_sum: float = 1e4,
        embedding_key: str = "X_pca",
        use_mmap_csr: bool = True,
        cache_dir: str | None = None,
    ):
        """
        Initialize Accessor.

        Args:
            path: Path to h5ad file
            gene_filter: Function to filter genes (default filters RPL, RPS, MT, etc.)
            gene_col: Column in adata.var containing gene names. If None, uses var.index.
                      Should contain gene symbols (e.g., TP53, CD8A) for LLM compatibility.
            chunk_size: Chunk size for computing reference stats
            target_sum: Target sum for normalization
            use_mmap_csr: If True (default), create temporary memory-cell_indices CSR arrays
                         for faster access to sparse expression data
        """
        self._path = Path(path)
        self._reader = H5adMatrixReader(
            str(self._path), x_key="X", use_mmap_csr=use_mmap_csr, cache_dir=cache_dir,
        )
        self._gene_filter = gene_filter
        self._gene_col = gene_col
        self._chunk_size = chunk_size
        self._target_sum = target_sum
        self._embedding_key = embedding_key
        # Read obs and var now; obsm entries (embeddings, coordinates) can be
        # gigabytes, so each is read on first use. Same locking flag as the
        # matrix reader, which may hold the file open.
        with h5py.File(self._path, "r", locking=False) as f:
            self._obs = read_elem(f["obs"])
            self._var = read_elem(f["var"])
            self._obsm_names = list(f["obsm"].keys()) if "obsm" in f else []
        self._obsm: dict[str, np.ndarray] = {}
        self._obsm_lock = threading.Lock()

        # Strip Ensembl IDs from gene names: "TP53_ENSG00000141510" -> "TP53"
        _ens_suffix = re.compile(r'_ENS[A-Z]*\d+$')
        self._var.index = pd.Index([_ens_suffix.sub('', g) for g in self._var.index])
        self._var.index = self._var.index.where(~self._var.index.duplicated(), self._var.index + '_dup')
        if gene_col is not None and gene_col in self._var.columns:
            self._var[gene_col] = self._var[gene_col].map(lambda g: _ens_suffix.sub('', str(g)))

        # Validate gene_col if specified
        if gene_col is not None and gene_col not in self._var.columns:
            raise ValueError(
                f"gene_col '{gene_col}' not found in adata.var.columns. "
                f"Available columns: {list(self._var.columns)}"
            )

        # Overlay: grows as clustering proceeds
        self._overlay = pd.DataFrame(index=self._obs.index)

        # Cell Ontology terms: cluster_id -> Cell Ontology term
        self._ont_cell_types: dict[str, str] = {}

        # Hypotheses: cluster_id -> list of hypothesis dicts with reasoning
        self._hypotheses: dict[str, list[dict]] = {}

        # Reference stats: lazily computed or set externally
        self._ref_mean: np.ndarray | None = None
        self._ref_var: np.ndarray | None = None
        self._ref_n: int | None = None
        self._total_counts: np.ndarray | None = None  # Per-cell total counts (for normalization)

        # Gene name to index mapping (lazily built)
        self._gene_name_to_idx: dict[str, int] | None = None

        # Gene filter mask (lazily computed, cached for performance - 10-20x speedup)
        self._gene_filter_mask: np.ndarray | None = None

        # Per-cell dropout rate across filtered genes (lazily computed, cached)
        self._cell_dropout_rate: np.ndarray | None = None

        # Cluster stats cache: cluster_id -> {mean, var, n, mean_genes_detected}
        # Computed over gene_filter_indices. Cleared when overlay changes.
        self._cluster_stats_cache: dict[str, dict] = {}

        # Normalization: determined once on first data access
        self._needs_normalization: bool | None = None

        # Validate gene names look like gene symbols
        self._validate_gene_names()

    # ============ PROPERTIES ============

    @property
    def n_cells(self) -> int:
        return len(self._obs)

    @property
    def n_genes(self) -> int:
        return len(self._var)

    @property
    def obs(self) -> pd.DataFrame:
        return self._obs

    @property
    def var(self) -> pd.DataFrame:
        return self._var

    @property
    def gene_names(self) -> np.ndarray:
        """Get gene names from gene_col or var.index."""
        if self._gene_col is not None:
            return self._var[self._gene_col].values
        return self._var.index.values

    @property
    def gene_col(self) -> str | None:
        """Column used for gene names (None if using var.index)."""
        return self._gene_col

    @property
    def obsm_keys(self) -> list[str]:
        return list(self._obsm_names)

    @property
    def gene_filter(self) -> Callable[[str], bool] | None:
        return self._gene_filter

    @property
    def gene_filter_mask(self) -> np.ndarray:
        """
        Boolean mask for filtered genes (cached for performance).

        Computes mask once and caches it. This provides 10-20x speedup
        over calling gene_filter() in a loop for each gene.

        Returns:
            Boolean array of shape (n_genes,) where True means gene passes filter
        """
        if self._gene_filter_mask is None:
            if self._gene_filter is not None:
                # Compute once using vectorized operations where possible
                gene_names = self.gene_names

                # Try vectorized approach for common filter patterns
                if self._gene_filter == DEFAULT_GENE_FILTER:
                    # Vectorized version of default filter (10x faster)
                    # A numpy unicode array: pandas 3 keeps Index.astype(str) as
                    # its own string dtype, which np.char rejects.
                    upper_genes = np.char.upper(np.asarray(gene_names, dtype=str))
                    self._gene_filter_mask = ~(
                        np.char.startswith(upper_genes, 'RPL') |
                        np.char.startswith(upper_genes, 'RPS') |
                        np.char.startswith(upper_genes, 'MT-') |
                        np.char.startswith(upper_genes, 'HSP') |
                        np.char.startswith(upper_genes, 'ENSG')
                    )
                else:
                    # Fallback to loop for custom filters (still cached)
                    self._gene_filter_mask = np.array(
                        [self._gene_filter(g) for g in gene_names],
                        dtype=bool
                    )
            else:
                # No filter - all genes pass
                self._gene_filter_mask = np.ones(self.n_genes, dtype=bool)

        return self._gene_filter_mask

    @property
    def gene_filter_indices(self) -> np.ndarray:
        """
        Integer indices of genes that pass the filter (cached).

        Returns:
            Array of gene indices where filter is True
        """
        return np.where(self.gene_filter_mask)[0]

    @property
    def cell_dropout_rate(self) -> np.ndarray:
        """Per-cell dropout rate across filtered genes (fraction of zero entries).

        Computed lazily on first access and cached. Uses chunked iteration to
        avoid loading the full expression matrix into memory at once. For sparse
        data only the stored column indices are examined — no values are read.

        Returns:
            Float32 array of shape (n_cells,) where 1.0 = all zeros, 0.0 = fully expressed.
        """
        if self._cell_dropout_rate is None:
            self._cell_dropout_rate = self._compute_cell_dropout_rate()
        return self._cell_dropout_rate

    def _compute_cell_dropout_rate(self) -> np.ndarray:
        """Fallback: compute dropout rate via a dedicated pass.

        Normally dropout rate is computed as a free by-product of
        ``_compute_reference_stats`` (same chunk loop). This method is only
        called if that path hasn't run yet (e.g. external reference was set
        but total_counts was never accessed).
        """
        _, _, _, nnz_filtered = self._reader.compute_col_stats_chunked(
            chunk_size=self._chunk_size,
            normalize=False,  # sparsity pattern is normalization-invariant
            col_filter_mask=self.gene_filter_mask,
        )
        n_filtered = int(self.gene_filter_mask.sum())
        if n_filtered == 0 or nnz_filtered is None:
            return np.ones(self.n_cells, dtype=np.float32)
        return (1.0 - nnz_filtered / n_filtered).astype(np.float32)

    @property
    def overlay(self) -> pd.DataFrame:
        return self._overlay

    @property
    def needs_normalization(self) -> bool:
        if self._needs_normalization is None:
            self._check_normalization()
        return self._needs_normalization

    # ============ NORMALIZATION ============

    def _check_normalization(self, sample_size: int = 100):
        """
        Check if data needs normalization.

        Criteria for already normalized:
        - Non-integer values
        - After exp(), row sums ≈ 1e4 or 1e6
        """
        n_sample = min(sample_size, self.n_cells)
        sample_indices = np.random.choice(self.n_cells, n_sample, replace=False)
        X_sample = self._reader.fetch_rows_dense(sample_indices)

        # Check 1: Integer values → raw counts
        is_integer = np.allclose(X_sample, X_sample.astype(int))
        if is_integer:
            self._needs_normalization = True
        else:
            self._needs_normalization = False
        return

        # # Check 2: After exp(), row sums ≈ target?
        # row_sums = np.expm1(X_sample).sum(axis=1)
        # median_sum = np.median(row_sums)

        # is_normalized = np.isclose(median_sum, 1e4, rtol=0.5) or np.isclose(median_sum, 1e6, rtol=0.5)
        # self._needs_normalization = not is_normalized

    def _validate_gene_names(self, threshold: float = 0.5):
        """
        Check if gene names look like gene symbols (e.g., TP53, CD8A, HLA-DRB1).

        LLMs work best with standard gene symbols. If most gene names don't match
        the expected pattern (likely Ensembl IDs, RefSeq IDs, etc.), warn the user.

        Args:
            threshold: Warn if proportion of matching gene names is below this (default 0.5)
        """
        gene_names = self.gene_names

        # Count how many gene names match the gene symbol pattern
        n_matches = sum(1 for name in gene_names if GENE_SYMBOL_PATTERN.search(str(name)))
        proportion = n_matches / len(gene_names) if len(gene_names) > 0 else 0

        if proportion < threshold:
            # Identify what kind of IDs these might be
            sample_genes = [str(g) for g in gene_names[:5]]
            id_type = "unknown format"

            # Check for common ID patterns
            if any(str(g).startswith("ENSG") for g in gene_names[:100]):
                id_type = "Ensembl IDs (ENSG...)"
            elif any(str(g).startswith("ENSMUS") for g in gene_names[:100]):
                id_type = "Ensembl IDs (ENSMUS...)"
            elif any(str(g).startswith(("NM_", "NR_", "XM_", "XR_")) for g in gene_names[:100]):
                id_type = "RefSeq IDs"

            source = f"var['{self._gene_col}']" if self._gene_col else "var.index"
            warnings.warn(
                f"Only {proportion:.1%} of gene names in {source} appear to be gene symbols. "
                f"Detected: {id_type}. Examples: {sample_genes}. "
                f"LLMs work best with standard gene symbols (e.g., TP53, CD8A). "
                f"Consider using a different gene_col or converting IDs to symbols.",
                UserWarning,
            )

    def _normalize_chunk(self, X: np.ndarray) -> np.ndarray:
        """Normalize: total count → target_sum → log1p"""
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        X_norm = X / row_sums * self._target_sum
        return np.log1p(X_norm, dtype=np.float32)

    def _fetch_normalized(self, cell_indices: np.ndarray) -> np.ndarray:
        """Fetch expression, normalizing if needed.

        Uses sparse path when available for better performance.
        """
        if self._reader.is_sparse:
            # Use sparse fetch then normalize
            sparse_mat = self._reader.fetch_rows_sparse(cell_indices)

            if self.needs_normalization:
                # Compute row sums from sparse
                row_sums = np.asarray(sparse_mat.sum(axis=1)).ravel()
                row_sums = np.where(row_sums == 0, 1, row_sums)

                # Normalize: scale rows then log1p
                sparse_mat = sparse_mat.multiply(1.0 / row_sums[:, np.newaxis] * self._target_sum)
                sparse_mat = sparse_mat.tocsr()
                sparse_mat.data = np.log1p(sparse_mat.data).astype(np.float32)

            return np.asarray(sparse_mat.toarray(), dtype=np.float32)
        else:
            # Dense path
            X = self._reader.fetch_rows_dense(cell_indices, dtype=np.float32)
            if self.needs_normalization:
                X = self._normalize_chunk(X)
            return X

    # ============ REFERENCE STATS ============

    @property
    def ref_mean(self) -> np.ndarray:
        if self._ref_mean is None:
            self._compute_reference_stats()
        return self._ref_mean

    @property
    def ref_var(self) -> np.ndarray:
        if self._ref_var is None:
            self._compute_reference_stats()
        return self._ref_var

    @property
    def ref_n(self) -> int:
        if self._ref_n is None:
            self._compute_reference_stats()
        return self._ref_n

    @property
    def total_counts(self) -> np.ndarray:
        """Per-cell total counts (raw, before normalization)."""
        if self._total_counts is None:
            self._compute_reference_stats()
        return self._total_counts

    def set_reference_stats(self, mean: np.ndarray, var: np.ndarray, n: int, total_counts: np.ndarray | None = None,
                            dropout: np.ndarray | None = None):
        """Set reference stats externally"""
        if len(mean) != self.n_genes or len(var) != self.n_genes:
            raise ValueError(f"Stats length must match n_genes ({self.n_genes})")
        self._ref_mean = mean.astype(np.float32)
        self._ref_var = var.astype(np.float32)
        self._ref_n = n
        if total_counts is not None:
            self._total_counts = total_counts.astype(np.float32)
        if dropout is not None:
            if len(dropout) != self.n_cells:
                raise ValueError("Dropout length must match the dataset cell count")
            self._cell_dropout_rate = dropout.astype(np.float32)

    def _compute_reference_stats(self):
        """Compute mean, variance, and total counts over entire dataset.

        Uses the optimized chunked computation in the matrix reader.
        Mean/var are column aggregates (row order irrelevant).
        total_counts (row sums) comes back in mmap order — if rows are
        sorted, unsort back to original-index order so that
        total_counts[original_idx] works correctly.
        """
        mean, var, total_counts, nnz_filtered = self._reader.compute_col_stats_chunked(
            chunk_size=self._chunk_size,
            normalize=self.needs_normalization,
            target_sum=self._target_sum,
            col_filter_mask=self.gene_filter_mask,
        )

        # Only overwrite reference stats if not already set externally (e.g. via set_reference_stats).
        # Accessing total_counts triggers this method, but must not clobber an externally-loaded reference.
        if self._ref_mean is None:
            self._ref_mean = mean
            self._ref_var = var
            self._ref_n = self.n_cells
        self._total_counts = total_counts

        # Compute dropout rate from the same pass — no second full scan needed
        if nnz_filtered is not None:
            n_filtered = int(self.gene_filter_mask.sum())
            if n_filtered > 0:
                self._cell_dropout_rate = (1.0 - nnz_filtered / n_filtered).astype(np.float32)
            else:
                self._cell_dropout_rate = np.ones(self.n_cells, dtype=np.float32)

    # ============ CLUSTER INDEXING ============

    def _parse_cluster_depth(self, cluster_id: str) -> int:
        return len(cluster_id.split("_"))

    def _level_column(self, depth: int) -> str:
        return f"L{depth}"

    def get_cell_indices(self, cluster_id: str, depth: int | None = None) -> np.ndarray:
        """Get cell indices for a cluster ID (e.g., "1_3") or "" for all.

        Parameters
        ----------
        cluster_id : str
            Cluster identifier. Empty string returns all cells.
        depth : int, optional
            Explicit depth level to look up. When *None* (default), the depth
            is inferred from the cluster_id format (underscore-separated).
            Pass an explicit depth when cluster IDs may contain underscores
            that are NOT level separators (e.g. spatial region names).
        """
        if not cluster_id:
            return np.arange(self.n_cells)

        if depth is None:
            depth = self._parse_cluster_depth(cluster_id)
        col = self._level_column(depth)

        if col not in self._overlay.columns:
            raise KeyError(f"Level {depth} not computed. Available: {list(self._overlay.columns)}")

        mask = self._overlay[col] == cluster_id
        return np.where(mask)[0]

    def get_n_cells_in_cluster(self, cluster_id: str) -> int:
        return len(self.get_cell_indices(cluster_id))

    def get_all_clusters(self, depth: int) -> list[str]:
        """All cluster IDs at a given depth"""
        col = self._level_column(depth)
        if col not in self._overlay.columns:
            return []
        return sorted(self._overlay[col].dropna().unique().tolist())

    def get_children(self, cluster_id: str) -> list[str]:
        """Get child cluster IDs"""
        depth = self._parse_cluster_depth(cluster_id) if cluster_id else 0
        child_col = self._level_column(depth + 1)

        if child_col not in self._overlay.columns:
            return []

        if not cluster_id:
            return sorted(self._overlay[child_col].dropna().unique().tolist())

        mask = self._overlay[self._level_column(depth)] == cluster_id
        return sorted(self._overlay.loc[mask, child_col].dropna().unique().tolist())

    def get_parent_cluster_id(self, cluster_id: str) -> str:
        """Get parent cluster ID (e.g., '1_3' -> '1', '1' -> '')"""
        if not cluster_id:
            return ""
        parts = cluster_id.split("_")
        if len(parts) <= 1:
            return ""
        return "_".join(parts[:-1])

    # ============ GENE INDEXING ============

    def _build_gene_index(self):
        """Build gene name to index mapping."""
        if self._gene_name_to_idx is None:
            self._gene_name_to_idx = {str(name): idx for idx, name in enumerate(self.gene_names)}

    def get_gene_indices(self, gene_names: Sequence[str]) -> tuple[np.ndarray, list[str]]:
        """
        Get indices for gene names.

        Args:
            gene_names: List of gene names to look up

        Returns:
            Tuple of (indices array, list of found gene names)
            Genes not found are silently skipped.
        """
        self._build_gene_index()
        indices = []
        found_names = []
        for name in gene_names:
            idx = self._gene_name_to_idx.get(str(name))
            if idx is not None:
                indices.append(idx)
                found_names.append(str(name))
        return np.array(indices, dtype=np.int64), found_names

    # ============ TARGETED GENE EXPRESSION ============

    def prepare_column_cache(self) -> bool:
        """Build (once per dataset) and attach a gene-major copy of a sparse matrix.

        Gene-panel DE against most of the dataset then reads only those genes'
        columns instead of every row. Call before starting worker threads.
        Returns False for dense data, which does not need it.
        """
        return self._reader.prepare_column_cache()

    def attach_column_cache(self) -> bool:
        """Use an already-built gene-major copy; never builds one."""
        return self._reader.attach_column_cache()

    def get_expression_for_genes(
        self,
        cell_indices: np.ndarray,
        gene_names: Sequence[str],
    ) -> tuple[np.ndarray, list[str]]:
        """
        Fetch normalized expression for specific genes only.

        More efficient than fetching all genes when only a few are needed.
        Handles normalization using pre-computed total counts.

        Args:
            cell_indices: Cell indices to fetch
            gene_names: Gene names to fetch

        Returns:
            Tuple of (expression array [n_cells x n_genes], list of found gene names)
        """
        gene_indices, found_names = self.get_gene_indices(gene_names)

        if len(gene_indices) == 0:
            return np.array([]).reshape(len(cell_indices), 0), []

        # Fetch raw data for specific genes
        X_raw = self._reader.fetch_rows_cols_dense(cell_indices, gene_indices, dtype=np.float32)

        if self.needs_normalization:
            # Normalize using pre-computed total counts
            cell_totals = self.total_counts[cell_indices]
            cell_totals = np.where(cell_totals == 0, 1, cell_totals)

            # Normalize: scale by target_sum, then log1p
            X_norm = X_raw / cell_totals[:, np.newaxis] * self._target_sum
            X_norm = np.log1p(X_norm, dtype=np.float32)
            return X_norm, found_names
        else:
            return X_raw, found_names

    # ============ SUBSETTING ============

    def get_expression(self, cell_indices: np.ndarray, genes: Sequence[str] | None = None) -> np.ndarray:
        """Fetch expression, normalizing if needed"""
        X = self._fetch_normalized(cell_indices)

        if genes is not None:
            gene_idx = self._var.index.get_indexer(genes)
            missing = [g for g, i in zip(genes, gene_idx) if i == -1]
            if missing:
                raise KeyError(f"Genes not found: {missing}")
            X = X[:, gene_idx]

        return X

    def get_obsm(self, key: str, cell_indices: np.ndarray | None = None) -> np.ndarray:
        """Fetch embedding, optionally subset to cells"""
        if key not in self._obsm_names:
            raise KeyError(f"'{key}' not found. Available: {self.obsm_keys}")
        if key not in self._obsm:
            with self._obsm_lock:
                if key not in self._obsm:
                    with h5py.File(self._path, "r", locking=False) as f:
                        self._obsm[key] = read_elem(f["obsm"][key])
        if cell_indices is None:
            return self._obsm[key]
        return self._obsm[key][cell_indices]

    def get_obs(self, columns: Sequence[str] | None = None, cell_indices: np.ndarray | None = None) -> pd.DataFrame:
        """Fetch obs columns, optionally subset to cells"""
        df = self._obs if columns is None else self._obs[list(columns)]
        if cell_indices is not None:
            df = df.iloc[cell_indices]
        return df

    # ============ CLUSTER STATS ============

    def compute_cluster_stats(
        self,
        cell_indices: np.ndarray,
        gene_indices: np.ndarray | None = None,
        return_detection: bool = False,
        per_gene_detection: bool = False,
    ) -> dict:
        """
        Compute mean, var, n for a subset of cells.

        Args:
            cell_indices: Indices of cells to compute stats for
            gene_indices: Optional indices of genes to compute stats for.
                If None, computes stats for all genes (slower).
                If provided, computes stats only for specified genes (5-6x faster).
            return_detection: If True, also return mean_genes_detected
                (average number of non-zero genes per cell). No extra I/O cost.
            per_gene_detection: If True, also return frac_nonzero — a length-
                n_genes array of the fraction of cells in which each gene is
                detected (i.e. has non-zero expression). No extra I/O cost
                on the sparse path. Used by callers that need DE-style
                per-gene detection rates (e.g. narrate_immune compute_de).

        Returns:
            Dict with 'mean', 'var', 'n' arrays/values
            (and 'mean_genes_detected' float if return_detection=True,
             and 'frac_nonzero' np.ndarray[n_genes] if per_gene_detection=True)

        Performance note:
            For sparse data, computes stats directly on the sparse matrix
            without densifying, using ~10-20x less memory.
            For large clusters (> chunk_size cells), stats are accumulated
            in chunks using Chan's parallel variance formula to avoid OOM.
        """
        n_cells = len(cell_indices)

        if n_cells == 0:
            n_genes = len(gene_indices) if gene_indices is not None else self.n_genes
            result = {
                "mean": np.zeros(n_genes, dtype=np.float32),
                "var": np.zeros(n_genes, dtype=np.float32),
                "n": 0,
            }
            if return_detection:
                result["mean_genes_detected"] = 0.0
            if per_gene_detection:
                result["frac_nonzero"] = np.zeros(n_genes, dtype=np.float32)
            return result

        # Accumulate stats in chunks using Chan's parallel variance
        global_mean = None
        global_M2 = None
        global_n = 0
        detected_sum = 0.0
        # For per_gene_detection we accumulate per-gene non-zero counts
        # additively across chunks (no Chan needed — straight sum).
        global_nonzero_per_gene = None

        for start in range(0, n_cells, self._chunk_size):
            chunk_idx = cell_indices[start: start + self._chunk_size]
            chunk_stats = self._compute_chunk_stats(
                chunk_idx, gene_indices, return_detection, per_gene_detection
            )
            chunk_n = chunk_stats["n"]
            chunk_mean = chunk_stats["mean"]
            chunk_var = chunk_stats["var"]

            if return_detection:
                detected_sum += chunk_stats["detected_sum"]
            if per_gene_detection:
                chunk_nz = chunk_stats["nonzero_per_gene"]
                if global_nonzero_per_gene is None:
                    global_nonzero_per_gene = chunk_nz.astype(np.float64)
                else:
                    global_nonzero_per_gene = global_nonzero_per_gene + chunk_nz

            if global_mean is None:
                global_mean = chunk_mean
                global_M2 = chunk_var * chunk_n
                global_n = chunk_n
            else:
                delta = chunk_mean - global_mean
                new_n = global_n + chunk_n
                global_mean = (global_n * global_mean + chunk_n * chunk_mean) / new_n
                global_M2 = global_M2 + chunk_var * chunk_n + delta ** 2 * global_n * chunk_n / new_n
                global_n = new_n

        result = {
            "mean": global_mean,
            "var": global_M2 / global_n,
            "n": global_n,
        }
        if return_detection:
            result["mean_genes_detected"] = detected_sum / global_n
        if per_gene_detection:
            result["frac_nonzero"] = (global_nonzero_per_gene / global_n).astype(np.float32)
        return result

    def _compute_chunk_stats(
        self,
        cell_indices: np.ndarray,
        gene_indices: np.ndarray | None,
        return_detection: bool = False,
        per_gene_detection: bool = False,
    ) -> dict:
        """Compute mean/var/n for a chunk, staying sparse when possible.

        For sparse readers with gene_indices, avoids densifying entirely:
        fetches sparse rows, slices to gene columns, and computes sum/sumsq
        on the sparse representation directly.
        """
        n = len(cell_indices)

        # Try sparse-only path: sparse reader + gene_indices subset
        if self._reader.is_sparse and gene_indices is not None:
            sparse_mat = self._reader.fetch_rows_sparse(cell_indices)

            # fetch_rows_sparse returns None if storage turned out to be dense
            # on lazy open (initial probe may disagree with actual format).
            if sparse_mat is not None:
                if self.needs_normalization:
                    if self._total_counts is not None:
                        row_sums = self._total_counts[cell_indices].copy()
                    else:
                        row_sums = np.asarray(sparse_mat.sum(axis=1)).ravel()
                    row_sums = np.where(row_sums == 0, 1, row_sums)
                    sparse_mat = sparse_mat.multiply(
                        1.0 / row_sums[:, np.newaxis] * self._target_sum
                    )
                    sparse_mat = sparse_mat.tocsr()
                    sparse_mat.data = np.log1p(sparse_mat.data).astype(np.float32)

                # Slice to requested gene columns (sparse column slice is cheap)
                sparse_mat = sparse_mat[:, gene_indices]

                col_sum = np.asarray(sparse_mat.sum(axis=0), dtype=np.float64).ravel()
                sparse_sq = sparse_mat.copy()
                sparse_sq.data = sparse_sq.data.astype(np.float64) ** 2
                col_sumsq = np.asarray(sparse_sq.sum(axis=0)).ravel()

                mean = (col_sum / n).astype(np.float32)
                var = ((col_sumsq / n) - mean.astype(np.float64) ** 2).astype(np.float32)

                result = {"mean": mean, "var": var, "n": n}
                if return_detection:
                    result["detected_sum"] = float(
                        (sparse_mat > 0).sum(axis=1).astype(np.float64).sum()
                    )
                if per_gene_detection:
                    result["nonzero_per_gene"] = np.asarray(
                        (sparse_mat > 0).sum(axis=0), dtype=np.float64
                    ).ravel()
                return result

        # Dense fallback (dense reader, or gene_indices=None, or sparse returned None)
        if gene_indices is not None:
            X = self._reader.fetch_rows_cols_dense(cell_indices, gene_indices, dtype=np.float32)
            if self.needs_normalization:
                cell_totals = self.total_counts[cell_indices]
                cell_totals = np.where(cell_totals == 0, 1, cell_totals)
                X = X / cell_totals[:, np.newaxis] * self._target_sum
                X = np.log1p(X, dtype=np.float32)
        else:
            X = self._fetch_normalized(cell_indices)

        result = {
            "mean": X.mean(axis=0),
            "var": X.var(axis=0),
            "n": n,
        }
        if return_detection:
            result["detected_sum"] = float(
                (X > 0).sum(axis=1).astype(np.float64).sum()
            )
        if per_gene_detection:
            result["nonzero_per_gene"] = np.asarray(
                (X > 0).sum(axis=0), dtype=np.float64
            ).ravel()
        return result

    def get_cluster_stats(self, cluster_id: str) -> dict:
        """Get cached cluster stats (mean, var, n, mean_genes_detected) over gene_filter_indices.

        Computes on first call for a cluster_id, returns cached result afterwards.
        Cache is cleared when overlay changes (set_cluster_labels).
        """
        if cluster_id in self._cluster_stats_cache:
            return self._cluster_stats_cache[cluster_id]

        cell_indices = self.get_cell_indices(cluster_id)
        stats = self.compute_cluster_stats(
            cell_indices, gene_indices=self.gene_filter_indices, return_detection=True,
        )
        self._cluster_stats_cache[cluster_id] = stats
        return stats

    def restore_overlay(self, overlay: pd.DataFrame) -> None:
        """Restore a checkpoint only when it refers to the same ordered cells."""
        if not overlay.index.equals(self._obs.index):
            raise ValueError("Overlay cell identities/order do not match the source dataset")
        self._overlay = overlay.copy()
        self._cluster_stats_cache.clear()

    # ============ OVERLAY MUTATIONS ============

    def set_cluster_labels(self, parent_id: str, labels: np.ndarray) -> dict:
        """Assign cluster labels under a parent"""
        parent_depth = self._parse_cluster_depth(parent_id) if parent_id else 0
        new_depth = parent_depth + 1
        new_col = self._level_column(new_depth)

        cell_indices = self.get_cell_indices(parent_id) if parent_id else np.arange(self.n_cells)

        if len(labels) != len(cell_indices):
            raise ValueError(f"Labels length {len(labels)} != cells {len(cell_indices)}")

        prefix = f"{parent_id}_" if parent_id else ""
        full_labels = [f"{prefix}{lbl}" for lbl in labels]

        if new_col not in self._overlay.columns:
            self._overlay[new_col] = None

        self._overlay.iloc[cell_indices, self._overlay.columns.get_loc(new_col)] = full_labels

        # Invalidate cluster stats cache (cell membership changed)
        self._cluster_stats_cache.clear()

        unique = sorted(set(full_labels))
        return {"depth": new_depth, "clusters": unique, "n_clusters": len(unique)}

    def _annotation_column(self, depth: int, base_column: str = "cell_type") -> str:
        """Get annotation column name for a depth level."""
        return f"{base_column}_L{depth}"

    def set_annotation(
        self,
        cluster_id: str,
        label: str,
        column: str = "cell_type",
    ) -> dict:
        """
        Assign annotation to cluster.

        Annotations are stored in per-depth columns (e.g., cell_type_L1, cell_type_L2)
        to preserve hierarchical annotations.

        Args:
            cluster_id: Cluster ID (e.g., "0", "0_1")
            label: Annotation label (e.g., "T cell")
            column: Base column name (default "cell_type")

        Returns:
            Dict with cluster, label, n_cells, and column info
        """
        depth = self._parse_cluster_depth(cluster_id) if cluster_id else 1
        ann_col = self._annotation_column(depth, column)

        mask = self.get_cell_mask(cluster_id)
        n_cells = mask.sum()

        if ann_col not in self._overlay.columns:
            self._overlay[ann_col] = None

        self._overlay.loc[mask, ann_col] = label

        return {
            "cluster": cluster_id,
            "label": label,
            "n_cells": int(n_cells),
            "column": ann_col,
            "depth": depth,
        }

    def get_annotations(self, depth: int | None = None, column: str = "cell_type") -> dict[str, str]:
        """
        Get cluster → annotation mapping.

        Args:
            depth: Specific depth level, or None for all depths
            column: Base column name (default "cell_type")

        Returns:
            Dict mapping cluster_id to annotation label
        """
        annotations = {}

        if depth is not None:
            depths = [depth]
        else:
            # Find all depths with annotations
            depths = []
            for col in self._overlay.columns:
                if col.startswith(f"{column}_L"):
                    try:
                        d = int(col.split("_L")[-1])
                        depths.append(d)
                    except ValueError:
                        pass

        for d in depths:
            ann_col = self._annotation_column(d, column)
            cluster_col = self._level_column(d)

            if ann_col not in self._overlay.columns:
                continue
            if cluster_col not in self._overlay.columns:
                continue

            # Group by cluster, get unique annotation
            for cluster_id in self._overlay[cluster_col].dropna().unique():
                mask = self._overlay[cluster_col] == cluster_id
                labels = self._overlay.loc[mask, ann_col].dropna().unique()
                if len(labels) == 1:
                    annotations[cluster_id] = labels[0]

        return annotations

    def get_annotation_columns(self) -> list[str]:
        """Get all annotation column names in the overlay."""
        return [col for col in self._overlay.columns if col.startswith("cell_type_L")]

    def get_cluster_columns(self) -> list[str]:
        """Get all cluster ID column names in the overlay."""
        return [col for col in self._overlay.columns if col.startswith("L") and not col.startswith("cell_type")]

    def set_ont_cell_type(self, cluster_id: str, ont_term: str | None) -> None:
        """Store Cell Ontology term for a cluster."""
        if ont_term:
            self._ont_cell_types[cluster_id] = ont_term
        elif cluster_id in self._ont_cell_types:
            del self._ont_cell_types[cluster_id]

    def get_ont_cell_type(self, cluster_id: str) -> str | None:
        """Get Cell Ontology term for a cluster."""
        return self._ont_cell_types.get(cluster_id)

    def set_hypotheses(self, cluster_id: str, hypotheses: list[dict]) -> None:
        """Store hypotheses with reasoning for a cluster."""
        if hypotheses:
            self._hypotheses[cluster_id] = hypotheses
        elif cluster_id in self._hypotheses:
            del self._hypotheses[cluster_id]

    def get_hypotheses(self, cluster_id: str) -> list[dict] | None:
        """Get hypotheses with reasoning for a cluster."""
        return self._hypotheses.get(cluster_id)

    def merge_clusters(
        self,
        cluster_ids_to_merge: list[str],
        target_cluster_id: str | None = None,
    ) -> dict:
        """
        Merge multiple clusters into a single cluster by reassigning cluster IDs.

        Args:
            cluster_ids_to_merge: List of cluster IDs to merge (must be at same depth)
            target_cluster_id: Target cluster ID to merge into (default: first cluster in list)

        Returns:
            Dict with merge info
        """
        if not cluster_ids_to_merge or len(cluster_ids_to_merge) < 2:
            return {"merged": False, "message": "Need at least 2 clusters to merge"}

        # Use first cluster as target if not specified
        if target_cluster_id is None:
            target_cluster_id = cluster_ids_to_merge[0]

        # Verify all clusters are at same depth
        depths = [self._parse_cluster_depth(cid) for cid in cluster_ids_to_merge]
        if len(set(depths)) != 1:
            raise ValueError(f"All clusters must be at same depth. Found depths: {set(depths)}")

        depth = depths[0]
        col = self._level_column(depth)

        if col not in self._overlay.columns:
            raise KeyError(f"Level {depth} not found in overlay")

        # Get all cell indices for clusters to merge
        all_cell_indices = []
        for cid in cluster_ids_to_merge:
            indices = self.get_cell_indices(cid)
            all_cell_indices.extend(indices)

        # Reassign all cells to target cluster ID
        self._overlay.iloc[all_cell_indices, self._overlay.columns.get_loc(col)] = target_cluster_id

        return {
            "merged": True,
            "target_cluster_id": target_cluster_id,
            "merged_clusters": cluster_ids_to_merge,
            "n_cells": len(all_cell_indices),
        }

    def dissolve_clusters(self, cluster_ids: list[str]) -> None:
        """Remove clusters from the overlay by setting their cells back to None.

        Used to drop small subclusters that fall below the minimum proportion
        threshold. Affected cells retain their parent-level assignment but have
        no entry at the child depth, so they are not processed further.

        Args:
            cluster_ids: Cluster IDs to dissolve (must all be at the same depth)
        """
        if not cluster_ids:
            return

        depth = self._parse_cluster_depth(cluster_ids[0])
        col = self._level_column(depth)

        if col not in self._overlay.columns:
            return

        col_loc = self._overlay.columns.get_loc(col)
        for cid in cluster_ids:
            cell_indices = self.get_cell_indices(cid)
            self._overlay.iloc[cell_indices, col_loc] = None

        self._cluster_stats_cache.clear()

    def reindex_clusters(self, parent_id: str, kept_ids: list[str]) -> dict[str, str]:
        """Reindex surviving clusters to consecutive 0..N-1 under *parent_id*.

        Called after ``dissolve_clusters`` so that cluster numbering has no gaps.

        Args:
            parent_id: Parent cluster whose children should be reindexed.
            kept_ids: The cluster IDs that survived the filter (will be sorted).

        Returns:
            Mapping ``{old_id: new_id}`` for IDs that changed.  Empty if
            the enumeration is already contiguous.
        """
        if not kept_ids:
            return {}

        depth = self._parse_cluster_depth(kept_ids[0])
        col = self._level_column(depth)
        prefix = f"{parent_id}_" if parent_id else ""

        if len(set(kept_ids)) != len(kept_ids):
            raise ValueError("Duplicate cluster IDs in reindex request")
        expected_depth = self._parse_cluster_depth(parent_id) + 1 if parent_id else 1
        if any(self._parse_cluster_depth(cid) != expected_depth or
               (cid.rsplit("_", 1)[0] if "_" in cid else "") != parent_id
               for cid in kept_ids):
            raise ValueError("All reindexed clusters must be immediate children of the parent")
        original = self._overlay[col]
        existing = set(original.dropna().unique())
        if not set(kept_ids).issubset(existing):
            raise ValueError("Cannot reindex missing clusters")
        # This operation is used before children are annotated or subdivided.
        # Refuse to invalidate an already established descendant hierarchy.
        affected = original.isin(kept_ids)
        deeper = [c for c in self._overlay if c.startswith("L") and c[1:].isdigit() and int(c[1:]) > depth]
        if deeper and self._overlay.loc[affected, deeper].notna().any().any():
            raise ValueError("Cannot reindex clusters with existing descendants")

        mapping = {old: f"{prefix}{i}" for i, old in enumerate(sorted(kept_ids))}
        if (set(mapping.values()) - set(kept_ids)) & existing:
            raise ValueError("Reindex target collides with a cluster outside the request")
        rename_map = {old: new for old, new in mapping.items() if old != new}
        if not rename_map:
            return {}

        # Derive all replacements from the original labels/stores. Sequential
        # in-place renames would merge 10 -> 2 with a later 2 -> 4 rename.
        replacements = original.loc[affected].map(mapping)
        before = original.loc[affected].value_counts()
        after = replacements.value_counts()
        if any(int(after.get(new, 0)) != int(before[old]) for old, new in mapping.items()):
            raise RuntimeError("Reindexing would change cluster membership counts")
        stores = []
        for store in (self._hypotheses, self._ont_cell_types):
            if (set(mapping.values()) - set(kept_ids)) & set(store):
                raise ValueError("Reindex target collides with existing cluster metadata")
            stores.append({mapping.get(cid, cid): value for cid, value in store.items()})
        self._overlay.loc[affected, col] = replacements
        self._hypotheses, self._ont_cell_types = stores

        self._cluster_stats_cache.clear()
        return rename_map

    def get_cell_mask(self, cluster_id: str) -> np.ndarray:
        """Boolean mask for cells in cluster"""
        if not cluster_id:
            return np.ones(self.n_cells, dtype=bool)

        depth = self._parse_cluster_depth(cluster_id)
        col = self._level_column(depth)

        if col not in self._overlay.columns:
            raise KeyError(f"Level {depth} not computed")

        return (self._overlay[col] == cluster_id).values
