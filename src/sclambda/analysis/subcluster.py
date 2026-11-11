from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
from enum import Enum
import logging
import os
import threading

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, pairwise_distances
from sklearn.neighbors import NearestNeighbors

from sclambda.analysis.timing import log_execution

logger = logging.getLogger(__name__)

_SUBSAMPLE_THRESHOLD = 300_000
_HNSWLIB_THRESHOLD = 800_000
_LARGE_DATASET_THRESHOLD = 1_000_000
_LARGE_DATASET_DEFAULT_RESOLUTION = 0.15
# Serializes GPU access across threads — prevents OOM when multiple clusters
# are processed in parallel (TaskRunner.map_parallel) with enable_rapids=True.
_GPU_LOCK = threading.Semaphore(1)


class SubclusterMethod(str, Enum):
    LEIDEN = "leiden"
    FROM_OBS = "from_obs"


@dataclass
class SubclusterResult:
    cluster_id: str
    new_depth: int
    clusters: list[str]
    n_clusters: int
    n_cells: int
    method: str
    resolution: float | None = None  # Only for leiden

    @property
    def summary(self) -> str:
        res_str = f" (resolution={self.resolution:.2f})" if self.resolution else ""
        return (
            f"Subclustered '{self.cluster_id or 'root'}' → "
            f"{self.n_clusters} clusters at L{self.new_depth}{res_str}: "
            f"{self.clusters[:5]}{'...' if len(self.clusters) > 5 else ''}"
        )


def _composite_scores(records: list[dict]) -> np.ndarray:
    """
    Compute normalized composite scores for a list of metric records.

    Silhouette and Calinski-Harabasz are normalized to [0,1] (higher=better).
    Davies-Bouldin is inverted after normalization (lower=better).
    Returns an array of composite scores, one per record.
    """
    df = pd.DataFrame(records)
    s = pd.DataFrame(index=df.index)
    for col in ["silhouette", "calinski_harabasz"]:
        s[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-9)
    s["davies_bouldin"] = 1 - (df["davies_bouldin"] - df["davies_bouldin"].min()) / (
        df["davies_bouldin"].max() - df["davies_bouldin"].min() + 1e-9
    )
    return s.mean(axis=1).values


def _resolution_sweep(
    adata_tmp: ad.AnnData,
    embeddings: np.ndarray,
    resolutions: list[float] | None = None,
    subsample_n: int = 10000,
    random_state: int = 42,
    patience: int = 3,
    leiden_fn: Callable | None = None,
) -> pd.DataFrame:
    """
    Sweep Leiden resolutions (low → high) with greedy early stopping.

    After each new resolution, the composite score (silhouette + CH - DB,
    all normalized to [0,1] using current min/max) is recomputed for all
    evaluated points.  If the current best index has not moved forward for
    `patience` consecutive steps, the sweep stops — we are confident the
    optimum has been passed.

    Args:
        adata_tmp: AnnData with neighbors already computed
        embeddings: Cell embeddings matrix
        resolutions: Ordered list of resolutions to try (low → high)
        subsample_n: Subsample size for silhouette score (O(n^2))
        random_state: Random seed
        patience: Stop after this many consecutive non-improving steps

    Returns:
        DataFrame with resolution, n_clusters, silhouette, calinski_harabasz,
        davies_bouldin for every evaluated resolution
    """
    if resolutions is None:
        resolutions = np.round(np.arange(0.1, 0.51, 0.05), 2).tolist()

    rng = np.random.default_rng(random_state)
    n = len(embeddings)
    sub = rng.choice(n, min(subsample_n, n), replace=False)

    # The same sampled cells are scored at every resolution. Cache their exact
    # distances once; cap persistent memory per concurrent worker.
    distance_cache = None
    sample = embeddings[sub]
    distance_bytes = len(sub) ** 2 * (4 if sample.dtype == np.float32 else 8)
    cache_budget = int(os.environ.get("SCLAMBDA_SILHOUETTE_CACHE_BYTES", 512 * 1024**2))
    results = []
    best_idx = 0  # index of the best composite score seen so far
    no_improve = 0  # consecutive steps the best index hasn't moved

    for res in resolutions:
        if leiden_fn is not None:
            leiden_fn(adata_tmp, res)
        else:
            sc.tl.leiden(adata_tmp, resolution=res, key_added="_sweep", random_state=random_state, flavor="igraph")
        labels = np.asarray(adata_tmp.obs["_sweep"].cat.codes.values)

        if labels.max() < 1:
            continue

        # Subsample may miss a tiny cluster — skip when that happens
        sub_labels = labels[sub]
        if len(set(sub_labels)) < 2:
            continue

        if distance_cache is None and distance_bytes <= cache_budget:
            distance_cache = pairwise_distances(sample, metric="euclidean")
            np.fill_diagonal(distance_cache, 0)
        silhouette = (
            silhouette_score(distance_cache, sub_labels, metric="precomputed")
            if distance_cache is not None else silhouette_score(sample, sub_labels)
        )
        results.append(
            {
                "resolution": res,
                "n_clusters": labels.max() + 1,
                "silhouette": silhouette,
                "calinski_harabasz": calinski_harabasz_score(embeddings, labels),
                "davies_bouldin": davies_bouldin_score(embeddings, labels),
            }
        )

        # Need ≥ 2 points for meaningful normalization
        if len(results) < 2:
            continue

        scores = _composite_scores(results)
        new_best_idx = int(np.argmax(scores))

        if new_best_idx > best_idx:
            best_idx = new_best_idx
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.debug(
                    "Resolution sweep stopped early at %.2f " "(best was %.2f, patience=%d).",
                    res,
                    results[best_idx]["resolution"],
                    patience,
                )
                break

    if "_sweep" in adata_tmp.obs.columns:
        del adata_tmp.obs["_sweep"]
    return pd.DataFrame(results).round(4)


def _find_optimal_resolution(df: pd.DataFrame) -> float:
    """
    Pick the best resolution via composite score.
    Normalizes silhouette and Calinski-Harabasz to [0,1] (higher=better),
    inverts Davies-Bouldin (lower=better), then averages.
    """
    scores = _composite_scores(df.to_dict("records"))
    return float(df.iloc[int(np.argmax(scores))]["resolution"])


def _leiden_on_embeddings(
    embeddings: np.ndarray,
    n_neighbors: int = 30,
    random_state: int = 42,
    skip_sweep: bool = False,
    enable_rapids: bool = False,
    resolution: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    Run neighbors + resolution sweep + Leiden on an embedding matrix.
    Returns (string_labels, optimal_resolution).

    When *resolution* is given it is used directly (no sweep, overrides the
    large-dataset default). Otherwise, when *skip_sweep* is True the resolution
    sweep is skipped and ``_LARGE_DATASET_DEFAULT_RESOLUTION`` is used.

    When *enable_rapids* is True, uses rapids_singlecell (backed by cuGraph)
    for GPU-accelerated neighbors and Leiden. GPU access is serialized via
    ``_GPU_LOCK`` so parallel calls from TaskRunner.map_parallel are safe.
    """
    n_cells = len(embeddings)
    adata_tmp = ad.AnnData(X=embeddings)
    adata_tmp.obsm["embedding"] = embeddings
    n_neighbors = min(n_neighbors, n_cells - 1)

    if enable_rapids:
        try:
            import rapids_singlecell as rsc
        except ImportError as exc:
            raise ImportError(
                "rapids_singlecell is required for enable_rapids=True. "
                "Install the RAPIDS stack: https://rapids.ai/start.html"
            ) from exc

        with _GPU_LOCK:
            rsc.get.anndata_to_GPU(adata_tmp)
            rsc.pp.neighbors(adata_tmp, use_rep="embedding", n_neighbors=n_neighbors)

            def leiden_fn(adata, res):
                rsc.tl.leiden(adata, resolution=res, key_added="_sweep", random_state=random_state)

            if resolution is not None:
                optimal_res = float(resolution)
                logger.info("Using explicit resolution %.2f (no sweep).", optimal_res)
            elif skip_sweep:
                optimal_res = _LARGE_DATASET_DEFAULT_RESOLUTION
                logger.info(
                    "Skipping resolution sweep (large dataset); using default resolution %.2f.",
                    optimal_res,
                )
            else:
                sweep_df = _resolution_sweep(adata_tmp, embeddings, random_state=random_state,
                                             leiden_fn=leiden_fn)
                if sweep_df.empty:
                    return np.zeros(n_cells, dtype=int).astype(str), 0.0
                optimal_res = _find_optimal_resolution(sweep_df)

            rsc.tl.leiden(adata_tmp, resolution=optimal_res, random_state=random_state)
            labels = adata_tmp.obs["leiden"].astype(str).values

        return labels, optimal_res

    else:
        sc.pp.neighbors(adata_tmp, use_rep="embedding", n_neighbors=n_neighbors)

        if resolution is not None:
            optimal_res = float(resolution)
            logger.info("Using explicit resolution %.2f (no sweep).", optimal_res)
        elif skip_sweep:
            optimal_res = _LARGE_DATASET_DEFAULT_RESOLUTION
            logger.info(
                "Skipping resolution sweep (large dataset); using default resolution %.2f.",
                optimal_res,
            )
        else:
            sweep_df = _resolution_sweep(adata_tmp, embeddings, random_state=random_state)
            if sweep_df.empty:
                return np.zeros(n_cells, dtype=int).astype(str), 0.0
            optimal_res = _find_optimal_resolution(sweep_df)

        sc.tl.leiden(adata_tmp, resolution=optimal_res, flavor="igraph", random_state=random_state)
        return adata_tmp.obs["leiden"].astype(str).values, optimal_res


def _knn_propagate(
    fit_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    fit_labels: np.ndarray,
    n_cells_total: int,
    enable_rapids: bool = False,
) -> np.ndarray:
    """
    Assign each query cell the label of its nearest fit cell.

    Priority: rapids (cuML) > hnswlib (large datasets) > sklearn.
    """
    if enable_rapids:
        try:
            from cuml.neighbors import NearestNeighbors as cuNN
        except ImportError as exc:
            raise ImportError(
                "cuML is required for enable_rapids=True. "
                "Install the RAPIDS stack: https://rapids.ai/start.html"
            ) from exc
        with _GPU_LOCK:
            nn = cuNN(n_neighbors=1, output_type="numpy")
            nn.fit(fit_embeddings.astype(np.float32))
            _, indices = nn.kneighbors(query_embeddings.astype(np.float32))
        return fit_labels[np.asarray(indices)[:, 0]]
    elif n_cells_total > _HNSWLIB_THRESHOLD and os.getenv("SCLAMBDA_DISABLE_HNSWLIB", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        try:
            import hnswlib  # Local import: avoid hard crash on incompatible CPUs unless explicitly enabled.
        except Exception as exc:  # noqa: BLE001
            logger.warning("hnswlib unavailable (%s) — falling back to sklearn 1-NN.", exc)
        else:
            dim = fit_embeddings.shape[1]
            index = hnswlib.Index(space="l2", dim=dim)
            index.init_index(max_elements=len(fit_embeddings), ef_construction=200, M=16)
            index.add_items(fit_embeddings, num_threads=int(os.getenv("SCLAMBDA_CLUSTER_THREADS", "1")))
            index.set_ef(max(50, 1))
            indices, _ = index.knn_query(query_embeddings, k=1, num_threads=int(os.getenv("SCLAMBDA_CLUSTER_THREADS", "1")))
            return fit_labels[indices[:, 0]]
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=1)
    nn.fit(fit_embeddings)
    _, indices = nn.kneighbors(query_embeddings)
    return fit_labels[indices[:, 0]]


def _run_leiden_with_retry(embeddings, resolution=1.0, min_clusters=2,
                           n_neighbors=30, random_state=42, enable_rapids=False,
                           force_resolution=None, **kwargs):
    """Isolate CPU clustering from annotation threads and process-global RNGs."""
    params = dict(resolution=resolution, min_clusters=min_clusters,
                  n_neighbors=n_neighbors, random_state=random_state,
                  enable_rapids=enable_rapids, force_resolution=force_resolution)
    if enable_rapids or len(embeddings) <= 1:
        return _run_leiden_local(embeddings, **params)
    from sclambda.analysis.clustering_process import cluster_in_process
    return cluster_in_process(embeddings, **params)


def _run_leiden_local(
    embeddings: np.ndarray,
    resolution: float = 1.0,
    min_clusters: int = 2,
    n_neighbors: int = 30,
    random_state: int = 42,
    enable_rapids: bool = False,
    force_resolution: float | None = None,
    **_kwargs,
) -> tuple[np.ndarray, float]:
    """
    Find optimal Leiden resolution via metric sweep, then cluster.

    When n_cells > _SUBSAMPLE_THRESHOLD (300k):
      1. Subsample _SUBSAMPLE_THRESHOLD cells.
      2. Run neighbors + Leiden on the subsample.
      3. Assign the remaining cells to the nearest subsampled cell's cluster
         via 1-NN lookup in the embedding space.

    Args:
        embeddings: Cell embeddings (n_cells, n_dims)
        resolution: Ignored (kept for backward compat); resolution is auto-determined
        min_clusters: Ignored (kept for backward compat)
        random_state: Random seed

    Returns:
        (labels, optimal_resolution)
    """
    n_cells = len(embeddings)

    if n_cells <= 1:
        return np.zeros(max(n_cells, 0), dtype=int), 0.0

    skip_sweep = n_cells > _LARGE_DATASET_THRESHOLD

    if n_cells <= _SUBSAMPLE_THRESHOLD:
        str_labels, optimal_res = _leiden_on_embeddings(
            embeddings, n_neighbors=n_neighbors, random_state=random_state,
            skip_sweep=skip_sweep, enable_rapids=enable_rapids, resolution=force_resolution,
        )
    else:
        logger.info(
            "Cluster has %d cells > %d; subsampling for Leiden clustering.",
            n_cells,
            _SUBSAMPLE_THRESHOLD,
        )
        rng = np.random.default_rng(random_state)
        sub_idx = rng.choice(n_cells, _SUBSAMPLE_THRESHOLD, replace=False)
        rest_mask = np.ones(n_cells, dtype=bool)
        rest_mask[sub_idx] = False
        rest_idx = np.where(rest_mask)[0]

        sub_labels, optimal_res = _leiden_on_embeddings(
            embeddings[sub_idx], n_neighbors=n_neighbors, random_state=random_state,
            skip_sweep=skip_sweep, enable_rapids=enable_rapids, resolution=force_resolution,
        )
        logger.info(
            "Leiden on subsample produced %d clusters (resolution=%.2f); "
            "propagating to %d remaining cells via 1-NN.",
            len(set(sub_labels)),
            optimal_res,
            len(rest_idx),
        )

        str_labels = np.empty(n_cells, dtype=sub_labels.dtype)
        str_labels[sub_idx] = sub_labels
        str_labels[rest_idx] = _knn_propagate(
            embeddings[sub_idx], embeddings[rest_idx], sub_labels, n_cells,
            enable_rapids=enable_rapids,
        )

    unique_labels = sorted(set(str_labels))
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    numeric_labels = np.array([label_map[lbl] for lbl in str_labels])

    return numeric_labels, optimal_res


def _encode_labels_numeric(labels: np.ndarray) -> np.ndarray:
    """Encode arbitrary labels to consecutive integers"""
    unique = sorted(set(labels))
    label_map = {lbl: i for i, lbl in enumerate(unique)}
    return np.array([label_map[lbl] for lbl in labels])


@log_execution
def subcluster(
    accessor,
    cluster_id: str = "",
    method: SubclusterMethod | str = SubclusterMethod.LEIDEN,
    # From obs params
    obs_columns: Sequence[str] | None = None,
    # Deprecated – kept for backward compat, ignored
    resolution: float = 1.0,
    min_clusters: int = 2,
    n_neighbors: int = 30,
    enable_rapids: bool = False,
    synchronize: Callable | None = None,
    **_kwargs,
) -> SubclusterResult:
    """
    Subcluster cells within a cluster.

    Leiden resolution is determined automatically via a metric-based sweep
    (silhouette, Calinski-Harabasz, Davies-Bouldin).

    Args:
        accessor: Data accessor
        cluster_id: Parent cluster to subcluster ("" for root)
        method: "leiden" or "from_obs"
        obs_columns: Hierarchical columns in obs (for from_obs method)
        synchronize: Optional callback such as runner.with_lock. Protects the
            membership snapshot and validated overlay commit, not computation.

    Returns:
        SubclusterResult with new cluster info
    """
    method = SubclusterMethod(method)

    # Snapshot membership under the same lock used to publish overlay updates.
    sync = synchronize or (lambda func, *args: func(*args))
    try:
        cell_indices = sync(lambda: accessor.get_cell_indices(cluster_id).copy())
    except KeyError:
        # Re-raise KeyError from get_cell_indices (e.g., level not computed)
        raise

    n_cells = len(cell_indices)

    if n_cells == 0:
        # Check if cluster exists - if cluster_id is provided but no cells found, raise KeyError
        if cluster_id:
            # Verify cluster doesn't exist by checking overlay
            depth = len(cluster_id.split("_"))
            col = f"L{depth}"
            if col in accessor._overlay.columns:
                # Column exists, check if cluster_id is in it
                unique_clusters = accessor._overlay[col].dropna().unique()
                if cluster_id not in unique_clusters:
                    raise KeyError(f"Cluster '{cluster_id}' not found")
        raise ValueError(f"No cells in cluster '{cluster_id}'")

    # Determine current depth and new depth
    if cluster_id:
        current_depth = len(cluster_id.split("_"))
    else:
        current_depth = 0
    new_depth = current_depth + 1

    # Determine which method to use (support mixed mode)
    use_from_obs = False
    use_leiden = False

    # Handle explicit FROM_OBS method
    if method == SubclusterMethod.FROM_OBS:
        if obs_columns is None:
            raise ValueError("obs_columns required for from_obs method")

        # Check if obs_columns are available for current depth
        if current_depth < len(obs_columns):
            use_from_obs = True
        else:
            # Obs columns exhausted - try to fall back to Leiden if embedding available
            try:
                embedding_key = accessor._embedding_key
                accessor.get_obsm(embedding_key, cell_indices[:1])  # Quick check with one cell
                use_leiden = True
            except (KeyError, IndexError):
                # No embedding available - skip clustering gracefully
                use_leiden = False

    # Handle explicit LEIDEN method
    elif method == SubclusterMethod.LEIDEN:
        # Check if embedding is available
        try:
            embedding_key = accessor._embedding_key
            accessor.get_obsm(embedding_key, cell_indices[:1])  # Quick check with one cell
            use_leiden = True
        except (KeyError, IndexError, AttributeError):
            # Explicit Leiden request but no embedding - raise error
            raise KeyError(f"Embedding not found. Available: {accessor.obsm_keys}")

    # Execute clustering
    labels = None
    used_resolution = None

    if use_from_obs:
        labels = _subcluster_from_obs(
            accessor=accessor,
            cell_indices=cell_indices,
            obs_columns=obs_columns,
            current_depth=current_depth,
        )
        used_method = SubclusterMethod.FROM_OBS

    elif use_leiden:
        labels, final_resolution = _subcluster_leiden(
            accessor=accessor,
            cell_indices=cell_indices,
            embedding_key=embedding_key,
            n_neighbors=n_neighbors,
            enable_rapids=enable_rapids,
        )
        used_resolution = final_resolution
        used_method = SubclusterMethod.LEIDEN

    else:
        # No clustering possible - return empty result
        # This happens when:
        # - obs_columns exhausted and no embedding available
        # - method is FROM_OBS but obs_columns exhausted and no embedding
        return SubclusterResult(
            cluster_id=cluster_id,
            new_depth=new_depth,
            clusters=[],
            n_clusters=0,
            n_cells=n_cells,
            method="none",
            resolution=None,
        )

    # The expensive computation above uses immutable expression/embedding data.
    # Validate the snapshot and publish labels together under the shared lock.
    def commit():
        if not np.array_equal(accessor.get_cell_indices(cluster_id), cell_indices):
            raise RuntimeError(f"Cluster {cluster_id!r} membership changed during subclustering")
        return accessor.set_cluster_labels(cluster_id, labels)

    result = sync(commit)

    return SubclusterResult(
        cluster_id=cluster_id,
        new_depth=new_depth,
        clusters=result["clusters"],
        n_clusters=result["n_clusters"],
        n_cells=n_cells,
        method=used_method.value,
        resolution=used_resolution,
    )


def _subcluster_leiden(
    accessor,
    cell_indices: np.ndarray,
    embedding_key: str,
    n_neighbors: int = 30,
    enable_rapids: bool = False,
) -> tuple[np.ndarray, float]:
    """Run leiden subclustering with automatic resolution selection."""

    embeddings = accessor.get_obsm(embedding_key, cell_indices)

    labels, optimal_resolution = _run_leiden_with_retry(
        embeddings=embeddings, n_neighbors=n_neighbors, enable_rapids=enable_rapids,
    )

    return labels, optimal_resolution


def _subcluster_from_obs(
    accessor,
    cell_indices: np.ndarray,
    obs_columns: Sequence[str],
    current_depth: int,
) -> np.ndarray:
    """
    Extract subclusters from user-provided obs columns.

    Note: This function assumes current_depth < len(obs_columns).
    The caller should check this before calling.
    """
    # Get the next level column
    next_col = obs_columns[current_depth]

    # Get labels for cells in this cluster
    obs_subset = accessor.get_obs([next_col], cell_indices)
    raw_labels = obs_subset[next_col].values

    # Encode to numeric
    labels = _encode_labels_numeric(raw_labels)

    return labels


def subcluster_all_at_depth(
    accessor,
    depth: int,
    method: SubclusterMethod | str = SubclusterMethod.LEIDEN,
    **kwargs,
) -> list[SubclusterResult]:
    """
    Subcluster all clusters at a given depth.

    Args:
        accessor: Data accessor
        depth: Depth of clusters to subcluster (0 = root)
        method: Clustering method
        **kwargs: Passed to subcluster()

    Returns:
        List of SubclusterResults
    """
    if depth == 0:
        clusters = [""]
    else:
        clusters = accessor.get_all_clusters(depth)

    results = []
    for cluster_id in clusters:
        try:
            result = subcluster(
                accessor=accessor,
                cluster_id=cluster_id,
                method=method,
                **kwargs,
            )
            results.append(result)
        except ValueError as e:
            # Skip empty clusters
            print(f"Skipping {cluster_id}: {e}")

    return results
