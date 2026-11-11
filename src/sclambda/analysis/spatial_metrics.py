"""
Spatial autocorrelation metrics for spatial transcriptomics data.

Provides Moran's I computation for:
1. Cell type distribution - is a cell type clustered or evenly spread within a region?
2. Gene expression - is gene expression spatially coherent within a region?

Uses k-nearest neighbor spatial weights for robustness across varying cell densities.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


def _build_knn_indices(
    coords: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Build self-excluded kNN index matrix of shape (n_cells, k_actual)."""
    from sklearn.neighbors import NearestNeighbors

    n = coords.shape[0]
    k_actual = min(k, n - 1)
    if k_actual < 1:
        return np.empty((n, 0), dtype=np.int32)

    # Query one extra neighbour and drop self to keep exactly k non-self neighbours.
    nn = NearestNeighbors(n_neighbors=min(k_actual + 1, n), algorithm="ball_tree")
    nn.fit(coords)
    _, indices_full = nn.kneighbors(coords)

    indices = np.empty((n, k_actual), dtype=np.int32)
    for i, row in enumerate(indices_full):
        row_wo_self = row[row != i]
        indices[i] = row_wo_self[:k_actual]

    return indices


def _weights_from_knn_indices(knn_indices: np.ndarray, n_cells: int) -> csr_matrix:
    """Construct row-standardized sparse weight matrix from kNN indices."""
    if n_cells == 0 or knn_indices.shape[1] == 0:
        return csr_matrix((n_cells, n_cells))

    k_actual = knn_indices.shape[1]
    rows = np.repeat(np.arange(n_cells), k_actual)
    cols = knn_indices.ravel()
    data = np.ones(len(rows), dtype=np.float64)

    W = csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))

    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    W = W.multiply(1.0 / row_sums[:, np.newaxis])

    return W


def _build_knn_structure(
    coords: np.ndarray,
    k: int = 10,
) -> tuple[csr_matrix, np.ndarray]:
    """Build both row-standardized kNN weights and neighbour index matrix."""
    knn_indices = _build_knn_indices(coords, k=k)
    W = _weights_from_knn_indices(knn_indices, n_cells=coords.shape[0])
    return W, knn_indices


def _build_knn_weights(
    coords: np.ndarray,
    k: int = 10,
) -> csr_matrix:
    """
    Build a sparse k-nearest neighbor spatial weight matrix.

    Args:
        coords: (n_cells, 2) array of spatial coordinates
        k: Number of nearest neighbors

    Returns:
        Sparse (n_cells, n_cells) binary weight matrix (row-standardized)
    """
    W, _ = _build_knn_structure(coords, k=k)
    return W


def _build_uniform_hop_candidates(
    knn_indices: np.ndarray,
    max_hops: int = 2,
    max_candidates: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build uniform swap candidates from an undirected kNN graph.

    For each source cell i, candidate set is all cells reachable within
    <= max_hops graph hops (excluding i). Swap partner is sampled uniformly.
    """
    n = knn_indices.shape[0]
    candidates = -np.ones((n, max_candidates), dtype=np.int32)
    lengths = np.zeros(n, dtype=np.int32)

    if n == 0 or knn_indices.shape[1] == 0 or max_hops < 1:
        return candidates, lengths

    # Undirected adjacency for hop queries
    adjacency_sets = [set(row.tolist()) for row in knn_indices]
    for i, row in enumerate(knn_indices):
        for j in row:
            adjacency_sets[int(j)].add(i)
    adjacency = [np.fromiter(s, dtype=np.int32) for s in adjacency_sets]

    for src in range(n):
        visited = {src}
        frontier = deque([(src, 0)])

        while frontier:
            node, depth = frontier.popleft()
            if depth >= max_hops:
                continue
            for nb in adjacency[node]:
                nb_int = int(nb)
                if nb_int not in visited:
                    visited.add(nb_int)
                    frontier.append((nb_int, depth + 1))

        hop_cells = sorted(visited - {src})
        if not hop_cells:
            continue
        if len(hop_cells) > max_candidates:
            hop_cells = hop_cells[:max_candidates]

        m = len(hop_cells)
        candidates[src, :m] = np.asarray(hop_cells, dtype=np.int32)
        lengths[src] = m

    return candidates, lengths


def _prepare_moran_edges(W: csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute edge arrays for fast Moran numerator evaluation."""
    W_coo = W.tocoo()
    return (
        W_coo.row.astype(np.int32),
        W_coo.col.astype(np.int32),
        W_coo.data.astype(np.float64),
    )


def _generate_hop_swap_permutations(
    n: int,
    n_permutations: int,
    hop_candidates: np.ndarray,
    hop_lengths: np.ndarray,
    n_swaps_per_permutation: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate permutation indices via random transpositions in hop neighborhoods.

    Vectorized across permutations to reduce Python-loop overhead.
    """
    perms = np.tile(np.arange(n, dtype=np.int32), (n_permutations, 1))
    if n == 0 or n_permutations == 0 or n_swaps_per_permutation <= 0:
        return perms

    perm_rows = np.arange(n_permutations, dtype=np.int32)
    max_candidates = hop_candidates.shape[1]

    for _ in range(n_swaps_per_permutation):
        src = rng.integers(0, n, size=n_permutations)
        lens = hop_lengths[src]
        valid = lens > 0
        if not np.any(valid):
            continue

        rows = perm_rows[valid]
        src_valid = src[valid]
        lens_valid = lens[valid]

        # Draw a uniform candidate index in [0, lens_i) per permutation row.
        if max_candidates == 1:
            local_idx = np.zeros(len(rows), dtype=np.int32)
        else:
            local_idx = rng.integers(0, max_candidates, size=len(rows)).astype(np.int32)
            local_idx %= lens_valid

        dst = hop_candidates[src_valid, local_idx]

        temp = perms[rows, src_valid].copy()
        perms[rows, src_valid] = perms[rows, dst]
        perms[rows, dst] = temp

    return perms


def _perm_numerators_from_indices(
    z: np.ndarray,
    perm_indices: np.ndarray,
    edge_rows: np.ndarray,
    edge_cols: np.ndarray,
    edge_weights: np.ndarray,
    max_block_elements: int = 8_000_000,
) -> np.ndarray:
    """Moran numerators z_p^T W z_p for each permutation p.

    Permutations are processed in blocks, each as one sparse-dense product,
    so memory stays bounded however large the niche is.
    """
    n_perm, n = perm_indices.shape
    W = csr_matrix((edge_weights, (edge_rows, edge_cols)), shape=(n, n))
    block = max(1, max_block_elements // max(n, 1))
    perm_nums = np.empty(n_perm, dtype=np.float64)
    for start in range(0, n_perm, block):
        z_block = z[perm_indices[start:start + block]].T  # (n, block)
        perm_nums[start:start + block] = np.einsum("ij,ij->j", z_block, W @ z_block)
    return perm_nums


def compute_morans_i(
    values: np.ndarray,
    W: csr_matrix,
) -> float:
    """
    Compute Moran's I spatial autocorrelation statistic.

    Moran's I ranges from -1 (dispersed) through 0 (random) to +1 (clustered).

    Args:
        values: (n,) array of values (e.g., binary cell type indicator or expression)
        W: (n, n) sparse spatial weight matrix (row-standardized)

    Returns:
        Moran's I value
    """
    n = len(values)
    if n < 3:
        return 0.0

    x = values.astype(np.float64)
    x_bar = x.mean()
    z = x - x_bar

    denominator = np.sum(z ** 2)
    if denominator == 0:
        return 0.0

    # Numerator: sum_i sum_j w_ij * z_i * z_j
    # Using sparse matrix: z^T W z
    Wz = W.dot(z)
    numerator = z.dot(Wz)

    # W_sum = sum of all weights
    W_sum = W.sum()
    if W_sum == 0:
        return 0.0

    I = (n / W_sum) * (numerator / denominator)
    return float(I)


def _permutation_morans_z(
    values: np.ndarray,
    W: csr_matrix,
    n_permutations: int = 199,
    seed: int | None = None,
    hop_candidates: np.ndarray | None = None,
    hop_lengths: np.ndarray | None = None,
    n_swaps_per_permutation: int | None = None,
    edge_rows: np.ndarray | None = None,
    edge_cols: np.ndarray | None = None,
    edge_weights: np.ndarray | None = None,
    perm_indices: np.ndarray | None = None,
    unavailable_as_none: bool = False,
) -> float | None:
    """
    Compute permutation-based z-score for Moran's I.

    Permutes values n_permutations times, recomputes Moran's I numerator
    each time, and returns z = (observed - mean_perm) / std_perm.

    The (n / W_sum) / denominator scaling is constant across permutations
    and cancels in the z-score, so only the numerator (z^T W z) is permuted.

    Args:
        values: (n,) array of values
        W: (n, n) sparse spatial weight matrix (row-standardized)
        n_permutations: Number of random permutations for null distribution
        seed: Random seed for reproducibility (None = random)
        hop_candidates: Optional (n, m) matrix of candidate swap indices.
        hop_lengths: Optional (n,) valid candidate counts per source cell.
        n_swaps_per_permutation: Number of random transpositions per permutation.
        edge_rows/edge_cols/edge_weights: Optional precomputed edge arrays.
        perm_indices: Optional precomputed permutation indices of shape
            (n_permutations, n).

    Returns:
        Permutation z-score (higher = more spatially autocorrelated than chance)
    """
    n = len(values)
    if n < 3:
        return None if unavailable_as_none else 0.0

    x = values.astype(np.float64)
    z = x - x.mean()

    if np.sum(z ** 2) == 0:
        return None if unavailable_as_none else 0.0

    if edge_rows is None or edge_cols is None or edge_weights is None:
        edge_rows, edge_cols, edge_weights = _prepare_moran_edges(W)

    # Observed numerator: z^T W z
    observed_num = np.sum(edge_weights * z[edge_rows] * z[edge_cols])

    rng = np.random.default_rng(seed)

    use_hop_null = (
        hop_candidates is not None
        and hop_lengths is not None
        and hop_candidates.shape[0] == n
        and hop_lengths.shape[0] == n
    )

    if perm_indices is not None:
        perm_nums = _perm_numerators_from_indices(
            z,
            perm_indices=perm_indices,
            edge_rows=edge_rows,
            edge_cols=edge_cols,
            edge_weights=edge_weights,
        )
    elif use_hop_null:
        if n_swaps_per_permutation is None:
            n_swaps_per_permutation = n

        perm_indices = _generate_hop_swap_permutations(
            n=n,
            n_permutations=n_permutations,
            hop_candidates=hop_candidates,
            hop_lengths=hop_lengths,
            n_swaps_per_permutation=n_swaps_per_permutation,
            rng=rng,
        )
        perm_nums = _perm_numerators_from_indices(
            z,
            perm_indices=perm_indices,
            edge_rows=edge_rows,
            edge_cols=edge_cols,
            edge_weights=edge_weights,
        )
    else:
        # Backward-compatible global shuffle null
        perm_indices = np.array([rng.permutation(n) for _ in range(n_permutations)])
        z_perms = z[perm_indices]  # (n_perm, n)
        Wz_perms = W.dot(z_perms.T)  # (n, n_perm)
        perm_nums = np.sum(z_perms.T * Wz_perms, axis=0)  # (n_perm,)

    std_perm = perm_nums.std()
    if std_perm < 1e-15:
        return None if unavailable_as_none else 0.0

    return float((observed_num - perm_nums.mean()) / std_perm)


@dataclass
class CellTypeSpatialMetrics:
    """Spatial metrics for a single cell type within a niche."""

    cell_type: str
    n_cells: int
    proportion: float  # absolute proportion in niche
    morans_z: float | None  # None when unavailable

    def as_dict(self) -> dict:
        return {
            "cell_type": self.cell_type,
            "n_cells": self.n_cells,
            "proportion": round(self.proportion, 4),
            "morans_z": round(self.morans_z, 3) if self.morans_z is not None else None,
        }


@dataclass
class SpatialProportionsResult:
    """Spatial proportion analysis result for one niche."""

    cluster_id: str
    n_total: int
    metrics: list[CellTypeSpatialMetrics]

    def as_dicts(self) -> list[dict]:
        return [m.as_dict() for m in self.metrics]


def compute_spatial_proportions(
    coords: np.ndarray,
    cell_types: np.ndarray,
    cluster_id: str = "",
    k_neighbors: int = 10,
    max_hops: int = 2,
    n_permutations: int = 199,
    seed: int | None = None,
) -> SpatialProportionsResult:
    """
    Compute absolute proportions and Moran's z-score for each cell type.

    Permutation test uses uniform random swaps within a max-hop neighborhood
    of the kNN graph to build a local null distribution for Moran's I.

    Args:
        coords: (n_cells, 2) spatial coordinates for cells in the niche
        cell_types: (n_cells,) cell type labels for cells in the niche
        cluster_id: Identifier for the niche/cluster
        k_neighbors: Number of neighbors for spatial weight matrix
        max_hops: Maximum hop radius for local label swaps (default: 2)
        n_permutations: Number of permutations for z-score computation
        seed: Random seed for reproducibility (None = random)

    Returns:
        SpatialProportionsResult with per-cell-type proportions and Moran's z-score
    """
    n = len(cell_types)
    if n == 0:
        return SpatialProportionsResult(
            cluster_id=cluster_id, n_total=0, metrics=[]
        )

    # Build spatial weight matrix and hop neighborhoods once for the whole niche
    W, knn_indices = _build_knn_structure(coords, k=k_neighbors)
    hop_candidates, hop_lengths = _build_uniform_hop_candidates(
        knn_indices,
        max_hops=max_hops,
    )
    edge_rows, edge_cols, edge_weights = _prepare_moran_edges(W)
    # One set of permutations per niche, shared by every cell type or gene.
    shared_perm_indices = _generate_hop_swap_permutations(
        n=n,
        n_permutations=n_permutations,
        hop_candidates=hop_candidates,
        hop_lengths=hop_lengths,
        n_swaps_per_permutation=n,
        rng=np.random.default_rng(seed),
    )

    unique_types, counts = np.unique(cell_types, return_counts=True)
    # Sort by abundance (most cells first)
    order = np.argsort(-counts)
    unique_types = unique_types[order]
    counts = counts[order]

    metrics = []
    for ct, count in zip(unique_types, counts):
        proportion = count / n

        # Binary indicator for this cell type
        indicator = (cell_types == ct).astype(np.float64)

        # Compute Moran's z-score only if cell type has enough cells and isn't 100%
        if count >= 3 and count < n:
            mz = _permutation_morans_z(
                indicator,
                W,
                n_permutations,
                seed,
                hop_candidates=hop_candidates,
                hop_lengths=hop_lengths,
                edge_rows=edge_rows,
                edge_cols=edge_cols,
                edge_weights=edge_weights,
                perm_indices=shared_perm_indices,
                unavailable_as_none=True,
            )
        else:
            mz = None

        metrics.append(CellTypeSpatialMetrics(
            cell_type=str(ct),
            n_cells=int(count),
            proportion=proportion,
            morans_z=mz,
        ))

    return SpatialProportionsResult(
        cluster_id=cluster_id,
        n_total=n,
        metrics=metrics,
    )


@dataclass
class GeneSpatialScore:
    """Permutation z-score for a single gene's spatial autocorrelation."""

    gene_name: str
    morans_z: float | None

    def as_dict(self) -> dict:
        return {
            "gene_name": self.gene_name,
            "morans_z": round(self.morans_z, 3) if self.morans_z is not None else None,
        }


@dataclass
class GeneSpatialScoresResult:
    """Moran's z-scores for multiple genes within a niche."""

    cluster_id: str
    scores: list[GeneSpatialScore]

    def as_dict(self) -> dict:
        return {s.gene_name: s.morans_z for s in self.scores}

    def get_score(self, gene_name: str) -> float | None:
        for s in self.scores:
            if s.gene_name == gene_name:
                return s.morans_z
        return None


def compute_gene_spatial_scores(
    coords: np.ndarray,
    expression_matrix: np.ndarray,
    gene_names: list[str],
    cluster_id: str = "",
    k_neighbors: int = 10,
    max_hops: int = 2,
    n_permutations: int = 199,
    seed: int | None = None,
) -> GeneSpatialScoresResult:
    """
    Compute permutation z-score for each gene's spatial autocorrelation.

    Expression values are permuted via local random swaps within a max-hop
    neighborhood on the provided-cell kNN graph.

    Args:
        coords: (n_cells, 2) spatial coordinates for cells in the niche
        expression_matrix: (n_cells, n_genes) expression matrix
        gene_names: Gene names corresponding to columns of expression_matrix
        cluster_id: Identifier for the niche/cluster
        k_neighbors: Number of neighbors for spatial weight matrix
        max_hops: Maximum hop radius for local value swaps (default: 2)
        n_permutations: Number of permutations for z-score computation
        seed: Random seed for reproducibility (None = random)

    Returns:
        GeneSpatialScoresResult with Moran's z-score per gene
    """
    n = coords.shape[0]
    if n < 3 or len(gene_names) == 0:
        return GeneSpatialScoresResult(cluster_id=cluster_id, scores=[])

    W, knn_indices = _build_knn_structure(coords, k=k_neighbors)
    hop_candidates, hop_lengths = _build_uniform_hop_candidates(
        knn_indices,
        max_hops=max_hops,
    )
    edge_rows, edge_cols, edge_weights = _prepare_moran_edges(W)
    # One set of permutations per niche, shared by every cell type or gene.
    shared_perm_indices = _generate_hop_swap_permutations(
        n=n,
        n_permutations=n_permutations,
        hop_candidates=hop_candidates,
        hop_lengths=hop_lengths,
        n_swaps_per_permutation=n,
        rng=np.random.default_rng(seed),
    )

    scores = []
    for i, gene in enumerate(gene_names):
        expr = expression_matrix[:, i]
        # Only compute if gene has non-zero variance
        if np.std(expr) > 1e-10:
            mz = _permutation_morans_z(
                expr,
                W,
                n_permutations,
                seed,
                hop_candidates=hop_candidates,
                hop_lengths=hop_lengths,
                edge_rows=edge_rows,
                edge_cols=edge_cols,
                edge_weights=edge_weights,
                perm_indices=shared_perm_indices,
                unavailable_as_none=True,
            )
        else:
            mz = None
        scores.append(GeneSpatialScore(gene_name=gene, morans_z=mz))

    return GeneSpatialScoresResult(cluster_id=cluster_id, scores=scores)


# --------------------------------------------------------------------------- #
#  Niche adjacency
# --------------------------------------------------------------------------- #


def compute_niche_adjacency(
    coords: np.ndarray,
    niche_labels: np.ndarray,
    k: int = 10,
) -> dict[str, list[str]]:
    """Compute spatial adjacency between niches using k-NN.

    For each cell, its k nearest neighbors are inspected.  If a neighbor
    belongs to a different niche, that niche pair is considered adjacent.
    Neighbors are sorted by the number of shared boundary cells
    (descending) so the first entry is the most strongly adjacent niche.

    Parameters
    ----------
    coords : ndarray of shape (n_cells, 2)
        Spatial coordinates for every cell.
    niche_labels : ndarray of shape (n_cells,)
        Niche membership label for every cell.
    k : int
        Number of nearest neighbours to consider.

    Returns
    -------
    dict mapping each niche label to a list of adjacent niche labels
    (ordered by shared boundary cell count, descending).
    """
    from sklearn.neighbors import NearestNeighbors
    from collections import Counter

    coords = np.asarray(coords, dtype=float)
    niche_labels = np.asarray(niche_labels, dtype=str)
    n = len(coords)
    k_actual = min(k, n - 1)
    if k_actual < 1:
        return {}

    nn = NearestNeighbors(n_neighbors=k_actual, algorithm="ball_tree")
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)

    # Count cross-niche neighbor relationships
    # adjacency_counts[(A, B)] = number of cells in A that have a neighbor in B
    adjacency_counts: dict[tuple[str, str], int] = Counter()
    for i in range(n):
        src = niche_labels[i]
        for j in indices[i]:
            dst = niche_labels[j]
            if dst != src:
                adjacency_counts[(src, dst)] += 1

    # Build per-niche neighbor lists sorted by shared boundary count
    all_niches = sorted(set(niche_labels))
    adjacency: dict[str, list[str]] = {n: [] for n in all_niches}
    for niche in all_niches:
        neighbors = [
            (other, adjacency_counts[(niche, other)])
            for other in all_niches
            if other != niche and (niche, other) in adjacency_counts
        ]
        neighbors.sort(key=lambda x: x[1], reverse=True)
        adjacency[niche] = [other for other, _ in neighbors]

    return adjacency


# --------------------------------------------------------------------------- #
#  Per-instance niche neighbourhood details
# --------------------------------------------------------------------------- #


@dataclass
class NicheInstanceDetail:
    """Description of a single spatial instantiation of a niche."""

    instance_index: int
    n_cells: int
    centroid: tuple[float, float]
    adjacent_niches: list[str]  # ordered by boundary cell count


@dataclass
class NicheInstancesResult:
    """All instances for a given niche, plus overall adjacency."""

    cluster_id: str
    n_instances: int
    instances: list[NicheInstanceDetail]


def compute_niche_instance_details(
    coords: np.ndarray,
    niche_labels: np.ndarray,
    target_niche: str,
    k: int = 10,
    dbscan_eps: float | None = None,
    dbscan_min_samples: int = 5,
) -> NicheInstancesResult:
    """Detect spatial instances of a niche and compute per-instance neighbours.

    Each spatially disconnected region of *target_niche* (detected via DBSCAN)
    is treated as a separate instance.  For each instance the function
    identifies adjacent niches by inspecting k-NN relationships of the
    instance's cells only.

    Parameters
    ----------
    coords : ndarray of shape (n_cells, 2)
        Spatial coordinates for **all** cells (not just the target niche).
    niche_labels : ndarray of shape (n_cells,)
        Niche label per cell.
    target_niche : str
        Niche to analyse.
    k : int
        Number of nearest neighbours.
    dbscan_eps : float or None
        DBSCAN epsilon. None = auto (5% of spatial diagonal of target cells).
    dbscan_min_samples : int
        DBSCAN min_samples.

    Returns
    -------
    NicheInstancesResult
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import DBSCAN
    from collections import Counter

    coords = np.asarray(coords, dtype=float)
    niche_labels = np.asarray(niche_labels, dtype=str)

    target_mask = niche_labels == target_niche
    target_indices = np.where(target_mask)[0]
    if len(target_indices) == 0:
        return NicheInstancesResult(cluster_id=target_niche, n_instances=0, instances=[])

    target_coords = coords[target_indices]

    # --- DBSCAN for instance detection ---
    if dbscan_eps is None:
        extent = target_coords.max(axis=0) - target_coords.min(axis=0)
        diag = np.linalg.norm(extent)
        dbscan_eps = max(diag * 0.05, 1e-6)

    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(target_coords)
    db_labels = db.labels_
    unique_db_labels = sorted(set(db_labels) - {-1})

    if not unique_db_labels:
        # Everything noise → treat as single instance
        unique_db_labels = [0]
        db_labels = np.zeros(len(target_coords), dtype=int)

    # Assign noise to nearest cluster centroid
    noise_mask = db_labels == -1
    if noise_mask.any() and len(unique_db_labels) > 0:
        centroids_arr = np.array([
            target_coords[db_labels == lab].mean(axis=0) for lab in unique_db_labels
        ])
        noise_idx = np.where(noise_mask)[0]
        noise_pts = target_coords[noise_idx]
        dists = np.linalg.norm(noise_pts[:, None, :] - centroids_arr[None, :, :], axis=2)
        nearest = dists.argmin(axis=1)
        for ni, ci in zip(noise_idx, nearest):
            db_labels[ni] = unique_db_labels[ci]

    # --- k-NN on all cells for neighbour lookup ---
    k_actual = min(k, len(coords) - 1)
    nn = NearestNeighbors(n_neighbors=k_actual, algorithm="ball_tree")
    nn.fit(coords)

    instances: list[NicheInstanceDetail] = []
    for inst_i, lab in enumerate(unique_db_labels):
        inst_local_idx = np.where(db_labels == lab)[0]
        inst_global_idx = target_indices[inst_local_idx]
        inst_coords = coords[inst_global_idx]
        centroid = tuple(float(v) for v in inst_coords.mean(axis=0))

        # Find neighbours of this instance's cells
        _, neighbor_idx = nn.kneighbors(inst_coords)
        adj_counts: Counter = Counter()
        for row in neighbor_idx:
            for j in row:
                nb_label = niche_labels[j]
                if nb_label != target_niche:
                    adj_counts[nb_label] += 1

        adjacent = [n for n, _ in adj_counts.most_common()]

        instances.append(NicheInstanceDetail(
            instance_index=inst_i + 1,
            n_cells=len(inst_global_idx),
            centroid=centroid,
            adjacent_niches=adjacent,
        ))

    return NicheInstancesResult(
        cluster_id=target_niche,
        n_instances=len(instances),
        instances=instances,
    )
