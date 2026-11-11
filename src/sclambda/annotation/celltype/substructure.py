"""
Deterministic, signature-light substructure probes for the subcluster decision.

Two graph statistics on a cluster's kNN graph (in ``embedding_key`` space) decide,
*before* an expensive split, whether and why a cluster should be subclustered —
without knowing the contaminant or the subtypes a priori:

1. **Self-depletion** — a connected pocket where the cluster's OWN committed-identity
   signature is *absent* flags an off-lineage minority / contamination. Needs only the
   self signature (which you always have: it's the annotation). Contaminant-agnostic.

2. **Structured variance** — unsupervised graph autocorrelation (Moran's I) over
   nuisance-masked highly-variable genes flags within-lineage subtype heterogeneity, and
   groups the structured genes into co-expression *modules* for the model to judge.

The model (not a threshold) then decides ``needs_subcluster`` and which modules are real
subtype axes vs. confounds (clonality, donor/sex, stress). See ``subcluster_decision.md``.

Also exposes ``parent_signature_retention`` for the formalized parent-match: scoring the
*parent's* self signature on a child to decide same_type / subtype / unrelated.

Pure numpy/scipy; no scanpy dependency.
"""

from __future__ import annotations

import re
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


# ---- minimal technical floor ONLY ----
# We do NOT hardcode biological nuisance (clonality, cell-cycle, activation, sex,
# housekeeping). Whether a structured program is a meaningful subdivision or a confound
# is a biological judgment made by the LLM from the module's genes (see
# subcluster_decision.md). The probe only drops pure-technical genes that dominate
# variance under every interpretation (ribosomal, mitochondrial) so the recovered
# modules are legible.
_NUISANCE_PREFIX = ("RPL", "RPS", "MRPL", "MRPS", "MT-")
_NUISANCE_SET = {"MALAT1", "NEAT1"}


def is_nuisance(gene: str) -> bool:
    g = str(gene).upper()
    return g in _NUISANCE_SET or g.startswith(_NUISANCE_PREFIX)


# ---- extract the cluster's own marker genes from the dossier text ----

_GENE_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,}(?:-[A-Z0-9]+)?\b")
_STOPWORDS = {
    "T", "B", "NK", "DE", "LFC", "IS", "NOT", "AND", "OR", "THE", "OF", "VS", "CD",
    "HIGH", "LOW", "FRAC", "MARKER", "MARKERS", "CELL", "CELLS", "TYPE", "GENE", "GENES",
    "MHC", "TCR", "BCR", "RNA", "ALL", "TF",
}


def extract_signature_genes(texts, valid_genes, max_genes: int = 40) -> list[str]:
    """Parse gene symbols cited in dossier text (confirmed traits / evidence_for),
    keep those that are real measured genes and not nuisance."""
    valid = set(map(str, valid_genes))
    out: list[str] = []
    seen: set[str] = set()
    for t in texts or []:
        for tok in _GENE_RE.findall(str(t)):
            if tok in _STOPWORDS or is_nuisance(tok):
                continue
            if tok in valid and tok not in seen:
                seen.add(tok)
                out.append(tok)
                if len(out) >= max_genes:
                    return out
    return out


def signature_genes_from_dossier(dossier, valid_genes, max_genes: int = 40) -> list[str]:
    """Self signature = genes the dossier cites for the committed identity:
    confirmed traits + the supported/leading candidate's evidence_for."""
    annotation = getattr(dossier, "annotation", {})
    if annotation:
        valid = set(valid_genes)
        return list(dict.fromkeys(g for ref in annotation.get("support", [])
                                  for g in ref["genes"] if g in valid))[:max_genes]
    texts = list(getattr(dossier, "confirmed", []) or [])
    lead = None
    try:
        lead = dossier.match(dossier.best_cell_type) if dossier.best_cell_type else dossier.leading()
    except Exception:  # noqa: BLE001
        lead = None
    if lead is not None:
        texts += list(getattr(lead, "evidence_for", []) or [])
    return extract_signature_genes(texts, valid_genes, max_genes=max_genes)


# ---- graph + Moran's I ----

def build_knn_graph(embedding: np.ndarray, k: int = 15) -> sp.csr_matrix:
    """Symmetric binary kNN adjacency from an embedding (self excluded)."""
    from sklearn.neighbors import NearestNeighbors
    n = embedding.shape[0]
    k = int(min(k, max(1, n - 1)))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(embedding)
    _, idx = nn.kneighbors(embedding)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    W = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return W.maximum(W.T)  # symmetrize


def morans_i(W: sp.csr_matrix, X: np.ndarray) -> np.ndarray:
    """Moran's I for each column of X (cells x features) on adjacency W.
    Returns an array of length n_features."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[0]
    S0 = float(W.sum())
    if S0 == 0:
        return np.zeros(X.shape[1])
    xc = X - X.mean(0, keepdims=True)
    num = np.einsum("ij,ij->j", xc, W @ xc)
    den = np.einsum("ij,ij->j", xc, xc)
    den = np.where(den == 0, np.nan, den)
    return (n / S0) * (num / den)


def _smooth(W: sp.csr_matrix, v: np.ndarray) -> np.ndarray:
    """Average a per-cell vector over kNN neighborhoods (incl. self)."""
    A = (W > 0).astype(float) + sp.eye(W.shape[0])
    A = sp.diags(1.0 / np.asarray(A.sum(1)).ravel()) @ A
    return np.asarray(A @ v).ravel()


# ---- the probe ----

def probe_substructure(
    accessor,
    cluster_id: str,
    self_genes,
    *,
    embedding_key: str | None = None,
    k: int = 15,
    absent_thr: float = 0.12,
    max_cells: int = 2000,
    n_hvg: int = 2000,
    min_module: int = 3,
    corr_thr: float = 0.30,
    min_subpop_frac: float = 0.10,
    rng_seed: int = 0,
) -> dict:
    """Return the deterministic substructure evidence for one cluster:

    {
      "n_cells", "self_genes",
      "depletion": {"frac_absent", "morans_i", "null_i", "present": bool},
      "structure": {"n_structured", "frac_structured", "chance_rate",
                    "modules": [{"genes": [...], "mean_i": float}], "present": bool},
    }
    """
    rng = np.random.RandomState(rng_seed)
    embedding_key = embedding_key or getattr(accessor, "_embedding_key", "X_pca")
    idx = np.asarray(accessor.get_cell_indices(cluster_id))
    n_all = len(idx)
    if n_all > max_cells:
        idx = np.sort(rng.choice(idx, max_cells, replace=False))
    n = len(idx)

    out: dict = {"n_cells": int(n_all), "self_genes": list(self_genes),
                 "depletion": None, "structure": None}
    if n < max(30, k + 5):
        return out  # too few cells to say anything

    emb = np.asarray(accessor.get_obsm(embedding_key, idx))
    W = build_knn_graph(emb, k=k)
    Xfull = np.asarray(accessor.get_expression(idx))
    gene_names = np.asarray(accessor.gene_names)
    gi = {g: i for i, g in enumerate(gene_names)}

    # --- self-depletion: connected pocket of absent committed-identity expression ---
    self_genes = [g for g in self_genes if g and g in gi]
    if self_genes:
        E = Xfull[:, [gi[g] for g in self_genes]]
        s = (E > 0).mean(1)                          # fraction of identity markers detected
        ss = _smooth(W, s)
        absent = (ss < absent_thr).astype(float)
        if absent.std() > 0:
            I = float(morans_i(W, absent)[0])
            null = float(morans_i(W, rng.permutation(absent))[0])
        else:
            I = null = 0.0
        present = bool(absent.mean() >= 0.02 and I > max(0.15, null + 0.1))
        pocket_markers: list[str] = []
        if present:
            # what is IN the pocket? top genes enriched in the absent cells vs the rest
            # (this names the unknown contaminant for the LLM — still self-supervised)
            pm = absent > 0
            diff = Xfull[pm].mean(0) - Xfull[~pm].mean(0)
            order = np.argsort(diff)[::-1]
            for j in order:
                g = str(gene_names[j])
                if not is_nuisance(g) and g not in self_genes and diff[j] > 0:
                    pocket_markers.append(g)
                if len(pocket_markers) >= 12:
                    break
        out["depletion"] = {
            "frac_absent": float(absent.mean()), "morans_i": I, "null_i": null,
            "present": present, "pocket_markers": pocket_markers,
        }

    # --- structured variance: HVGs autocorrelated on the graph ---
    mean = Xfull.mean(0)
    var = Xfull.var(0)
    expressed = mean > 0.02
    keep = np.array([expressed[i] and not is_nuisance(gene_names[i]) for i in range(len(gene_names))])
    cand = np.where(keep)[0]
    if len(cand) > n_hvg:
        cand = cand[np.argsort(var[cand])[::-1][:n_hvg]]
    if len(cand) >= min_module:
        Xc = Xfull[:, cand]
        I = morans_i(W, Xc)
        # chance rate from a permutation null
        sample = rng.choice(len(cand), min(40, len(cand)), replace=False)
        null = morans_i(W, np.column_stack([rng.permutation(Xc[:, j]) for j in sample]))
        chance = float(np.nanquantile(null, 0.99))
        thr = max(0.15, chance)
        struct_mask = np.nan_to_num(I) > thr
        struct_idx = cand[struct_mask]
        struct_I = I[struct_mask]
        modules = _modules(Xfull[:, struct_idx], gene_names[struct_idx], struct_I,
                           corr_thr=corr_thr, min_module=min_module,
                           min_subpop_frac=min_subpop_frac)
        out["structure"] = {
            "n_structured": int(struct_mask.sum()),
            "frac_structured": float(struct_mask.sum() / max(1, len(cand))),
            "chance_rate": chance,
            "modules": modules,
            "present": bool(len(modules) > 0),
        }
    return out


def _modules(Xs: np.ndarray, genes: np.ndarray, gene_I: np.ndarray,
             corr_thr: float, min_module: int, min_subpop_frac: float = 0.10) -> list[dict]:
    """Group structured genes into co-expression modules via correlation
    connected-components. Keep only modules that mark a **discrete subpopulation**
    of >= min_subpop_frac of cells — a diffuse gradient (e.g. ambient
    contamination spread across all cells) is NOT a splittable subpopulation and
    is dropped. Each kept module reports its discrete high-fraction (frac_high)."""
    m = Xs.shape[1]
    if m < min_module:
        return []
    C = np.nan_to_num(np.corrcoef(Xs.T))
    A = sp.csr_matrix((np.abs(C) >= corr_thr).astype(int))
    n_comp, labels = connected_components(A, directed=False)
    mods = []
    for c in range(n_comp):
        sel = np.where(labels == c)[0]
        if len(sel) < min_module:
            continue
        frac_high, discrete = _subpop_fraction(Xs[:, sel])
        if not discrete or frac_high < min_subpop_frac:
            continue  # gradient / too-small -> not a splittable subpopulation
        order = sel[np.argsort(gene_I[sel])[::-1]]
        mods.append({
            "genes": [str(genes[i]) for i in order[:12]],
            "size": int(len(sel)),
            "mean_i": float(np.mean(gene_I[sel])),
            "frac_high": round(float(frac_high), 3),
        })
    mods.sort(key=lambda d: d["mean_i"], reverse=True)
    return mods


def _subpop_fraction(Xmod: np.ndarray) -> tuple[float, bool]:
    """Does a module mark a DISCRETE subpopulation? Score = mean z-scored module
    expression per cell; fit a 2-component GMM. Returns (fraction of cells in the
    high mode, is_discrete). ``is_discrete`` is False for a continuous gradient
    (modes not separated), which should not trigger a split."""
    Z = (Xmod - Xmod.mean(0)) / (Xmod.std(0) + 1e-9)
    score = Z.mean(1)
    try:
        from sklearn.mixture import GaussianMixture
        gm = GaussianMixture(2, random_state=0).fit(score.reshape(-1, 1))
        mu = gm.means_.ravel()
        w = gm.weights_.ravel()
        lo, hi = np.argsort(mu)
        gap = (mu[hi] - mu[lo]) / (score.std() + 1e-9)
        return float(w[hi]), bool(gap >= 1.0)  # >=1 SD separation = discrete
    except Exception:  # noqa: BLE001
        return 1.0, False


# ---- formalized parent-match support ----

def signature_score(accessor, cell_indices: np.ndarray, genes) -> float:
    """Mean fraction of a gene set detected across the given cells (0..1).
    Used to score a parent's self signature on a child cluster."""
    genes = [g for g in (genes or []) if g]
    if not genes or len(cell_indices) == 0:
        return float("nan")
    E, found = accessor.get_expression_for_genes(np.asarray(cell_indices), genes)
    if not found:
        return float("nan")
    return float((np.asarray(E) > 0).mean())
