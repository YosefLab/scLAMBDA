"""
Targeted differential expression for specific genes.

Two levels of API:
1. compute_de_for_genes() - Low-level: DE for arbitrary gene list given cell indices
2. targeted_de_signatures() - High-level: DE for signatures with automatic reference selection

Used after marker refinement to evaluate suggested gene signatures
against proper reference populations (parent cluster excluding current cluster).
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np

if TYPE_CHECKING:
    from sclambda.core.accessor import Accessor

from sclambda.analysis.timing import log_execution


# Legend for compact metric abbreviations (included once per DE block in prompts)
DE_METRICS_LEGEND = (
    "DE metrics: LFC=log fold change, FAR=frac. above ref. median, "
    "DFE=delta frac. expressing (target−ref; positive=more cells expressing in target), "
    "ZE-LFC=zero-excluded LFC"
)

_DE_TABLE_LEGEND = (
    "DE table columns: Expression=mean expression in cluster, "
    "LFC=log fold change vs reference, "
    "Frac Expressing=fraction of cells in cluster with non-zero expression, "
    "Delta Frac Expressing=difference in fraction expressing (target−reference; positive=more cells expressing in target)"
)


def format_de_as_table(
    rows: list[tuple[str, list["GeneDE"]]],
) -> str:
    """Format DE results as a markdown table with one row per gene.

    Genes appearing in multiple rows (signatures/cell types) are merged into
    a single table row with all associated labels listed in the last column.

    Args:
        rows: List of (label, genes) tuples. Label is a signature name,
              cell type hypothesis, or other grouping. Genes are GeneDE objects.

    Returns:
        Markdown table string with legend.
    """
    # Collect genes and track which labels each gene belongs to
    gene_map: dict[str, "GeneDE"] = {}  # name -> best GeneDE (first seen)
    gene_labels: dict[str, list[str]] = {}  # name -> list of labels

    for label, genes in rows:
        for gene in genes:
            if gene.name not in gene_map:
                gene_map[gene.name] = gene
                gene_labels[gene.name] = []
            gene_labels[gene.name].append(label)

    if not gene_map:
        return "No DE data available."

    # Sort: enriched first (by LFC desc), then depleted (by LFC asc)
    sorted_genes = sorted(gene_map.values(), key=lambda g: -g.lfc)

    # Build table
    header = "| Gene | Expression | LFC | Frac Expressing | Delta Frac Expressing | Associated with |"
    separator = "|------|-----------|------|-----------------|----------------------|-----------------|"
    table_lines = [_DE_TABLE_LEGEND, "", header, separator]

    for g in sorted_genes:
        # Skip genes absent in the target cluster
        if g.frac_non_zero_target < 0.005 and abs(g.mean_target) < 0.005:
            continue
        labels = ", ".join(gene_labels[g.name])
        expr = f"{g.mean_target:.2f}" if g.mean_target != 0.0 else "--"
        frac_expr = f"{g.frac_non_zero_target:.2f}"
        zrd = f"{g.zero_rate_diff:+.2f}" if not math.isnan(g.zero_rate_diff) else "N/A"
        table_lines.append(
            f"| {g.name} | {expr} | {g.lfc:+.2f} | {frac_expr} | {zrd} | {labels} |"
        )

    return "\n".join(table_lines)


def _compute_extended_metrics(X_target: np.ndarray, X_reference: np.ndarray) -> dict:
    """
    Compute extended DE metrics from expression matrices.

    Args:
        X_target: Expression matrix for target/comparator cells (n_target x n_genes)
        X_reference: Expression matrix for reference cells (n_ref x n_genes)

    Returns:
        Dict with per-gene arrays:
          frac_above_ref_median, zero_rate_diff, zero_excl_lfc
    """
    # 1. Fraction above reference median
    ref_medians = np.median(X_reference, axis=0)
    frac_above_ref_median = np.mean(X_target > ref_medians[np.newaxis, :], axis=0)

    # 2. Delta non-zero fraction (target − reference)
    # Positive = more cells expressing in target (same direction as LFC)
    frac_nonzero_target = np.mean(X_target != 0, axis=0)
    frac_nonzero_ref = np.mean(X_reference != 0, axis=0)
    zero_rate_diff = frac_nonzero_target - frac_nonzero_ref

    # 3. Zero-excluded LFC (vectorised)
    target_nz_mask = X_target > 0
    ref_nz_mask = X_reference > 0

    target_nz_sum = np.where(target_nz_mask, X_target, 0).sum(axis=0)
    target_nz_count = target_nz_mask.sum(axis=0)
    ref_nz_sum = np.where(ref_nz_mask, X_reference, 0).sum(axis=0)
    ref_nz_count = ref_nz_mask.sum(axis=0)

    with np.errstate(invalid="ignore"):
        target_nz_mean = np.where(target_nz_count > 0, target_nz_sum / target_nz_count, 0)
        ref_nz_mean = np.where(ref_nz_count > 0, ref_nz_sum / ref_nz_count, 0)
    zero_excl_lfc = np.where(
        (target_nz_count > 0) & (ref_nz_count > 0),
        target_nz_mean - ref_nz_mean,
        float("nan"),
    )

    return {
        "frac_non_zero_target": frac_nonzero_target.astype(float),
        "frac_above_ref_median": frac_above_ref_median.astype(float),
        "zero_rate_diff": zero_rate_diff.astype(float),
        "zero_excl_lfc": zero_excl_lfc.astype(float),
    }


def classify_effect(lfc: float) -> str:
    """
    Classify the effect size based on log fold change.

    Returns a string with LFC value and effect category.
    """
    lfc_str = f"LFC={lfc:.2f};"
    if lfc > 1.5:
        return lfc_str + "very_high"
    elif lfc > 0.5:
        return lfc_str + "high"
    elif lfc > 0.001:
        return lfc_str + "medium"
    else:
        return lfc_str + "depleted"


# ============ BINNED TABLE FORMATTING ============

# Thresholds for binned table format used in reevaluation prompts
_HIGH_LFC_THRESHOLD = 1.0
_HIGH_ZELFC_THRESHOLD = 0.5
_SPATIAL_Z_THRESHOLD = 2.0

DE_BINNED_TABLE_LEGEND = (
    "Expression bins: Highly Enriched (strong upregulation confirmed by zero-excluded LFC), "
    "Moderately Enriched (mild upregulation or strong but unconfirmed by ZE-LFC), "
    "Moderately Depleted (mild downregulation), "
    "Highly Depleted (strong downregulation confirmed by zero-excluded LFC). "
    "Mean log-normalized expression in target shown in parentheses (higher = more highly expressed in absolute terms)"
)


def bin_gene_de(gene: "GeneDE") -> str:
    """Classify a GeneDE into one of four expression bins.

    Uses LFC as the primary signal. When zero-excluded LFC is available,
    genes with high |LFC| but low |ZE-LFC| are demoted to the moderate bin
    (the enrichment/depletion may be driven by dropout differences).

    Returns:
        One of: "highly_enriched", "moderately_enriched",
                "moderately_depleted", "highly_depleted"
    """
    lfc = gene.lfc
    ze_lfc = gene.zero_excl_lfc
    ze_available = not math.isnan(ze_lfc)

    if lfc >= _HIGH_LFC_THRESHOLD:
        if not ze_available or ze_lfc >= _HIGH_ZELFC_THRESHOLD:
            return "highly_enriched"
        return "moderately_enriched"
    elif lfc > 0:
        return "moderately_enriched"
    elif lfc > -_HIGH_LFC_THRESHOLD:
        return "moderately_depleted"
    else:
        if not ze_available or ze_lfc <= -_HIGH_ZELFC_THRESHOLD:
            return "highly_depleted"
        return "moderately_depleted"


def format_genes_as_binned_table(
    rows: list[tuple[str, list["GeneDE"]]],
    spatial_scores: dict[str, dict[str, float]] | None = None,
) -> str:
    """Format gene DE results as a binned markdown table.

    Args:
        rows: List of (row_label, genes) tuples. row_label is a cell type
              name, signature name, or hypothesis label. genes is a list
              of GeneDE objects for that row.
        spatial_scores: Optional dict mapping row_label -> {gene_name: morans_z}.
                        When provided, adds a "Localized" column.

    Returns:
        Markdown table string.
    """
    include_spatial = spatial_scores is not None

    columns = [
        "Highly Enriched",
        "Moderately Enriched",
        "Moderately Depleted",
        "Highly Depleted",
    ]
    if include_spatial:
        columns.append("Localized")

    _BIN_TO_COL = {
        "highly_enriched": "Highly Enriched",
        "moderately_enriched": "Moderately Enriched",
        "moderately_depleted": "Moderately Depleted",
        "highly_depleted": "Highly Depleted",
    }

    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    table_lines = [header, separator]

    for row_label, genes in rows:
        bins: dict[str, list[str]] = {c: [] for c in columns}

        seen: set[str] = set()
        for gene in genes:
            if gene.name in seen:
                continue
            seen.add(gene.name)
            col = _BIN_TO_COL[bin_gene_de(gene)]
            expr_str = f"{gene.mean_target:.2f}"
            bins[col].append(f"{gene.name}({expr_str})")

        if include_spatial:
            row_scores = (spatial_scores or {}).get(row_label, {})
            localized = [g for g, mz in row_scores.items() if mz is not None and mz > _SPATIAL_Z_THRESHOLD]
            bins["Localized"] = sorted(localized)

        cells = [", ".join(bins[c]) if bins[c] else "--" for c in columns]
        table_lines.append(f"| **{row_label}** | " + " | ".join(cells) + " |")

    return "\n".join(table_lines)


def format_genes_as_gene_table(
    rows: list[tuple[str, list["GeneDE"]]],
    spatial_scores: dict[str, dict[str, float]] | None = None,
) -> str:
    """Format gene DE results as per-cell-type markdown tables.

    Each cell type gets its own section heading and table.

    Args:
        rows: List of (cell_type, genes) tuples.  genes is a list of GeneDE
              objects.
        spatial_scores: Optional dict mapping cell_type -> {gene_name: morans_z}.
                        When provided, adds a "Moran's Z" column.

    Returns:
        Markdown string with one table per cell type.
    """
    include_spatial = spatial_scores is not None

    if include_spatial:
        header = "| Gene | Expression | Frac Expressed | LFC | Moran's Z |"
        sep    = "|------|-----------|---------------|-----|-----------|"
    else:
        header = "| Gene | Expression | Frac Expressed | LFC |"
        sep    = "|------|-----------|---------------|-----|"

    sections = []
    for cell_type, genes in rows:
        if not genes:
            continue
        row_scores = (spatial_scores or {}).get(cell_type, {})
        table_lines = [f"### {cell_type}", "", header, sep]
        for gene in genes:
            mz = row_scores.get(gene.name, float("nan"))
            line = (
                f"| {gene.name} "
                f"| {gene.mean_target:.2f} "
                f"| {gene.frac_non_zero_target:.2f} "
                f"| {gene.lfc:+.2f} "
            )
            if include_spatial:
                mz_str = f"{mz:.2f}" if mz is not None and not math.isnan(mz) else "--"
                line += f"| {mz_str} "
            line += "|"
            table_lines.append(line)
        sections.append("\n".join(table_lines))

    if not sections:
        return "No gene DE results available."

    return "\n\n".join(sections)


@dataclass
class GeneDE:
    """Differential expression result for a single gene."""

    name: str
    mean_target: float
    mean_reference: float
    lfc: float  # log fold change (mean_target - mean_reference in log space)
    pval: float
    n_target: int
    n_reference: int
    frac_non_zero_target: float = 1.0  # Fraction of target cells where gene is expressed (non-zero)
    frac_non_zero_thresh: float = 0.2  # Threshold for considering gene as depleted based on expression fraction
    # Extended metrics (NaN when not computed)
    frac_above_ref_median: float = float("nan")  # fraction of target cells > reference median
    zero_rate_diff: float = float("nan")  # frac_nonzero_target − frac_nonzero_reference (positive = more expressing in target)
    zero_excl_lfc: float = float("nan")  # LFC using only non-zero cells in both groups

    @property
    def is_enriched(self) -> bool:
        """True if gene is significantly upregulated in target."""
        return self.lfc > 0 and self.pval < 0.05

    @property
    def is_depleted(self) -> bool:
        """True if gene is significantly downregulated in target or expressed in too few cells."""
        # Consider depleted if:
        # 1. Significantly downregulated (LFC < 0 and pval < 0.05), OR
        # 2. Expressed in less than frac_non_zero_thresh fraction of target cells
        #    (but only when not significantly enriched — avoid double-counting)
        if self.lfc < 0 and self.pval < 0.05:
            return True
        if self.frac_non_zero_target < self.frac_non_zero_thresh and not self.is_enriched:
            return True
        return False

    @property
    def effect(self) -> str:
        """Effect classification with LFC value."""
        return classify_effect(self.lfc)

    def format_compact(self, include_direction: bool = False) -> str:
        """Format gene with all metrics using compact abbreviations.

        Uses the abbreviations defined in DE_METRICS_LEGEND.
        """
        parts = []
        if include_direction:
            direction = "\u2191" if self.is_enriched else "\u2193" if self.is_depleted else "="
            parts.append(direction)
        parts.append(f"LFC={self.lfc:+.2f}")
        if not math.isnan(self.frac_above_ref_median):
            parts.append(f"FAR={self.frac_above_ref_median:.2f}")
        if not math.isnan(self.zero_rate_diff):
            parts.append(f"DFE={self.zero_rate_diff:+.2f}")
        if not math.isnan(self.zero_excl_lfc):
            parts.append(f"ZE-LFC={self.zero_excl_lfc:+.2f}")
        return f"{self.name} ({' '.join(parts)})"

    def __str__(self) -> str:
        """String representation with name and effect."""
        return f"{self.name} ({self.effect})"


@dataclass
class GenesDEResult:
    """DE results for a set of genes."""

    genes: list[GeneDE]
    n_target: int
    n_reference: int
    genes_not_found: list[str] = field(default_factory=list)
    # Global detection (avg filtered genes detected per cell); NaN when not available
    mean_genes_detected_target: float = float("nan")
    mean_genes_detected_reference: float = float("nan")

    @property
    def n_enriched(self) -> int:
        return sum(1 for g in self.genes if g.is_enriched)

    @property
    def n_depleted(self) -> int:
        return sum(1 for g in self.genes if g.is_depleted)

    @property
    def mean_lfc(self) -> float:
        if not self.genes:
            return 0.0
        return float(np.mean([g.lfc for g in self.genes]))

    def get_gene(self, name: str) -> GeneDE | None:
        """Get DE result for a specific gene."""
        for g in self.genes:
            if g.name == name:
                return g
        return None

    def format_for_prompt(self, cluster_id: str, max_genes: int = 10) -> str:
        """
        Format DE results for prompt inclusion.

        Args:
            cluster_id: Cluster identifier for the header
            max_genes: Maximum number of genes to show per category

        Returns:
            Formatted string with enriched and depleted genes, including all metrics
        """
        if not self.genes:
            return f"**{cluster_id}:** No significant differential expression"

        # Get enriched and depleted genes
        enriched = [g for g in self.genes if g.is_enriched]
        depleted = [g for g in self.genes if g.is_depleted]

        # Sort by absolute LFC (most significant first)
        enriched.sort(key=lambda g: abs(g.lfc), reverse=True)
        depleted.sort(key=lambda g: abs(g.lfc), reverse=True)

        parts = [f"**{cluster_id}:**"]
        if enriched:
            gene_strs = [g.format_compact() for g in enriched[:max_genes]]
            enriched_str = ", ".join(gene_strs)
            if len(enriched) > max_genes:
                enriched_str += "..."
            parts.append(f"Enriched: {enriched_str}")
        if depleted:
            gene_strs = [g.format_compact() for g in depleted[:max_genes]]
            depleted_str = ", ".join(gene_strs)
            if len(depleted) > max_genes:
                depleted_str += "..."
            parts.append(f"Depleted: {depleted_str}")
        if not enriched and not depleted:
            parts.append("No significant differential expression")

        return " ".join(parts)


@dataclass
class SignatureDE:
    """DE results for a gene signature."""

    signature_name: str
    genes: list[GeneDE]
    justification: str

    @property
    def n_enriched(self) -> int:
        return sum(1 for g in self.genes if g.is_enriched)

    @property
    def n_depleted(self) -> int:
        return sum(1 for g in self.genes if g.is_depleted)

    @property
    def mean_lfc(self) -> float:
        if not self.genes:
            return 0.0
        return float(np.mean([g.lfc for g in self.genes]))

    @property
    def summary(self) -> str:
        if not self.genes:
            return f"{self.signature_name}: no genes found"
        enriched = [g.name for g in self.genes if g.is_enriched]
        depleted = [g.name for g in self.genes if g.is_depleted]
        parts = [f"{self.signature_name} ({len(self.genes)} genes)"]
        if enriched:
            parts.append(f"↑ {', '.join(enriched)}")
        if depleted:
            parts.append(f"↓ {', '.join(depleted)}")
        return " | ".join(parts)


@dataclass
class SignaturesDEResult:
    """Complete targeted DE results for signatures."""

    cluster_id: str
    reference_type: str  # "parent", "all_other", "custom", or "insufficient_cells"
    n_target: int
    n_reference: int
    signatures: list[SignatureDE] = field(default_factory=list)

    def to_serializable(self) -> list[dict]:
        """
        Convert results to serializable format (list of dicts).

        Returns:
            List of dictionaries with signature DE results
        """
        return [
            {
                "signature_name": sig.signature_name,
                "justification": sig.justification,
                "n_enriched": sig.n_enriched,
                "n_depleted": sig.n_depleted,
                "mean_lfc": sig.mean_lfc,
                "genes": [
                    {
                        "name": g.name,
                        "lfc": g.lfc,
                        "pval": g.pval,
                        "effect": g.effect,
                        "is_enriched": g.is_enriched,
                        "is_depleted": g.is_depleted,
                        "mean_target": g.mean_target,
                        "mean_reference": g.mean_reference,
                        "frac_above_ref_median": g.frac_above_ref_median,
                        "zero_rate_diff": g.zero_rate_diff,
                        "zero_excl_lfc": g.zero_excl_lfc,
                    }
                    for g in sig.genes
                ],
            }
            for sig in self.signatures
        ]



# ============ LOW-LEVEL API ============


def _ols_de(
    X_target: np.ndarray,
    X_reference: np.ndarray,
    dropout_target: np.ndarray,
    dropout_reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized OLS DE with per-cell dropout rate as covariate.

    Fits ``expression ~ group + dropout_rate`` for all genes simultaneously.
    The design matrix (intercept, group, dropout_rate) is identical across
    genes, so ``(X'X)⁻¹`` is computed once and all coefficients are solved
    with a single ``np.linalg.lstsq`` call — comparable speed to the t-test.

    Dropout rate is provided externally, pre-computed over all filtered genes
    in the dataset — a stable, unbiased per-cell quality metric.

    Args:
        X_target: Expression matrix for target cells (n_target × n_genes).
        X_reference: Expression matrix for reference cells (n_ref × n_genes).
        dropout_target: Per-cell dropout rate for target cells (n_target,).
        dropout_reference: Per-cell dropout rate for reference cells (n_ref,).

    Returns:
        Tuple of (coefs, pvals), each shape (n_genes,).
        coefs: dropout-corrected group effect (LFC proxy).
        pvals: two-sided p-value for the group coefficient.
    """
    from scipy.stats import t as t_dist

    n_target = X_target.shape[0]
    n_ref = X_reference.shape[0]
    n_total = n_target + n_ref

    Y = np.vstack([X_target, X_reference]).astype(np.float64)  # (n_total, n_genes)
    group = np.array([1.0] * n_target + [0.0] * n_ref)
    dropout_rate = np.concatenate([dropout_target, dropout_reference]).astype(np.float64)

    # Design matrix — same for every gene
    X = np.column_stack([np.ones(n_total), group, dropout_rate])  # (n_total, 3)

    # Solve for all genes at once: β shape (3, n_genes)
    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    # Residual standard errors (vectorized)
    residuals = Y - X @ beta                          # (n_total, n_genes)
    rss = np.sum(residuals ** 2, axis=0)              # (n_genes,)
    df_resid = n_total - 3
    s2 = rss / df_resid                               # (n_genes,)

    # SE of group coefficient: sqrt(s² × (X'X)⁻¹[1,1])
    XtXinv = np.linalg.inv(X.T @ X)                  # (3, 3) — computed once
    se_group = np.sqrt(np.maximum(s2 * XtXinv[1, 1], 0.0))  # (n_genes,)

    coefs = beta[1]                                   # group coefficients
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(se_group > 0, coefs / se_group, 0.0)
    pvals = 2 * t_dist.sf(np.abs(t_stats), df=df_resid)

    return coefs, pvals


@log_execution
def compute_de_for_genes(
    accessor: "Accessor",
    gene_names: Sequence[str],
    target_indices: np.ndarray,
    reference_indices: np.ndarray,
    frac_non_zero_thresh: float = 0.2,
    de_method: Literal["ttest", "ols"] = "ttest",
) -> GenesDEResult:
    """
    Compute differential expression for specific genes.

    Low-level function that takes explicit cell indices.
    Use this when you have custom target/reference populations.

    Args:
        accessor: Data accessor
        gene_names: List of gene names to test
        target_indices: Cell indices for target population
        reference_indices: Cell indices for reference population
        frac_non_zero_thresh: Threshold for fraction of non-zero cells to consider gene as depleted (default: 0.2)
        de_method: Statistical test — ``"ttest"`` (default, fast, vectorized Welch's t-test)
                   or ``"ols"`` (OLS with per-cell dropout rate as covariate; requires statsmodels).

    Returns:
        GenesDEResult with DE for each gene
    """
    n_target = len(target_indices)
    n_reference = len(reference_indices)

    if n_target == 0 or n_reference == 0:
        return GenesDEResult(
            genes=[],
            n_target=n_target,
            n_reference=n_reference,
            genes_not_found=list(gene_names),
        )

    gene_names_list = list(gene_names)

    # Fetch expression for target genes
    X_target, found_genes = accessor.get_expression_for_genes(target_indices, gene_names_list)
    X_reference, _ = accessor.get_expression_for_genes(reference_indices, gene_names_list)

    # Track genes not found
    found_set = set(found_genes)
    genes_not_found = [g for g in gene_names_list if g not in found_set]

    if len(found_genes) == 0:
        return GenesDEResult(
            genes=[],
            n_target=n_target,
            n_reference=n_reference,
            genes_not_found=genes_not_found,
        )

    # ========== PER-GENE STATISTICS ==========

    mean_target = np.mean(X_target, axis=0)
    mean_ref = np.mean(X_reference, axis=0)
    frac_non_zero = np.sum(X_target > 0, axis=0) / n_target

    if de_method == "ols":
        # OLS: dropout-corrected effect size as LFC, p-value from group coefficient.
        # Dropout rate is taken from the accessor (pre-computed over all filtered genes)
        # so it reflects global per-cell quality, not just the tested gene slice.
        cell_dropout = accessor.cell_dropout_rate
        lfc, pvals = _ols_de(
            X_target, X_reference,
            dropout_target=cell_dropout[target_indices],
            dropout_reference=cell_dropout[reference_indices],
        )
    else:
        # Welch's t-test (vectorized)
        from scipy.stats import ttest_ind_from_stats

        std_target = np.std(X_target, axis=0, ddof=1)
        std_ref = np.std(X_reference, axis=0, ddof=1)
        lfc = mean_target - mean_ref

        both_zero = (std_target == 0) & (std_ref == 0)
        _, pvals = ttest_ind_from_stats(
            mean_target, std_target, n_target,
            mean_ref, std_ref, n_reference,
            equal_var=False,
        )
        pvals = np.nan_to_num(pvals, nan=1.0)
        pvals[both_zero] = 1.0

    # ========== EXTENDED METRICS ==========
    ext = _compute_extended_metrics(X_target, X_reference)

    # Build results from vectorized arrays
    gene_results = [
        GeneDE(
            name=gene,
            mean_target=float(mean_target[i]),
            mean_reference=float(mean_ref[i]),
            lfc=float(lfc[i]),
            pval=float(pvals[i]),
            n_target=n_target,
            n_reference=n_reference,
            frac_non_zero_target=float(frac_non_zero[i]),
            frac_non_zero_thresh=frac_non_zero_thresh,
            frac_above_ref_median=float(ext["frac_above_ref_median"][i]),
            zero_rate_diff=float(ext["zero_rate_diff"][i]),
            zero_excl_lfc=float(ext["zero_excl_lfc"][i]),
        )
        for i, gene in enumerate(found_genes)
    ]

    return GenesDEResult(
        genes=gene_results,
        n_target=n_target,
        n_reference=n_reference,
        genes_not_found=genes_not_found,
    )


def get_reference_indices(
    accessor: "Accessor",
    cluster_id: str,
    target_indices: np.ndarray,
) -> tuple[np.ndarray, str]:
    """
    Get reference cell indices for a cluster.

    Reference population is:
    - At every depth: all other cells (global reference), matching rank_genes.
    - For parent-relative or peer contrasts, pass explicit reference indices
      to compute_de_for_genes.

    Args:
        accessor: Data accessor with overlay
        cluster_id: Current cluster ID
        target_indices: Target cell indices (to exclude from reference)

    Returns:
        Tuple of (reference_indices, reference_type)
    """
    # Always use all other cells as reference (global reference), consistent with rank_genes
    all_indices = np.arange(accessor.n_cells)
    reference_indices = np.setdiff1d(all_indices, target_indices)
    reference_type = "all_other"

    return reference_indices, reference_type


# ============ HIGH-LEVEL API ============


@log_execution
def targeted_de_signatures(
    accessor: "Accessor",
    cluster_id: str,
    signatures: dict[str, dict],
    min_cells: int = 10,
    frac_non_zero_thresh: float = 0.2,
    de_method: Literal["ttest", "ols"] = "ttest",
) -> SignaturesDEResult:
    """
    Compute differential expression for gene signatures, grouped by signature.

    The target is the cluster's cells and the reference is every other cell,
    matching rank_genes. For other contrasts, call compute_de_for_genes with
    explicit indices.

    Args:
        accessor: Data accessor with overlay
        cluster_id: Current cluster to analyze
        signatures: Dict of signature_name -> {"genes": [...], "justification": "..."}
        min_cells: Minimum cells required in each group
        frac_non_zero_thresh: Threshold for fraction of non-zero cells to consider gene as depleted (default: 0.2)
        de_method: Statistical test — ``"ttest"`` (default) or ``"ols"`` (dropout-corrected).

    Returns:
        SignaturesDEResult with DE for each signature
    """
    if not cluster_id:
        raise ValueError("cluster_id required for targeted DE (use rank_genes for global)")

    target_indices = accessor.get_cell_indices(cluster_id)
    n_target = len(target_indices)

    if n_target < min_cells:
        return SignaturesDEResult(
            cluster_id=cluster_id,
            reference_type="insufficient_cells",
            n_target=n_target,
            n_reference=0,
            signatures=[],
        )

    reference_indices, reference_type = get_reference_indices(accessor, cluster_id, target_indices)
    n_reference = len(reference_indices)

    if n_reference < min_cells:
        return SignaturesDEResult(
            cluster_id=cluster_id,
            reference_type=reference_type,
            n_target=n_target,
            n_reference=n_reference,
            signatures=[],
        )

    # Collect all unique genes across signatures
    all_genes = set()
    for sig_data in signatures.values():
        all_genes.update(sig_data.get("genes", []))

    # Compute DE for all genes at once (efficient)
    # Even if all_genes is empty, we still want to create SignatureDE objects
    # for each signature (with empty gene lists) so they appear in the output
    if all_genes:
        de_result = compute_de_for_genes(
            accessor=accessor,
            gene_names=list(all_genes),
            target_indices=target_indices,
            reference_indices=reference_indices,
            frac_non_zero_thresh=frac_non_zero_thresh,
            de_method=de_method,
        )
    else:
        # Create empty DE result when no genes to test
        de_result = GenesDEResult(
            genes=[],
            n_target=n_target,
            n_reference=n_reference,
            genes_not_found=[],
        )

    # Organize results by signature
    signature_results = []
    for sig_name, sig_data in signatures.items():
        sig_genes = sig_data.get("genes", [])
        justification = sig_data.get("justification", "")

        # Get DE results for this signature's genes
        gene_results = []
        for gene in sig_genes:
            gene_de = de_result.get_gene(gene)
            if gene_de is not None:
                gene_results.append(gene_de)

        signature_results.append(
            SignatureDE(
                signature_name=sig_name,
                genes=gene_results,
                justification=justification,
            )
        )

    return SignaturesDEResult(
        cluster_id=cluster_id,
        reference_type=reference_type,
        n_target=n_target,
        n_reference=n_reference,
        signatures=signature_results,
    )


# Alias for backward compatibility
targeted_de = targeted_de_signatures


@dataclass
class PathologyDEResult:
    """DE results for pathology: both enriched and depleted genes vs healthy reference."""

    cluster_id: str
    cluster_cell_type: str
    n_target: int
    n_reference: int
    enriched_genes: list[GeneDE] = field(default_factory=list)
    depleted_genes: list[GeneDE] = field(default_factory=list)
    # Global detection (avg filtered genes detected per cell)
    mean_genes_detected_target: float = float("nan")
    mean_genes_detected_reference: float = float("nan")

    def format_for_prompt(self) -> str:
        """Format enriched + depleted genes for LLM prompt with full metrics."""
        header_parts = [DE_METRICS_LEGEND]
        if not math.isnan(self.mean_genes_detected_target):
            header_parts.append(
                f"Detection: {self.mean_genes_detected_target:.0f} genes/cell (target) "
                f"vs {self.mean_genes_detected_reference:.0f} (ref)"
            )

        gene_parts = []
        if self.enriched_genes:
            enriched_str = ", ".join(g.format_compact() for g in self.enriched_genes)
            gene_parts.append(
                f"**Upregulated in disease ({len(self.enriched_genes)} genes):** {enriched_str}"
            )
        if self.depleted_genes:
            depleted_str = ", ".join(g.format_compact() for g in self.depleted_genes)
            gene_parts.append(
                f"**Downregulated in disease ({len(self.depleted_genes)} genes):** {depleted_str}"
            )
        if not gene_parts:
            gene_parts.append("No significant differential expression detected.")

        return "\n".join(header_parts + gene_parts)


@log_execution
def compute_pathology_de_from_indices(
    accessor: "Accessor",
    target_indices: np.ndarray,
    reference_indices: np.ndarray,
    cell_type_label: str,
    n_top_genes: int = 20,
    min_cells: int = 10,
    alpha: float = 0.05,
    min_lfc: float = 0.25,
) -> PathologyDEResult:
    """
    Compute pathology DE given explicit target and reference indices.

    Used by the spatial pipeline and CLI for niche-vs-rest contrasts.

    Args:
        accessor: Data accessor
        target_indices: Cell indices for pathological cells
        reference_indices: Cell indices for healthy reference cells
        cell_type_label: Cell type name (for result labeling)
        n_top_genes: Number of top genes per direction
        min_cells: Minimum cells in each group
        alpha: FDR significance threshold
        min_lfc: Minimum absolute LFC threshold

    Returns:
        PathologyDEResult with top enriched and depleted genes
    """
    from scipy.stats import ttest_ind_from_stats
    from statsmodels.stats.multitest import multipletests

    n_target = len(target_indices)
    n_reference = len(reference_indices)

    if n_target < min_cells or n_reference < min_cells:
        return PathologyDEResult(
            cluster_id="",
            cluster_cell_type=cell_type_label,
            n_target=n_target,
            n_reference=n_reference,
        )

    gene_filter_indices = accessor.gene_filter_indices
    cluster_stats = accessor.compute_cluster_stats(
        target_indices, gene_indices=gene_filter_indices, return_detection=True,
    )
    ref_stats = accessor.compute_cluster_stats(
        reference_indices, gene_indices=gene_filter_indices, return_detection=True,
    )

    mean_target = cluster_stats["mean"]
    var_target = cluster_stats["var"]
    mean_ref = ref_stats["mean"]
    var_ref = ref_stats["var"]
    mean_genes_detected_target = cluster_stats["mean_genes_detected"]
    mean_genes_detected_ref = ref_stats["mean_genes_detected"]

    scores, p_values = ttest_ind_from_stats(
        mean1=mean_target,
        std1=np.sqrt(var_target),
        nobs1=n_target,
        mean2=mean_ref,
        std2=np.sqrt(var_ref),
        nobs2=n_reference,
        equal_var=False,
    )
    p_values = np.nan_to_num(p_values, nan=1.0)
    scores = np.nan_to_num(scores, nan=0.0)

    lfcs = mean_target - mean_ref

    pvals_adj = multipletests(p_values, alpha=alpha, method="fdr_bh")[1]

    significant_mask = pvals_adj < alpha
    gene_names = accessor.gene_names

    enriched_mask = significant_mask & (lfcs > min_lfc)
    enriched_indices = np.where(enriched_mask)[0]
    enriched_sorted = enriched_indices[np.argsort(lfcs[enriched_indices])[::-1]]

    depleted_mask = significant_mask & (lfcs < -min_lfc)
    depleted_indices = np.where(depleted_mask)[0]
    depleted_sorted = depleted_indices[np.argsort(lfcs[depleted_indices])]

    # Compute extended metrics for top genes
    top_enriched = enriched_sorted[:n_top_genes]
    top_depleted = depleted_sorted[:n_top_genes]
    all_top_local = np.concatenate([top_enriched, top_depleted])

    ext_metrics_map: dict[int, dict] = {}
    if len(all_top_local) > 0:
        top_gene_names = [gene_names[gene_filter_indices[idx]] for idx in all_top_local]
        X_top_target, found_top = accessor.get_expression_for_genes(target_indices, top_gene_names)
        X_top_ref, _ = accessor.get_expression_for_genes(reference_indices, top_gene_names)

        if len(found_top) > 0:
            ext = _compute_extended_metrics(X_top_target, X_top_ref)
            name_to_ext_idx = {name: i for i, name in enumerate(found_top)}
            for local_idx in all_top_local:
                gname = gene_names[gene_filter_indices[local_idx]]
                ext_idx = name_to_ext_idx.get(gname)
                if ext_idx is not None:
                    ext_metrics_map[local_idx] = {
                        "frac_non_zero_target": float(ext["frac_non_zero_target"][ext_idx]),
                        "frac_above_ref_median": float(ext["frac_above_ref_median"][ext_idx]),
                        "zero_rate_diff": float(ext["zero_rate_diff"][ext_idx]),
                        "zero_excl_lfc": float(ext["zero_excl_lfc"][ext_idx]),
                    }

    def _build_gene_de(idx: int) -> GeneDE:
        global_idx = gene_filter_indices[idx]
        em = ext_metrics_map.get(idx, {})
        return GeneDE(
            name=gene_names[global_idx],
            mean_target=float(mean_target[idx]),
            mean_reference=float(mean_ref[idx]),
            lfc=float(lfcs[idx]),
            pval=float(pvals_adj[idx]),
            n_target=n_target,
            n_reference=n_reference,
            # Real fraction of niche cells expressing the gene (was hard-defaulted
            # to 1.0, making the "Frac Expressed" column a constant 1.00).
            frac_non_zero_target=em.get("frac_non_zero_target", float("nan")),
            frac_above_ref_median=em.get("frac_above_ref_median", float("nan")),
            zero_rate_diff=em.get("zero_rate_diff", float("nan")),
            zero_excl_lfc=em.get("zero_excl_lfc", float("nan")),
        )

    enriched_genes = [_build_gene_de(idx) for idx in top_enriched]
    depleted_genes = [_build_gene_de(idx) for idx in top_depleted]

    return PathologyDEResult(
        cluster_id="",
        cluster_cell_type=cell_type_label,
        n_target=n_target,
        n_reference=n_reference,
        enriched_genes=enriched_genes,
        depleted_genes=depleted_genes,
        mean_genes_detected_target=mean_genes_detected_target,
        mean_genes_detected_reference=mean_genes_detected_ref,
    )
