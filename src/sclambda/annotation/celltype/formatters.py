"""
Data formatting utilities for cell-typing prompts.

Provides consistent formatting for:
- Gene lists
- Cell type hypotheses
- Refinement signatures
- DE results
- Subcluster markers

Eliminates manual string formatting scattered throughout nodes.
"""

from __future__ import annotations
from typing import Any
import math


def format_gene_list(
    genes: list[str],
    numbered: bool = True,
    max_genes: int | None = None,
    separator: str = ", ",
) -> str:
    """
    Format gene list for prompts.

    Args:
        genes: List of gene names
        numbered: Whether to number genes (e.g., "1. GENE1, 2. GENE2")
        max_genes: Maximum genes to include (None = all)
        separator: Separator between genes

    Returns:
        Formatted gene string

    Examples:
        >>> format_gene_list(["CD3D", "CD8A"], numbered=True)
        "1. CD3D, 2. CD8A"
        >>> format_gene_list(["CD3D", "CD8A"], numbered=False)
        "CD3D, CD8A"
    """
    if not genes:
        return "No genes"

    genes_to_format = genes[:max_genes] if max_genes else genes

    if numbered:
        formatted = separator.join(f"{i}. {gene}" for i, gene in enumerate(genes_to_format, 1))
    else:
        formatted = separator.join(genes_to_format)

    return formatted


def format_top_genes_de_table(rank_genes_result) -> str:
    """
    Format top enriched genes with full DE statistics for hypothesis generation.

    Args:
        rank_genes_result: RankGenesResult object with ranked genes
            (each gene has name, lfc, effect, pval_adj, score, frac_non_zero)

    Returns:
        Markdown table with gene name, LFC, fraction non-zero, and effect classification
    """
    if rank_genes_result is None or not rank_genes_result.genes:
        return "No DE data available"

    header = "| # | Gene | LFC | Frac Non-Zero | Adjusted p |"
    sep = "|---|------|-----|---------------|------------|"
    lines = [f"Reference: {getattr(rank_genes_result, 'reference', 'unspecified legacy reference')}", header, sep]
    for i, gene in enumerate(rank_genes_result.genes, 1):
        lines.append(
            f"| {i} | {gene.name} | {gene.lfc:+.6g} | {gene.frac_non_zero:.6g} | {gene.pval_adj:.6g} |"
        )
    return "\n".join(lines)


def format_hypotheses(
    hypotheses: list[dict],
    include_genes: bool = True,
    include_reasoning: bool = True,
) -> str:
    """
    Format cell type hypotheses for prompts.

    Uses the ``justification`` field (structured paragraph) when available;
    falls back to ``supporting_genes`` list + ``reasoning`` for backward
    compatibility.

    Args:
        hypotheses: List of hypothesis dicts with keys:
            - cell_type: str
            - confidence: str (high/medium/low)
            - justification: str (optional, preferred)
            - supporting_genes: list[str] (optional, legacy fallback)
            - reasoning: str (optional, legacy fallback)
        include_genes: Whether to include supporting genes (when no justification)
        include_reasoning: Whether to include reasoning (when no justification)

    Returns:
        Formatted hypotheses string (one per line)
    """
    if not hypotheses:
        return "No hypotheses available"

    lines = []
    for h in hypotheses:
        cell_type = h.get("cell_type", "unknown")
        confidence = h.get("confidence", "unknown")
        justification = h.get("justification", "")

        # Main line
        line = f"- **{cell_type}** (confidence: {confidence})"

        if justification:
            line += f"\n  Justification: {justification}"
        else:
            # Legacy fallback: supporting_genes + reasoning
            supporting_genes = h.get("supporting_genes", [])
            reasoning = h.get("reasoning", "N/A")

            if include_genes and supporting_genes:
                genes_str = ", ".join(supporting_genes)
                line += f"\n  Supporting genes: {genes_str}"

            if include_reasoning and reasoning:
                line += f"\n  Reasoning: {reasoning}"

        lines.append(line)

    return "\n".join(lines)


def format_signatures(signatures: dict) -> str:
    """
    Format refinement signatures for prompts.

    Args:
        signatures: Dict mapping signature name -> signature info:
            {
                "signature_name": {
                    "genes": ["GENE1", "GENE2", ...],
                    "justification": "Why these genes..."
                }
            }

    Returns:
        Formatted signatures string

    Example:
        >>> sigs = {
        ...     "T_cell_markers": {
        ...         "genes": ["CD3D", "CD8A"],
        ...         "justification": "Canonical T cell markers"
        ...     }
        ... }
        >>> print(format_signatures(sigs))
        **T_cell_markers** (Canonical T cell markers): CD3D, CD8A
    """
    if not signatures:
        return "No refinement signatures provided"

    lines = []
    for sig_name, sig_data in signatures.items():
        genes = sig_data.get("genes", [])
        justification = sig_data.get("justification", "N/A")

        gene_list = ", ".join(genes) if genes else "(No genes specified)"
        lines.append(f"**{sig_name}** ({justification}): {gene_list}")

    return "\n\n".join(lines)


def format_subcluster_markers(markers_dict: dict[str, list[str]]) -> str:
    """
    Format subcluster markers for prompts.

    Args:
        markers_dict: Dict mapping cluster_id -> list of gene names

    Returns:
        Formatted markers string

    Example:
        >>> markers = {"cluster_0": ["CD3D", "CD8A"], "cluster_1": ["CD79A"]}
        >>> print(format_subcluster_markers(markers))
        - **cluster_0**: CD3D, CD8A
        - **cluster_1**: CD79A
    """
    if not markers_dict:
        return "No markers available"

    lines = []
    for cluster_id, markers in markers_dict.items():
        markers_str = ", ".join(markers) if markers else "No markers"
        lines.append(f"- **{cluster_id}**: {markers_str}")

    return "\n".join(lines)


class SignatureFormatter:
    """
    Formatter for signature DE results in ct_reevaluation prompts.

    Formats DE results with full metrics (LFC, Frac Expressing, Delta Frac) for each gene.
    Handles three data sources in priority order:
    1. Live SignaturesDEResult Python object
    2. Serialized list[dict] fallback (after state transport)
    3. Raw signatures without DE (DE analysis pending)
    """

    def __init__(
        self,
        refinement_signatures: dict | None = None,
        signature_de_result_obj=None,
        signature_de_results: list[dict] | None = None,
        signature_de_results_present: bool = False,
    ):
        self.refinement_signatures = refinement_signatures or {}
        self.signature_de_result_obj = signature_de_result_obj
        self.signature_de_results = signature_de_results or []
        self.signature_de_results_present = signature_de_results_present

    @staticmethod
    def _to_float(value: Any, default: float) -> float:
        try:
            if value is None:
                return default
            v = float(value)
            if math.isnan(v):
                return default
            return v
        except (TypeError, ValueError):
            return default

    def _build_rows_from_signatures(self, signatures) -> list[tuple[str, list]]:
        """Build (label, genes) rows from SignatureDE objects."""
        rows = []
        for sig in signatures:
            rows.append((sig.signature_name, sig.genes))
        return rows

    def _format_justification_header(self, signatures) -> str | None:
        """Build justification header from signatures that have justifications."""
        justification_lines = []
        for sig in signatures:
            justification = getattr(sig, "justification", None) or ""
            if justification:
                justification_lines.append(f"- **{sig.signature_name}**: {justification}")
        if justification_lines:
            return (
                "**Signature justifications (what each signature was designed to test):**\n"
                + "\n".join(justification_lines)
            )
        return None

    def format(self) -> str:
        """Format signatures as a DE table with one row per gene."""
        from sclambda.analysis.targeted_de import (
            GeneDE, SignatureDE, format_de_as_table,
        )

        # Preferred path: full SignaturesDEResult object in state.
        if self.signature_de_result_obj and self.signature_de_result_obj.signatures:
            rows = self._build_rows_from_signatures(self.signature_de_result_obj.signatures)
            parts = []
            justification_header = self._format_justification_header(self.signature_de_result_obj.signatures)
            if justification_header:
                parts.append(justification_header)
            parts.append(format_de_as_table(rows))
            return "\n\n".join(parts)

        # Fallback: serialized results (object may be unavailable after state transport).
        if self.signature_de_results:
            signatures = []
            for sig in self.signature_de_results:
                sig_name = str(sig.get("signature_name", "signature"))
                justification = str(sig.get("justification") or "")

                genes = []
                for gene in sig.get("genes", []) or []:
                    name = gene.get("name")
                    if not name:
                        continue
                    genes.append(GeneDE(
                        name=str(name),
                        mean_target=self._to_float(gene.get("mean_target"), 0.0),
                        mean_reference=self._to_float(gene.get("mean_reference"), 0.0),
                        lfc=self._to_float(gene.get("lfc"), 0.0),
                        pval=self._to_float(gene.get("pval"), 1.0),
                        n_target=0,
                        n_reference=0,
                        frac_non_zero_target=self._to_float(gene.get("frac_non_zero_target"), 1.0),
                        frac_above_ref_median=self._to_float(gene.get("frac_above_ref_median"), float("nan")),
                        zero_rate_diff=self._to_float(gene.get("zero_rate_diff"), float("nan")),
                        zero_excl_lfc=self._to_float(gene.get("zero_excl_lfc"), float("nan")),
                    ))
                signatures.append(SignatureDE(
                    signature_name=sig_name,
                    genes=genes,
                    justification=justification,
                ))

            if signatures:
                rows = self._build_rows_from_signatures(signatures)
                parts = []
                justification_header = self._format_justification_header(signatures)
                if justification_header:
                    parts.append(justification_header)
                parts.append(format_de_as_table(rows))
                return "\n\n".join(parts)

        # Signature DE ran but yielded no rows (e.g. insufficient cells/reference).
        if self.signature_de_result_obj is not None:
            reference_type = getattr(self.signature_de_result_obj, "reference_type", "unknown")
            n_target = getattr(self.signature_de_result_obj, "n_target", "unknown")
            n_reference = getattr(self.signature_de_result_obj, "n_reference", "unknown")
            return (
                "No signature DE results available "
                f"(reference type: {reference_type}, n_target: {n_target}, n_reference: {n_reference})."
            )

        if self.signature_de_results_present and not self.signature_de_results:
            return "No signature DE results available."

        if self.refinement_signatures:
            lines = []
            for sig_name, sig_data in self.refinement_signatures.items():
                genes = sig_data.get("genes", [])
                justification = sig_data.get("justification", "N/A")
                gene_list = ", ".join(genes) if genes else "(No genes specified)"
                lines.append(f"**{sig_name}** ({justification}): {gene_list} (DE analysis pending)")
            return "\n\n".join(lines)

        return "No signature evidence available."
