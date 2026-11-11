"""
Data formatting utilities for spatial annotation pipeline prompts.

Formats spatial proportions with Moran's z-score, per-cell-type DE with gene
spatial coherence scores, spatial hypotheses, and signature DE results.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sclambda.analysis.targeted_de import SignaturesDEResult


def format_spatial_proportions(proportions: list[dict], k: int = 10) -> str:
    """
    Format cell type proportions with Moran's z-score for prompt inclusion.

    Shows top *k* cell types ranked by absolute proportion.

    Args:
        proportions: List of dicts with keys: cell_type, n_cells, proportion, morans_z
        k: Number of top cell types to show.

    Returns:
        Formatted table string
    """
    if not proportions:
        return "No cell type proportion data available."

    ranked = sorted(proportions, key=lambda p: p["proportion"], reverse=True)
    top = ranked[:k]

    lines = [
        f"Top {len(top)} cell types by abundance (of {len(proportions)} total):",
        "| Cell Type | Proportion | Moran's Z |",
        "|-----------|-----------|-----------|",
    ]
    for p in top:
        mz = "unavailable" if p.get("morans_z") is None else f"{p['morans_z']:.2f}"
        lines.append(
            f"| {p['cell_type']} | {p['proportion']*100:.1f}% | {mz} |"
        )
    return "\n".join(lines)


def format_spatial_celltype_de(
    per_celltype_de: dict,
    gene_spatial_scores: dict | None = None,
    max_genes_per_direction: int | None = None,
    detailed: bool = True,
    enriched_only: bool = False,
) -> str:
    """
    Format per-cell-type DE results with gene spatial coherence scores.

    Args:
        per_celltype_de: Dict mapping cell_type -> PathologyDEResult
        gene_spatial_scores: Dict mapping cell_type -> {gene_name: morans_z}
        max_genes_per_direction: Max genes to show per direction per cell type.
            None (default) shows all genes.
        detailed: If True, show binned DE table with spatial column.
                  If False, show gene names grouped by direction.
        enriched_only: If True, only include enriched genes (no depleted).

    Returns:
        Combined formatted DE summary across all cell types
    """
    if not per_celltype_de:
        return "No differential expression data available."

    if detailed:
        from sclambda.analysis.targeted_de import format_genes_as_gene_table

        rows = []
        spatial_for_table = {} if gene_spatial_scores is not None else None
        for cell_type, de_result in per_celltype_de.items():
            enriched = de_result.enriched_genes[:max_genes_per_direction] if max_genes_per_direction and de_result.enriched_genes else (de_result.enriched_genes or [])
            if enriched_only:
                genes = list(enriched)
            else:
                depleted = de_result.depleted_genes[:max_genes_per_direction] if max_genes_per_direction and de_result.depleted_genes else (de_result.depleted_genes or [])
                genes = list(enriched) + list(depleted)
            if genes:
                rows.append((cell_type, genes))
                if spatial_for_table is not None:
                    spatial_for_table[cell_type] = (gene_spatial_scores or {}).get(cell_type, {})

        if not rows:
            return "No differential expression data available."

        return format_genes_as_gene_table(rows, spatial_scores=spatial_for_table)

    sections = []
    for cell_type, de_result in per_celltype_de.items():
        section_lines = [f"### {cell_type}"]

        if de_result.enriched_genes:
            genes = de_result.enriched_genes[:max_genes_per_direction] if max_genes_per_direction else de_result.enriched_genes
            gene_str = ", ".join(g.name for g in genes)
            section_lines.append(f"**Upregulated in niche ({len(de_result.enriched_genes)} genes):** {gene_str}")

        if not enriched_only and de_result.depleted_genes:
            genes = de_result.depleted_genes[:max_genes_per_direction] if max_genes_per_direction else de_result.depleted_genes
            gene_str = ", ".join(g.name for g in genes)
            section_lines.append(f"**Downregulated in niche ({len(de_result.depleted_genes)} genes):** {gene_str}")

        if not de_result.enriched_genes and (enriched_only or not de_result.depleted_genes):
            section_lines.append("No significant differential expression detected.")

        sections.append("\n".join(section_lines))

    return "\n\n".join(sections)


def format_spatial_hypotheses(
    hypotheses: list[dict],
    max_justified: int | None = None,
) -> str:
    """
    Format spatial annotation hypotheses for prompts.

    Uses the ``justification`` field (structured paragraph) when available;
    falls back to ``supporting_evidence`` list + ``reasoning`` for backward
    compatibility.

    Args:
        hypotheses: List of hypothesis dicts with keys:
            - annotation: str
            - confidence: str (high/medium/low)
            - justification: str (optional, preferred)
            - supporting_evidence: list[dict] (optional, legacy fallback)
            - reasoning: str (optional, legacy fallback)
            - pathways: list[str] (optional)
        max_justified: If set, only the first N hypotheses get full
            justification; the rest are listed as name + confidence only.

    Returns:
        Formatted hypotheses string (one per line)
    """
    if not hypotheses:
        return "No spatial annotation hypotheses available."

    lines = []
    for idx, h in enumerate(hypotheses):
        annotation = h.get("annotation", "unknown")
        confidence = h.get("confidence", "unknown")
        justification = h.get("justification", "")

        line = f"- **{annotation}** (confidence: {confidence})"

        # If max_justified is set and we're past the limit, skip justification
        if max_justified is not None and idx >= max_justified:
            lines.append(line)
            continue

        if justification:
            line += f"\n  Justification: {justification}"
        else:
            # Legacy fallback: supporting_evidence + reasoning
            evidence = h.get("supporting_evidence", [])
            pathways = h.get("pathways", [])
            reasoning = h.get("reasoning", "N/A")

            if evidence:
                for ev in evidence:
                    ct = ev.get("cell_type", "unknown")
                    genes = ev.get("genes", [])
                    direction = ev.get("direction", "")
                    spatial_note = ev.get("spatial_pattern", "")
                    dir_str = f" [{direction}]" if direction else ""
                    sp_str = f" ({spatial_note})" if spatial_note else ""
                    if genes:
                        line += f"\n  {ct}{dir_str}{sp_str}: {', '.join(genes)}"

            if pathways:
                line += f"\n  Pathways: {', '.join(pathways)}"

            if reasoning:
                line += f"\n  Reasoning: {reasoning}"

        lines.append(line)

    return "\n".join(lines)


class SpatialSignatureFormatter:
    """Format signature DE results with spatial scores for reevaluation prompts."""

    def __init__(
        self,
        refinement_signatures: dict,
        signature_de_by_celltype: dict[str, "SignaturesDEResult"],
        signature_spatial_scores: dict[str, dict] | None = None,
    ):
        self.refinement_signatures = refinement_signatures or {}
        self.signature_de_by_celltype = signature_de_by_celltype or {}
        self.signature_spatial_scores = signature_spatial_scores or {}

    def format(self) -> str:
        """Format all signature DE results with justifications and a (cell type, gene) table."""
        from sclambda.analysis.targeted_de import format_genes_as_gene_table

        if not self.signature_de_by_celltype:
            return "No signature evidence available."

        rows = []
        sig_spatial: dict[str, dict] = {}
        justification_lines = []
        for cell_type, sde_result in self.signature_de_by_celltype.items():
            ct_scores = self.signature_spatial_scores.get(cell_type, {})
            for sig in sde_result.signatures:
                label = f"{cell_type}: {sig.signature_name}"
                rows.append((label, sig.genes))
                sig_spatial[label] = ct_scores
                if sig.justification:
                    justification_lines.append(f"- **{label}**: {sig.justification}")

        if not rows:
            return "No signature evidence available."

        parts = []
        if justification_lines:
            parts.append("**Signature justifications (what each signature was designed to test):**\n" + "\n".join(justification_lines))
        parts.append(format_genes_as_gene_table(rows, spatial_scores=sig_spatial))
        return "\n\n".join(parts)
