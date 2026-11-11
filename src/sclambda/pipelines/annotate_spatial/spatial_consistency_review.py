"""Per-slide global observations; only the main critic may revise identities."""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re

import numpy as np

logger = logging.getLogger(__name__)

_PROMPT_FILE = os.path.join(
    os.path.dirname(__file__), "prompts", "14_spatial_consistency_review.md"
)


def _parse_json_array(text: str) -> list[dict]:
    """Extract one complete array, never regenerate a biological assessment.

    Prose around a single array is harmless. Multiple or incomplete arrays
    are ambiguous and fail this optional review; no model repair is requested.
    """
    try:
        data = json.loads(text.strip())
    except (ValueError, TypeError):
        arrays = []
        position = 0
        decoder = json.JSONDecoder()
        while (start := text.find("[", position)) >= 0:
            try:
                value, end = decoder.raw_decode(text[start:])
            except ValueError as exc:
                raise ValueError("Global review contains an incomplete or ambiguous array") from exc
            arrays.append(value)
            position = start + end
        if len(arrays) != 1:
            raise ValueError("Global review needs exactly one complete JSON array")
        data = arrays[0]
    if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
        raise ValueError("Global observations must be a JSON list of objects")
    return data


def _review_slide(
    accessor,
    llm_client,
    results: list[dict],
    adjacency: dict[str, list[str]],
    *,
    organism: str = "human",
    tissue: str = "unknown",
    spatial_key: str = "spatial",
    slice_key: str | None = None,
    image_px: int = 1500,
    flip: bool = False,
    slide: str | None = None,
) -> dict[str, dict]:
    """Review all standardized labels against the global tissue map.

    Returns ``{cluster_id: {"sub_region", "major_region", "reason"}}`` for the
    niches the model recommends changing (consistent ones are omitted).
    """
    from sclambda.analysis.spatial_visualization import generate_niche_image_base64

    by_id = {r.get("cluster_id"): r for r in results}
    niche_to_major = {cid: (r.get("major_region") or "") for cid, r in by_id.items()}
    niche_to_sub = {cid: (r.get("annotation") or "") for cid, r in by_id.items()}

    idx = np.arange(accessor.n_cells)
    if slice_key and slide is not None:
        s = accessor.get_obs([slice_key], idx)[slice_key].values.astype(str)
        idx = np.where(s == str(slide))[0]
    coords = np.asarray(accessor.get_obsm(spatial_key, idx))[:, :2]
    l1 = accessor._level_column(1)
    niche_cell = accessor._overlay.iloc[idx][l1].values.astype(str)
    # Neutral niche-ID map: predicted anatomy remains in the revisable roster,
    # rather than being painted into the image as apparent visual evidence.
    try:
        image_b64 = generate_niche_image_base64(
            coords=coords,
            niche_labels=niche_cell,
            number_niches=True,
            target_niche="\x00__no_target__",
            background_labels=None,
            image_px=image_px,
            flip=flip,
        )
    except Exception as e:
        raise RuntimeError("Global review image could not be rendered") from e

    # Position is inferred by the model from the whole-section image + each
    # niche's neighbour compartments (below) — both tissue-agnostic. We do NOT
    # compute a geometric "outerness" / radial proxy: notions like inner vs outer
    # are specific to hollow / layered organs and don't generalize, so the model
    # reads position from the tissue it actually sees, not an injected assumption.

    # Roster: only niches present on this slide (visible on the map).
    present = [cid for cid in by_id if (niche_cell == cid).any()]
    roster_lines = []
    for cid in present:
        nbrs = [niche_to_major.get(n, "") for n in adjacency.get(cid, [])]
        nbrs = [n for n in nbrs if n]
        roster_lines.append(
            f"- niche {cid}: sub_region=\"{niche_to_sub.get(cid,'')}\" | "
            f"major_region=\"{niche_to_major.get(cid,'')}\" | "
            f"annotated neighbour major_regions=[{', '.join(sorted(set(nbrs)))}]; "
            f"unannotated neighbour IDs={[n for n in adjacency.get(cid, []) if not niche_to_major.get(n)]}; "
            "an empty annotated list does not establish spatial isolation"
        )
    roster = "\n".join(roster_lines)

    with open(_PROMPT_FILE) as f:
        template = f.read()
    prompt = template.format(organism=organism, tissue=tissue, roster=roster)

    from sclambda.pipelines.celltype.nodes.agent import CTDossierAgentNode
    response = llm_client.query(prompt, power="medium", image_base64=image_b64)
    if getattr(response, "failed_after_retries", False):
        raise RuntimeError("Global observation call failed")
    if CTDossierAgentNode._truncated(response):
        raise RuntimeError("Global observation response truncated")
    parsed = _parse_json_array(response.content)
    changes: dict[str, dict] = {}
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("Global observation must be an object")
        cid = str(item.get("niche", "")).strip()
        if cid not in present:
            continue
        new_sub = str(item.get("corrected_sub_region", "")).strip()
        new_major = str(item.get("corrected_major_region", "")).strip()
        if not new_sub and not new_major:
            continue
        from sclambda.annotation.niche.measurements import observation_record
        evidence = observation_record(cid, str(item.get("reason", "")),
            {"slice_key": slice_key, "spatial_slide": slide}, kind="global_review",
            views=[{"slide": slide, "mode": "whole_neutral_niche_ids", "sha256": hashlib.sha256(image_b64.encode()).hexdigest()}])
        evidence["images"] = [image_b64]
        changes[cid] = {
            "evidence": evidence,
            "sub_region": new_sub or niche_to_sub.get(cid, ""),
            "major_region": new_major or niche_to_major.get(cid, ""),
            "reason": str(item.get("reason", "")).strip(),
        }

    if changes:
        logger.info("Spatial-consistency review: %d label(s) corrected", len(changes))
    return changes


def review_spatial_consistency(accessor, llm_client, results, adjacency=None, **kwargs):
    """Return proposals per slide. No labels are written by this observer."""
    from sclambda.analysis.spatial_metrics import compute_niche_adjacency
    slice_key = kwargs.get("slice_key")
    indices = np.arange(accessor.n_cells)
    slides = (accessor.get_obs([slice_key], indices)[slice_key].astype(str).values
              if slice_key else np.full(len(indices), "single"))
    proposals = {}
    requested = kwargs.pop("selected_slides", None)
    for slide in (requested if requested is not None else sorted(set(slides))):
        idx = indices[slides == str(slide)] if slice_key else indices
        if not len(idx):
            continue
        coords = np.asarray(accessor.get_obsm(kwargs.get("spatial_key", "spatial"), idx))[:, :2]
        labels = accessor._overlay.iloc[idx][accessor._level_column(1)].astype(str).values
        neighbours = compute_niche_adjacency(coords, labels)
        changes = _review_slide(accessor, llm_client, results, neighbours,
                               slide=slide if slice_key else None, **kwargs)
        for cid, proposal in changes.items():
            proposals.setdefault(cid, []).append(proposal)
    return proposals
