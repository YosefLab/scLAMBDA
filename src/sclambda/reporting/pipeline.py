"""File exports and prompt histories for the Python pipeline adapter."""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

class PipelineReports:
    def _get_usage_events(self):
        if hasattr(self, "_usage_events"):
            return self._usage_events
        from sclambda.llm.usage import UsageLedger
        ledger = getattr(getattr(self, "llm_client", None), "usage_ledger", None)
        return ledger.snapshot(getattr(self, "_usage_start", 0)) if isinstance(ledger, UsageLedger) else None

    def _write_usage_report(self, output_path, kwargs):
        from pathlib import Path
        from .usage import write_usage
        events = self._get_usage_events()
        if events is None or kwargs.get("usage_path") is False:
            return None
        base = Path(kwargs.get("excel_path") or output_path or f"{self.name}.xlsx")
        path = kwargs.get("usage_path") or base.with_name(base.stem + "_usage.json")
        return write_usage(path, events)

    def write_outputs(self, accessor, output_path=None, **kwargs) -> list:
        """Write pipeline results to file(s).

        Produces two outputs automatically for every pipeline:

        1. **Annotation files** – an Excel workbook with one sheet per
           annotation depth (cluster IDs × cell_type / condition /
           annotation, justification, Cell Ontology term) plus an optional
           CSV/TSV with per-cell annotations when ``output_path`` is given.

        2. **Prompt-history Markdown** – a hierarchical markdown document
           containing every recorded LLM attempt, including parse and actor/critic
           prompts, with provider-reported usage. Legacy saved states without a
           request ledger retain their older, incomplete text-only history.
           Written alongside the annotation Excel by default; pass
           ``prompts_path=False`` to disable.

        Args:
            accessor: Data accessor containing overlay and annotation data.
            output_path: If provided, write per-cell annotations to this
                CSV/TSV path.
            **kwargs:
                excel_path: Path for the Excel annotation file (default:
                    ``"lambda_annotations.xlsx"``).
                prompts_path: Explicit path for the prompt-history markdown.
                    Pass ``False`` to skip writing prompt history.
                    If omitted, derived from ``excel_path`` automatically.
                usage_path: JSON request ledger and usage summary destination;
                    defaults to a ``*_usage.json`` sidecar. False disables export.

        Returns:
            List of Paths to files written.
        """
        written = []
        written.extend(self._write_annotation_files(accessor, output_path, **kwargs))
        usage_path = self._write_usage_report(output_path, kwargs)
        if usage_path is not None:
            written.append(usage_path)

        prompts_path = self._resolve_prompts_path(output_path, kwargs)
        if prompts_path is not None:
            p = self._write_prompt_history(prompts_path)
            if p is not None:
                written.append(p)

        return written

    def _resolve_prompts_path(self, output_path, kwargs) -> "Path | None":
        """Derive the prompt-history markdown path from available path hints."""
        from pathlib import Path

        prompts_path = kwargs.get("prompts_path")
        if prompts_path is False:
            return None
        if prompts_path:
            return Path(prompts_path)

        excel_path = kwargs.get("excel_path")
        if excel_path:
            p = Path(excel_path)
            return p.parent / (p.stem + "_prompts.md")
        if output_path:
            p = Path(output_path)
            return p.parent / (p.stem + "_prompts.md")
        return Path(f"{self.name}_prompts.md")

    def _write_annotation_files(self, accessor, output_path=None, **kwargs) -> list:
        """Write annotation Excel and optional CSV/TSV.

        Reads from ``accessor.overlay`` and the annotation/cluster columns
        stored there by the pipeline nodes.  Works for all built-in
        pipelines because they all write their results into the overlay.

        Returns:
            List of Paths written.
        """
        from pathlib import Path
        import pandas as pd

        excel_path = kwargs.get("excel_path")

        overlay = accessor.overlay.copy()
        ann_columns = accessor.get_annotation_columns()
        cluster_columns = accessor.get_cluster_columns()

        def _get_depth(col: str) -> int:
            if col.startswith("L") and col[1:].isdigit():
                return int(col[1:])
            if "_L" in col:
                try:
                    return int(col.split("_L")[-1])
                except ValueError:
                    pass
            return 0

        def _add_depth_columns(df, columns, name_fn) -> None:
            for col in sorted(columns, key=_get_depth, reverse=True):
                df[name_fn(_get_depth(col))] = overlay[col]

        # Build cell-level dataframe
        cell_df = pd.DataFrame(index=overlay.index)
        cell_df.index.name = "obs_index"

        # Check whether any ontology terms exist (e.g., spatial pipeline has none)
        has_ontology = bool(getattr(accessor, "_ont_cell_types", None))

        # cluster_id_L{depth}: numeric cluster ID for each cell at each level.
        # cluster_name_L{depth}: original (pre-encoding) cluster name when an
        #   encoding was applied; omitted when already numeric / Leiden mode.
        inv_cluster_map = {v: k for k, v in self._cluster_name_map.items()} if self._cluster_name_map else {}
        for col in sorted(cluster_columns, key=_get_depth):
            depth = _get_depth(col)
            if depth > 0:
                cell_df[f"cluster_id_L{depth}"] = overlay[col]
                if inv_cluster_map:
                    cell_df[f"cluster_name_L{depth}"] = overlay[col].map(
                        lambda x, m=inv_cluster_map: m.get(str(x), str(x))
                    )

        if ann_columns:
            _add_depth_columns(cell_df, ann_columns, lambda d: f"lambda_{d}" if d > 0 else "cell_type")

        pathology_cols = [col for col in overlay.columns if col.startswith("pathology_L")]
        if pathology_cols:
            _add_depth_columns(cell_df, pathology_cols, lambda d: f"pathology_{d}" if d > 0 else "pathology")

        if has_ontology:
            for col in cluster_columns:
                depth = _get_depth(col)
                ont_col = f"lambda_ont_{depth}" if depth > 0 else "lambda_ont"
                cluster_map = {
                    cid: ont_term for cid in overlay[col].dropna().unique() if (ont_term := accessor.get_ont_cell_type(cid))
                }
                if cluster_map:
                    cell_df[ont_col] = overlay[col].map(cluster_map)

        # Forward-fill annotations from shallower to deeper levels so that
        # clusters not subclustered carry their parent label instead of NaN.
        ann_col_names = sorted(
            [c for c in cell_df.columns if c == "cell_type" or c.startswith("lambda_")],
            key=lambda c: 0 if c == "cell_type" else int(c.split("_")[-1]),
        )
        for prev, cur in zip(ann_col_names, ann_col_names[1:]):
            if cur in cell_df.columns:
                cell_df[cur] = cell_df[cur].fillna(cell_df[prev])

        if has_ontology:
            ont_col_names = sorted(
                [c for c in cell_df.columns if c == "lambda_ont" or c.startswith("lambda_ont_")],
                key=lambda c: 0 if c == "lambda_ont" else int(c.rsplit("_", 1)[-1]),
            )
            for prev, cur in zip(ont_col_names, ont_col_names[1:]):
                if cur in cell_df.columns:
                    cell_df[cur] = cell_df[cur].fillna(cell_df[prev])

        pathology_col_names = sorted(
            [c for c in cell_df.columns if c == "pathology" or c.startswith("pathology_")],
            key=lambda c: 0 if c == "pathology" else int(c.split("_")[-1]),
        )
        for prev, cur in zip(pathology_col_names, pathology_col_names[1:]):
            if cur in cell_df.columns:
                cell_df[cur] = cell_df[cur].fillna(cell_df[prev])

        # Build cluster-level Excel: one sheet per depth
        pathology_ann_columns = [col for col in overlay.columns if col.startswith("pathology_L")]
        all_ann_columns = ann_columns + pathology_ann_columns

        depths = set()
        for col in all_ann_columns:
            depths.add(_get_depth(col))

        cluster_data = {}

        for depth in sorted(depths):
            ann_col = accessor._annotation_column(depth, "cell_type")
            if ann_col not in overlay.columns:
                ann_col = accessor._annotation_column(depth, "pathology")
            cluster_col = accessor._level_column(depth)

            if ann_col not in overlay.columns or cluster_col not in overlay.columns:
                continue

            unique_clusters = overlay[cluster_col].value_counts(dropna=True).index.tolist()

            base_columns = ["cluster_id", "cell_type", "justification"]
            if has_ontology:
                base_columns.append("cell_type_ont")
            depth_df = pd.DataFrame(
                index=unique_clusters,
                columns=base_columns,
            )
            depth_df.index.name = "cluster_id_index"
            depth_df["cluster_id"] = depth_df.index
            depth_df["cell_type"] = None
            depth_df["justification"] = ""
            if has_ontology:
                depth_df["cell_type_ont"] = None

            for cid in unique_clusters:
                # Overlay is the canonical cell_type source: update_annotation()
                # always writes there, including post-broadening and post-correction
                # corrections that bypass the hypothesis list.
                mask = overlay[cluster_col] == cid
                cell_types = overlay.loc[mask, ann_col].dropna()
                if len(cell_types) > 0:
                    depth_df.loc[cid, "cell_type"] = cell_types.value_counts().index[0]
                    continue

                # Fallback: hypotheses (clusters never passed through update_annotation)
                hypotheses = accessor.get_hypotheses(str(cid))
                if hypotheses:
                    best_cell_type = (
                        hypotheses[0].get("cell_type")
                        or hypotheses[0].get("condition")
                        or hypotheses[0].get("annotation")
                    )
                    if best_cell_type:
                        depth_df.loc[cid, "cell_type"] = best_cell_type

            if has_ontology:
                for cid in unique_clusters:
                    ont_term = accessor.get_ont_cell_type(str(cid))
                    depth_df.loc[cid, "cell_type_ont"] = ont_term

            def _get_justification(cid) -> str:
                hypotheses = accessor.get_hypotheses(str(cid))
                if hypotheses and len(hypotheses) > 0:
                    return hypotheses[0].get("justification", "") or hypotheses[0].get("reasoning", "") or ""
                return ""

            for cid in unique_clusters:
                depth_df.loc[cid, "justification"] = _get_justification(cid)

            depth_df = depth_df[base_columns]
            def _sort_key(x):
                try:
                    return (0, tuple(int(p) for p in str(x).split("_")))
                except ValueError:
                    return (1, (str(x),))

            depth_df = depth_df.sort_values(
                "cluster_id",
                key=lambda s: s.map(_sort_key),
            )
            cluster_data[f"Level_{depth}"] = depth_df

        # Optional Cluster Mapping sheet (numeric_id → original_name)
        mapping_df: "pd.DataFrame | None" = None
        if self._cluster_name_map:
            mapping_df = pd.DataFrame(
                sorted(inv_cluster_map.items(), key=lambda x: int(x[0]) if x[0].isdigit() else float("inf")),
                columns=["cluster_id", "original_name"],
            )

        written = []
        excel_path = Path(excel_path or "lambda_annotations.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            if cluster_data:
                for sheet_name, df in cluster_data.items():
                    df.to_excel(writer, sheet_name=sheet_name)
            else:
                pd.DataFrame(columns=["cluster_id", "cell_type", "justification"]).to_excel(
                    writer, sheet_name="Summary", index=False
                )
            if mapping_df is not None:
                mapping_df.to_excel(writer, sheet_name="Cluster_Mapping", index=False)
        written.append(excel_path)

        if output_path:
            output_path = Path(output_path)
            cell_df.to_csv(output_path, sep="\t" if output_path.suffix == ".tsv" else ",")
            written.append(output_path)

        return written

    def _write_prompt_history(self, prompts_path) -> "Path | None":
        """Write per-cluster prompt history to a hierarchical markdown file.

        Includes:
        - Cluster metadata (cluster ID, annotation, Cell Ontology, justification, token totals)
        - Every recorded request, grouped by cluster, with provider usage
        - Legacy states without a ledger use incomplete recorded-text estimates

        Returns the path written, or None if there was nothing to write.
        """
        from pathlib import Path
        import re

        events = self._get_usage_events()
        if events is not None:
            from .usage import write_history
            return write_history(prompts_path, self.name, events,
                                 getattr(self, "_last_results", []), getattr(self, "_cluster_name_map", {}))

        if not self._last_results:
            return None

        def _cluster_sort_key(cluster_id: str):
            try:
                return (0, tuple(int(p) for p in str(cluster_id).split("_")))
            except ValueError:
                return (1, (str(cluster_id),))

        def _stage_title_from_stem(stem: str) -> str:
            mapping = {
                "generation": "Hypothesis Generation",
                "refinement": "Refinement",
                "reevaluation": "Reevaluation",
            }
            return mapping.get(stem, stem.replace("_", " ").strip().title())

        def _slugify_heading(text: str) -> str:
            """Approximate GitHub heading slug for markdown TOC links."""
            slug = text.strip().lower()
            slug = re.sub(r"[^\w\s-]", "", slug)
            slug = re.sub(r"\s+", "-", slug)
            slug = re.sub(r"-+", "-", slug)
            return slug.strip("-")

        def _add_separator(lines: list[str]) -> None:
            if lines and lines[-1] != "":
                lines.append("")
            lines.append("---")
            lines.append("")

        token_encoder = None
        try:
            import tiktoken  # type: ignore

            # Stable general-purpose encoding used by GPT tokenizers.
            token_encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            token_encoder = None

        def _token_count(text: str) -> int:
            content = str(text or "")
            if not content:
                return 0
            if token_encoder is not None:
                try:
                    return len(token_encoder.encode(content))
                except Exception:
                    pass
            # Lightweight fallback when tokenizer package is unavailable.
            return max(1, len(content) // 4)

        def _add_collapsible_stage(
            lines: list[str],
            stage_title: str,
            input_prompt: str,
            output_response: str,
            input_tokens: int,
            output_tokens: int,
        ) -> None:

            lines.append(f"### {stage_title}")
            lines.append("")
            lines.append("<details>")
            lines.append("<summary>Show Prompt Exchange</summary>")
            lines.append("")
            lines.append(f"#### Input ({input_tokens} tokens)")
            lines.append("")
            lines.append("````text")
            lines.append(str(input_prompt))
            lines.append("````")
            lines.append("")
            lines.append(f"#### Output ({output_tokens} tokens)")
            lines.append("")
            lines.append("````text")
            lines.append(str(output_response))
            lines.append("````")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        def _extract_prompt_events(state: dict) -> list[dict]:
            """Get prompt events in execution order, with legacy fallback."""
            execution_log = state.get("_prompt_execution_log", [])
            if isinstance(execution_log, list) and execution_log:
                events = []
                for entry in execution_log:
                    if not isinstance(entry, dict):
                        continue
                    prompt = entry.get("input_prompt")
                    if prompt is None or str(prompt).strip() == "":
                        continue
                    events.append(
                        {
                            "stage_stem": str(entry.get("stage_stem") or "").strip(),
                            "input_prompt": str(prompt),
                            "output_response": str(entry.get("output_response") or ""),
                        }
                    )
                if events:
                    return events

            # Legacy fallback: reconstruct from final-state prompt fields
            ordered_prompt_keys = [
                field for field in self._prompt_history_fields
                if field.endswith("_prompt") and "_parse_" not in field
            ]
            prompt_keys = list(ordered_prompt_keys)
            if not prompt_keys:
                prompt_keys = sorted(
                    k for k in state.keys()
                    if k.endswith("_prompt") and "_parse_" not in k
                )

            events = []
            for prompt_key in prompt_keys:
                prompt_text = state.get(prompt_key)
                if not prompt_text:
                    continue
                stem = prompt_key[:-7]
                response_key = f"{stem}_response"
                response_text = state.get(response_key, "")
                if stem == "generation" and (not response_text):
                    response_text = state.get("raw_llm_response", "")

                events.append(
                    {
                        "stage_stem": stem,
                        "input_prompt": str(prompt_text),
                        "output_response": str(response_text or ""),
                    }
                )
            return events

        def _normalize_prompt_events(events: list[dict]) -> list[dict]:
            """Collapse ontology-selection retries to a single final event.

            If retry-attempt events exist, keep only the last attempt and label
            it as the canonical ``ontology_selection`` stage.
            """
            last_attempt_idx: int | None = None
            attempt_re = re.compile(r"^ontology_selection_attempt_\d+$")
            for idx, event in enumerate(events):
                stem = str(event.get("stage_stem", ""))
                if attempt_re.match(stem):
                    last_attempt_idx = idx

            if last_attempt_idx is None:
                return events

            normalized: list[dict] = []
            for idx, event in enumerate(events):
                stem = str(event.get("stage_stem", ""))
                if stem == "ontology_selection":
                    continue
                if attempt_re.match(stem):
                    if idx != last_attempt_idx:
                        continue
                    event = dict(event)
                    event["stage_stem"] = "ontology_selection"
                normalized.append(event)
            return normalized

        def _extract_justification(state: dict) -> str:
            justification = (
                state.get("justification_summary")
                or state.get("refinement_reasoning")
                or state.get("reasoning")
            )
            if justification:
                return str(justification)

            hypotheses = state.get("updated_hypotheses") or state.get("hypotheses") or []
            if isinstance(hypotheses, list) and hypotheses:
                top = hypotheses[0]
                if isinstance(top, dict):
                    return str(top.get("justification") or top.get("reasoning") or "N/A")
            return "N/A"

        # Numeric encoded id -> original cluster name.
        id_to_name = {v: k for k, v in self._cluster_name_map.items()} if self._cluster_name_map else {}

        states_by_cluster = {}
        for state in self._last_results:
            cluster_id = str(state.get("cluster_id", "unknown"))
            states_by_cluster[cluster_id] = state

        sorted_cluster_ids = sorted(states_by_cluster, key=_cluster_sort_key)
        cluster_events_by_id: dict[str, list[dict]] = {}
        cluster_token_totals: dict[str, tuple[int, int]] = {}
        global_input_tokens = 0
        global_output_tokens = 0
        for cluster_id in sorted_cluster_ids:
            events = _normalize_prompt_events(_extract_prompt_events(states_by_cluster[cluster_id]))
            total_input_tokens = sum(_token_count(str(e.get("input_prompt", ""))) for e in events)
            total_output_tokens = sum(_token_count(str(e.get("output_response", ""))) for e in events)
            cluster_events_by_id[cluster_id] = events
            cluster_token_totals[cluster_id] = (total_input_tokens, total_output_tokens)
            global_input_tokens += total_input_tokens
            global_output_tokens += total_output_tokens

        lines: list[str] = []
        lines.append(f"# {self.name} Prompt History")
        lines.append("")
        lines.append("## Legacy Recorded-Text Estimates (Incomplete)")
        lines.append("")
        lines.append("Provider usage was not recorded. These estimates cover saved text only; actor/critic calls, "
                     "parsing, retries, system prompts, images and reasoning may be missing. Do not use as total run usage.")
        lines.append(f"- Estimated recorded-text input tokens: `{global_input_tokens}`")
        lines.append(f"- Estimated recorded-text output tokens: `{global_output_tokens}`")
        lines.append("")

        # Top-level navigation for quick skimming.
        lines.append("## Table of Contents")
        lines.append("")
        if id_to_name:
            lines.append("- [Cluster ID Mapping](#cluster-id-mapping)")
        for cluster_id in sorted_cluster_ids:
            state = states_by_cluster[cluster_id]
            annotation = (
                state.get("cell_type")
                or state.get("condition")
                or state.get("annotation")
                or "N/A"
            )
            full_cell_type = str(state.get("full_cell_type") or annotation or "N/A")
            has_ont = bool(getattr(self.accessor, "_ont_cell_types", None))
            if has_ont:
                cell_ont = (
                    state.get("cell_type_ont")
                    or self.accessor.get_ont_cell_type(cluster_id)
                    or "N/A"
                )
                cluster_heading = f"Cluster {cluster_id} -> {cell_ont} [{full_cell_type}]"
            else:
                cluster_heading = f"Cluster {cluster_id} -> {full_cell_type}"
            lines.append(f"- [{cluster_heading}](#{_slugify_heading(cluster_heading)})")
        lines.append("")

        if id_to_name:
            lines.append("## Cluster ID Mapping")
            lines.append("")
            for numeric_id in sorted(id_to_name, key=lambda x: int(x) if str(x).isdigit() else float("inf")):
                lines.append(f"- `{numeric_id}` -> `{id_to_name[numeric_id]}`")
            lines.append("")

        has_ont = bool(getattr(self.accessor, "_ont_cell_types", None))
        for idx, cluster_id in enumerate(sorted_cluster_ids):
            if idx > 0:
                _add_separator(lines)

            state = states_by_cluster[cluster_id]
            annotation = (
                state.get("cell_type")
                or state.get("condition")
                or state.get("annotation")
                or "N/A"
            )
            full_cell_type = str(state.get("full_cell_type") or annotation or "N/A")
            if has_ont:
                cell_ont = (
                    state.get("cell_type_ont")
                    or self.accessor.get_ont_cell_type(cluster_id)
                    or "N/A"
                )
            justification = _extract_justification(state)
            original_name = id_to_name.get(cluster_id)
            events = cluster_events_by_id.get(cluster_id, [])
            total_input_tokens, total_output_tokens = cluster_token_totals.get(cluster_id, (0, 0))

            if has_ont:
                lines.append(f"## Cluster {cluster_id} -> {cell_ont} [{full_cell_type}]")
            else:
                lines.append(f"## Cluster {cluster_id} -> {full_cell_type}")
            lines.append("")
            lines.append("### Metadata")
            lines.append("")
            lines.append(f"- Cluster ID: `{cluster_id}`")
            if original_name is not None:
                lines.append(f"- Original Cluster Name: `{original_name}`")
            lines.append(f"- Cell type: {annotation}")
            if has_ont:
                lines.append(f"- Cell ontology: {cell_ont}")
            lines.append(f"- Justification: {justification}")
            lines.append(f"- Estimated recorded-text input tokens: `{total_input_tokens}`")
            lines.append(f"- Estimated recorded-text output tokens: `{total_output_tokens}`")
            lines.append("")

            # Append iteration labels for repeated stages and loop stages.
            total_per_stage: dict[str, int] = {}
            for e in events:
                stem = e.get("stage_stem", "")
                if not stem:
                    continue
                total_per_stage[stem] = total_per_stage.get(stem, 0) + 1

            seen_per_stage: dict[str, int] = {}
            for event in events:
                stem = event.get("stage_stem", "")
                stage_title = _stage_title_from_stem(stem or "prompt")
                seen_per_stage[stem] = seen_per_stage.get(stem, 0) + 1
                iteration = seen_per_stage[stem]
                is_loop_stage = stem in {"refinement", "reevaluation"}
                if is_loop_stage or total_per_stage.get(stem, 0) > 1:
                    stage_title = f"{stage_title} (iteration {iteration})"

                input_prompt = str(event.get("input_prompt", ""))
                output_response = str(event.get("output_response", ""))
                _add_collapsible_stage(
                    lines=lines,
                    stage_title=stage_title,
                    input_prompt=input_prompt,
                    output_response=output_response,
                    input_tokens=_token_count(input_prompt),
                    output_tokens=_token_count(output_response),
                )

            if not events:
                lines.append("_No prompt input/output pairs recorded for this cluster._")
                lines.append("")

        prompts_path = Path(prompts_path)
        with open(prompts_path, "w") as f:
            f.write("\n".join(lines).rstrip() + "\n")
        return prompts_path
