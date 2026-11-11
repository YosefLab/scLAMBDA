"""Reports from actual LLM requests, including actor/critic and repair calls."""
import json
from pathlib import Path

from sclambda.llm.usage import summarize
from sclambda.storage.atomic import write_text


def write_usage(path, events):
    write_text(Path(path), json.dumps({"schema_version": 1, "summary": summarize(events),
                                      "events": events}, indent=2, ensure_ascii=False, default=str))
    return Path(path)


def write_history(path, pipeline, events, states=(), name_map=None):
    summary = summarize(events)
    lines = [f"# {pipeline} Prompt History", "", "## Provider-reported Token Usage", "",
             f"- Total input tokens: `{summary['input_tokens']}`",
             f"- Total output tokens: `{summary['output_tokens']}`",
             f"- Recorded attempts: `{summary['attempts']}`",
             f"- Attempts with missing usage: `{summary['attempts_missing_usage']}`",
             "- These are reported totals, not tokenized prompt estimates. Missing counts are unknown, not zero.",
             "- All calls are included: actor, critic, parsing, repairs, final reviews and shared run calls.",
             "- System and image tokens and reasoning output are included where reported by the provider.",
             "- Cache/reasoning detail counters are subsets and are not added a second time.",
             f"- {summary['limitations']}", "",
             "| Provider/model | Attempts | Input | Output | Missing usage |",
             "|---|---:|---:|---:|---:|"]
    for model, bucket in summary["by_model"].items():
        lines.append(f"| {model} | {bucket['attempts']} | {bucket['input_tokens']} | {bucket['output_tokens']} | {bucket['attempts_missing_usage']} |")
    states_by_id = {str(s.get("cluster_id")): s for s in states}
    names = {str(v): k for k, v in (name_map or {}).items()}
    groups = {cid: [] for cid in states_by_id}
    for event in {e['event_id']: e for e in events}.values():
        groups.setdefault(event.get("cluster_id") or "run_setup_or_global", []).append(event)
    for cluster, calls in groups.items():
        totals = summarize(calls)
        lines += ["", f"## Cluster {cluster}" if cluster != "run_setup_or_global" else "## Shared run calls", "",
                  f"- Input tokens: `{totals['input_tokens']}`; output tokens: `{totals['output_tokens']}`; "
                  f"missing usage: `{totals['attempts_missing_usage']}`", ""]
        state = states_by_id.get(cluster, {})
        label = state.get("cell_type") or state.get("annotation") or state.get("condition")
        if label:
            lines += [f"- Annotation: {label}",
                      f"- Assessment: {state.get('annotation_assessment', state.get('annotation_status', ''))}"]
        if cluster in names:
            lines.append(f"- Original cluster name: {names[cluster]}")
        if state.get("cell_type_ont"):
            lines.append(f"- Cell Ontology: {state['cell_type_ont']}")
        hypotheses = state.get("updated_hypotheses") or state.get("hypotheses") or []
        reason = state.get("justification_summary") or (hypotheses[0].get("justification") or hypotheses[0].get("reasoning") if hypotheses else None)
        if reason:
            lines += ["", str(reason), ""]
        for i, event in enumerate(calls, 1):
            stage = event.get('role') or event.get('node') or 'shared call'
            lines += [f"### {i}. {stage}", "", f"- Event: `{event['event_id']}`",
                      f"- Model: `{event['model']}`; request attempt: `{event['attempt']}`; outcome: `{event['outcome']}`",
                      f"- Images: `{event['image_count']}` (image payloads omitted from this text report)",
                      f"- Usage: `{json.dumps(event['usage'])}`", "", "<details>", "<summary>Request and response</summary>", "",
                      "Input (including system messages):", "", "`````text", event['input_prompt'], "`````", "",
                      "Output (visible text; reasoning tokens remain in provider usage):", "", "`````text",
                      event['output_response'], "`````", "", "</details>", ""]
    write_text(Path(path), "\n".join(lines) + "\n")
    return Path(path)
