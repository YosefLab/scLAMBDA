"""`lambdakit usage` — what the annotation cost in tokens.

An annotation run is expensive in a way that is invisible while it happens: the
orchestrator reasons over every cluster roster and DE table, and each cluster
subagent burns its own context on reference documents, measurements and belief
updates. By the time the run finishes, the user has no idea whether it consumed
a tenth of the session or most of it.

Nothing in the engine can measure that. The token counts live in the *harness*,
so this command reads them back out of Claude Code's own transcripts:

    ~/.claude/projects/<encoded-cwd>/<session-id>.jsonl                main loop
    ~/.claude/projects/<encoded-cwd>/<session-id>/subagents/agent-*.jsonl

The two halves are attributed differently. Subagent transcripts name the run
directory they were given, so their cost is attributed **exactly**. The
orchestrator's messages carry no such marker — a session interleaves annotation
with whatever else the user was doing — so its share is bounded by the run's
time window and reported as an upper bound rather than a measurement.

## Why cached tokens are not summed into one number

Prompt caching dominates these totals and is priced in three tiers, so adding
the counters together would be arithmetic on incomparable units. A run that
re-reads a 200k-token cached prefix fifty times reports ten million cache-read
tokens — a true count of what was billed, but only about a million tokens'
worth of *input-priced* work, and nothing like ten million tokens of context.
Reporting that headline unqualified would make every annotation look
catastrophically expensive.

So each tier is counted separately and converted to a common unit — the
**input-token equivalent** — using the published multipliers in `WEIGHTS`. Those
are ratios against the model's own base input price, not dollar amounts: they
hold across models and survive price changes, which a hard-coded per-model rate
table would not. Pass `--price-in` / `--price-out` when you actually want money.

This is the one part of lambdakit that depends on how another program writes
files it does not own. Every failure path here degrades to "unavailable" with a
reason; none of them raises, because a missing token count must never be the
thing that breaks a finished run.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from sclambda.storage.session import Session, SessionError

# Cost of each token class relative to one base input token. Cache writes carry
# a premium for the write itself and are charged per TTL — a 1-hour entry costs
# more to create than a 5-minute one — while a cache read is an order of
# magnitude cheaper than fresh input. These are Anthropic's published ratios;
# they are deliberately not dollar amounts, since the ratios outlive the prices.
WEIGHTS = {
    "input_tokens": 1.0,
    "cache_write_5m": 1.25,
    "cache_write_1h": 2.0,
    "cache_read": 0.1,
}

# Order matters: this is the column order in every table below.
COUNTERS = ("input_tokens", "cache_write_5m", "cache_write_1h", "cache_read", "output_tokens")

PROJECTS = Path.home() / ".claude" / "projects"


class Unavailable(Exception):
    """No transcripts to read. Carries the reason shown to the user."""


# ---------------------------------------------------------------- transcripts


def encode_cwd(path: Path) -> list[str]:
    """Candidate directory names Claude Code may have used for this cwd.

    The encoding has changed between versions — older ones kept underscores,
    newer ones fold them to dashes — and a project that predates the change has
    a directory under each spelling. Return both, newest convention first.
    """
    s = str(path)
    return [re.sub(r"[^A-Za-z0-9]", "-", s), re.sub(r"[^A-Za-z0-9_]", "-", s)]


def project_dirs(cwd: Path) -> list[Path]:
    return [PROJECTS / name for name in dict.fromkeys(encode_cwd(cwd)) if (PROJECTS / name).is_dir()]


def _iter_usage(path: Path) -> Iterator[tuple[str | None, str | None, dict[str, Any]]]:
    """Yield (timestamp, model, usage) for each assistant message in a transcript.

    The model matters as much as the token count. A run can span models — an
    orchestrator on one, subagents on another, a mid-run `/model` switch — and
    their prices differ by 2x or more, so a single blended dollar figure at one
    assumed rate is wrong by however far the mix departs from that assumption.

    Transcripts reach hundreds of megabytes, so lines are filtered on a
    substring before being parsed — the JSON cost of the user turns and tool
    results is most of the file and none of it is wanted here. A corrupt or
    half-written line is skipped rather than aborting the scan: the current
    session's transcript is being appended to *while this runs*.
    """
    try:
        handle = path.open("r", errors="replace")
    except OSError:
        return

    # One API response is logged as several records — a `thinking` fragment, a
    # `text` fragment, a `tool_use` fragment — and every one of them carries a
    # *copy* of the same `usage` object. Summing records therefore counts the
    # same billed tokens two or three times over (measured: 2.30x on a real
    # run, and it turns 665 API calls into 1,638). Deduplicate on the message
    # id, which identifies the response rather than the fragment.
    seen: dict[str, tuple[str | None, dict[str, Any]]] = {}
    order: list[str] = []
    with handle:
        for line in handle:
            if '"usage"' not in line:
                continue
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if rec.get("type") != "assistant":
                continue
            message = rec.get("message", {})
            usage = message.get("usage")
            if not isinstance(usage, dict):
                continue
            # A response with no id cannot be deduplicated; keep it rather than
            # drop it, keyed uniquely so it is never merged with another.
            key = message.get("id") or f"__anon_{len(order)}"
            if key not in seen:
                order.append(key)
            seen[key] = (rec.get("timestamp"), message.get("model"), usage)

    for key in order:
        yield seen[key]


def _split_cache_writes(usage: dict[str, Any]) -> tuple[int, int]:
    """Cache-creation tokens split into (5-minute, 1-hour) TTL buckets.

    The two are priced differently, so they cannot be added together. The
    breakdown lives in a `cache_creation` sub-object; when it is absent — an
    older transcript, or a provider that does not report it — fall back to the
    flat total and attribute it to the 5-minute tier, which is the default TTL
    and the cheaper of the two. That under-states rather than inflates.
    """
    detail = usage.get("cache_creation")
    if isinstance(detail, dict):
        five = detail.get("ephemeral_5m_input_tokens")
        hour = detail.get("ephemeral_1h_input_tokens")
        if isinstance(five, int) or isinstance(hour, int):
            return (five if isinstance(five, int) else 0), (hour if isinstance(hour, int) else 0)
    total = usage.get("cache_creation_input_tokens")
    return (total if isinstance(total, int) else 0), 0


def _tally(rows) -> dict[str, int]:
    """Sum one transcript's usage records into the counters in `COUNTERS`.

    Only the top-level fields are read. Each record also carries an
    `iterations` array whose entries sum to those same top-level values, so
    adding both would double-count every message.
    """
    total: dict[str, Any] = dict.fromkeys(COUNTERS, 0)
    total["messages"] = 0
    per_model: dict[str, dict[str, int]] = {}
    for _, model, usage in rows:
        total["messages"] += 1
        bucket = per_model.setdefault(model or "unknown", dict(dict.fromkeys(COUNTERS, 0), messages=0))
        bucket["messages"] += 1
        for field in ("input_tokens", "output_tokens"):
            value = usage.get(field)
            if isinstance(value, int):
                total[field] += value
                bucket[field] += value
        read = usage.get("cache_read_input_tokens")
        if isinstance(read, int):
            total["cache_read"] += read
            bucket["cache_read"] += read
        five, hour = _split_cache_writes(usage)
        total["cache_write_5m"] += five
        total["cache_write_1h"] += hour
        bucket["cache_write_5m"] += five
        bucket["cache_write_1h"] += hour
    total["by_model"] = per_model
    return total


def input_equivalent(t: dict[str, int]) -> float:
    """The input-side counters expressed as one comparable number.

    Output tokens are excluded — they are priced on a different scale entirely
    (typically 5x input) and are reported in their own column.
    """
    return sum(t.get(field, 0) * weight for field, weight in WEIGHTS.items())


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


# ---------------------------------------------------------------- attribution


def _run_window(sess: Session) -> tuple[datetime, datetime]:
    """When this run was active.

    `created_at` is stamped at init, but runs made before that field existed
    fall back to the manifest's own mtime — which is close enough, since the
    manifest is written once at init and rewritten only during setup. The end
    is the newest committed result, or now for a run still in progress.
    """
    started = _parse_ts(sess.get("created_at"))
    if started is None:
        started = datetime.fromtimestamp((sess.run / "manifest.json").stat().st_mtime, timezone.utc)

    ends = [p.stat().st_mtime for p in (sess.run / "results").glob("*.json")]
    ended = datetime.fromtimestamp(max(ends), timezone.utc) if ends else datetime.now(timezone.utc)
    return started, max(ended, started)


def _mentions_run(path: Path, needles: list[str]) -> bool:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return False
    return any(n in text for n in needles)


def _cluster_of(path: Path) -> str | None:
    """Which cluster a subagent worked, read back from its own commands.

    The agent was handed a cluster id and passes it to every `lambdakit`
    call, so the most frequent `--cluster X` in its transcript identifies it
    without the orchestrator having to record the mapping separately.
    """
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    hits = re.findall(r"--cluster[= ]+([A-Za-z0-9_.-]+)", text)
    if not hits:
        return None
    return max(set(hits), key=hits.count)


def collect(sess: Session) -> dict[str, Any]:
    """Gather every token count attributable to this run.

    Raises `Unavailable` when the harness left nothing to read.
    """
    cwd = Path(sess.get("cwd") or Path.cwd())
    dirs = project_dirs(cwd)
    if not dirs:
        raise Unavailable(
            f"no Claude Code transcript directory for `{cwd}` under `{PROJECTS}` — "
            "token counts are only recorded when the run is driven from a Claude Code session"
        )

    started, ended = _run_window(sess)
    run_dir = Path(sess.run)
    # Match on full paths only. The bare directory name is tempting as a third
    # needle, but run directories get generic names ("run1", "probe", "test")
    # that collide with ordinary prose in a transcript — and a false match
    # silently bills one run's subagents to a different run, which is far worse
    # than missing one. Both spellings are kept because the orchestrator may
    # hand a subagent either the relative path or the resolved absolute one.
    needles = [n for n in dict.fromkeys([str(run_dir), str(run_dir.resolve())]) if os.sep in n]

    sessions: list[dict[str, Any]] = []
    agents: list[dict[str, Any]] = []

    for project in dirs:
        for transcript in sorted(project.glob("*.jsonl")):
            # A session that ended before this run began cannot have produced
            # it. Skipping on mtime keeps old multi-hundred-megabyte
            # transcripts from being read at all.
            if datetime.fromtimestamp(transcript.stat().st_mtime, timezone.utc) < started:
                continue

            rows = list(_iter_usage(transcript))
            if not rows:
                continue
            within = [r for r in rows if (ts := _parse_ts(r[0])) and started <= ts <= ended]
            sessions.append({
                "id": transcript.stem,
                "whole": _tally(rows),
                "in_window": _tally(within),
            })

            sub = transcript.with_suffix("") / "subagents"
            for agent in sorted(sub.glob("agent-*.jsonl")) if sub.is_dir() else []:
                if not _mentions_run(agent, needles):
                    continue
                agents.append({
                    "id": agent.stem.replace("agent-", ""),
                    "cluster": _cluster_of(agent),
                    "session": transcript.stem,
                    "tally": _tally(list(_iter_usage(agent))),
                })

    # Which of those sessions actually drove this run? A project directory
    # accumulates one transcript per Claude Code session, and several can
    # overlap the run's window while having nothing to do with it — a second
    # terminal working in the same repo would otherwise be charged to the
    # annotation and inflate the denominator below it.
    #
    # A subagent naming this run's directory is proof, so prefer the sessions
    # that own one. Without any (a run small enough that the orchestrator did
    # every cluster itself), fall back to sessions that were at least active
    # during the window; one idle throughout cannot have produced the run.
    if not sessions:
        raise Unavailable(
            f"no session transcript touched since {started:%Y-%m-%d %H:%M} was found under "
            f"`{dirs[0]}` — the transcript may have been pruned"
        )

    owning = {a["session"] for a in agents}
    if owning:
        sessions = [s for s in sessions if s["id"] in owning]
    else:
        sessions = [s for s in sessions if s["in_window"]["messages"]]

    # An empty result here is not an error — a run that was initialised but
    # never annotated has genuinely spent nothing, and reporting zero is more
    # useful than claiming the transcripts are missing.

    def _sum(tallies):
        out: dict[str, Any] = dict.fromkeys(("messages",) + COUNTERS, 0)
        merged: dict[str, dict[str, int]] = {}
        for t in tallies:
            for k in ("messages",) + COUNTERS:
                out[k] += t.get(k, 0)
            for model, bucket in (t.get("by_model") or {}).items():
                dest = merged.setdefault(model, dict(dict.fromkeys(COUNTERS, 0), messages=0))
                for k, v in bucket.items():
                    dest[k] += v
        out["by_model"] = merged
        return out

    return {
        "started": started,
        "ended": ended,
        "sessions": sessions,
        "agents": agents,
        "orchestrator": _sum([s["in_window"] for s in sessions]),
        "subagents": _sum([a["tally"] for a in agents]),
        "session_whole": _sum([s["whole"] for s in sessions]),
    }


# ---------------------------------------------------------------- rendering


def _row(label: str, t: dict[str, int]) -> str:
    return (f"| {label} | {t['messages']:,} | {t['input_tokens']:,} | "
            f"{t['cache_write_5m']:,} | {t['cache_write_1h']:,} | "
            f"{t['cache_read']:,} | {t['output_tokens']:,} |")


def render(data: dict[str, Any], run: Path, rates: dict[str, tuple[float, float]]) -> str:
    orch, subs, whole = data["orchestrator"], data["subagents"], data["session_whole"]
    # `by_model` is a nested dict, not a counter — merge it separately or the
    # scalar comprehension tries to add two dicts together.
    total = {k: orch.get(k, 0) + subs.get(k, 0) for k in ("messages",) + COUNTERS}
    merged: dict[str, dict[str, int]] = {}
    for part in (orch, subs):
        for name, bucket in (part.get("by_model") or {}).items():
            dest = merged.setdefault(name, dict(dict.fromkeys(COUNTERS, 0), messages=0))
            for k, v in bucket.items():
                dest[k] += v
    total["by_model"] = merged

    minutes = (data["ended"] - data["started"]).total_seconds() / 60
    span = f"{minutes / 60:.1f} h" if minutes >= 90 else f"{minutes:.0f} min"

    out = [f"# Token usage — `{run}`", ""]
    out.append(f"- window: {data['started']:%Y-%m-%d %H:%M} → {data['ended']:%Y-%m-%d %H:%M} UTC ({span})")
    out.append(f"- subagents attributed to this run: {len(data['agents'])}")
    out.append("")

    if not any(total[c] for c in COUNTERS):
        out.append("No annotation work has been attributed to this run yet.")
        return "\n".join(out)

    out.append("| where | msgs | input | cache-write 5m | cache-write 1h | cache-read | output |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    out.append(_row("orchestrator _(upper bound)_", orch))
    out.append(_row(f"{len(data['agents'])} cluster subagent(s)", subs))
    out.append(_row("**run total**", total))
    out.append("")

    # The weighted view. Raw counts above are what was billed; this is what it
    # cost, with each cache tier converted to the input-token scale.
    eq_total = input_equivalent(total)
    out.append("## Weighted cost")
    out.append("")
    out.append("Cached tokens are not interchangeable with fresh input, so the raw counts above "
               "cannot be added up. Priced against base input (write 5m ×1.25, write 1h ×2.0, "
               "read ×0.1):")
    out.append("")
    out.append("| where | input-token equivalent | output tokens |")
    out.append("|---|---:|---:|")
    for label, t in (("orchestrator _(upper bound)_", orch),
                     (f"{len(data['agents'])} cluster subagent(s)", subs),
                     ("**run total**", total)):
        out.append(f"| {label} | {input_equivalent(t):,.0f} | {t['output_tokens']:,} |")
    out.append("")

    if total["cache_read"]:
        raw_in = sum(total[f] for f in ("input_tokens", "cache_write_5m", "cache_write_1h", "cache_read"))
        served = 100 * total["cache_read"] / raw_in if raw_in else 0
        out.append(f"Caching did its job: {served:.0f}% of input tokens were served from cache at "
                   f"a tenth of the price, so {raw_in:,} raw input tokens cost "
                   f"{eq_total:,.0f} tokens' worth.")
        out.append("")

    # Per-model split. A run that spans models cannot be priced with one rate:
    # pricing a Fable-5-dominated run at Opus rates understated it by 1.92x.
    models = total.get("by_model") or {}
    if models:
        out.append("## By model")
        out.append("")
        out.append("| model | calls | input-equivalent | output | cost |")
        out.append("|---|---:|---:|---:|---:|")
        priced_total, unpriced = 0.0, []
        for name, bucket in sorted(models.items(), key=lambda kv: -kv[1]["messages"]):
            eq = input_equivalent(bucket)
            rate = rates.get(name)
            if rate:
                cost = eq / 1e6 * rate[0] + bucket["output_tokens"] / 1e6 * rate[1]
                priced_total += cost
                money = f"${cost:,.2f}"
            else:
                unpriced.append(name)
                money = "—"
            out.append(f"| `{name}` | {bucket['messages']:,} | {eq:,.0f} | {bucket['output_tokens']:,} | {money} |")
        if rates and not unpriced:
            out.append(f"| **total** | {total['messages']:,} | {eq_total:,.0f} | "
                       f"{total['output_tokens']:,} | **${priced_total:,.2f}** |")
        out.append("")
        if unpriced:
            out.append("No rate given for " + ", ".join(f"`{m}`" for m in unpriced)
                       + " — pass `--price MODEL=IN,OUT` (dollars per million tokens) to price "
                       "it. Rates change, so none is assumed.")
            out.append("")

    if data["agents"]:
        out.append("## Per subagent")
        out.append("")
        out.append("| cluster | agent | msgs | input-equiv | output |")
        out.append("|---|---|---:|---:|---:|")
        for a in sorted(data["agents"], key=lambda x: (x["cluster"] or "", x["id"])):
            t = a["tally"]
            out.append(f"| {a['cluster'] or '—'} | `{a['id'][:12]}` | {t['messages']:,} | "
                       f"{input_equivalent(t):,.0f} | {t['output_tokens']:,} |")
        out.append("")

    share = 100 * total["output_tokens"] / whole["output_tokens"] if whole["output_tokens"] else 0
    out.append("## Share of the session")
    out.append("")
    out.append(f"- session output tokens in total: **{whole['output_tokens']:,}**")
    out.append(f"- of which this annotation: **{total['output_tokens']:,}** (~{share:.0f}%)")
    if data["agents"] and total["output_tokens"]:
        sub_share = 100 * subs["output_tokens"] / total["output_tokens"]
        out.append(f"- the fan-out did {sub_share:.0f}% of the annotation's output work, in parallel")
    out.append("")
    out.append("Subagent cost is exact — their transcripts name this run. The orchestrator line is an "
               "**upper bound**: it counts every main-loop message in the window above, including any "
               "unrelated work done in the same session.")
    return "\n".join(out)


def _parse_rates(specs) -> dict[str, tuple[float, float]]:
    """Parse repeated `--price MODEL=IN,OUT` into per-model dollar rates.

    Prices are never hard-coded: they change, and a stale table baked into the
    engine would quietly misreport every run made after the next price change.
    """
    rates: dict[str, tuple[float, float]] = {}
    for spec in specs or []:
        try:
            model, pair = spec.split("=", 1)
            first, second = pair.split(",", 1)
            rates[model.strip()] = (float(first), float(second))
        except ValueError:
            raise SessionError(
                f"could not read --price {spec!r}; expected MODEL=INPUT,OUTPUT "
                "in dollars per million tokens, e.g. --price claude-opus-5=5,25"
            ) from None
    return rates


def cmd_usage(args) -> int:
    sess = Session(args.run)
    try:
        data = collect(sess)
    except Unavailable as exc:
        print(f"# Token usage — `{sess.run}`\n\nUnavailable: {exc}.")
        return 0

    if getattr(args, "json", False):
        payload = {k: v for k, v in data.items() if k not in ("started", "ended")}
        payload["started"] = data["started"].isoformat()
        payload["ended"] = data["ended"].isoformat()
        payload["weights"] = WEIGHTS
        for key in ("orchestrator", "subagents", "session_whole"):
            payload[key] = dict(payload[key], input_token_equivalent=input_equivalent(payload[key]))
        print(json.dumps(payload, indent=2))
        return 0

    print(render(data, Path(sess.run), _parse_rates(getattr(args, "price", None))))
    return 0
