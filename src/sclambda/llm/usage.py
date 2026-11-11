"""Provider-reported usage at the request boundary, independent of prompt fields."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from threading import Lock
from uuid import uuid4

_context = ContextVar("sclambda_usage_context", default={})


@contextmanager
def usage_context(**fields):
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") == "text")
    return ""


def summarize(events):
    """Sum normalized totals once per attempt; cache/reasoning details are subsets."""
    unique = {e["event_id"]: e for e in events}.values()
    total = {"attempts": 0, "responses": 0, "input_tokens": 0, "output_tokens": 0,
             "attempts_missing_usage": 0}
    by_model, by_role = {}, {}
    for event in unique:
        usage = event.get("usage") or {}
        known = {k: usage.get(k) for k in ("input_tokens", "output_tokens")}
        complete = all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in known.values())
        model = f"{event['provider']}/{event['model']}"
        role = event.get("role") or event.get("node") or "run_setup_or_global"
        buckets = [total, by_model.setdefault(model, dict.fromkeys(total, 0)),
                   by_role.setdefault(role, dict.fromkeys(total, 0))]
        for bucket in buckets:
            bucket["attempts"] += 1
            bucket["responses"] += event.get("outcome") == "response"
            bucket["attempts_missing_usage"] += not complete
            for key, value in known.items():
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    bucket[key] += value
    return {**total, "complete": total["attempts_missing_usage"] == 0,
            "by_model": by_model, "by_role": by_role,
            "basis": "provider-reported normalized usage; missing usage is not estimated",
            "limitations": "Transport failures and provider/SDK-internal retries may have unreported usage. "
                           "Cache and reasoning breakdowns are subsets of input/output totals, not additional tokens."}


class UsageLedger:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        self._lock = Lock()
        self._events = []
        self._run = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8"):
                pass

    def begin_run(self, pipeline):
        with self._lock:
            if self._run is not None:
                raise RuntimeError("Concurrent annotate calls on the same LLM client are not supported")
            self._run = {"run_id": uuid4().hex, "pipeline": pipeline}
            return self._run["run_id"]

    def end_run(self, run_id):
        with self._lock:
            events = deepcopy([e for e in self._events if e.get("run_id") == run_id])
            if self._run and self._run["run_id"] == run_id:
                self._run = None
            return events

    def snapshot(self, start=0):
        with self._lock:
            return deepcopy(self._events[start:])

    @property
    def position(self):
        with self._lock:
            return len(self._events)

    def record(self, *, request_id, attempt, provider, model, power, messages, response=None, error=None):
        usage = getattr(response, "usage_metadata", None)
        usage = deepcopy(usage) if isinstance(usage, dict) else {}
        metadata = getattr(response, "response_metadata", None)
        metadata = metadata if isinstance(metadata, dict) else {}
        event = {"event_id": uuid4().hex, "request_id": request_id, "attempt": attempt,
                 "timestamp": datetime.now(timezone.utc).isoformat(), **_context.get(),
                 "provider": provider, "model": model, "power": power,
                 "outcome": "response" if response is not None else "error",
                 "usage": usage, "provider_usage": deepcopy(metadata.get("usage") or metadata.get("token_usage") or {}),
                 "response_id": getattr(response, "id", None) if isinstance(getattr(response, "id", None), str) else None,
                 "stop_reason": metadata.get("stop_reason") or metadata.get("finish_reason"),
                 "input_prompt": "\n\n".join(f"[{getattr(m, 'type', 'message')}]\n{_text(m.content)}" for m in messages),
                 "image_count": sum(sum(isinstance(b, dict) and b.get("type") in {"image", "image_url"}
                                        for b in m.content) for m in messages if isinstance(m.content, list)),
                 "output_response": _text(getattr(response, "content", None)),
                 "error_type": type(error).__name__ if error is not None else None}
        with self._lock:
            event.update(self._run or {})
            self._events.append(event)
            if self.path:
                try:
                    with self.path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                except OSError:
                    # Never repeat a billed request because the local log could not be written.
                    logging.getLogger(__name__).exception("Could not persist usage event; retained in memory")
