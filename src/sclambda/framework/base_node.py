"""
Base node classes for building pipeline nodes.

Provides:
- BaseNode: Base class for all nodes (LLM and non-LLM)
- LLMNode: Base class for nodes that interact with LLMs
- ParseSchema: Declarative parsing configuration for LLM responses

Usage:
    class MyNode(LLMNode):
        prompt_dir = "/path/to/prompts"
        default_output = {"result": "unknown"}

        def execute(self, state):
            # self.output is pre-initialized from default_output by __call__
            result = self.chain.execute(...)
            self.output["result"] = result.data
            return self.output
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass, field
import copy
import json
import re
import logging
import threading

if TYPE_CHECKING:
    from sclambda.core.accessor import Accessor
    from sclambda.core.task_runner import TaskRunner
    from sclambda.llm.api import LLMClient

logger = logging.getLogger(__name__)


# ============ GENE SYMBOL CONVENTION ============


def gene_symbol_convention(organism: str) -> str:
    """Return gene symbol naming convention instruction for the given organism.

    Human genes follow HGNC (all uppercase: TP53, CD8A).
    Mouse/rat genes follow MGI/RGD (first letter uppercase, rest lowercase: Tp53, Cd8a).
    Other organisms default to a generic instruction.
    """
    organism_lower = organism.lower().strip()
    if organism_lower in ("human", "homo sapiens"):
        return (
            "Use human gene symbol convention: ALL UPPERCASE "
            "(e.g. TP53, CD8A, BRCA1, IL6)"
        )
    if organism_lower in ("mouse", "mus musculus"):
        return (
            "Use mouse gene symbol convention: first letter uppercase, "
            "rest lowercase (e.g. Tp53, Cd8a, Brca1, Il6)"
        )
    if organism_lower in ("rat", "rattus norvegicus"):
        return (
            "Use rat gene symbol convention: first letter uppercase, "
            "rest lowercase (e.g. Tp53, Cd8a, Brca1, Il6)"
        )
    return (
        f"Use the standard gene symbol convention for {organism}. "
        "Ensure gene symbols match the naming used in the organism's reference databases."
    )


# ============ PARSE SCHEMA ============


@dataclass
class ParseSchema:
    """
    Declarative parsing configuration for LLM responses.

    Defines HOW to extract and post-process structured data from raw LLM text.
    Used with LLMNode.parse_llm_response() to replace per-node parsing functions.

    Args:
        format: JSON structure to extract - "list" for [...] or "dict" for {...}
        defaults: Default values for missing/empty fields.
            - For "list" format: applied to each item in the list.
            - For "dict" format: applied to the top-level dict.
            - Values can be callables: fn(item) -> default_value.
              Use this for dynamic defaults like generated reasoning strings.
        sort_by: Field name to sort list items by (only for "list" format).
        sort_order: Mapping of field values to sort priority (higher = first).
            Example: {"high": 3, "medium": 2, "low": 1}
    """

    format: str = "list"  # "list" or "dict"
    defaults: dict = field(default_factory=dict)
    sort_by: str | None = None
    sort_order: dict | None = None

    def apply_defaults(self, data):
        """Apply defaults to parsed data based on format."""
        if self.format == "list" and isinstance(data, list):
            for item in data:
                self._apply_item_defaults(item)
            if self.sort_by and self.sort_order:
                data.sort(
                    key=lambda x: self.sort_order.get(
                        str(x.get(self.sort_by, "unknown")).lower(), 0
                    ),
                    reverse=True,
                )
        elif self.format == "dict" and isinstance(data, dict):
            self._apply_item_defaults(data)
        return data

    def _apply_item_defaults(self, item: dict):
        """Fill missing/falsy fields with defaults. Callables get the item as arg."""
        for key, default in self.defaults.items():
            if key not in item or not item[key]:
                item[key] = default(item) if callable(default) else default


# ============ BASE NODE ============


class BaseNode:
    """
    Base class for all pipeline nodes.

    Provides:
    - Access to accessor, runner, and optionally llm_client
    - Helper methods for common operations (logging)
    - LangGraph-compatible __call__ interface
    - default_output class attribute for failure fallbacks

    Subclasses must implement execute(state) -> dict.
    Subclasses should override default_output with their failure defaults.
    """

    # Override in subclasses with the dict returned on failure.
    # This is the "if everything fails, return this" baseline.
    default_output: dict = {}

    def __init__(
        self,
        accessor: "Accessor",
        runner: "TaskRunner",
        llm_client: "LLMClient | None" = None,
    ):
        self.accessor = accessor
        self.runner = runner
        self.llm_client = llm_client
        # Thread-local storage for output: each thread that concurrently invokes
        # the same compiled graph (and therefore the same node instance) gets its
        # own independent `output` dict, preventing cross-cluster data mixing.
        self._output_local = threading.local()

    @property
    def output(self) -> dict:
        return getattr(self._output_local, "value", {})

    @output.setter
    def output(self, value: dict) -> None:
        self._output_local.value = value

    def execute(self, state: dict) -> dict:
        raise NotImplementedError(f"{self.__class__.__name__} must implement execute()")

    def __call__(self, state: dict) -> dict:
        self.output = copy.deepcopy(self.default_output)
        from sclambda.llm.usage import usage_context
        with usage_context(cluster_id=str(state.get("cluster_id", "")), node=self.__class__.__name__):
            result = self.execute(state)
        if not isinstance(result, dict):
            return result

        prev_log = state.get("_prompt_execution_log", [])
        if not isinstance(prev_log, list):
            prev_log = []

        new_entries = self._collect_prompt_entries(result)
        if new_entries:
            result = dict(result)
            result["_prompt_execution_log"] = [*prev_log, *new_entries]
        elif prev_log:
            # Preserve previously collected execution log when this node does
            # not emit prompts.
            result = dict(result)
            result["_prompt_execution_log"] = list(prev_log)

        return result

    def _collect_prompt_entries(self, result: dict) -> list[dict]:
        """Extract non-parse prompt input/output pairs from a node output.

        Entries are emitted in insertion order, which matches execution order
        for prompts written by the node.
        """
        entries: list[dict] = []

        for key, prompt_text in result.items():
            if not key.endswith("_prompt") or "_parse_" in key:
                continue
            if prompt_text is None or str(prompt_text).strip() == "":
                continue

            stem = key[:-7]  # strip "_prompt"
            response_key = f"{stem}_response"
            response_text = result.get(response_key, "")
            if stem == "generation" and (not response_text):
                response_text = result.get("raw_llm_response", "")

            entries.append(
                {
                    "prompt_key": key,
                    "stage_stem": stem,
                    "input_prompt": str(prompt_text),
                    "output_response": str(response_text or ""),
                    "node": self.__class__.__name__,
                }
            )

        return entries

    # Helper methods for common operations

    def log_info(self, cluster_id: str, message: str):
        logger.info(f"Cluster {cluster_id}: {message}")

    def log_warning(self, cluster_id: str, message: str):
        logger.warning(f"Cluster {cluster_id}: {message}")

    def log_error(self, cluster_id: str, message: str):
        logger.error(f"Cluster {cluster_id}: {message}")


# ============ LLM NODE ============


class LLMNode(BaseNode):
    """
    Base class for nodes that interact with LLMs.

    Extends BaseNode with:
    - LLM chain for prompt execution
    - parse_llm_response() for schema-based JSON extraction

    Subclasses must set the `prompt_dir` class attribute to the directory
    containing their prompt templates.
    """

    prompt_dir: str = ""  # Must be set by subclass

    def __init__(
        self,
        accessor: "Accessor",
        runner: "TaskRunner",
        llm_client: "LLMClient",
    ):
        if llm_client is None:
            raise ValueError(f"{self.__class__.__name__} requires llm_client")

        super().__init__(accessor, runner, llm_client)

        if not self.prompt_dir:
            raise ValueError(
                f"{self.__class__.__name__} must define prompt_dir class attribute"
            )

        from sclambda.framework.llm_chain import LLMPromptChain

        self.chain = LLMPromptChain(llm_client, self.prompt_dir)

    def parse_llm_response(
        self,
        text: str,
        schema: ParseSchema,
        fallback: Any = None,
    ) -> list | dict:
        """
        Parse LLM response text according to a schema.

        1. Extract JSON from text using regex (based on schema.format)
        2. Apply defaults to missing/empty fields
        3. Sort if schema defines sort_by

        Args:
            text: Raw LLM response text to parse
            schema: ParseSchema defining format, defaults, and sorting
            fallback: Value to return if parsing fails entirely.
                Defaults to [] for list format, {} for dict format.

        Returns:
            Parsed and post-processed data (list or dict)
        """
        if fallback is None:
            fallback = [] if schema.format == "list" else {}

        try:
            if schema.format == "list":
                json_match = re.search(r"\[.*\]", text, re.DOTALL)
                if not json_match:
                    return fallback

                data = json.loads(json_match.group())
                if not isinstance(data, list):
                    return fallback

            else:  # "dict"
                json_match = re.search(r"\{.*\}", text, re.DOTALL)
                if not json_match:
                    return fallback

                data = json.loads(json_match.group())
                if not isinstance(data, dict):
                    return fallback

            return schema.apply_defaults(data) or fallback

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return fallback
