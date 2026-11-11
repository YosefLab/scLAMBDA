"""
LLM prompt chain abstraction.

Encapsulates the common pattern of:
1. Load prompt template from file
2. Format with variables
3. Query LLM (with retries handled by LLMClient)
4. Optionally parse response with second LLM call
5. Extract JSON data from response
6. Return structured result

This eliminates ~30-50 lines of boilerplate per node.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
import json
import re
import logging


def _safe_format(template: str, **kwargs) -> str:
    """Format template, substituting only {identifier} placeholders that match known variables.

    Unlike str.format(), this leaves unrecognized {…} blocks (e.g. JSON examples)
    untouched instead of raising KeyError.
    """
    def replace_match(m: re.Match) -> str:
        key = m.group(1)
        return str(kwargs[key]) if key in kwargs else m.group(0)

    return re.sub(r"\{(\w+)\}", replace_match, template)

if TYPE_CHECKING:
    from sclambda.llm.api import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ChainResult:
    """
    Result from LLM prompt chain execution.

    Attributes:
        success: Whether chain completed successfully
        data: Parsed data (dict or list) from response
        raw_response: Raw LLM response text
        parse_response: Optional parsing LLM response
        formatted_prompt: The formatted main prompt
        formatted_parse_prompt: Optional formatted parse prompt
        error_message: Error message if failed
    """

    success: bool
    data: dict | list = field(default_factory=dict)
    raw_response: str = ""
    parse_response: str = ""
    formatted_prompt: str = ""
    formatted_parse_prompt: str = ""
    error_message: str | None = None

    def as_state_dict(self) -> dict:
        """
        Convert to state dictionary for LangGraph.

        Returns dict suitable for merging into pipeline state.
        """
        return {
            "raw_llm_response": self.raw_response,
            "parse_response": self.parse_response,
        }


class JSONResponseParser:
    """Parser for extracting JSON from LLM responses."""

    @staticmethod
    def parse_list(text: str, item_defaults: dict | None = None) -> list[dict]:
        """
        Extract JSON list from text and add defaults to items.

        Args:
            text: Text containing JSON list
            item_defaults: Default values to add to each item

        Returns:
            Parsed list of dicts (empty list if parsing fails)
        """
        try:
            json_match = re.search(r"\[.*\]", text, re.DOTALL)
            if not json_match:
                logger.warning("No JSON array found in response")
                return []

            data = json.loads(json_match.group())

            if not isinstance(data, list):
                logger.warning(f"Expected list, got {type(data)}")
                return []

            if item_defaults:
                for item in data:
                    for key, default_value in item_defaults.items():
                        if key not in item or not item[key]:
                            item[key] = default_value

            return data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON list: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            return []

    @staticmethod
    def parse_dict(text: str, defaults: dict | None = None) -> dict:
        """
        Extract JSON dict from text and add defaults.

        Tries fenced code blocks first (last ```json ... ``` block),
        then falls back to greedy regex match.

        Args:
            text: Text containing JSON dict
            defaults: Default values to add

        Returns:
            Parsed dict (empty dict if parsing fails)
        """
        try:
            # Try fenced code blocks first (use last one — most likely the final output)
            fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if fenced:
                for candidate in reversed(fenced):
                    try:
                        data = json.loads(candidate)
                        if isinstance(data, dict):
                            if defaults:
                                for key, default_value in defaults.items():
                                    if key not in data or data[key] is None:
                                        data[key] = default_value
                            return data
                    except json.JSONDecodeError:
                        continue

            # Fallback: greedy match from first { to last }
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if not json_match:
                logger.warning("No JSON object found in response")
                return defaults or {}

            data = json.loads(json_match.group())

            if not isinstance(data, dict):
                logger.warning(f"Expected dict, got {type(data)}")
                return defaults or {}

            if defaults:
                for key, default_value in defaults.items():
                    if key not in data or data[key] is None:
                        data[key] = default_value

            return data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON dict: {e}")
            return defaults or {}
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            return defaults or {}


class LLMPromptChain:
    """
    Orchestrates LLM prompt execution pipeline.

    Handles:
    - Loading and caching prompt templates
    - Formatting prompts with variables
    - Querying LLM (retries handled by LLMClient)
    - Optional parsing with second LLM call
    - JSON extraction and validation
    """

    def __init__(self, llm_client: "LLMClient", prompt_dir: str):
        self.llm_client = llm_client
        self.prompt_dir = prompt_dir
        self._prompt_cache: dict[str, str] = {}

    def _load_prompt(self, filename: str) -> str:
        """Load prompt template from file (with caching)."""
        if filename in self._prompt_cache:
            return self._prompt_cache[filename]

        path = f"{self.prompt_dir}/{filename}"
        with open(path, "r") as f:
            content = f.read().rstrip()

        self._prompt_cache[filename] = content
        return content

    def execute(
        self,
        prompt_file: str,
        variables: dict,
        parse_prompt_file: str | None = None,
        parse_as: str | None = "list",  # "list", "dict", or None to skip extraction
        power: str = "medium",
        parse_power: str = "weak",
        image_base64: str | None = None,
    ) -> ChainResult:
        """
        Execute LLM prompt chain.

        Steps:
        1. Load and format main prompt
        2. Query LLM
        3. Optionally query parsing LLM
        4. Extract JSON data
        5. Return structured result

        Args:
            prompt_file: Main prompt template filename
            variables: Variables to format into prompt
            parse_prompt_file: Optional parsing prompt filename
            parse_as: Parse response as "list" or "dict"
            power: LLM power level for main query
            parse_power: LLM power level for parsing query
            image_base64: Optional base64-encoded PNG image to attach to
                the main LLM query (not the parse query).

        Returns:
            ChainResult with parsed data and metadata
        """
        # Step 1: Load and format main prompt
        try:
            prompt_template = self._load_prompt(prompt_file)
            formatted_prompt = _safe_format(prompt_template, **variables)
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_file}")
            return ChainResult(
                success=False,
                error_message=f"Prompt file not found: {prompt_file}",
            )
        except KeyError as e:
            logger.error(f"Missing variable in prompt {prompt_file}: {e}")
            return ChainResult(
                success=False,
                error_message=f"Missing variable: {e}",
            )

        # Step 2: Query main LLM
        gen_response = self.llm_client.query(
            formatted_prompt, power=power, image_base64=image_base64,
        )

        if gen_response.failed_after_retries:
            logger.error(f"LLM query failed: {gen_response.error_message}")
            return ChainResult(
                success=False,
                formatted_prompt=formatted_prompt,
                error_message=gen_response.error_message,
            )

        raw_response = gen_response.content

        # Step 3: Optionally parse response
        parse_response_text = ""
        formatted_parse_prompt = ""

        if parse_prompt_file:
            parse_template = self._load_prompt(parse_prompt_file)
            formatted_parse_prompt = _safe_format(
                parse_template,
                **variables,
                **{f"{prompt_file.split('_')[0]}_response": raw_response},
                debate_response=raw_response,
                refinement_response=raw_response,
                reevaluation_response=raw_response,
                evaluation_response=raw_response,
            )

            if formatted_parse_prompt:
                parse_resp = self.llm_client.query(formatted_parse_prompt, power=parse_power)

                if not parse_resp.failed_after_retries:
                    parse_response_text = parse_resp.content
                else:
                    logger.warning(f"Parse LLM query failed: {parse_resp.error_message}")
                    parse_response_text = raw_response
            else:
                parse_response_text = raw_response
        else:
            parse_response_text = raw_response

        # Step 4: Extract JSON data. parse_as=None means the caller handles
        # parsing itself (e.g. the dossier-update step) — skip extraction so we
        # don't emit spurious "No JSON array found" warnings on free-text output.
        if parse_as is None:
            data = None
        elif parse_as == "list":
            data = JSONResponseParser.parse_list(parse_response_text)
        else:
            data = JSONResponseParser.parse_dict(parse_response_text)

        return ChainResult(
            success=True,
            data=data,
            raw_response=raw_response,
            parse_response=parse_response_text,
            formatted_prompt=formatted_prompt,
            formatted_parse_prompt=formatted_parse_prompt,
        )
