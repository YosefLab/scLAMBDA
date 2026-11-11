"""
LLM API abstraction layer.

Provides unified interface to multiple LLM providers:
- OpenAI (GPT-4, GPT-4o, GPT-4o-mini)
- Anthropic (Claude Sonnet, Claude Opus)
- Google (Gemini Pro, Gemini Flash)

Power levels:
- weak: Fast, cheap models for summarization (GPT-4o-mini, Gemini Flash, Haiku)
- medium: Balanced models (GPT-4o, Gemini Pro, Claude Sonnet)
- strong: Reasoning models (GPT-4, o1, Claude Opus)
"""

from __future__ import annotations
from enum import Enum
from typing import Literal
from dataclasses import dataclass, field
import os
import time
import logging
from copy import deepcopy
from uuid import uuid4
from .usage import UsageLedger

logger = logging.getLogger(__name__)


class Provider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class Power(str, Enum):
    """Model power levels."""

    WEAK = "weak"  # Fast, cheap - summarization tasks
    MEDIUM = "medium"  # Balanced - general tasks
    STRONG = "strong"  # Reasoning - complex tasks


# Model mappings per provider and power level
# Model mappings per provider and power level
# Notes:
# - OpenAI: "o1-preview" is deprecated; "o3" is the modern strong reasoning model.
# - Anthropic: use snapshot IDs for stability in production; aliases auto-migrate.
# - Google: Gemini 2.5 is the current stable family; 2.0 is being retired.

MODEL_REGISTRY: dict[Provider, dict[Power, str]] = {
    Provider.OPENAI: {
        Power.WEAK: "gpt-5-nano-2025-08-07",
        Power.MEDIUM: "gpt-5-mini-2025-08-07",
        Power.STRONG: "gpt-5.2-2025-12-11",
    },
    Provider.ANTHROPIC: {
        Power.WEAK: "claude-haiku-4-5-20251001",
        Power.MEDIUM: "claude-sonnet-4-5-20250929",
        Power.STRONG: "claude-opus-4-5-20251101",
    },
    Provider.GOOGLE: {
        Power.WEAK: "gemini-2.5-flash-lite",
        Power.MEDIUM: "gemini-2.5-flash",
        Power.STRONG: "gemini-2.5-pro",
    },
}


@dataclass
class LLMConfig:
    """Configuration for LLM API."""

    provider: Provider = Provider.OPENAI
    power: Power = Power.MEDIUM
    temperature: float = 1.0
    max_tokens: int = 40960

    # Retry configuration
    max_retries: int = 3
    initial_retry_delay: float = 1.0  # seconds
    max_retry_delay: float = 60.0  # seconds
    retry_exponential_base: float = 2.0

    # API keys (loaded from env if not provided)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None

    def __post_init__(self):
        # Convert string values to enums if needed
        if isinstance(self.provider, str):
            self.provider = Provider(self.provider)
        if isinstance(self.power, str):
            self.power = Power(self.power)


@dataclass
class LLMResponse:
    """Response from LLM call."""

    content: str
    model: str
    provider: Provider
    power: Power
    usage: dict = field(default_factory=dict)
    failed_after_retries: bool = False
    error_message: str | None = None
    retry_count: int = 0
    stop_reason: str | None = None

    @property
    def input_tokens(self) -> int:
        return self.usage.get("input_tokens", 0)

    @property
    def output_tokens(self) -> int:
        return self.usage.get("output_tokens", 0)

    @property
    def is_success(self) -> bool:
        """Check if the query succeeded."""
        return not self.failed_after_retries


class LLMClient:
    """
    Unified LLM client with multi-provider support.

    Usage:
        # Create with defaults
        client = LLMClient(provider="openai", power="medium")

        # Query with defaults
        response = client.query("What is a T cell?")

        # Override provider/power for specific call
        response = client.query(
            "Complex reasoning task...",
            provider="anthropic",
            power="strong"
        )

    Environment variables:
        OPENAI_API_KEY: OpenAI API key
        ANTHROPIC_API_KEY: Anthropic API key
        GEMINI_API_KEY: Google AI API key
    """

    def __init__(
        self,
        provider: str = "openai",
        power: str = "medium",
        temperature: float = 0.0,
        max_tokens: int = 40960,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
        max_retry_delay: float = 60.0,
        retry_exponential_base: float = 2.0,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        google_api_key: str | None = None,
        model_overrides: dict[str, str] | None = None,
        usage_log_path: str | None = None,
    ):
        """
        Initialize LLM client.

        Args:
            provider: Default provider (openai, anthropic, google)
            power: Default power level (weak, medium, strong)
            temperature: Default temperature for generation
            max_tokens: Default max tokens for response
            max_retries: Maximum number of retry attempts for timeout errors
            initial_retry_delay: Initial delay in seconds before first retry
            max_retry_delay: Maximum delay in seconds between retries
            retry_exponential_base: Base for exponential backoff calculation
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            anthropic_api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            google_api_key: Google API key (or set GEMINI_API_KEY env var)
            model_overrides: Optional dict mapping power level to model ID,
                e.g. {"medium": "claude-sonnet-4-6-20260320"}
        """
        # Convert string to enum internally
        provider_enum = Provider(provider.lower())
        power_enum = Power(power.lower())

        self.config = LLMConfig(
            provider=provider_enum,
            power=power_enum,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            initial_retry_delay=initial_retry_delay,
            max_retry_delay=max_retry_delay,
            retry_exponential_base=retry_exponential_base,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            google_api_key=google_api_key,
        )

        # Per-instance model overrides: {Power -> model_id}
        self._model_overrides: dict[Power, str] = {}
        if model_overrides:
            for power_key, model_id in model_overrides.items():
                self._model_overrides[Power(power_key.lower())] = model_id

        self.usage_ledger = UsageLedger(usage_log_path)

    def _get_api_key(self, provider: Provider) -> str:
        """Get API key for provider from config or environment."""
        if provider == Provider.OPENAI:
            key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY env var.")
            return key
        elif provider == Provider.ANTHROPIC:
            key = self.config.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY env var.")
            return key
        elif provider == Provider.GOOGLE:
            key = self.config.google_api_key or os.getenv("GEMINI_API_KEY")
            if not key:
                raise ValueError("Google API key not found. Set GEMINI_API_KEY env var.")
            return key
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _get_model(self, provider: Provider, power: Power) -> str:
        """Get model name for provider and power level."""
        if power in self._model_overrides:
            return self._model_overrides[power]
        return MODEL_REGISTRY[provider][power]

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Check if an exception is retryable (timeout or overload/rate-limit).

        Handles various exceptions from different providers:
        - Timeouts: Timeout, APITimeoutError, DeadlineExceeded, etc.
        - Overload: HTTP 529 (Anthropic overloaded), HTTP 429 (rate limit)
        """
        error_type = type(error).__name__
        error_str = str(error).lower()

        # Check exception type names for timeouts
        timeout_types = [
            "TimeoutError",
            "Timeout",
            "APITimeoutError",
            "DeadlineExceeded",
            "ReadTimeout",
            "ConnectTimeout",
        ]

        if any(timeout_type in error_type for timeout_type in timeout_types):
            return True

        # Check error message for timeout indicators
        timeout_keywords = [
            "timeout",
            "deadline exceeded",
            "timed out",
            "connection timeout",
            "read timeout",
            "request timeout",
        ]

        if any(keyword in error_str for keyword in timeout_keywords):
            return True

        # Check for overload / rate-limit indicators (HTTP 529, 429)
        overload_keywords = [
            "overloaded",
            "overload_error",
            "529",
            "rate limit",
            "rate_limit",
            "too many requests",
            "429",
        ]

        if any(keyword in error_str for keyword in overload_keywords):
            return True

        return False

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay for retry attempt."""
        delay = self.config.initial_retry_delay * (self.config.retry_exponential_base**attempt)
        return min(delay, self.config.max_retry_delay)

    def _create_llm(
        self,
        provider: Provider,
        power: Power,
        temperature: float,
        max_tokens: int,
        extended_thinking: bool = False,
        thinking_budget: int = 8000,
    ):
        """Create LangChain LLM instance for provider.

        When extended_thinking=True and provider is Anthropic, enables Claude's
        extended thinking feature. temperature is forced to 1.0 (Anthropic requirement)
        and max_tokens is bumped to accommodate both thinking and the response.
        Extended thinking is silently ignored for non-Anthropic providers.
        """
        model = self._get_model(provider, power)
        api_key = self._get_api_key(provider)

        if provider == Provider.OPENAI:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )
        elif provider == Provider.ANTHROPIC:
            from langchain_anthropic import ChatAnthropic

            # 5-generation Claude models reject the `temperature` param outright
            # ("temperature is deprecated for this model"), so omit it entirely
            # for them (even the thinking-mode 1.0 default is rejected).
            omit_temperature = any(
                model.startswith(p)
                for p in ("claude-sonnet-5", "claude-opus-5", "claude-haiku-5", "claude-fable-5")
            )
            kwargs: dict = dict(
                model=model,
                max_tokens=max_tokens,
                api_key=api_key,
            )
            if not omit_temperature:
                kwargs["temperature"] = temperature
            if extended_thinking:
                if not omit_temperature:
                    kwargs["temperature"] = 1.0  # Required by Anthropic
                kwargs["max_tokens"] = max(max_tokens, thinking_budget + 4096)
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }
            return ChatAnthropic(**kwargs)
        elif provider == Provider.GOOGLE:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                max_output_tokens=max_tokens,
                google_api_key=api_key,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _invoke_with_retry(
        self,
        llm,
        messages: list,
        resolved_provider: Provider,
        resolved_power: Power,
    ) -> LLMResponse:
        """
        Invoke LLM with retry logic for timeout errors.

        Args:
            llm: LangChain LLM instance
            messages: List of messages to send
            resolved_provider: Provider being used
            resolved_power: Power level being used

        Returns:
            LLMResponse with content or failure flag set
        """
        last_error = None
        request_id = uuid4().hex

        for attempt in range(self.config.max_retries + 1):
            response = None
            try:
                response = llm.invoke(messages)
                self.usage_ledger.record(request_id=request_id, attempt=attempt + 1,
                    provider=resolved_provider.value, model=self._get_model(resolved_provider, resolved_power),
                    power=resolved_power.value, messages=messages, response=response)

                # Success - extract usage and return
                usage = {}
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage = deepcopy(response.usage_metadata)

                # Extended thinking returns a list of content blocks; extract text only
                raw_content = response.content
                if isinstance(raw_content, list):
                    content = "".join(
                        block.get("text", "")
                        for block in raw_content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                else:
                    content = raw_content

                return LLMResponse(
                    content=content,
                    model=self._get_model(resolved_provider, resolved_power),
                    provider=resolved_provider,
                    power=resolved_power,
                    usage=usage,
                    stop_reason=(getattr(response, "response_metadata", {}) or {}).get("stop_reason")
                        or (getattr(response, "response_metadata", {}) or {}).get("finish_reason"),
                    failed_after_retries=False,
                    retry_count=attempt,
                )

            except Exception as e:
                last_error = e
                if response is None:
                    self.usage_ledger.record(request_id=request_id, attempt=attempt + 1,
                        provider=resolved_provider.value, model=self._get_model(resolved_provider, resolved_power),
                        power=resolved_power.value, messages=messages, error=e)

                # Only retry on retryable errors (timeout / overload / rate-limit)
                if not self._is_retryable_error(e):
                    logger.error(
                        f"Non-retryable error during LLM query (attempt {attempt + 1}): {e}",
                        exc_info=True,
                    )
                    return LLMResponse(
                        content="",
                        model=self._get_model(resolved_provider, resolved_power),
                        provider=resolved_provider,
                        power=resolved_power,
                        failed_after_retries=True,
                        error_message=f"Non-timeout error: {str(e)}",
                        retry_count=attempt,
                    )

                # Timeout error - retry if attempts remaining
                if attempt < self.config.max_retries:
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(
                        f"Timeout error during LLM query (attempt {attempt + 1}/{self.config.max_retries + 1}). "
                        f"Retrying in {delay:.2f}s... Error: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"LLM query failed after {self.config.max_retries + 1} attempts due to timeout: {e}",
                        exc_info=True,
                    )

        # All retries exhausted
        return LLMResponse(
            content="",
            model=self._get_model(resolved_provider, resolved_power),
            provider=resolved_provider,
            power=resolved_power,
            failed_after_retries=True,
            error_message=f"Timeout after {self.config.max_retries + 1} attempts: {str(last_error)}",
            retry_count=self.config.max_retries,
        )

    def query(
        self,
        prompt: str,
        *,
        system: (
            str | None
        ) = "You are an expert cell biologist whose task is to perform analysis of gene expression data.",
        provider: str | None = None,
        power: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        image_base64: str | list[str] | None = None,
        extended_thinking: bool = False,
        thinking_budget: int = 8000,
    ) -> LLMResponse:
        """
        Query the LLM with automatic retry on timeout errors.

        Args:
            prompt: User prompt/question
            system: Optional system message
            provider: Override default provider (openai, anthropic, google)
            power: Override default power level (weak, medium, strong)
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            image_base64: Optional base64-encoded PNG image to include in the
                message.  Passed as a ``data:`` URL so it works with all
                LangChain chat model providers.
            extended_thinking: Enable extended thinking (Anthropic only). Forces
                temperature=1.0 and bumps max_tokens to fit thinking + response.
                Silently ignored for non-Anthropic providers.
            thinking_budget: Token budget for the thinking phase (default 8000).
                Only used when extended_thinking=True.

        Returns:
            LLMResponse with content and metadata. Check `failed_after_retries` flag
            to determine if query succeeded after retries.
        """
        # Resolve parameters (use overrides or defaults)
        # Convert strings to enums internally
        resolved_provider = Provider(provider.lower()) if provider is not None else self.config.provider
        resolved_power = Power(power.lower()) if power is not None else self.config.power
        resolved_temperature = temperature if temperature is not None else self.config.temperature
        resolved_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        # Create LLM instance
        llm = self._create_llm(
            provider=resolved_provider,
            power=resolved_power,
            temperature=resolved_temperature,
            max_tokens=resolved_max_tokens,
            extended_thinking=extended_thinking,
            thinking_budget=thinking_budget,
        )

        # Build messages
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = []
        if system:
            messages.append(SystemMessage(content=system))

        if image_base64:
            # Multimodal message: one or more images + text. ``image_base64`` may
            # be a single base64 PNG or a list (e.g. one image per tissue slide).
            imgs = image_base64 if isinstance(image_base64, (list, tuple)) else [image_base64]
            content_blocks: list[dict] = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b}"},
                }
                for b in imgs if b
            ]
            content_blocks.append({"type": "text", "text": prompt})
            messages.append(HumanMessage(content=content_blocks))
        else:
            messages.append(HumanMessage(content=prompt))

        # Invoke with retry logic
        return self._invoke_with_retry(
            llm=llm,
            messages=messages,
            resolved_provider=resolved_provider,
            resolved_power=resolved_power,
        )

    async def _ainvoke_with_retry(
        self,
        llm,
        messages: list,
        resolved_provider: Provider,
        resolved_power: Power,
    ) -> LLMResponse:
        """
        Async invoke LLM with retry logic for timeout errors.

        Args:
            llm: LangChain LLM instance
            messages: List of messages to send
            resolved_provider: Provider being used
            resolved_power: Power level being used

        Returns:
            LLMResponse with content or failure flag set
        """
        import asyncio

        last_error = None
        request_id = uuid4().hex

        for attempt in range(self.config.max_retries + 1):
            response = None
            try:
                response = await llm.ainvoke(messages)
                self.usage_ledger.record(request_id=request_id, attempt=attempt + 1,
                    provider=resolved_provider.value, model=self._get_model(resolved_provider, resolved_power),
                    power=resolved_power.value, messages=messages, response=response)

                # Success - extract usage and return
                usage = {}
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage = deepcopy(response.usage_metadata)

                return LLMResponse(
                    content=response.content,
                    model=self._get_model(resolved_provider, resolved_power),
                    provider=resolved_provider,
                    power=resolved_power,
                    usage=usage,
                    stop_reason=(getattr(response, "response_metadata", {}) or {}).get("stop_reason")
                        or (getattr(response, "response_metadata", {}) or {}).get("finish_reason"),
                    failed_after_retries=False,
                    retry_count=attempt,
                )

            except Exception as e:
                last_error = e
                if response is None:
                    self.usage_ledger.record(request_id=request_id, attempt=attempt + 1,
                        provider=resolved_provider.value, model=self._get_model(resolved_provider, resolved_power),
                        power=resolved_power.value, messages=messages, error=e)

                # Only retry on retryable errors (timeout / overload / rate-limit)
                if not self._is_retryable_error(e):
                    logger.error(
                        f"Non-retryable error during async LLM query (attempt {attempt + 1}): {e}",
                        exc_info=True,
                    )
                    return LLMResponse(
                        content="",
                        model=self._get_model(resolved_provider, resolved_power),
                        provider=resolved_provider,
                        power=resolved_power,
                        failed_after_retries=True,
                        error_message=f"Non-timeout error: {str(e)}",
                        retry_count=attempt,
                    )

                # Timeout error - retry if attempts remaining
                if attempt < self.config.max_retries:
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(
                        f"Timeout error during async LLM query (attempt {attempt + 1}/{self.config.max_retries + 1}). "
                        f"Retrying in {delay:.2f}s... Error: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Async LLM query failed after {self.config.max_retries + 1} attempts due to timeout: {e}",
                        exc_info=True,
                    )

        # All retries exhausted
        return LLMResponse(
            content="",
            model=self._get_model(resolved_provider, resolved_power),
            provider=resolved_provider,
            power=resolved_power,
            failed_after_retries=True,
            error_message=f"Timeout after {self.config.max_retries + 1} attempts: {str(last_error)}",
            retry_count=self.config.max_retries,
        )

    async def aquery(
        self,
        prompt: str,
        *,
        system: str | None = None,
        provider: str | None = None,
        power: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Async query the LLM with automatic retry on timeout errors.

        Same as query() but async. Check `failed_after_retries` flag to determine
        if query succeeded after retries.

        Args:
            prompt: User prompt/question
            system: Optional system message
            provider: Override default provider (openai, anthropic, google)
            power: Override default power level (weak, medium, strong)
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            LLMResponse with content and metadata. Check `failed_after_retries` flag
            to determine if query succeeded after retries.
        """
        # Resolve parameters
        # Convert strings to enums internally
        resolved_provider = Provider(provider.lower()) if provider is not None else self.config.provider
        resolved_power = Power(power.lower()) if power is not None else self.config.power
        resolved_temperature = temperature if temperature is not None else self.config.temperature
        resolved_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        # Create LLM instance
        llm = self._create_llm(
            provider=resolved_provider,
            power=resolved_power,
            temperature=resolved_temperature,
            max_tokens=resolved_max_tokens,
        )

        # Build messages
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        # Async invoke with retry logic
        return await self._ainvoke_with_retry(
            llm=llm,
            messages=messages,
            resolved_provider=resolved_provider,
            resolved_power=resolved_power,
        )

    @staticmethod
    def available_providers() -> list[str]:
        """List available providers."""
        return [p.value for p in Provider]

    @staticmethod
    def available_powers() -> list[str]:
        """List available power levels."""
        return [p.value for p in Power]

    @staticmethod
    def get_model_info() -> dict:
        """Get model registry info."""
        return {
            provider.value: {power.value: model for power, model in models.items()}
            for provider, models in MODEL_REGISTRY.items()
        }
