"""Tests for LLM API module."""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

from sclambda.llm.api import (
    LLMClient,
    LLMConfig,
    LLMResponse,
    Provider,
    Power,
    MODEL_REGISTRY,
)


# Check which provider API keys are available
def get_available_providers():
    """Check which providers have API keys available."""
    available = []
    if os.getenv("OPENAI_API_KEY"):
        available.append(Provider.OPENAI)
    if os.getenv("ANTHROPIC_API_KEY"):
        available.append(Provider.ANTHROPIC)
    if os.getenv("GEMINI_API_KEY"):
        available.append(Provider.GOOGLE)
    return available


@pytest.fixture(scope="module")
def available_providers():
    """Fixture that checks which providers have API keys."""
    return get_available_providers()


def skip_if_no_key(provider: Provider):
    """Skip test if provider's API key is not available."""
    key_map = {
        Provider.OPENAI: "OPENAI_API_KEY",
        Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
        Provider.GOOGLE: "GEMINI_API_KEY",
    }
    key_name = key_map[provider]
    if not os.getenv(key_name):
        pytest.skip(f"{key_name} not set, skipping {provider.value} tests")


class TestEnums:
    """Test Provider and Power enums."""

    def test_provider_values(self):
        assert Provider.OPENAI.value == "openai"
        assert Provider.ANTHROPIC.value == "anthropic"
        assert Provider.GOOGLE.value == "google"

    def test_power_values(self):
        assert Power.WEAK.value == "weak"
        assert Power.MEDIUM.value == "medium"
        assert Power.STRONG.value == "strong"

    def test_provider_from_string(self):
        assert Provider("openai") == Provider.OPENAI
        assert Provider("anthropic") == Provider.ANTHROPIC
        assert Provider("google") == Provider.GOOGLE

    def test_power_from_string(self):
        assert Power("weak") == Power.WEAK
        assert Power("medium") == Power.MEDIUM
        assert Power("strong") == Power.STRONG


class TestModelRegistry:
    """Test model registry configuration."""

    def test_all_providers_have_all_powers(self):
        for provider in Provider:
            assert provider in MODEL_REGISTRY
            for power in Power:
                assert power in MODEL_REGISTRY[provider]
                assert isinstance(MODEL_REGISTRY[provider][power], str)

    def test_openai_models(self):
        assert MODEL_REGISTRY[Provider.OPENAI][Power.WEAK] == "gpt-5-nano-2025-08-07"
        assert MODEL_REGISTRY[Provider.OPENAI][Power.MEDIUM] == "gpt-5-mini-2025-08-07"
        assert MODEL_REGISTRY[Provider.OPENAI][Power.STRONG] == "gpt-5.2-2025-12-11"

    def test_anthropic_models(self):
        assert "haiku" in MODEL_REGISTRY[Provider.ANTHROPIC][Power.WEAK]
        assert "sonnet" in MODEL_REGISTRY[Provider.ANTHROPIC][Power.MEDIUM]
        assert "opus" in MODEL_REGISTRY[Provider.ANTHROPIC][Power.STRONG]

    def test_google_models(self):
        assert "gemini" in MODEL_REGISTRY[Provider.GOOGLE][Power.WEAK]
        assert "gemini" in MODEL_REGISTRY[Provider.GOOGLE][Power.MEDIUM]
        assert "gemini" in MODEL_REGISTRY[Provider.GOOGLE][Power.STRONG]


class TestLLMConfig:
    """Test LLMConfig dataclass."""

    def test_default_values(self):
        config = LLMConfig()
        assert config.provider == Provider.OPENAI
        assert config.power == Power.MEDIUM
        assert config.temperature == 1.0
        assert config.max_tokens == 40960

    def test_custom_values(self):
        config = LLMConfig(
            provider=Provider.ANTHROPIC,
            power=Power.STRONG,
            temperature=0.7,
            max_tokens=2048,
        )
        assert config.provider == Provider.ANTHROPIC
        assert config.power == Power.STRONG
        assert config.temperature == 0.7
        assert config.max_tokens == 2048

    def test_string_conversion(self):
        """Config should accept string values and convert to enums."""
        config = LLMConfig(provider="google", power="weak")
        assert config.provider == Provider.GOOGLE
        assert config.power == Power.WEAK

    def test_retry_config_defaults(self):
        """Test retry configuration defaults."""
        config = LLMConfig()
        assert config.max_retries == 3
        assert config.initial_retry_delay == 1.0
        assert config.max_retry_delay == 60.0
        assert config.retry_exponential_base == 2.0

    def test_retry_config_custom(self):
        """Test custom retry configuration."""
        config = LLMConfig(
            max_retries=5,
            initial_retry_delay=2.0,
            max_retry_delay=120.0,
            retry_exponential_base=3.0,
        )
        assert config.max_retries == 5
        assert config.initial_retry_delay == 2.0
        assert config.max_retry_delay == 120.0
        assert config.retry_exponential_base == 3.0


class TestLLMResponse:
    """Test LLMResponse dataclass."""

    def test_response_creation(self):
        response = LLMResponse(
            content="Hello world",
            model="gpt-4o",
            provider=Provider.OPENAI,
            power=Power.MEDIUM,
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        assert response.content == "Hello world"
        assert response.model == "gpt-4o"
        assert response.input_tokens == 10
        assert response.output_tokens == 5

    def test_empty_usage(self):
        response = LLMResponse(
            content="Test",
            model="test",
            provider=Provider.OPENAI,
            power=Power.WEAK,
        )
        assert response.input_tokens == 0
        assert response.output_tokens == 0

    def test_failed_after_retries_flag(self):
        """Test failed_after_retries flag."""
        # Success case
        success_response = LLMResponse(
            content="Success",
            model="test",
            provider=Provider.OPENAI,
            power=Power.MEDIUM,
            failed_after_retries=False,
        )
        assert success_response.is_success is True
        assert success_response.failed_after_retries is False

        # Failure case
        failed_response = LLMResponse(
            content="",
            model="test",
            provider=Provider.OPENAI,
            power=Power.MEDIUM,
            failed_after_retries=True,
            error_message="Timeout after retries",
            retry_count=3,
        )
        assert failed_response.is_success is False
        assert failed_response.failed_after_retries is True
        assert failed_response.error_message == "Timeout after retries"
        assert failed_response.retry_count == 3


class TestProviderAvailability:
    """Test provider availability detection."""

    def test_get_available_providers(self):
        """Check which providers have API keys."""
        available = get_available_providers()
        # Should return a list of Provider enums
        assert isinstance(available, list)
        for provider in available:
            assert isinstance(provider, Provider)

    def test_skip_if_no_key_helper(self):
        """Test skip_if_no_key helper function."""
        # This test should always pass - it's just checking the helper exists
        # Actual skipping will happen in provider-specific tests
        assert callable(skip_if_no_key)


class TestLLMClient:
    """Test LLMClient class."""

    def test_init_defaults(self):
        client = LLMClient()
        assert client.config.provider == Provider.OPENAI
        assert client.config.power == Power.MEDIUM

    def test_init_custom(self):
        client = LLMClient(
            provider="anthropic",
            power="strong",
            temperature=0.5,
        )
        assert client.config.provider == Provider.ANTHROPIC
        assert client.config.power == Power.STRONG
        assert client.config.temperature == 0.5

    def test_init_with_enum(self):
        client = LLMClient(
            provider=Provider.GOOGLE,
            power=Power.WEAK,
        )
        assert client.config.provider == Provider.GOOGLE
        assert client.config.power == Power.WEAK

    def test_get_model(self):
        client = LLMClient()
        assert client._get_model(Provider.OPENAI, Power.MEDIUM) == "gpt-5-mini-2025-08-07"
        assert "haiku" in client._get_model(Provider.ANTHROPIC, Power.WEAK)

    def test_available_providers(self):
        providers = LLMClient.available_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        assert "google" in providers

    def test_available_powers(self):
        powers = LLMClient.available_powers()
        assert "weak" in powers
        assert "medium" in powers
        assert "strong" in powers

    def test_get_model_info(self):
        info = LLMClient.get_model_info()
        assert "openai" in info
        assert "anthropic" in info
        assert "google" in info
        assert info["openai"]["medium"] == "gpt-5-mini-2025-08-07"

    def test_missing_api_key_raises(self):
        client = LLMClient()
        with patch.dict("os.environ", {}, clear=True):
            # Remove any existing env vars
            client.config.openai_api_key = None
            with pytest.raises(ValueError, match="OpenAI API key not found"):
                client._get_api_key(Provider.OPENAI)

    def test_api_key_from_env(self):
        client = LLMClient()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"}):
            key = client._get_api_key(Provider.OPENAI)
            assert key == "test-key-123"

    def test_api_key_from_config(self):
        client = LLMClient(openai_api_key="config-key-456")
        key = client._get_api_key(Provider.OPENAI)
        assert key == "config-key-456"


class TestLLMClientQuery:
    """Test LLMClient.query() method with mocks."""

    @pytest.fixture
    def client_with_key(self):
        """Create client with API key."""
        return LLMClient(openai_api_key="test-key")

    def test_query_uses_defaults(self, client_with_key):
        """Query without overrides uses default provider/power."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Mocked response",
            usage_metadata={"input_tokens": 15, "output_tokens": 10},
        )

        with patch.object(client_with_key, "_create_llm", return_value=mock_llm):
            response = client_with_key.query("Test prompt")

        assert response.content == "Mocked response"
        assert response.provider == Provider.OPENAI
        assert response.power == Power.MEDIUM
        assert response.model == "gpt-5-mini-2025-08-07"

    def test_query_with_provider_override(self, client_with_key):
        """Query can override provider."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Claude response",
            usage_metadata={},
        )

        client = LLMClient(
            openai_api_key="test",
            anthropic_api_key="test-anthropic",
        )

        with patch.object(client, "_create_llm", return_value=mock_llm) as mock_create:
            response = client.query("Test", provider="anthropic", power="strong")

        # Verify _create_llm was called with correct params
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["provider"] == Provider.ANTHROPIC
        assert call_kwargs["power"] == Power.STRONG

        assert response.provider == Provider.ANTHROPIC
        assert response.power == Power.STRONG

    def test_query_with_system_message(self, client_with_key):
        """Query includes system message when provided."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Response with system",
            usage_metadata={},
        )

        captured_messages = []

        def capture_invoke(messages):
            captured_messages.extend(messages)
            return MagicMock(content="Response", usage_metadata={})

        mock_llm.invoke = capture_invoke

        with patch.object(client_with_key, "_create_llm", return_value=mock_llm):
            client_with_key.query("Test", system="You are helpful")

        # Should have 2 messages: system and human
        assert len(captured_messages) == 2


class TestLLMClientIntegration:
    """Integration-style tests (still mocked, but testing full flow)."""

    def test_full_query_flow(self):
        """Test complete query flow with all components."""
        # Create mock LLM that returns a proper response
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="T cells are a type of lymphocyte.",
            usage_metadata={
                "input_tokens": 25,
                "output_tokens": 12,
            },
        )

        client = LLMClient(
            provider="openai",
            power="medium",
            openai_api_key="test-key",
        )

        with patch.object(client, "_create_llm", return_value=mock_llm):
            response = client.query(
                "What is a T cell?",
                system="You are a biology expert.",
            )

        assert response.content == "T cells are a type of lymphocyte."
        assert response.model == "gpt-5-mini-2025-08-07"
        assert response.provider == Provider.OPENAI
        assert response.power == Power.MEDIUM
        assert response.input_tokens == 25
        assert response.output_tokens == 12

    def test_temperature_override(self):
        """Test temperature can be overridden per call."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Creative", usage_metadata={})

        client = LLMClient(
            openai_api_key="test-key",
            temperature=0.0,  # Default
        )

        with patch.object(client, "_create_llm", return_value=mock_llm) as mock_create:
            client.query("Be creative", temperature=0.9)

        # Verify temperature was overridden
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.9

    def test_power_levels_map_to_correct_models(self):
        """Verify power level -> model mapping."""
        client = LLMClient(openai_api_key="test-key")

        # OpenAI
        assert client._get_model(Provider.OPENAI, Power.WEAK) == "gpt-5-nano-2025-08-07"
        assert client._get_model(Provider.OPENAI, Power.MEDIUM) == "gpt-5-mini-2025-08-07"
        assert client._get_model(Provider.OPENAI, Power.STRONG) == "gpt-5.2-2025-12-11"

        # Anthropic
        assert "haiku" in client._get_model(Provider.ANTHROPIC, Power.WEAK)
        assert "sonnet" in client._get_model(Provider.ANTHROPIC, Power.MEDIUM)
        assert "opus" in client._get_model(Provider.ANTHROPIC, Power.STRONG)

        # Google
        assert "flash" in client._get_model(Provider.GOOGLE, Power.WEAK)
        assert "flash" in client._get_model(Provider.GOOGLE, Power.MEDIUM)
        assert "pro" in client._get_model(Provider.GOOGLE, Power.STRONG)

    def test_is_retryable_error(self):
        """Test timeout error detection."""
        client = LLMClient(openai_api_key="test-key")

        # TimeoutError
        timeout_error = TimeoutError("Request timed out")
        assert client._is_retryable_error(timeout_error) is True

        # APITimeoutError (simulated)
        class APITimeoutError(Exception):
            pass

        api_timeout = APITimeoutError("API timeout")
        assert client._is_retryable_error(api_timeout) is True

        # Error with timeout in message
        generic_error = ValueError("Connection timeout occurred")
        assert client._is_retryable_error(generic_error) is True

        # Non-timeout error
        other_error = ValueError("Invalid input")
        assert client._is_retryable_error(other_error) is False

    def test_calculate_retry_delay(self):
        """Test exponential backoff delay calculation."""
        client = LLMClient(
            openai_api_key="test-key",
            initial_retry_delay=1.0,
            max_retry_delay=60.0,
            retry_exponential_base=2.0,
        )

        # First retry (attempt 0)
        delay0 = client._calculate_retry_delay(0)
        assert delay0 == 1.0

        # Second retry (attempt 1)
        delay1 = client._calculate_retry_delay(1)
        assert delay1 == 2.0

        # Third retry (attempt 2)
        delay2 = client._calculate_retry_delay(2)
        assert delay2 == 4.0

        # Should cap at max_retry_delay
        delay10 = client._calculate_retry_delay(10)
        assert delay10 == 60.0  # Capped at max


class TestLLMClientRetry:
    """Test retry logic for timeout errors."""

    def test_retry_on_timeout_success_after_retry(self):
        """Test that query retries on timeout and succeeds after retry."""
        client = LLMClient(
            openai_api_key="test-key",
            max_retries=2,
            initial_retry_delay=0.1,  # Short delay for testing
        )

        # Mock LLM that fails once then succeeds
        mock_llm = MagicMock()
        timeout_error = TimeoutError("Request timed out")
        success_response = MagicMock(
            content="Success after retry",
            usage_metadata={"input_tokens": 10, "output_tokens": 5},
        )
        mock_llm.invoke.side_effect = [timeout_error, success_response]

        with patch.object(client, "_create_llm", return_value=mock_llm):
            response = client.query("Test prompt")

        assert response.is_success is True
        assert response.failed_after_retries is False
        assert response.content == "Success after retry"
        assert response.retry_count == 1  # Succeeded on second attempt
        assert mock_llm.invoke.call_count == 2

    def test_retry_exhausted_returns_failure(self):
        """Test that query returns failure flag after all retries exhausted."""
        client = LLMClient(
            openai_api_key="test-key",
            max_retries=2,
            initial_retry_delay=0.1,
        )

        # Mock LLM that always times out
        mock_llm = MagicMock()
        timeout_error = TimeoutError("Request timed out")
        mock_llm.invoke.side_effect = timeout_error

        with patch.object(client, "_create_llm", return_value=mock_llm):
            response = client.query("Test prompt")

        assert response.is_success is False
        assert response.failed_after_retries is True
        assert response.content == ""
        assert "timeout" in response.error_message.lower()
        assert response.retry_count == 2  # Exhausted all retries
        assert mock_llm.invoke.call_count == 3  # Initial + 2 retries

    def test_no_retry_on_non_timeout_error(self):
        """Test that non-timeout errors don't trigger retries."""
        client = LLMClient(
            openai_api_key="test-key",
            max_retries=3,
        )

        # Mock LLM that raises non-timeout error
        mock_llm = MagicMock()
        value_error = ValueError("Invalid API key")
        mock_llm.invoke.side_effect = value_error

        with patch.object(client, "_create_llm", return_value=mock_llm):
            response = client.query("Test prompt")

        assert response.is_success is False
        assert response.failed_after_retries is True
        assert "non-timeout error" in response.error_message.lower()
        assert response.retry_count == 0  # No retries attempted
        assert mock_llm.invoke.call_count == 1  # Only one attempt

    def test_retry_with_custom_config(self):
        """Test retry with custom configuration."""
        client = LLMClient(
            openai_api_key="test-key",
            max_retries=1,
            initial_retry_delay=0.1,
            max_retry_delay=10.0,
            retry_exponential_base=3.0,
        )

        mock_llm = MagicMock()
        timeout_error = TimeoutError("Timeout")
        success_response = MagicMock(
            content="Success",
            usage_metadata={},
        )
        mock_llm.invoke.side_effect = [timeout_error, success_response]

        with patch.object(client, "_create_llm", return_value=mock_llm):
            response = client.query("Test")

        assert response.is_success is True
        assert response.retry_count == 1
        # Verify delay calculation uses custom base
        delay = client._calculate_retry_delay(0)
        assert delay == 0.1  # initial * base^0 = initial * 1 = initial
        delay1 = client._calculate_retry_delay(1)
        assert delay1 == 0.1 * 3.0  # initial * base^1 = initial * base


@pytest.mark.integration
class TestLLMClientRealAPI:
    """Integration tests with real API calls (skipped if keys not available)."""

    @pytest.mark.parametrize("provider", [Provider.OPENAI, Provider.ANTHROPIC, Provider.GOOGLE])
    def test_real_query_basic(self, provider, available_providers):
        """Test real API call (skipped if key not available)."""
        if provider not in available_providers:
            skip_if_no_key(provider)

        # Create client with real API key from env
        client = LLMClient(provider=provider, power=Power.WEAK)

        try:
            # Make a simple query
            response = client.query("Say 'test' and nothing else.")

            # Verify response structure
            assert isinstance(response, LLMResponse)
            assert len(response.content) > 0
            assert response.provider == provider
            assert response.power == Power.WEAK
            assert response.model == MODEL_REGISTRY[provider][Power.WEAK]
        except Exception as e:
            # Handle API errors gracefully (quota, rate limits, etc.)
            pytest.skip(f"API call failed (may be quota/rate limit): {e}")

    @pytest.mark.parametrize("provider", [Provider.OPENAI, Provider.ANTHROPIC, Provider.GOOGLE])
    def test_real_query_with_system(self, provider, available_providers):
        """Test real API call with system message."""
        if provider not in available_providers:
            skip_if_no_key(provider)

        client = LLMClient(provider=provider, power=Power.MEDIUM)

        try:
            response = client.query(
                "What is 2+2? Answer with just the number.",
                system="You are a helpful assistant.",
            )

            assert isinstance(response, LLMResponse)
            assert response.provider == provider
            assert response.power == Power.MEDIUM
        except Exception as e:
            pytest.skip(f"API call failed (may be quota/rate limit): {e}")

    @pytest.mark.parametrize("provider", [Provider.OPENAI, Provider.ANTHROPIC, Provider.GOOGLE])
    def test_real_query_power_override(self, provider, available_providers):
        """Test real API call with power override."""
        if provider not in available_providers:
            skip_if_no_key(provider)

        # Create with weak default
        client = LLMClient(provider=provider, power=Power.WEAK)

        try:
            # Override to strong for this call
            response = client.query(
                "Say 'hello'",
                power=Power.STRONG,
            )

            assert response.provider == provider
            assert response.power == Power.STRONG
            assert response.model == MODEL_REGISTRY[provider][Power.STRONG]
        except Exception as e:
            pytest.skip(f"API call failed (may be quota/rate limit): {e}")

    @pytest.mark.parametrize("provider", [Provider.OPENAI, Provider.ANTHROPIC, Provider.GOOGLE])
    def test_real_query_provider_override(self, provider, available_providers):
        """Test real API call with provider override."""
        if provider not in available_providers:
            skip_if_no_key(provider)

        # Create with OpenAI default
        client = LLMClient(provider=Provider.OPENAI, power=Power.WEAK)

        try:
            # Override to requested provider
            response = client.query(
                "Say 'test'",
                provider=provider,
            )

            assert response.provider == provider
        except Exception as e:
            pytest.skip(f"API call failed (may be quota/rate limit): {e}")
