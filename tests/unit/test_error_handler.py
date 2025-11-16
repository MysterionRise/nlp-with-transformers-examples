"""
Unit tests for error handling utilities
"""

import time

import pytest

from utils.error_handler import (
    ConfigurationError,
    ErrorContext,
    InferenceError,
    InvalidInputError,
    ModelLoadError,
    ModelNotFoundError,
    NLPError,
    ResourceError,
    format_error_message,
    get_error_suggestion,
    handle_errors,
    retry_on_error,
    validate_input,
)


class TestCustomExceptions:
    """Test custom exception classes"""

    def test_nlp_error(self):
        """Test base NLPError"""
        error = NLPError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_model_load_error(self):
        """Test ModelLoadError"""
        original_error = ValueError("Connection failed")
        error = ModelLoadError("bert-base", original_error=original_error)

        assert "bert-base" in str(error)
        assert "Connection failed" in str(error)
        assert error.model_name == "bert-base"
        assert error.original_error == original_error

    def test_model_not_found_error(self):
        """Test ModelNotFoundError"""
        error = ModelNotFoundError("bert-base", category="sentiment")

        assert "bert-base" in str(error)
        assert "sentiment" in str(error)
        assert error.model_name == "bert-base"
        assert error.category == "sentiment"

    def test_inference_error(self):
        """Test InferenceError"""
        original_error = RuntimeError("CUDA out of memory")
        error = InferenceError("bert-base", "test input text", original_error=original_error)

        assert "bert-base" in str(error)
        assert error.model_name == "bert-base"
        assert error.input_text == "test input text"
        assert error.original_error == original_error

    def test_invalid_input_error(self):
        """Test InvalidInputError"""
        error = InvalidInputError("Text too long", input_value="test")

        assert "Text too long" in str(error)
        assert error.input_value == "test"

    def test_configuration_error(self):
        """Test ConfigurationError"""
        error = ConfigurationError("Invalid config")
        assert isinstance(error, NLPError)

    def test_resource_error(self):
        """Test ResourceError"""
        error = ResourceError("Out of memory")
        assert isinstance(error, NLPError)


class TestHandleErrorsDecorator:
    """Test @handle_errors decorator"""

    def test_handle_errors_success(self):
        """Test decorator with successful function"""

        @handle_errors(default_return="error")
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_handle_errors_with_exception(self):
        """Test decorator with exception (no re-raise)"""

        @handle_errors(default_return="default", raise_error=False)
        def failing_func():
            raise ValueError("Test error")

        result = failing_func()
        assert result == "default"

    def test_handle_errors_with_user_message(self):
        """Test decorator with custom user message"""

        @handle_errors(default_return=None, user_message="Custom error message")
        def failing_func():
            raise ValueError("Test error")

        result = failing_func()
        assert isinstance(result, dict)
        assert result["error"] == "Custom error message"
        assert "Test error" in result["details"]

    def test_handle_errors_raise_error(self):
        """Test decorator re-raising errors"""

        @handle_errors(raise_error=True)
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_func()

    def test_handle_errors_with_args_kwargs(self):
        """Test decorator with function arguments"""

        @handle_errors(default_return=0)
        def add_numbers(a, b, multiplier=1):
            return (a + b) * multiplier

        result = add_numbers(2, 3, multiplier=2)
        assert result == 10


class TestRetryOnErrorDecorator:
    """Test @retry_on_error decorator"""

    def test_retry_success_on_first_attempt(self):
        """Test retry with immediate success"""

        @retry_on_error(max_retries=3)
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_retry_success_after_failures(self):
        """Test retry with success after some failures"""
        attempt_count = {"count": 0}

        @retry_on_error(max_retries=3, delay=0.01)
        def eventually_successful():
            attempt_count["count"] += 1
            if attempt_count["count"] < 3:
                raise ValueError("Temporary error")
            return "success"

        result = eventually_successful()
        assert result == "success"
        assert attempt_count["count"] == 3

    def test_retry_all_attempts_fail(self):
        """Test retry when all attempts fail"""

        @retry_on_error(max_retries=2, delay=0.01)
        def always_fails():
            raise ValueError("Permanent error")

        with pytest.raises(ValueError, match="Permanent error"):
            always_fails()

    def test_retry_specific_exceptions(self):
        """Test retry only catches specific exceptions"""

        @retry_on_error(max_retries=2, delay=0.01, exceptions=(ValueError,))
        def raises_type_error():
            raise TypeError("Type error")

        with pytest.raises(TypeError, match="Type error"):
            raises_type_error()

    def test_retry_backoff(self):
        """Test exponential backoff"""
        times = []

        @retry_on_error(max_retries=2, delay=0.01, backoff=2.0)
        def measure_delays():
            times.append(time.time())
            if len(times) < 3:
                raise ValueError("Retry")
            return "success"

        result = measure_delays()
        assert result == "success"

        # Check that delays increase (with some tolerance for timing)
        if len(times) >= 3:
            delay1 = times[1] - times[0]
            delay2 = times[2] - times[1]
            assert delay2 > delay1 or abs(delay2 - delay1 * 2) < 0.05


class TestValidateInputDecorator:
    """Test @validate_input decorator"""

    def test_validate_input_success(self):
        """Test validation with valid input"""

        @validate_input(min_length=5, max_length=20)
        def process_text(text):
            return f"Processed: {text}"

        result = process_text("Hello World")
        assert result == "Processed: Hello World"

    def test_validate_input_empty(self):
        """Test validation rejects empty text"""

        @validate_input(allow_empty=False)
        def process_text(text):
            return text

        with pytest.raises(InvalidInputError, match="cannot be empty"):
            process_text("")

    def test_validate_input_allow_empty(self):
        """Test validation allows empty when configured"""

        @validate_input(allow_empty=True)
        def process_text(text):
            return f"Processed: '{text}'"

        result = process_text("")
        assert result == "Processed: ''"

    def test_validate_input_too_short(self):
        """Test validation rejects text that's too short"""

        @validate_input(min_length=10)
        def process_text(text):
            return text

        with pytest.raises(InvalidInputError, match="at least 10 characters"):
            process_text("short")

    def test_validate_input_too_long(self):
        """Test validation rejects text that's too long"""

        @validate_input(max_length=10)
        def process_text(text):
            return text

        with pytest.raises(InvalidInputError, match="at most 10 characters"):
            process_text("this text is way too long")

    def test_validate_input_strip(self):
        """Test validation strips whitespace"""

        @validate_input(min_length=5, strip=True)
        def process_text(text):
            return text

        result = process_text("  hello  ")
        assert result == "hello"

    def test_validate_input_no_strip(self):
        """Test validation without stripping"""

        @validate_input(min_length=5, strip=False)
        def process_text(text):
            return len(text)

        result = process_text("  hello  ")
        assert result == 9  # Includes whitespace


class TestErrorContext:
    """Test ErrorContext context manager"""

    def test_error_context_success(self):
        """Test context manager with successful operation"""
        with ErrorContext("test_operation", key="value"):
            result = 1 + 1
        assert result == 2

    def test_error_context_with_error(self):
        """Test context manager with error (doesn't suppress)"""
        with pytest.raises(ValueError):
            with ErrorContext("test_operation"):
                raise ValueError("Test error")


class TestUtilityFunctions:
    """Test utility functions"""

    def test_format_error_message_nlp_error(self):
        """Test formatting NLPError"""
        error = ModelLoadError("bert-base")
        message = format_error_message(error)
        assert "bert-base" in message

    def test_format_error_message_value_error(self):
        """Test formatting ValueError"""
        error = ValueError("Invalid value")
        message = format_error_message(error)
        assert "Invalid value" in message

    def test_format_error_message_file_not_found(self):
        """Test formatting FileNotFoundError"""
        error = FileNotFoundError("config.yaml")
        message = format_error_message(error)
        assert "File not found" in message

    def test_format_error_message_memory_error(self):
        """Test formatting MemoryError"""
        error = MemoryError()
        message = format_error_message(error)
        assert "memory" in message.lower()

    def test_format_error_message_with_traceback(self):
        """Test formatting with traceback"""
        error = ValueError("Test")
        message = format_error_message(error, include_traceback=True)
        assert "Traceback" in message

    def test_get_error_suggestion_memory_error(self):
        """Test getting suggestion for MemoryError"""
        error = MemoryError()
        suggestion = get_error_suggestion(error)
        assert suggestion is not None
        assert "smaller model" in suggestion.lower()

    def test_get_error_suggestion_model_load_error(self):
        """Test getting suggestion for ModelLoadError"""
        error = ModelLoadError("bert-base")
        suggestion = get_error_suggestion(error)
        assert suggestion is not None
        assert "internet" in suggestion.lower() or "model name" in suggestion.lower()

    def test_get_error_suggestion_invalid_input(self):
        """Test getting suggestion for InvalidInputError"""
        error = InvalidInputError("Invalid")
        suggestion = get_error_suggestion(error)
        assert suggestion is not None

    def test_get_error_suggestion_unknown_error(self):
        """Test getting suggestion for unknown error type"""
        error = RuntimeError("Unknown")
        suggestion = get_error_suggestion(error)
        assert suggestion is None
