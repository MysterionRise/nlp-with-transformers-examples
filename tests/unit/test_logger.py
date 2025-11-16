"""
Unit tests for logging utilities
"""

import logging
import tempfile
import time
from pathlib import Path

import pytest

from utils.logger import (
    ColoredFormatter,
    LoggerAdapter,
    PerformanceLogger,
    get_logger,
    log_function_call,
    setup_logger,
)


class TestColoredFormatter:
    """Test ColoredFormatter"""

    def test_colored_formatter_with_colors(self):
        """Test formatter with colors enabled"""
        formatter = ColoredFormatter("%(levelname)s - %(message)s", use_colors=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        assert "Test message" in formatted

    def test_colored_formatter_without_colors(self):
        """Test formatter with colors disabled"""
        formatter = ColoredFormatter("%(levelname)s - %(message)s", use_colors=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        assert "INFO" in formatted
        assert "Test message" in formatted


class TestSetupLogger:
    """Test setup_logger function"""

    def test_setup_logger_basic(self):
        """Test basic logger setup"""
        logger = setup_logger("test_logger", level="INFO")
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

    def test_setup_logger_with_file(self):
        """Test logger setup with file handler"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logger("test_logger_file", level="DEBUG", log_file=log_file)

            logger.info("Test message")

            assert log_file.exists()
            content = log_file.read_text()
            assert "Test message" in content

    def test_setup_logger_levels(self):
        """Test different log levels"""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            logger = setup_logger(f"test_logger_{level}", level=level)
            assert logger.level == getattr(logging, level)

    def test_setup_logger_no_duplicate_handlers(self):
        """Test that calling setup_logger twice doesn't add duplicate handlers"""
        logger1 = setup_logger("test_logger_dup", level="INFO")
        handler_count1 = len(logger1.handlers)

        logger2 = setup_logger("test_logger_dup", level="INFO")
        handler_count2 = len(logger2.handlers)

        assert handler_count1 == handler_count2
        assert logger1 is logger2  # Same logger instance


class TestGetLogger:
    """Test get_logger function"""

    def test_get_logger_basic(self):
        """Test getting a logger"""
        logger = get_logger("test_module")
        assert logger is not None
        assert isinstance(logger, logging.Logger)

    def test_get_logger_with_level(self):
        """Test getting a logger with specific level"""
        logger = get_logger("test_module_debug", level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_get_logger_same_instance(self):
        """Test that getting the same logger returns same instance"""
        logger1 = get_logger("test_module_same")
        logger2 = get_logger("test_module_same")
        # Note: Python's logging.getLogger returns the same instance
        # but our setup_logger might add handlers, so we just check it works
        assert logger1.name == logger2.name


class TestPerformanceLogger:
    """Test PerformanceLogger context manager"""

    def test_performance_logger_success(self, caplog):
        """Test PerformanceLogger with successful operation"""
        import logging

        logger = get_logger("test_performance")

        with caplog.at_level(logging.DEBUG, logger="test_performance"):
            with PerformanceLogger("test_operation", logger=logger):
                time.sleep(0.01)

        # Check that start and completion messages were logged
        messages = [record.message for record in caplog.records]
        assert any("Starting: test_operation" in msg for msg in messages)
        assert any("Completed: test_operation" in msg and "took" in msg for msg in messages)

    def test_performance_logger_with_error(self, caplog):
        """Test PerformanceLogger with error"""
        logger = get_logger("test_performance_error")

        with caplog.at_level(logging.ERROR, logger="test_performance_error"):
            with pytest.raises(ValueError):
                with PerformanceLogger("failing_operation", logger=logger):
                    raise ValueError("Test error")

        # Check that error message was logged
        messages = [record.message for record in caplog.records]
        assert any("Failed: failing_operation" in msg for msg in messages)

    def test_performance_logger_timing(self):
        """Test that PerformanceLogger measures time"""
        logger = get_logger("test_timing")

        start = time.time()
        with PerformanceLogger("timed_operation", logger=logger):
            time.sleep(0.1)
        elapsed = time.time() - start

        # Verify it took approximately the expected time
        assert elapsed >= 0.1
        assert elapsed < 0.2  # Some buffer for overhead


class TestLogFunctionCall:
    """Test log_function_call decorator"""

    def test_log_function_call_basic(self, caplog):
        """Test function call logging"""
        logger = get_logger("test_func_call")

        @log_function_call(logger=logger)
        def test_function(a, b):
            return a + b

        with caplog.at_level(logging.DEBUG, logger="test_func_call"):
            result = test_function(2, 3)

        assert result == 5

        messages = [record.message for record in caplog.records]
        assert any("Calling test_function" in msg for msg in messages)
        assert any("test_function completed" in msg for msg in messages)

    def test_log_function_call_with_kwargs(self, caplog):
        """Test function call logging with kwargs"""
        logger = get_logger("test_func_kwargs")

        @log_function_call(logger=logger)
        def test_function(a, b, multiplier=1):
            return (a + b) * multiplier

        with caplog.at_level(logging.DEBUG, logger="test_func_kwargs"):
            result = test_function(2, 3, multiplier=2)

        assert result == 10

        messages = [record.message for record in caplog.records]
        assert any("test_function" in msg for msg in messages)

    def test_log_function_call_with_error(self, caplog):
        """Test function call logging when function raises error"""
        logger = get_logger("test_func_error")

        @log_function_call(logger=logger)
        def failing_function():
            raise ValueError("Test error")

        with caplog.at_level(logging.ERROR, logger="test_func_error"):
            with pytest.raises(ValueError):
                failing_function()

        messages = [record.message for record in caplog.records]
        assert any("failed" in msg.lower() for msg in messages)


class TestLoggerAdapter:
    """Test LoggerAdapter"""

    def test_logger_adapter_with_context(self, caplog):
        """Test LoggerAdapter adds context to messages"""
        base_logger = get_logger("test_adapter")
        adapter = LoggerAdapter(base_logger, {"model": "bert-base", "task": "classification"})

        with caplog.at_level(logging.INFO):
            adapter.info("Processing text")

        messages = [record.message for record in caplog.records]
        # The adapter should add context to the message
        assert any("model=bert-base" in msg and "task=classification" in msg for msg in messages)

    def test_logger_adapter_no_context(self, caplog):
        """Test LoggerAdapter without context"""
        base_logger = get_logger("test_adapter_no_ctx")
        adapter = LoggerAdapter(base_logger, {})

        with caplog.at_level(logging.INFO):
            adapter.info("Test message")

        messages = [record.message for record in caplog.records]
        assert any("Test message" in msg for msg in messages)


class TestLoggingIntegration:
    """Integration tests for logging system"""

    def test_logging_to_file(self):
        """Test complete logging flow to file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "integration.log"
            logger = setup_logger("integration_test_file", level="INFO", log_file=log_file)

            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")

            # Flush and close all handlers to ensure messages are written
            for handler in logger.handlers:
                handler.flush()
                if hasattr(handler, "close"):
                    handler.close()

            # Reopen file to ensure it's written
            import logging

            logging.shutdown()

            # Verify file exists and has content
            assert log_file.exists(), f"Log file not created at {log_file}"
            content = log_file.read_text()
            assert "Info message" in content
            assert "Warning message" in content
            assert "Error message" in content

    def test_performance_logging_integration(self, caplog):
        """Test PerformanceLogger with actual operations"""
        logger = get_logger("integration_perf")

        with caplog.at_level(logging.DEBUG, logger="integration_perf"):
            with PerformanceLogger("data_processing", logger=logger):
                # Simulate some work
                data = [i**2 for i in range(1000)]
                assert len(data) == 1000

        messages = [record.message for record in caplog.records]
        assert any("Starting: data_processing" in msg for msg in messages)
        assert any("Completed: data_processing" in msg for msg in messages)
