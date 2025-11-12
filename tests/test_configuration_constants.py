"""
Tests for configuration constants - Task #4 Magic Numbers Extraction

This test validates that magic numbers have been extracted to named constants
for improved maintainability. Initially, this test will FAIL because the
constants don't exist yet. After implementation, this test should PASS.
"""

import pytest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from chain_of_thought.core import (
    RateLimiter,
    ConfidenceCalibrator,
    _safe_json_dumps
)


def test_rate_limiter_uses_default_constants():
    """Test that RateLimiter uses configuration constants instead of magic numbers"""
    # Create a rate limiter with default values
    limiter = RateLimiter()

    # These should use configuration constants, not magic numbers
    assert limiter.max_requests_per_minute == 60  # Should use DEFAULT_MAX_REQUESTS_PER_MINUTE
    assert limiter.max_requests_per_hour == 1000   # Should use DEFAULT_MAX_REQUESTS_PER_HOUR
    assert limiter.max_burst_size == 10            # Should use DEFAULT_MAX_BURST_SIZE

    # Verify constants exist and are accessible
    from chain_of_thought.core import (
        DEFAULT_MAX_REQUESTS_PER_MINUTE,
        DEFAULT_MAX_REQUESTS_PER_HOUR,
        DEFAULT_MAX_BURST_SIZE
    )

    assert DEFAULT_MAX_REQUESTS_PER_MINUTE == 60
    assert DEFAULT_MAX_REQUESTS_PER_HOUR == 1000
    assert DEFAULT_MAX_BURST_SIZE == 10


def test_configuration_constants_exist():
    """Test that core module configuration constants exist"""
    # These constants should be defined in the core module
    # Initially this will fail because they don't exist
    try:
        from chain_of_thought.core import (
            DEFAULT_MAX_REQUESTS_PER_MINUTE,
            DEFAULT_MAX_REQUESTS_PER_HOUR,
            DEFAULT_MAX_BURST_SIZE,
            MAX_RECURSION_DEPTH,
            MAX_LIST_SIZE,
            MAX_STRING_LENGTH,
            MAX_JSON_SIZE,
            HIGH_CONFIDENCE_THRESHOLD,
            MEDIUM_CONFIDENCE_THRESHOLD,
            MAX_PREDICTION_WORDS
        )
        # If import succeeds, verify values
        assert DEFAULT_MAX_REQUESTS_PER_MINUTE == 60
        assert DEFAULT_MAX_REQUESTS_PER_HOUR == 1000
        assert DEFAULT_MAX_BURST_SIZE == 10
        assert MAX_RECURSION_DEPTH == 50
        assert MAX_LIST_SIZE == 100
        assert MAX_STRING_LENGTH == 1000
        assert MAX_JSON_SIZE == 100000
        assert HIGH_CONFIDENCE_THRESHOLD == 0.15
        assert MEDIUM_CONFIDENCE_THRESHOLD == 0.05
        assert MAX_PREDICTION_WORDS == 20
    except ImportError as e:
        pytest.fail(f"Configuration constants not defined in core module: {e}")


def test_json_size_limit_constant():
    """Test that JSON size limit uses configuration constant"""
    try:
        from chain_of_thought.core import MAX_JSON_SIZE
        assert MAX_JSON_SIZE == 100000

        # Test the _safe_json_dumps function respects this limit
        large_data = {"data": "x" * 200000}  # Much larger than limit
        result = _safe_json_dumps(large_data)

        # Should handle large data gracefully
        assert result is not None
        assert isinstance(result, str)
    except ImportError as e:
        pytest.fail(f"MAX_JSON_SIZE constant not defined: {e}")


def test_confidence_calibrator_constants():
    """Test that ConfidenceCalibrator uses threshold constants"""
    try:
        from chain_of_thought.core import (
            HIGH_CONFIDENCE_THRESHOLD,
            MEDIUM_CONFIDENCE_THRESHOLD
        )
        assert HIGH_CONFIDENCE_THRESHOLD == 0.15
        assert MEDIUM_CONFIDENCE_THRESHOLD == 0.05

        # Test that calibration logic works with these thresholds
        calibrator = ConfidenceCalibrator()
        # This exercises the _generate_calibration_reasoning method which uses the thresholds
        reasoning = calibrator._generate_calibration_reasoning(0.20, "high")
        assert "Significant confidence reduction" in reasoning
        assert "0.20" in reasoning

        reasoning = calibrator._generate_calibration_reasoning(0.10, "medium")
        assert "Moderate confidence adjustment" in reasoning

        reasoning = calibrator._generate_calibration_reasoning(0.02, "low")
        assert "Minor confidence adjustment" in reasoning
    except ImportError as e:
        pytest.fail(f"Confidence threshold constants not defined: {e}")


def test_all_core_configuration_constants_exist():
    """Test that all expected core module configuration constants are defined and accessible"""
    try:
        # Core module constants
        from chain_of_thought.core import (
            DEFAULT_MAX_REQUESTS_PER_MINUTE,
            DEFAULT_MAX_REQUESTS_PER_HOUR,
            DEFAULT_MAX_BURST_SIZE,
            MAX_RECURSION_DEPTH,
            MAX_LIST_SIZE,
            MAX_STRING_LENGTH,
            MAX_JSON_SIZE,
            HIGH_CONFIDENCE_THRESHOLD,
            MEDIUM_CONFIDENCE_THRESHOLD,
            MAX_PREDICTION_WORDS
        )

        # Verify all constants have expected values
        constants_map = {
            DEFAULT_MAX_REQUESTS_PER_MINUTE: 60,
            DEFAULT_MAX_REQUESTS_PER_HOUR: 1000,
            DEFAULT_MAX_BURST_SIZE: 10,
            MAX_RECURSION_DEPTH: 50,
            MAX_LIST_SIZE: 100,
            MAX_STRING_LENGTH: 1000,
            MAX_JSON_SIZE: 100000,
            HIGH_CONFIDENCE_THRESHOLD: 0.15,
            MEDIUM_CONFIDENCE_THRESHOLD: 0.05,
            MAX_PREDICTION_WORDS: 20
        }

        for constant, expected_value in constants_map.items():
            assert constant == expected_value, f"Constant {constant} should be {expected_value}"
    except ImportError as e:
        pytest.fail(f"Core configuration constants not defined: {e}")


def test_validators_configuration_constants_exist():
    """Test that validators module configuration constants exist"""
    try:
        from chain_of_thought.validators import MAX_THOUGHT_LENGTH
        assert MAX_THOUGHT_LENGTH == 10000
    except ImportError as e:
        pytest.fail(f"Validators configuration constants not defined: {e}")


if __name__ == "__main__":
    # Run these tests to verify configuration constants
    pytest.main([__file__, "-v"])