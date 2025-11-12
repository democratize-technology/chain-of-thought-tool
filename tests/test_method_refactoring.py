#!/usr/bin/env python3
"""
Test to verify that refactored methods maintain identical functionality.

These tests ensure that method refactoring doesn't change any behavior
or output format. All tests must pass before and after refactoring.
"""

import unittest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from chain_of_thought.core import ChainOfThought, ConfidenceCalibrator

def test_add_step_functionality_preserved():
    """Test that add_step method maintains identical behavior after refactoring."""

    # Create a ChainOfThought instance
    cot = ChainOfThought()

    # Test data for comprehensive coverage
    test_cases = [
        {
            "name": "Basic step",
            "params": {
                "thought": "This is a test step",
                "step_number": 1,
                "total_steps": 3,
                "next_step_needed": True,
                "reasoning_stage": "Analysis",
                "confidence": 0.8
            }
        },
        {
            "name": "Complex step with all parameters",
            "params": {
                "thought": "Complex reasoning about market trends",
                "step_number": 2,
                "total_steps": 5,
                "next_step_needed": True,
                "reasoning_stage": "Research",
                "confidence": 0.7,
                "dependencies": [1],
                "contradicts": [],
                "evidence": ["Market data Q1", "Industry report"],
                "assumptions": ["Market stability", "No major disruptions"]
            }
        },
        {
            "name": "Step revision",
            "params": {
                "thought": "Revised analysis with more confidence",
                "step_number": 1,  # Same as first step - should replace
                "total_steps": 3,
                "next_step_needed": False,
                "reasoning_stage": "Conclusion",
                "confidence": 0.95
            }
        }
    ]

    for test_case in test_cases:
        result = cot.add_step(**test_case["params"])

        # Verify response structure
        assert isinstance(result, dict), f"Response should be dict for {test_case['name']}"
        assert "status" in result, f"Missing 'status' field for {test_case['name']}"
        assert result["status"] == "success", f"Status should be 'success' for {test_case['name']}"

        # Verify required fields
        required_fields = [
            "step_processed", "progress", "confidence", "feedback",
            "next_step_needed", "total_steps_recorded", "is_revision"
        ]
        for field in required_fields:
            assert field in result, f"Missing field '{field}' for {test_case['name']}"

        # Verify field types
        assert isinstance(result["step_processed"], int), f"step_processed should be int for {test_case['name']}"
        assert isinstance(result["confidence"], (int, float)), f"confidence should be number for {test_case['name']}"
        assert isinstance(result["feedback"], str), f"feedback should be string for {test_case['name']}"
        assert isinstance(result["is_revision"], bool), f"is_revision should be bool for {test_case['name']}"

        # Verify progress format
        progress = result["progress"]
        assert "/" in progress, f"Progress should contain '/' for {test_case['name']}"
        step_num, total = progress.split("/")
        assert step_num.isdigit(), f"Progress step should be digit for {test_case['name']}"
        assert total.isdigit(), f"Progress total should be digit for {test_case['name']}"

    print("✅ add_step functionality preserved test passed!")

def test_calibrate_confidence_functionality_preserved():
    """Test that calibrate_confidence method maintains identical behavior after refactoring."""

    calibrator = ConfidenceCalibrator()

    test_cases = [
        {
            "name": "High confidence prediction",
            "params": {
                "prediction": "This will definitely succeed",
                "initial_confidence": 0.95,
                "context": "Based on strong evidence"
            }
        },
        {
            "name": "Medium confidence with AI context",
            "params": {
                "prediction": "AI technology will transform education by 2030",
                "initial_confidence": 0.7,
                "context": ""
            }
        },
        {
            "name": "Complex prediction with limited data",
            "params": {
                "prediction": "Market will grow 25% next quarter based on limited data",
                "initial_confidence": 0.8,
                "context": "limited data available for analysis"
            }
        }
    ]

    for test_case in test_cases:
        result = calibrator.calibrate_confidence(**test_case["params"])

        # Verify response structure
        assert isinstance(result, dict), f"Response should be dict for {test_case['name']}"
        assert "status" in result, f"Missing 'status' field for {test_case['name']}"
        assert result["status"] == "success", f"Status should be 'success' for {test_case['name']}"

        # Verify required top-level fields
        required_fields = [
            "prediction", "original_confidence", "calibrated_confidence",
            "confidence_band", "adjustment", "overconfidence_analysis",
            "uncertainty_factors", "insights", "metadata"
        ]
        for field in required_fields:
            assert field in result, f"Missing field '{field}' for {test_case['name']}"

        # Verify confidence band structure
        confidence_band = result["confidence_band"]
        assert "lower_bound" in confidence_band, f"Missing lower_bound in confidence_band for {test_case['name']}"
        assert "upper_bound" in confidence_band, f"Missing upper_bound in confidence_band for {test_case['name']}"
        assert "range" in confidence_band, f"Missing range in confidence_band for {test_case['name']}"

        # Verify adjustment structure
        adjustment = result["adjustment"]
        assert "magnitude" in adjustment, f"Missing magnitude in adjustment for {test_case['name']}"
        assert "direction" in adjustment, f"Missing direction in adjustment for {test_case['name']}"
        assert "reasoning" in adjustment, f"Missing reasoning in adjustment for {test_case['name']}"

        # Verify overconfidence analysis structure
        overconfidence = result["overconfidence_analysis"]
        assert "risk_level" in overconfidence, f"Missing risk_level in overconfidence_analysis for {test_case['name']}"
        assert "indicators" in overconfidence, f"Missing indicators in overconfidence_analysis for {test_case['name']}"
        assert "score" in overconfidence, f"Missing score in overconfidence_analysis for {test_case['name']}"

        # Verify insights structure
        insights = result["insights"]
        assert "confidence_appropriate" in insights, f"Missing confidence_appropriate in insights for {test_case['name']}"
        assert "high_uncertainty" in insights, f"Missing high_uncertainty in insights for {test_case['name']}"
        assert "needs_more_evidence" in insights, f"Missing needs_more_evidence in insights for {test_case['name']}"

    print("✅ calibrate_confidence functionality preserved test passed!")

def test_method_line_counts_reduced():
    """Test that refactored methods have fewer than 50 lines each."""

    import inspect

    # Check add_step method line count
    add_step_source = inspect.getsource(ChainOfThought.add_step)
    add_step_lines = [line for line in add_step_source.split('\n') if line.strip() and not line.strip().startswith('#') or line.strip().startswith('def') or line.strip().startswith('"""')]

    # Check calibrate_confidence method line count
    calibrate_confidence_source = inspect.getsource(ConfidenceCalibrator.calibrate_confidence)
    calibrate_confidence_lines = [line for line in calibrate_confidence_source.split('\n') if line.strip() and not line.strip().startswith('#') or line.strip().startswith('def') or line.strip().startswith('"""')]

    # Assert methods are under 50 lines
    assert len(add_step_lines) < 50, f"add_step method still has {len(add_step_lines)} lines, should be < 50"
    assert len(calibrate_confidence_lines) < 50, f"calibrate_confidence method still has {len(calibrate_confidence_lines)} lines, should be < 50"

    print(f"✅ Method line count test passed! add_step: {len(add_step_lines)} lines, calibrate_confidence: {len(calibrate_confidence_lines)} lines")

if __name__ == "__main__":
    """Run all refactoring functionality tests."""

    print("🧪 Testing method refactoring functionality preservation...")

    try:
        # Run tests that check behavior is preserved
        test_add_step_functionality_preserved()
        test_calibrate_confidence_functionality_preserved()

        # Run test that checks line count is reduced (will fail before refactoring)
        test_method_line_counts_reduced()

        print("\n🎉 All refactoring tests passed successfully!")
        print("   ✓ add_step functionality preserved")
        print("   ✓ calibrate_confidence functionality preserved")
        print("   ✓ Method line counts reduced below 50")

    except AssertionError as e:
        print(f"\n❌ Refactoring test failed: {e}")
        print("   This is expected before refactoring is complete.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during refactoring testing: {e}")
        sys.exit(1)