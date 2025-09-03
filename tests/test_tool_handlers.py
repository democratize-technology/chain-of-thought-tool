"""
Unit tests for tool handler functions.

Tests cover all 6 tool handlers:
- chain_of_thought_step_handler
- get_chain_summary_handler  
- clear_chain_handler
- generate_hypotheses_handler
- map_assumptions_handler
- calibrate_confidence_handler

Includes input validation, error handling, and JSON output format validation.
"""
import json
import pytest
from chain_of_thought.core import (
    chain_of_thought_step_handler,
    get_chain_summary_handler,
    clear_chain_handler,
    generate_hypotheses_handler,
    map_assumptions_handler,
    calibrate_confidence_handler,
    _chain_processor,
    _hypothesis_generator,
    _assumption_mapper,
    _confidence_calibrator
)


class TestChainOfThoughtStepHandler:
    """Test chain_of_thought_step_handler function."""
    
    def setup_method(self):
        """Clear the global chain processor before each test."""
        _chain_processor.clear_chain()
    
    def test_valid_step_addition(self):
        """Test adding a valid step through handler."""
        result_json = chain_of_thought_step_handler(
            thought="Test reasoning step",
            step_number=1,
            total_steps=3,
            next_step_needed=True,
            reasoning_stage="Analysis",
            confidence=0.8
        )
        
        result = json.loads(result_json)
        assert result["status"] == "success"
        assert result["step_processed"] == 1
        assert result["progress"] == "1/3"
        assert result["confidence"] == 0.8
        assert result["next_step_needed"] is True
        assert result["is_revision"] is False
        assert "feedback" in result
    
    def test_step_with_metadata(self):
        """Test adding step with dependencies, contradictions, evidence, and assumptions."""
        result_json = chain_of_thought_step_handler(
            thought="Complex analysis step",
            step_number=1,
            total_steps=2,
            next_step_needed=True,
            dependencies=[],
            contradicts=[],
            evidence=["Market research", "User surveys"],
            assumptions=["Market stability", "User behavior patterns"]
        )
        
        result = json.loads(result_json)
        assert result["status"] == "success"
        
        # Verify the step was added with metadata
        assert len(_chain_processor.steps) == 1
        step = _chain_processor.steps[0]
        assert step.evidence == ["Market research", "User surveys"]
        assert step.assumptions == ["Market stability", "User behavior patterns"]
    
    def test_step_revision(self):
        """Test revising an existing step."""
        # Add initial step
        chain_of_thought_step_handler(
            thought="Original thought",
            step_number=1,
            total_steps=2,
            next_step_needed=True
        )
        
        # Revise the step
        result_json = chain_of_thought_step_handler(
            thought="Revised thought", 
            step_number=1,
            total_steps=2,
            next_step_needed=True
        )
        
        result = json.loads(result_json)
        assert result["is_revision"] is True
        assert len(_chain_processor.steps) == 1
        assert _chain_processor.steps[0].thought == "Revised thought"
    
    def test_invalid_parameters_handled(self):
        """Test that invalid parameters don't crash the handler."""
        # Missing required parameters should be handled gracefully
        result_json = chain_of_thought_step_handler()
        result = json.loads(result_json)
        assert result["status"] == "error"
        assert "message" in result
    
    def test_json_output_format(self):
        """Test that output is properly formatted JSON."""
        result_json = chain_of_thought_step_handler(
            thought="Test",
            step_number=1,
            total_steps=1,
            next_step_needed=False
        )
        
        # Should be valid JSON
        result = json.loads(result_json)
        assert isinstance(result, dict)
        
        # Should be formatted with indentation
        assert "\n" in result_json
        assert "  " in result_json


class TestGetChainSummaryHandler:
    """Test get_chain_summary_handler function."""
    
    def setup_method(self):
        """Clear the global chain processor before each test."""
        _chain_processor.clear_chain()
    
    def test_empty_chain_summary(self):
        """Test getting summary of empty chain."""
        result_json = get_chain_summary_handler()
        result = json.loads(result_json)
        
        assert result["status"] == "empty"
        assert "message" in result
    
    def test_chain_with_steps_summary(self):
        """Test getting summary of chain with steps."""
        # Add some steps
        _chain_processor.add_step("Step 1", 1, 3, True, confidence=0.7)
        _chain_processor.add_step("Step 2", 2, 3, True, confidence=0.8) 
        _chain_processor.add_step("Step 3", 3, 3, False, confidence=0.9)
        
        result_json = get_chain_summary_handler()
        result = json.loads(result_json)
        
        assert result["status"] == "success"
        assert result["total_steps"] == 3
        assert result["overall_confidence"] == 0.8  # (0.7 + 0.8 + 0.9) / 3
        assert len(result["chain"]) == 3
        assert "insights" in result
        assert "metadata" in result
    
    def test_error_handling(self):
        """Test error handling in summary generation."""
        # This is harder to trigger since summary is robust, 
        # but we can test the exception handling structure
        result_json = get_chain_summary_handler()
        result = json.loads(result_json)
        
        # Should always return valid JSON, even in error cases
        assert isinstance(result, dict)
    
    def test_json_formatting(self):
        """Test JSON output formatting."""
        result_json = get_chain_summary_handler()
        
        # Should be valid, formatted JSON
        result = json.loads(result_json)
        assert isinstance(result, dict)
        assert "\n" in result_json


class TestClearChainHandler:
    """Test clear_chain_handler function."""
    
    def setup_method(self):
        """Setup chain with some steps for testing."""
        _chain_processor.clear_chain()
        _chain_processor.add_step("Test step", 1, 1, False)
    
    def test_clear_chain_success(self):
        """Test successful chain clearing."""
        # Verify chain has content
        assert len(_chain_processor.steps) == 1
        
        result_json = clear_chain_handler()
        result = json.loads(result_json)
        
        assert result["status"] == "success"
        assert "message" in result
        assert len(_chain_processor.steps) == 0
    
    def test_clear_empty_chain(self):
        """Test clearing already empty chain."""
        _chain_processor.clear_chain()  # Make sure it's empty
        
        result_json = clear_chain_handler()
        result = json.loads(result_json)
        
        assert result["status"] == "success"
        assert len(_chain_processor.steps) == 0
    
    def test_json_formatting(self):
        """Test JSON output formatting."""
        result_json = clear_chain_handler()
        
        result = json.loads(result_json)
        assert isinstance(result, dict)
        assert "\n" in result_json


class TestGenerateHypothesesHandler:
    """Test generate_hypotheses_handler function."""
    
    def test_valid_hypothesis_generation(self):
        """Test generating hypotheses with valid input."""
        result_json = generate_hypotheses_handler(
            observation="Sales dropped 30% this quarter",
            hypothesis_count=4
        )
        
        result = json.loads(result_json)
        assert result["status"] == "success"
        assert "hypotheses" in result
        assert len(result["hypotheses"]) <= 4
        assert "insights" in result
        
        # Check hypothesis structure (using actual field names from implementation)
        for hypothesis in result["hypotheses"]:
            assert "confidence" in hypothesis
            assert "evidence_needed" in hypothesis
            assert "rank" in hypothesis
            assert "reasoning" in hypothesis
    
    def test_hypothesis_generation_with_defaults(self):
        """Test hypothesis generation with default parameters."""
        result_json = generate_hypotheses_handler(
            observation="Website traffic increased significantly"
        )
        
        result = json.loads(result_json)
        assert result["status"] == "success"
        assert "hypotheses" in result
        # Default should generate 4 hypotheses
        assert len(result["hypotheses"]) <= 4
    
    def test_hypothesis_generation_limited_count(self):
        """Test generating limited number of hypotheses."""
        result_json = generate_hypotheses_handler(
            observation="Customer complaints increased",
            hypothesis_count=2
        )
        
        result = json.loads(result_json)
        assert result["status"] == "success"
        assert len(result["hypotheses"]) <= 2
    
    def test_invalid_observation_handling(self):
        """Test handling of invalid or missing observation."""
        result_json = generate_hypotheses_handler()
        result = json.loads(result_json)
        assert result["status"] == "error"
        assert "message" in result
    
    def test_invalid_hypothesis_count(self):
        """Test handling of invalid hypothesis count."""
        result_json = generate_hypotheses_handler(
            observation="Test observation",
            hypothesis_count=10  # Above maximum of 4
        )
        
        result = json.loads(result_json)
        # Should handle gracefully, possibly clamping to max
        assert isinstance(result, dict)
    
    def test_empty_observation(self):
        """Test handling of empty observation."""
        result_json = generate_hypotheses_handler(observation="")
        result = json.loads(result_json)
        
        # Should handle gracefully
        assert isinstance(result, dict)


class TestMapAssumptionsHandler:
    """Test map_assumptions_handler function."""
    
    def test_valid_assumption_mapping(self):
        """Test mapping assumptions with valid input."""
        result_json = map_assumptions_handler(
            statement="Our new feature will increase user engagement by 25%",
            depth="surface"
        )
        
        result = json.loads(result_json)
        assert result["status"] == "success"
        assert "assumptions_found" in result
        assert "explicit" in result
        assert "implicit" in result
        
        # Check assumption structure (using actual field names)
        assert "critical" in result
        assert "depth" in result
    
    def test_deep_assumption_analysis(self):
        """Test deep assumption mapping."""
        result_json = map_assumptions_handler(
            statement="The market will remain stable for our product launch",
            depth="deep"
        )
        
        result = json.loads(result_json)
        assert result["status"] == "success"
        assert "assumptions_found" in result
        assert "explicit" in result
        assert "implicit" in result
    
    def test_assumption_mapping_defaults(self):
        """Test assumption mapping with default parameters."""
        result_json = map_assumptions_handler(
            statement="This strategy will work well"
        )
        
        result = json.loads(result_json)
        assert result["status"] == "success"
        # Default depth should be "surface"
        assert "assumptions_found" in result
        assert result["depth"] == "surface"
    
    def test_missing_statement_handling(self):
        """Test handling of missing statement."""
        result_json = map_assumptions_handler()
        result = json.loads(result_json)
        assert result["status"] == "error"
        assert "message" in result
    
    def test_empty_statement(self):
        """Test handling of empty statement."""
        result_json = map_assumptions_handler(statement="")
        result = json.loads(result_json)
        
        # Should handle gracefully
        assert isinstance(result, dict)
    
    def test_invalid_depth(self):
        """Test handling of invalid depth parameter."""
        result_json = map_assumptions_handler(
            statement="Test statement",
            depth="invalid_depth"
        )
        
        result = json.loads(result_json)
        # Should handle gracefully, possibly defaulting
        assert isinstance(result, dict)


class TestCalibrateConfidenceHandler:
    """Test calibrate_confidence_handler function."""
    
    def test_valid_confidence_calibration(self):
        """Test confidence calibration with valid input."""
        result_json = calibrate_confidence_handler(
            prediction="Our revenue will increase by 15% next quarter",
            initial_confidence=0.8,
            context="Based on current market trends and sales data"
        )
        
        result = json.loads(result_json)
        assert result["status"] == "success"
        assert "calibrated_confidence" in result
        assert "confidence_band" in result
        assert "adjustment" in result
        assert "insights" in result
        
        # Calibrated confidence should be between 0 and 1
        assert 0.0 <= result["calibrated_confidence"] <= 1.0
    
    def test_confidence_calibration_without_context(self):
        """Test confidence calibration without context."""
        result_json = calibrate_confidence_handler(
            prediction="This approach will be successful",
            initial_confidence=0.9
        )
        
        result = json.loads(result_json)
        assert result["status"] == "success"
        assert "calibrated_confidence" in result
    
    def test_edge_case_confidence_values(self):
        """Test calibration with edge case confidence values."""
        # Very low confidence
        result_json = calibrate_confidence_handler(
            prediction="Unlikely outcome",
            initial_confidence=0.1
        )
        result = json.loads(result_json)
        assert result["status"] == "success"
        
        # Very high confidence  
        result_json = calibrate_confidence_handler(
            prediction="Certain outcome",
            initial_confidence=0.99
        )
        result = json.loads(result_json)
        assert result["status"] == "success"
    
    def test_missing_parameters_handling(self):
        """Test handling of missing required parameters."""
        result_json = calibrate_confidence_handler()
        result = json.loads(result_json)
        assert result["status"] == "error"
        assert "message" in result
    
    def test_invalid_confidence_values(self):
        """Test handling of invalid confidence values."""
        # Confidence above 1.0
        result_json = calibrate_confidence_handler(
            prediction="Test prediction",
            initial_confidence=1.5
        )
        result = json.loads(result_json)
        # Should handle gracefully
        assert isinstance(result, dict)
        
        # Negative confidence
        result_json = calibrate_confidence_handler(
            prediction="Test prediction", 
            initial_confidence=-0.5
        )
        result = json.loads(result_json)
        assert isinstance(result, dict)
    
    def test_empty_prediction(self):
        """Test handling of empty prediction."""
        result_json = calibrate_confidence_handler(
            prediction="",
            initial_confidence=0.5
        )
        result = json.loads(result_json)
        
        # Should handle gracefully
        assert isinstance(result, dict)


class TestToolHandlerIntegration:
    """Test integration between tool handlers."""
    
    def setup_method(self):
        """Clear global state before each test."""
        _chain_processor.clear_chain()
    
    def test_chain_workflow_integration(self):
        """Test complete workflow using chain-related handlers."""
        # Add a step
        step_result = chain_of_thought_step_handler(
            thought="Initial analysis",
            step_number=1,
            total_steps=2,
            next_step_needed=True
        )
        step_data = json.loads(step_result)
        assert step_data["status"] == "success"
        
        # Get summary
        summary_result = get_chain_summary_handler()
        summary_data = json.loads(summary_result)
        assert summary_data["status"] == "success"
        assert summary_data["total_steps"] == 1
        
        # Add another step
        step_result = chain_of_thought_step_handler(
            thought="Final conclusion",
            step_number=2,
            total_steps=2,
            next_step_needed=False
        )
        step_data = json.loads(step_result)
        assert step_data["status"] == "success"
        
        # Get updated summary
        summary_result = get_chain_summary_handler()
        summary_data = json.loads(summary_result)
        assert summary_data["total_steps"] == 2
        
        # Clear chain
        clear_result = clear_chain_handler()
        clear_data = json.loads(clear_result)
        assert clear_data["status"] == "success"
        
        # Verify empty
        summary_result = get_chain_summary_handler()
        summary_data = json.loads(summary_result)
        assert summary_data["status"] == "empty"
    
    def test_error_propagation(self):
        """Test that errors are properly caught and formatted."""
        # All handlers should return proper JSON even on errors
        handlers = [
            lambda: chain_of_thought_step_handler(invalid_param="test"),
            lambda: generate_hypotheses_handler(invalid_param="test"),
            lambda: map_assumptions_handler(invalid_param="test"),
            lambda: calibrate_confidence_handler(invalid_param="test")
        ]
        
        for handler in handlers:
            result_json = handler()
            result = json.loads(result_json)  # Should not raise JSON decode error
            assert isinstance(result, dict)
            # Most likely will have error status
            if "status" in result:
                assert result["status"] in ["success", "error"]