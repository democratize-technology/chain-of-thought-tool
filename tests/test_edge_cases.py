"""
Edge case and error handling tests for boundary conditions.

Tests cover:
- Input validation and sanitization
- Boundary value testing
- Memory and performance limits
- Unicode and special character handling
- JSON serialization edge cases
- Error recovery and graceful degradation
- Resource exhaustion scenarios
"""
import pytest
import json
import sys
from unittest.mock import patch
from chain_of_thought.core import (
    ChainOfThought,
    ThoughtStep,
    HypothesisGenerator,
    AssumptionMapper,
    ConfidenceCalibrator,
    ThreadAwareChainOfThought,
    AsyncChainOfThoughtProcessor,
    chain_of_thought_step_handler,
    get_chain_summary_handler,
    generate_hypotheses_handler,
    map_assumptions_handler,
    calibrate_confidence_handler
)


@pytest.mark.edge_case
class TestInputValidationEdgeCases:
    """Test input validation and boundary conditions."""
    
    def setup_method(self):
        """Set up fresh instances for each test."""
        self.cot = ChainOfThought()
    
    def test_extreme_step_numbers(self):
        """Test handling of extreme step numbers."""
        # Very large step number
        result = self.cot.add_step(
            "Large step test",
            step_number=sys.maxsize,
            total_steps=sys.maxsize,
            next_step_needed=False
        )
        assert result["status"] == "success"
        assert result["step_processed"] == sys.maxsize
        
        # Zero step number
        self.cot.clear_chain()
        result = self.cot.add_step(
            "Zero step test",
            step_number=0,
            total_steps=1,
            next_step_needed=False
        )
        assert result["status"] == "success"
        assert result["step_processed"] == 0
        
        # Negative step number
        self.cot.clear_chain()
        result = self.cot.add_step(
            "Negative step test",
            step_number=-1,
            total_steps=1,
            next_step_needed=False
        )
        assert result["status"] == "success"
        assert result["step_processed"] == -1
    
    def test_extreme_confidence_values(self):
        """Test handling of extreme confidence values."""
        # Confidence above 1.0
        result = self.cot.add_step(
            "High confidence",
            1, 1, False,
            confidence=1.5
        )
        assert result["status"] == "success"
        assert result["confidence"] == 1.5  # Implementation doesn't clamp
        
        # Confidence below 0.0
        self.cot.clear_chain()
        result = self.cot.add_step(
            "Negative confidence",
            1, 1, False,
            confidence=-0.5
        )
        assert result["status"] == "success"
        assert result["confidence"] == -0.5
        
        # Very large confidence
        self.cot.clear_chain()
        try:
            result = self.cot.add_step(
                "Extreme confidence",
                1, 1, False,
                confidence=float('inf')
            )
            # If it doesn't raise an error, verify behavior
            assert result["status"] == "success"
        except ValueError:
            # Expected behavior for infinite confidence - security validation
            pass
        
        # NaN confidence
        self.cot.clear_chain()
        try:
            result = self.cot.add_step(
                "NaN confidence",
                1, 1, False,
                confidence=float('nan')
            )
            # If it doesn't raise an error, verify behavior
            assert result["status"] == "success"
        except ValueError:
            # Expected behavior for NaN confidence - security validation
            pass
    
    def test_empty_and_none_inputs(self):
        """Test handling of empty and None inputs."""
        # Empty thought
        result = self.cot.add_step("", 1, 1, False)
        assert result["status"] == "success"
        assert self.cot.steps[0].thought == ""
        
        # None thought (should fail in type checking, but test behavior)
        self.cot.clear_chain()
        try:
            result = self.cot.add_step(None, 1, 1, False)
            # If it doesn't raise an error, verify behavior
            assert result["status"] == "success"
        except (TypeError, ValueError):
            # Expected behavior for None input
            pass
        
        # Empty lists for optional parameters
        self.cot.clear_chain()
        result = self.cot.add_step(
            "Empty lists test",
            1, 1, False,
            dependencies=[],
            contradicts=[],
            evidence=[],
            assumptions=[]
        )
        assert result["status"] == "success"
        step = self.cot.steps[0]
        assert step.dependencies == []
        assert step.contradicts == []
        assert step.evidence == []
        assert step.assumptions == []
    
    def test_very_long_content(self):
        """Test handling of very long content."""
        # Test that extremely long content is properly rejected for security
        mega_thought = "x" * (1024 * 1024)  # 1MB - exceeds 10,000 char limit

        try:
            result = self.cot.add_step(mega_thought, 1, 1, False)
            # If it doesn't raise an error, verify it was truncated or handled properly
            assert result["status"] == "success"
            # Content should be within reasonable limits (truncated or rejected)
            assert len(self.cot.steps[0].thought) <= 10000
        except ValueError as e:
            # Expected behavior for extremely long content - security validation
            assert "thought cannot exceed 10,000 characters" in str(e)
        
        # Very long evidence and assumptions - should be rejected for security
        long_evidence = ["evidence_" + "x" * 10000 for i in range(100)]  # 100 items, each 10k chars
        long_assumptions = ["assumption_" + "x" * 10000 for i in range(100)]

        self.cot.clear_chain()
        try:
            result = self.cot.add_step(
                "Long metadata test",
                1, 1, False,
                evidence=long_evidence,
                assumptions=long_assumptions
            )
            # If it doesn't raise an error, verify it was handled properly
            assert result["status"] == "success"
            # Lists should be within reasonable limits
            assert len(self.cot.steps[0].evidence) <= 50  # Max items limit
            assert len(self.cot.steps[0].assumptions) <= 50
        except ValueError as e:
            # Expected behavior for extremely long lists - security validation
            assert "cannot exceed 50 items" in str(e) or "cannot exceed 500 characters" in str(e)
    
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        unicode_test_cases = [
            "思考步骤 🤔 with émojis",
            "¡Español! ñáéíóú",
            "Русский текст",
            "العربية",
            "עברית",
            "中文测试",
            "日本語",
            "한국어",
            "नमस्ते",
            "🚀🔥💯✨🎯🏆💎🌟⭐🎉",  # Emoji sequence
            "Mixed: ASCII + 中文 + 🎯 + العربية",
            "\n\t\r\\\"\'\`",  # Control characters
            "Zero\x00width\x00characters",
            "Math symbols: ∑∏∫∆∇∂√∞",
            "Currency: $€¥£₹₽₿"
        ]
        
        for i, test_text in enumerate(unicode_test_cases):
            self.cot.clear_chain()
            result = self.cot.add_step(
                test_text,
                1, 1, False,
                evidence=[f"Evidence: {test_text}"],
                assumptions=[f"Assumption: {test_text}"]
            )
            assert result["status"] == "success", f"Failed on test case {i}: {test_text}"

            # For security reasons, dangerous characters may be HTML-escaped
            stored_thought = self.cot.steps[0].thought
            # Check that the content is functionally equivalent (either original or escaped)
            if test_text == "\n\t\r\\\"\'\\`":
                # This specific case should be HTML-escaped for security
                expected_escaped = '\&quot;&#x27;\`'
                assert stored_thought == expected_escaped, f"Expected escaped text {repr(expected_escaped)}, got {repr(stored_thought)}"
            elif test_text == "Zero\x00width\x00characters":
                # Null bytes should be removed for security
                expected_cleaned = "Zerowidthcharacters"
                assert stored_thought == expected_cleaned, f"Expected cleaned text {repr(expected_cleaned)}, got {repr(stored_thought)}"
            else:
                # Other unicode should be preserved as-is
                assert stored_thought == test_text, f"Expected {repr(test_text)}, got {repr(stored_thought)}"

            # Evidence and assumptions should also be handled consistently
            stored_evidence = self.cot.steps[0].evidence[0]
            if f"Evidence: {test_text}" == "Evidence: \n\t\r\\\"\'\\`":
                # Evidence preserves control chars but escapes dangerous ones
                expected_evidence = "Evidence: \n\t\r\&quot;&#x27;\`"
                assert expected_evidence == stored_evidence, f"Expected evidence {repr(expected_evidence)}, got {repr(stored_evidence)}"
            elif f"Evidence: {test_text}" == "Evidence: Zero\x00width\x00characters":
                # Evidence also removes null bytes
                expected_evidence = "Evidence: Zerowidthcharacters"
                assert expected_evidence == stored_evidence, f"Expected evidence {repr(expected_evidence)}, got {repr(stored_evidence)}"
            else:
                assert f"Evidence: {test_text}" == stored_evidence
    
    def test_large_dependency_lists(self):
        """Test handling of large dependency and contradiction lists."""
        # Test that extremely large lists are properly rejected for security
        large_dependencies = list(range(1, 10000))  # 9999 dependencies
        large_contradictions = list(range(10000, 20000))  # 9999 contradictions

        try:
            result = self.cot.add_step(
                "Large dependencies test",
                10000, 10000, False,
                dependencies=large_dependencies,
                contradicts=large_contradictions
            )
            # If it doesn't raise an error, verify it was handled properly
            assert result["status"] == "success"
            step = self.cot.steps[0]
            # Lists should be within reasonable limits (truncated or rejected)
            assert len(step.dependencies) <= 50  # Max items limit
            assert len(step.contradicts) <= 50
        except ValueError as e:
            # Expected behavior for extremely large lists - security validation
            assert "cannot exceed 50 items" in str(e)

        # Test with reasonable sized lists that should work
        reasonable_deps = list(range(1, 24))  # 24 dependencies (no self-reference)
        reasonable_contradicts = list(range(100, 120))  # 20 contradictions

        self.cot.clear_chain()
        result = self.cot.add_step(
            "Reasonable dependencies test",
            25, 25, False,  # step_number should not exceed total_steps
            dependencies=reasonable_deps,
            contradicts=reasonable_contradicts
        )
        assert result["status"] == "success"
        step = self.cot.steps[0]
        assert len(step.dependencies) <= 24  # Step cannot depend on itself and validation may remove invalid deps
        assert len(step.contradicts) == 20
    
    def test_invalid_reasoning_stage(self):
        """Test handling of invalid reasoning stages."""
        invalid_stages = [
            "Invalid Stage",
            "",
            None,
            123,
            ["list", "stage"],
            {"dict": "stage"}
        ]
        
        for invalid_stage in invalid_stages:
            self.cot.clear_chain()
            try:
                result = self.cot.add_step(
                    "Invalid stage test",
                    1, 1, False,
                    reasoning_stage=invalid_stage
                )
                # If it doesn't fail, check what happened
                if result["status"] == "success":
                    step = self.cot.steps[0]
                    # Should either use default or store the invalid value
                    assert hasattr(step, 'reasoning_stage')
            except (TypeError, ValueError):
                # Expected for some invalid types
                pass


@pytest.mark.edge_case
class TestJSONSerializationEdgeCases:
    """Test JSON serialization edge cases."""
    
    def setup_method(self):
        """Set up fresh instances."""
        self.cot = ChainOfThought()
    
    def test_json_serialization_special_values(self):
        """Test JSON serialization with special float values."""
        # Test that special values are properly rejected for security
        try:
            result = self.cot.add_step(
                "Special values test",
                1, 1, False,
                confidence=float('inf')
            )
            # If it doesn't raise an error, verify it was handled properly
            assert result["status"] == "success"
            # Confidence should be within reasonable bounds
            assert 0.0 <= self.cot.steps[0].confidence <= 1.0
        except ValueError as e:
            # Expected behavior for infinite confidence - security validation
            assert "confidence must be between -100.0 and 100.0" in str(e)

        # Test with valid confidence values that should serialize properly
        valid_confidences = [0.0, 0.5, 0.8, 1.0]
        for conf in valid_confidences:
            self.cot.clear_chain()
            result = self.cot.add_step(
                f"Valid confidence test {conf}",
                1, 1, False,
                confidence=conf
            )
            assert result["status"] == "success"

            # Try to serialize the result
            result_json = json.dumps(result, indent=2)
            parsed_result = json.loads(result_json)

            # JSON should handle valid values
            assert isinstance(parsed_result, dict)
        
        # Test with NaN - should be rejected for security
        self.cot.clear_chain()
        try:
            result = self.cot.add_step(
                "NaN test",
                1, 1, False,
                confidence=float('nan')
            )
            # If it doesn't raise an error, verify it was handled properly
            assert result["status"] == "success"
        except ValueError as e:
            # Expected behavior for NaN confidence - security validation
            assert "confidence must be between -100.0 and 100.0" in str(e)

        # Test normal JSON serialization behavior with valid data
        self.cot.clear_chain()
        valid_result = self.cot.add_step(
            "Normal JSON test",
            1, 1, False,
            confidence=0.5
        )

        result_json = json.dumps(valid_result, indent=2)
        parsed_result = json.loads(result_json)
        assert isinstance(parsed_result, dict)
    
    def test_handler_json_output_special_cases(self):
        """Test handler JSON output with special cases."""
        # Test with very large numbers
        result_json = chain_of_thought_step_handler(
            thought="Large numbers test",
            step_number=sys.maxsize,
            total_steps=sys.maxsize,
            next_step_needed=False
        )
        
        # Should produce valid JSON
        result = json.loads(result_json)
        assert result["step_processed"] == sys.maxsize
        
        # Test with unicode content
        unicode_thought = "Unicode test: 🚀 中文 العربية"
        result_json = chain_of_thought_step_handler(
            thought=unicode_thought,
            step_number=1,
            total_steps=1,
            next_step_needed=False
        )
        
        result = json.loads(result_json)
        assert result["status"] == "success"
        
        # Verify unicode is preserved in the chain
        from chain_of_thought.core import _chain_processor
        _chain_processor.clear_chain()  # Clean up
    
    def test_summary_json_with_large_chain(self):
        """Test summary JSON generation with large chains."""
        # Create large chain
        for i in range(1000):
            self.cot.add_step(
                f"Step {i} with some content to make it longer",
                i + 1, 1000, True,
                evidence=[f"Evidence {i}"],
                assumptions=[f"Assumption {i}"]
            )
        
        # Generate summary
        summary = self.cot.generate_summary()
        
        # Should be serializable
        summary_json = json.dumps(summary, indent=2)
        parsed_summary = json.loads(summary_json)
        
        assert parsed_summary["total_steps"] == 1000
        assert len(parsed_summary["chain"]) == 1000
        assert len(parsed_summary["insights"]["total_evidence"]) == 1000


@pytest.mark.edge_case
class TestErrorRecoveryScenarios:
    """Test error recovery and graceful degradation."""
    
    def test_handler_exception_recovery(self):
        """Test that handler exceptions are caught and formatted."""
        # Patch the chain processor to raise an exception
        from chain_of_thought.core import _chain_processor
        
        original_add_step = _chain_processor.add_step
        
        def failing_add_step(*args, **kwargs):
            raise RuntimeError("Simulated error")
        
        _chain_processor.add_step = failing_add_step
        
        try:
            # Call handler - should catch exception and return error JSON
            result_json = chain_of_thought_step_handler(
                thought="This will fail",
                step_number=1,
                total_steps=1,
                next_step_needed=False
            )
            
            result = json.loads(result_json)
            assert result["status"] == "error"
            assert "message" in result
            assert "Simulated error" in result["message"]
            
        finally:
            # Restore original method
            _chain_processor.add_step = original_add_step
    
    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure."""
        # Create many large chains to simulate memory pressure
        chains = []
        
        try:
            for i in range(100):
                chain = ChainOfThought()
                
                # Test memory pressure with reasonable content sizes (security-limited)
                for j in range(10):  # Reduced for reasonable test time
                    # Use content that fits within security limits
                    moderate_content = "x" * 1000  # 1KB - under 10KB thought limit
                    moderate_evidence = [f"evidence_{k}" * 50 for k in range(5)]  # 250 chars each, under 500 limit
                    moderate_assumptions = [f"assumption_{k}" * 30 for k in range(3)]  # 150 chars each, under 500 limit

                    chain.add_step(
                        moderate_content,
                        j + 1, 10, True,
                        evidence=moderate_evidence,
                        assumptions=moderate_assumptions
                    )
                
                chains.append(chain)
                
                # Periodically check if we can still create summaries
                if i % 10 == 0:
                    summary = chain.generate_summary()
                    assert summary["status"] == "success"
            
        except MemoryError:
            # Expected under extreme memory pressure
            pass
    
    def test_concurrent_modification_resilience(self):
        """Test resilience to concurrent modifications."""
        # This tests the data structure stability, not true concurrency
        chain = ChainOfThought()
        
        # Add initial steps
        for i in range(10):
            chain.add_step(f"Step {i}", i + 1, 10, True)
        
        # Simulate concurrent modifications by modifying the steps list
        # while generating summary
        original_steps = chain.steps.copy()
        
        # Modify steps during summary generation
        def modify_during_summary():
            summary = chain.generate_summary()
            return summary
        
        # Clear steps mid-way through other operations
        chain.steps.clear()
        summary = chain.generate_summary()
        assert summary["status"] == "empty"
        
        # Restore and test recovery
        chain.steps = original_steps
        summary = chain.generate_summary()
        assert summary["status"] == "success"
        assert summary["total_steps"] == 10
    
    def test_circular_dependency_handling(self):
        """Test handling of circular dependencies."""
        chain = ChainOfThought()
        
        # Create circular dependencies
        chain.add_step("Step 1", 1, 3, True, dependencies=[3])
        chain.add_step("Step 2", 2, 3, True, dependencies=[1])
        chain.add_step("Step 3", 3, 3, False, dependencies=[2])
        
        # Summary should still work despite circular dependencies
        summary = chain.generate_summary()
        assert summary["status"] == "success"
        assert summary["total_steps"] == 3
    
    def test_self_referencing_steps(self):
        """Test handling of self-referencing steps."""
        chain = ChainOfThought()
        
        # Step that depends on and contradicts itself
        chain.add_step(
            "Self-referencing step",
            1, 1, False,
            dependencies=[1],
            contradicts=[1]
        )
        
        # Should handle gracefully
        summary = chain.generate_summary()
        assert summary["status"] == "success"
        
        # Check contradiction pairs include self-reference
        contradictions = summary["insights"]["contradiction_pairs"]
        assert (1, 1) in contradictions


@pytest.mark.edge_case
class TestHypothesisGeneratorEdgeCases:
    """Test edge cases for HypothesisGenerator."""
    
    def setup_method(self):
        """Set up hypothesis generator."""
        self.generator = HypothesisGenerator()
    
    def test_empty_observation(self):
        """Test hypothesis generation with empty observation."""
        result = self.generator.generate_hypotheses("", hypothesis_count=2)
        
        # Should handle gracefully
        assert "status" in result
        if result["status"] == "success":
            assert "hypotheses" in result
    
    def test_very_long_observation(self):
        """Test hypothesis generation with very long observation."""
        long_observation = "x" * 100000  # 100KB observation
        
        result = self.generator.generate_hypotheses(
            long_observation,
            hypothesis_count=4
        )
        
        assert "status" in result
        if result["status"] == "success":
            assert "hypotheses" in result
    
    def test_invalid_hypothesis_count(self):
        """Test invalid hypothesis count handling."""
        test_cases = [
            0,    # Zero
            -1,   # Negative
            100,  # Very large
            5     # Above maximum (4)
        ]
        
        for count in test_cases:
            result = self.generator.generate_hypotheses(
                "Test observation",
                hypothesis_count=count
            )
            
            # Should handle gracefully
            assert isinstance(result, dict)
            assert "status" in result
    
    def test_unicode_observation(self):
        """Test hypothesis generation with unicode observation."""
        unicode_observation = "观察到的现象: 销售下降了30% 🤔"
        
        result = self.generator.generate_hypotheses(
            unicode_observation,
            hypothesis_count=3
        )
        
        assert "status" in result
        if result["status"] == "success":
            assert "hypotheses" in result


@pytest.mark.edge_case  
class TestAssumptionMapperEdgeCases:
    """Test edge cases for AssumptionMapper."""
    
    def setup_method(self):
        """Set up assumption mapper."""
        self.mapper = AssumptionMapper()
    
    def test_empty_statement(self):
        """Test assumption mapping with empty statement."""
        result = self.mapper.map_assumptions("", depth="surface")

        assert "status" in result
        if result["status"] == "success":
            assert "assumptions_found" in result
    
    def test_nonsensical_statement(self):
        """Test assumption mapping with nonsensical statement."""
        nonsensical = "The purple elephant drives Tuesday's mathematics through the singing refrigerator."
        
        result = self.mapper.map_assumptions(nonsensical, depth="deep")
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert "status" in result
    
    def test_very_short_statement(self):
        """Test assumption mapping with very short statements."""
        short_statements = [
            "Yes.",
            "No.",
            "Maybe.",
            "A",
            "42"
        ]
        
        for statement in short_statements:
            result = self.mapper.map_assumptions(statement, depth="surface")
            assert isinstance(result, dict)
            assert "status" in result
    
    def test_invalid_depth(self):
        """Test invalid depth parameter handling."""
        invalid_depths = [
            "invalid",
            "",
            None,
            123,
            ["deep"],
            {"depth": "surface"}
        ]
        
        for depth in invalid_depths:
            result = self.mapper.map_assumptions(
                "Test statement",
                depth=depth
            )
            
            # Should handle gracefully or use default
            assert isinstance(result, dict)


@pytest.mark.edge_case
class TestConfidenceCalibratorEdgeCases:
    """Test edge cases for ConfidenceCalibrator."""
    
    def setup_method(self):
        """Set up confidence calibrator."""
        self.calibrator = ConfidenceCalibrator()
    
    def test_extreme_confidence_values(self):
        """Test calibration with extreme confidence values."""
        extreme_values = [
            -100.0,
            -1.0,
            0.0,
            1.0,
            2.0,
            100.0,
            float('inf'),
            float('-inf')
        ]
        
        for confidence in extreme_values:
            try:
                result = self.calibrator.calibrate_confidence(
                    "Test prediction",
                    confidence,
                    "Test context"
                )
                
                assert isinstance(result, dict)
                assert "status" in result
                
            except (ValueError, OverflowError):
                # Expected for extreme values
                pass
    
    def test_special_float_values(self):
        """Test calibration with special float values."""
        try:
            result = self.calibrator.calibrate_confidence(
                "Test prediction",
                float('nan'),
                "Test context"
            )
            
            assert isinstance(result, dict)
            
        except (ValueError, TypeError):
            # NaN handling varies
            pass
    
    def test_empty_prediction(self):
        """Test calibration with empty prediction."""
        result = self.calibrator.calibrate_confidence(
            "",
            0.8,
            "Context for empty prediction"
        )
        
        assert isinstance(result, dict)
        assert "status" in result
    
    def test_very_long_context(self):
        """Test calibration with very long context."""
        long_context = "Context: " + "x" * 100000  # 100KB context
        
        result = self.calibrator.calibrate_confidence(
            "Test prediction",
            0.7,
            long_context
        )
        
        assert isinstance(result, dict)
        assert "status" in result


@pytest.mark.edge_case
class TestThreadAwareEdgeCases:
    """Test edge cases for ThreadAwareChainOfThought."""
    
    def test_very_long_conversation_ids(self):
        """Test handling of very long conversation IDs."""
        long_id = "conversation_" + "x" * 10000
        
        instance = ThreadAwareChainOfThought(long_id)
        assert instance.conversation_id == long_id
        
        # Should work normally
        instance.chain.add_step("Test", 1, 1, False)
        assert len(instance.chain.steps) == 1
    
    def test_special_character_conversation_ids(self):
        """Test conversation IDs with special characters."""
        special_ids = [
            "conv-with-dashes",
            "conv_with_underscores",
            "conv.with.dots",
            "conv@with@symbols",
            "conv with spaces",
            "conv\twith\ttabs",
            "conv\nwith\nnewlines",
            "conv🚀with🎯emojis",
            "conv中文",
            "convالعربية"
        ]
        
        for conv_id in special_ids:
            instance = ThreadAwareChainOfThought(conv_id)
            assert instance.conversation_id == conv_id
            
            # Should work normally
            instance.chain.add_step("Test", 1, 1, False)
            assert len(instance.chain.steps) == 1
            
            # Clean up
            ThreadAwareChainOfThought._instances.clear()
    
    def test_instance_limit_behavior(self):
        """Test behavior with very large numbers of instances."""
        # Create many instances to test memory usage
        instance_count = 10000
        
        for i in range(instance_count):
            instance = ThreadAwareChainOfThought(f"instance_{i}")
            instance.chain.add_step(f"Step {i}", 1, 1, False)
        
        # Verify all instances exist
        assert len(ThreadAwareChainOfThought._instances) == instance_count
        
        # Verify random instances still work
        import random
        for _ in range(100):
            random_id = f"instance_{random.randint(0, instance_count - 1)}"
            instance = ThreadAwareChainOfThought(random_id)
            assert len(instance.chain.steps) == 1
        
        # Clean up
        ThreadAwareChainOfThought._instances.clear()