"""
Comprehensive security tests for chain-of-thought library security fixes.

This test module specifically targets the security improvements implemented:
1. Race condition fix in ThreadAwareChainOfThought (threading.RLock)
2. Input validation with HTML escaping in add_step method
3. JSON injection prevention in tool handlers with safe serialization  
4. AWS security improvements with environment configuration and validation

Tests include both positive (functionality works) and negative (attacks prevented) cases.
"""
import pytest
import threading
import time
import json
import html
import os
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional AWS imports for testing AWS integration
try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

from chain_of_thought.core import (
    ChainOfThought, 
    ThreadAwareChainOfThought, 
    _safe_json_dumps,
    chain_of_thought_step_handler,
    get_chain_summary_handler,
    clear_chain_handler,
    generate_hypotheses_handler,
    map_assumptions_handler,
    calibrate_confidence_handler
)


@pytest.mark.security
class TestRaceConditionFixes:
    """Test race condition fixes in ThreadAwareChainOfThought."""
    
    def setup_method(self):
        """Clear global instances before each test."""
        ThreadAwareChainOfThought._instances.clear()
    
    def teardown_method(self):
        """Clear global instances after each test."""
        ThreadAwareChainOfThought._instances.clear()
    
    def test_rlock_prevents_race_conditions(self):
        """Test that RLock prevents race conditions during concurrent instance creation."""
        conversation_id = "race_test_conv"
        num_threads = 50
        created_instances = []
        
        def create_instance():
            """Create instance and record it."""
            instance = ThreadAwareChainOfThought(conversation_id)
            created_instances.append(id(instance.chain))  # Record actual chain object id
            return instance
        
        # Create many instances concurrently to stress test the lock
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(create_instance) for _ in range(num_threads)]
            instances = [future.result() for future in as_completed(futures)]
        
        # Verify that all instances share the same underlying chain (no race condition)
        unique_chain_ids = set(created_instances)
        assert len(unique_chain_ids) == 1, f"Race condition detected: {len(unique_chain_ids)} unique chains created"
        
        # Verify all instances point to the same chain
        first_chain = instances[0].chain
        for instance in instances[1:]:
            assert instance.chain is first_chain, "Instances do not share the same chain object"
    
    def test_rlock_allows_reentrant_access(self):
        """Test that RLock allows reentrant access from the same thread."""
        conversation_id = "reentrant_test"
        
        def nested_creation():
            """Nested function that creates instances recursively."""
            instance1 = ThreadAwareChainOfThought(conversation_id)
            # This should work because RLock allows reentrant access
            instance2 = ThreadAwareChainOfThought(conversation_id) 
            return instance1, instance2
        
        instance1, instance2 = nested_creation()
        
        # Both instances should share the same chain
        assert instance1.chain is instance2.chain
        assert len(ThreadAwareChainOfThought._instances) == 1
    
    def test_concurrent_different_conversations_isolated(self):
        """Test that concurrent access to different conversations remains isolated."""
        num_conversations = 10
        num_threads_per_conv = 5
        
        def access_conversation(conv_id):
            """Access a specific conversation and add a step."""
            instance = ThreadAwareChainOfThought(f"conv_{conv_id}")
            instance.chain.add_step(f"Step from conv {conv_id}", 1, 1, False)
            return instance
        
        with ThreadPoolExecutor(max_workers=num_conversations * num_threads_per_conv) as executor:
            futures = []
            for conv_id in range(num_conversations):
                for _ in range(num_threads_per_conv):
                    futures.append(executor.submit(access_conversation, conv_id))
            
            # Wait for all to complete
            instances = [future.result() for future in as_completed(futures)]
        
        # Verify we have exactly the expected number of conversations
        assert len(ThreadAwareChainOfThought._instances) == num_conversations
        
        # Verify each conversation has its own isolated chain
        for conv_id in range(num_conversations):
            conv_key = f"conv_{conv_id}"
            assert conv_key in ThreadAwareChainOfThought._instances
            chain = ThreadAwareChainOfThought._instances[conv_key]
            # Each conversation should have steps from multiple threads
            assert len(chain.steps) >= 1
    
    def test_lock_performance_impact(self):
        """Test that lock doesn't significantly impact performance."""
        conversation_id = "performance_test"
        num_operations = 1000
        
        start_time = time.time()
        
        # Perform many sequential operations
        for i in range(num_operations):
            instance = ThreadAwareChainOfThought(conversation_id)
            # Small operation to test lock overhead
            _ = len(instance.chain.steps)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete quickly (less than 1 second for 1000 operations)
        assert total_time < 1.0, f"Lock causing performance issues: {total_time:.3f}s for {num_operations} operations"
        
        # Verify only one instance was created
        assert len(ThreadAwareChainOfThought._instances) == 1


@pytest.mark.security
class TestInputValidationSecurity:
    """Test input validation and HTML escaping security fixes."""
    
    def setup_method(self):
        """Set up fresh ChainOfThought instance for each test."""
        self.cot = ChainOfThought()
    
    def test_xss_prevention_in_thought_parameter(self):
        """Test that XSS attempts in thought parameter are properly escaped."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "<img src='x' onerror='alert(1)'>",
            "<svg onload='alert(1)'>",
            "javascript:alert('xss')",
            "<iframe src='javascript:alert(1)'></iframe>",
            "'\"><script>alert('xss')</script>",
            "<body onload='alert(1)'>",
            "<div onclick='alert(1)'>click me</div>"
        ]
        
        for payload in xss_payloads:
            result = self.cot.add_step(
                thought=payload,
                step_number=1,
                total_steps=1,
                next_step_needed=False
            )
            
            # Should succeed (not throw error)
            assert result["status"] == "success"
            
            # Verify it was actually escaped properly
            stored_thought = self.cot.steps[-1].thought
            expected_escaped = html.escape(payload)
            assert stored_thought == expected_escaped
            
            # Verify dangerous HTML tags are escaped
            if "<script>" in payload:
                assert "&lt;script&gt;" in stored_thought
                assert "<script>" not in stored_thought
            if "<img" in payload:
                assert "&lt;img" in stored_thought  
                assert "<img" not in stored_thought
            if "<svg" in payload:
                assert "&lt;svg" in stored_thought
                assert "<svg" not in stored_thought
            if "<iframe" in payload:
                assert "&lt;iframe" in stored_thought
                assert "<iframe" not in stored_thought
            if "<body" in payload:
                assert "&lt;body" in stored_thought
                assert "<body" not in stored_thought
            if "<div" in payload:
                assert "&lt;div" in stored_thought
                assert "<div" not in stored_thought
            
            # Clear for next test
            self.cot.steps.clear()
    
    def test_xss_prevention_in_evidence_list(self):
        """Test XSS prevention in evidence list items."""
        malicious_evidence = [
            "<script>alert('evidence xss')</script>",
            "<img src=x onerror=alert(1)>",
            "Normal evidence",
            "<svg/onload=alert(1)>"
        ]
        
        result = self.cot.add_step(
            thought="Test step",
            step_number=1,
            total_steps=1,
            next_step_needed=False,
            evidence=malicious_evidence
        )
        
        assert result["status"] == "success"
        stored_evidence = self.cot.steps[0].evidence
        
        # All evidence should be HTML escaped
        assert "&lt;script&gt;alert(&#x27;evidence xss&#x27;)&lt;/script&gt;" == stored_evidence[0]
        assert "&lt;img src=x onerror=alert(1)&gt;" == stored_evidence[1]
        assert "Normal evidence" == stored_evidence[2]  # Normal text unchanged
        assert "&lt;svg/onload=alert(1)&gt;" == stored_evidence[3]
    
    def test_xss_prevention_in_assumptions_list(self):
        """Test XSS prevention in assumptions list items."""
        malicious_assumptions = [
            "<script>document.location='http://evil.com'</script>",
            "Normal assumption",
            "<iframe src='data:text/html,<script>alert(1)</script>'></iframe>"
        ]
        
        result = self.cot.add_step(
            thought="Test step",
            step_number=1,
            total_steps=1,
            next_step_needed=False,
            assumptions=malicious_assumptions
        )
        
        assert result["status"] == "success"
        stored_assumptions = self.cot.steps[0].assumptions
        
        # All assumptions should be HTML escaped
        assert "&lt;script&gt;document.location=&#x27;http://evil.com&#x27;&lt;/script&gt;" == stored_assumptions[0]
        assert "Normal assumption" == stored_assumptions[1]
        assert "&lt;iframe src=&#x27;data:text/html,&lt;script&gt;alert(1)&lt;/script&gt;&#x27;&gt;&lt;/iframe&gt;" == stored_assumptions[2]
    
    def test_length_limit_validation(self):
        """Test that length limits are properly enforced."""
        # Test thought length limit (10,000 characters)
        long_thought = "A" * 10001
        with pytest.raises(ValueError, match="thought cannot exceed 10,000 characters"):
            self.cot.add_step(long_thought, 1, 1, False)
        
        # Test reasoning_stage length limit (100 characters)
        long_stage = "A" * 101
        with pytest.raises(ValueError, match="reasoning_stage cannot exceed 100 characters"):
            self.cot.add_step("test", 1, 1, False, reasoning_stage=long_stage)
        
        # Test evidence item length limit (500 characters)
        long_evidence_item = "A" * 501
        with pytest.raises(ValueError, match="evidence items cannot exceed 500 characters"):
            self.cot.add_step("test", 1, 1, False, evidence=[long_evidence_item])
        
        # Test assumptions item length limit (500 characters)  
        long_assumption_item = "A" * 501
        with pytest.raises(ValueError, match="assumptions items cannot exceed 500 characters"):
            self.cot.add_step("test", 1, 1, False, assumptions=[long_assumption_item])
        
        # Test evidence list length limit (50 items)
        too_many_evidence = ["item"] * 51
        with pytest.raises(ValueError, match="evidence list cannot exceed 50 items"):
            self.cot.add_step("test", 1, 1, False, evidence=too_many_evidence)
        
        # Test assumptions list length limit (50 items)
        too_many_assumptions = ["assumption"] * 51
        with pytest.raises(ValueError, match="assumptions list cannot exceed 50 items"):
            self.cot.add_step("test", 1, 1, False, assumptions=too_many_assumptions)
    
    def test_type_validation(self):
        """Test that type validation works correctly."""
        # Test invalid thought type
        with pytest.raises(ValueError, match="thought must be a string"):
            self.cot.add_step(123, 1, 1, False)
        
        with pytest.raises(ValueError, match="thought must be a string"):
            self.cot.add_step(None, 1, 1, False)
        
        # Test invalid step_number type
        with pytest.raises(ValueError, match="step_number must be an integer"):
            self.cot.add_step("test", "not_int", 1, False)
        
        # Test step_number range limits (now relaxed for backward compatibility)
        with pytest.raises(ValueError, match="step_number must be between -10000 and 10000000"):
            self.cot.add_step("test", -10001, 1, False)
        
        with pytest.raises(ValueError, match="step_number must be between -10000 and 10000000"):
            self.cot.add_step("test", 10000001, 1, False)
        
        # Test invalid total_steps type
        with pytest.raises(ValueError, match="total_steps must be an integer"):
            self.cot.add_step("test", 1, "not_int", False)
        
        # Test total_steps range limits
        with pytest.raises(ValueError, match="total_steps must be between -10000 and 10000000"):
            self.cot.add_step("test", 1, -10001, False)
        
        # Test step_number > total_steps
        with pytest.raises(ValueError, match="step_number cannot exceed total_steps"):
            self.cot.add_step("test", 5, 3, False)
        
        # Test invalid confidence type and range
        with pytest.raises(ValueError, match="confidence must be a number"):
            self.cot.add_step("test", 1, 1, False, confidence="not_number")
        
        with pytest.raises(ValueError, match="confidence must be between -100.0 and 100.0"):
            self.cot.add_step("test", 1, 1, False, confidence=-101.0)
        
        with pytest.raises(ValueError, match="confidence must be between -100.0 and 100.0"):
            self.cot.add_step("test", 1, 1, False, confidence=101.0)
        
        # Test invalid next_step_needed type
        with pytest.raises(ValueError, match="next_step_needed must be a boolean"):
            self.cot.add_step("test", 1, 1, "not_bool")
        
        # Test invalid dependencies type
        with pytest.raises(ValueError, match="dependencies must be a list"):
            self.cot.add_step("test", 1, 1, False, dependencies="not_list")
        
        with pytest.raises(ValueError, match="dependencies values must be integers"):
            self.cot.add_step("test", 1, 1, False, dependencies=["not_int"])
        
        with pytest.raises(ValueError, match="dependencies values must be integers between -10000 and 10000000"):
            self.cot.add_step("test", 1, 1, False, dependencies=[-10001])
        
        with pytest.raises(ValueError, match="dependencies values must be integers between -10000 and 10000000"):
            self.cot.add_step("test", 1, 1, False, dependencies=[10000001])
        
        # Test invalid evidence type
        with pytest.raises(ValueError, match="evidence must be a list"):
            self.cot.add_step("test", 1, 1, False, evidence="not_list")
        
        with pytest.raises(ValueError, match="evidence items must be strings"):
            self.cot.add_step("test", 1, 1, False, evidence=[123])
        
        # Test invalid assumptions type
        with pytest.raises(ValueError, match="assumptions must be a list"):
            self.cot.add_step("test", 1, 1, False, assumptions="not_list")
        
        with pytest.raises(ValueError, match="assumptions items must be strings"):
            self.cot.add_step("test", 1, 1, False, assumptions=[123])
    
    def test_reasoning_stage_security(self):
        """Test that reasoning_stage parameter is properly validated."""
        # Test invalid reasoning_stage type
        with pytest.raises(ValueError, match="reasoning_stage must be a string"):
            self.cot.add_step("test", 1, 1, False, reasoning_stage=123)
        
        # Test reasoning_stage with invalid characters (injection attempt)
        invalid_stages = [
            "<script>alert('xss')</script>",
            "Stage'; DROP TABLE steps; --",
            "Stage\n\rmalicious",
            "Stage\x00null",
            "Stage\x1fcontrol_char"
        ]
        
        for invalid_stage in invalid_stages:
            with pytest.raises(ValueError, match="reasoning_stage can only contain letters, numbers, spaces, underscores, and hyphens"):
                self.cot.add_step("test", 1, 1, False, reasoning_stage=invalid_stage)
        
        # Test valid reasoning_stage formats
        valid_stages = [
            "Analysis",
            "Problem Definition", 
            "Stage_1",
            "Stage-2",
            "Analysis123",
            "Problem Definition Phase 1"
        ]
        
        for valid_stage in valid_stages:
            result = self.cot.add_step("test", 1, 1, False, reasoning_stage=valid_stage)
            assert result["status"] == "success"
            self.cot.steps.clear()  # Clear for next test
    
    def test_empty_and_whitespace_validation(self):
        """Test validation of empty strings and whitespace-only inputs."""
        # Empty thought should be allowed for backward compatibility
        result = self.cot.add_step("", 1, 1, False)
        assert result["status"] == "success"
        assert self.cot.steps[0].thought == ""  # HTML escaped empty string is still empty
        self.cot.steps.clear()
        
        # Whitespace-only thought should be trimmed to empty and HTML escaped
        result = self.cot.add_step("   \t\n  ", 1, 1, False)
        assert result["status"] == "success"
        assert self.cot.steps[0].thought == ""  # Trimmed and HTML escaped
        self.cot.steps.clear()
        
        # Valid thought with whitespace should be trimmed
        result = self.cot.add_step("  valid thought  ", 1, 1, False)
        assert result["status"] == "success"
        assert self.cot.steps[0].thought == "valid thought"
    
    def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters."""
        unicode_inputs = [
            "思考步骤 with Chinese characters",
            "Étape de réflexion with accents",
            "Шаг размышления with Cyrillic",
            "🤔 Thinking step with emoji",
            "Math symbols: ∑∏∫∆√",
            "Special chars: «»""''—–…"
        ]
        
        for unicode_input in unicode_inputs:
            result = self.cot.add_step(unicode_input, 1, 1, False)
            assert result["status"] == "success"
            # Verify Unicode is properly preserved after HTML escaping
            assert self.cot.steps[-1].thought == html.escape(unicode_input)
            self.cot.steps.clear()


@pytest.mark.security  
class TestJSONInjectionPrevention:
    """Test JSON injection prevention in tool handlers."""
    
    def test_safe_json_dumps_prevents_injection(self):
        """Test that _safe_json_dumps prevents JSON injection attacks."""
        # Test basic functionality
        safe_data = {"status": "success", "value": "normal"}
        result = _safe_json_dumps(safe_data)
        parsed = json.loads(result)
        assert parsed == safe_data
        
        # Test that JSON is safely serialized with security parameters
        data_with_potential_issues = {
            "user_input": 'Contains "quotes" and \\ backslashes',
            "unicode": "Contains unicode: 测试 ñ á",
            "status": "test"
        }
        
        result = _safe_json_dumps(data_with_potential_issues)
        parsed = json.loads(result)
        
        # Should maintain data integrity
        assert parsed["user_input"] == 'Contains "quotes" and \\ backslashes'
        assert parsed["unicode"] == "Contains unicode: 测试 ñ á"
        assert parsed["status"] == "test"
        
        # Test security parameters: ensure_ascii=True, sort_keys=True, proper separators
        assert '"status": "test"' in result  # Proper separator
        assert '", ' in result or '",' in result  # Proper item separator
        
        # Keys should be sorted for consistency (security feature)
        keys_order = list(parsed.keys())
        assert keys_order == sorted(keys_order)
        
        # Test that dangerous control characters in data are properly escaped
        dangerous_data = {"input": "\x00\x08\x0c"}  # null, backspace, form feed
        result = _safe_json_dumps(dangerous_data)
        parsed = json.loads(result)
        
        # Should not contain raw control characters in the actual data value
        # (Note: JSON formatting may add legitimate newlines for indentation)
        assert '\x00' not in result  # Null character should be escaped as \u0000
        assert '\x08' not in result  # Backspace should be escaped as \b
        assert '\x0c' not in result  # Form feed should be escaped as \f
        
        # But should preserve them when parsed back
        assert parsed["input"] == "\x00\x08\x0c"
        
        # Verify the escaping is visible in the JSON string
        assert "\\u0000" in result or "\\u0000" in repr(result)  # Null is Unicode escaped
        assert "\\b" in result  # Backspace is escaped
        assert "\\f" in result  # Form feed is escaped
    
    def test_safe_json_dumps_with_unicode(self):
        """Test safe JSON dumping with Unicode characters."""
        unicode_data = {
            "chinese": "测试数据",
            "emoji": "🔒🛡️",
            "accents": "café résumé naïve",
            "symbols": "α β γ ∑ ∏ ∫"
        }
        
        result = _safe_json_dumps(unicode_data)
        parsed = json.loads(result)
        
        # Unicode should be preserved and properly encoded
        assert parsed["chinese"] == "测试数据"
        assert parsed["emoji"] == "🔒🛡️"
        assert parsed["accents"] == "café résumé naïve"
        assert parsed["symbols"] == "α β γ ∑ ∏ ∫"
    
    def test_safe_json_dumps_error_handling(self):
        """Test safe JSON error handling for non-serializable data."""
        # Test with non-serializable object
        class NonSerializable:
            def __init__(self):
                self.circular_ref = self
        
        non_serializable = NonSerializable()
        result = _safe_json_dumps(non_serializable)
        parsed = json.loads(result)
        
        # Should return an error response, not crash
        assert parsed["status"] == "error"
        assert "JSON serialization failed" in parsed["message"]
        assert "error_type" in parsed
    
    def test_tool_handler_injection_prevention(self):
        """Test that all tool handlers produce valid JSON and don't execute injected code."""
        handlers_to_test = [
            (chain_of_thought_step_handler, {
                "thought": '{"injected": true}',
                "step_number": 1,
                "total_steps": 1,
                "next_step_needed": False
            }),
            (map_assumptions_handler, {
                "statement": '<script>alert("xss")</script>',
                "depth": "surface"
            }),
            (calibrate_confidence_handler, {
                "prediction": 'Test prediction with special chars: "quotes" & <tags>',
                "initial_confidence": 0.8
            })
        ]
        
        for handler, test_params in handlers_to_test:
            result_json = handler(**test_params)
            
            # Should be valid JSON (no parse errors)
            parsed_result = json.loads(result_json)
            
            # Should have proper structure
            assert "status" in parsed_result
            assert parsed_result["status"] == "success"
            
            # JSON should be properly formed (no injection succeeded)
            # The input data may appear in the output (that's expected)
            # but the JSON structure should be intact
            assert isinstance(parsed_result, dict)
            
            # Verify no actual code execution occurred (handlers returned normally)
            # If injection worked, these handlers would have thrown exceptions or crashed
    
    def test_handler_error_responses_safe(self):
        """Test that handler error responses don't leak sensitive information."""
        # Test with parameters that will cause errors
        error_params = [
            (chain_of_thought_step_handler, {
                "thought": None,  # Will cause type error
                "step_number": "invalid",
                "total_steps": 1,
                "next_step_needed": False
            }),
            (calibrate_confidence_handler, {
                "prediction": "test",
                "initial_confidence": "not_a_number"  # Will cause validation error
            })
        ]
        
        for handler, params in error_params:
            result_json = handler(**params)
            parsed_result = json.loads(result_json)
            
            # Should have error structure
            assert "status" in parsed_result
            assert parsed_result["status"] == "error"
            assert "message" in parsed_result
            
            # Error message should not contain sensitive system information
            error_msg = parsed_result["message"].lower()
            assert "/users/" not in error_msg  # No file paths
            assert "password" not in error_msg  # No credential info
            assert "secret" not in error_msg    # No secret keys
            assert "token" not in error_msg     # No auth tokens
    
    def test_consistent_json_structure(self):
        """Test that all handlers return consistent, secure JSON structures."""
        # Test successful responses
        success_result = chain_of_thought_step_handler(
            thought="Test thought",
            step_number=1,
            total_steps=1,
            next_step_needed=False
        )
        
        parsed = json.loads(success_result)
        
        # Should have consistent structure
        assert "status" in parsed
        assert parsed["status"] == "success"
        
        # Keys should be sorted (security feature of _safe_json_dumps)
        keys_list = list(parsed.keys())
        assert keys_list == sorted(keys_list)
        
        # Should use consistent separators
        # With indent=2, separators are different than compact JSON
        if '\n' in success_result:  # Formatted JSON
            assert ',' in success_result  # Item separator exists
            assert ': ' in success_result  # Key-value separator
        else:  # Compact JSON
            assert ', ' in success_result  # Proper separator
            assert ': ' in success_result  # Proper key-value separator


@pytest.mark.security  
@pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not available")
class TestAWSSecurityConfiguration:
    """Test AWS security configuration and credential validation."""
    
    def test_get_aws_region_environment_variables(self):
        """Test AWS region configuration from environment variables."""
        from example_bedrock_integration import get_aws_region
        
        # Test AWS_REGION takes priority
        with patch.dict(os.environ, {"AWS_REGION": "us-west-2", "AWS_DEFAULT_REGION": "eu-west-1"}):
            region = get_aws_region()
            assert region == "us-west-2"
        
        # Test AWS_DEFAULT_REGION fallback
        with patch.dict(os.environ, {"AWS_DEFAULT_REGION": "ap-southeast-1"}, clear=True):
            if "AWS_REGION" in os.environ:
                del os.environ["AWS_REGION"]
            region = get_aws_region()
            assert region == "ap-southeast-1"
        
        # Test default fallback
        with patch.dict(os.environ, {}, clear=True):
            region = get_aws_region()
            assert region == "us-east-1"
    
    @pytest.mark.asyncio
    async def test_credential_validation_no_credentials(self):
        """Test credential validation when no AWS credentials are available."""
        from example_bedrock_integration import validate_aws_credentials
        
        # Mock boto3 to simulate no credentials
        with patch('boto3.client') as mock_client:
            mock_sts = MagicMock()
            mock_sts.get_caller_identity.side_effect = NoCredentialsError()
            mock_client.return_value = mock_sts
            
            with pytest.raises(RuntimeError, match="No AWS credentials found"):
                await validate_aws_credentials("us-east-1")
    
    @pytest.mark.asyncio
    async def test_credential_validation_invalid_credentials(self):
        """Test credential validation with invalid AWS credentials."""
        from example_bedrock_integration import validate_aws_credentials
        
        # Mock boto3 to simulate invalid credentials
        with patch('boto3.client') as mock_client:
            mock_sts = MagicMock()
            mock_sts.get_caller_identity.side_effect = ClientError(
                {'Error': {'Code': 'InvalidAccessKeyId', 'Message': 'Invalid access key'}},
                'GetCallerIdentity'
            )
            mock_client.return_value = mock_sts
            
            with pytest.raises(RuntimeError, match="AWS credentials invalid: InvalidAccessKeyId"):
                await validate_aws_credentials("us-east-1")
    
    @pytest.mark.asyncio
    async def test_credential_validation_unsupported_region(self):
        """Test credential validation with unsupported Bedrock region."""
        from example_bedrock_integration import validate_aws_credentials
        
        # Mock successful STS but failed Bedrock client creation
        with patch('boto3.client') as mock_client:
            def client_side_effect(service, **kwargs):
                if service == 'sts':
                    mock_sts = MagicMock()
                    mock_sts.get_caller_identity.return_value = {
                        'Account': 'YOUR_AWS_ACCOUNT_ID',
                        'Arn': 'arn:aws:iam::YOUR_AWS_ACCOUNT_ID:user/your-username'
                    }
                    return mock_sts
                elif service == 'bedrock-runtime':
                    raise ClientError(
                        {'Error': {'Code': 'UnrecognizedClientException', 'Message': 'Region not supported'}},
                        'CreateClient'
                    )
            
            mock_client.side_effect = client_side_effect
            
            with pytest.raises(RuntimeError, match="Bedrock service not available in region 'unsupported-region'"):
                await validate_aws_credentials("unsupported-region")
    
    @pytest.mark.asyncio
    async def test_credential_validation_success(self):
        """Test successful credential validation."""
        from example_bedrock_integration import validate_aws_credentials
        
        # Mock successful validation
        with patch('boto3.client') as mock_client:
            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {
                'Account': 'YOUR_AWS_ACCOUNT_ID',
                'Arn': 'arn:aws:iam::YOUR_AWS_ACCOUNT_ID:user/your-username'
            }
            
            mock_bedrock = MagicMock()
            
            def client_side_effect(service, **kwargs):
                if service == 'sts':
                    return mock_sts
                elif service == 'bedrock-runtime':
                    return mock_bedrock
            
            mock_client.side_effect = client_side_effect
            
            result = await validate_aws_credentials("us-east-1")
            assert result == mock_bedrock
    
    def test_no_hardcoded_credentials(self):
        """Test that no credentials are hardcoded in the example file."""
        # Get the project root directory relative to this test file
        project_root = os.path.dirname(os.path.dirname(__file__))
        example_file_path = os.path.join(project_root, 'example_bedrock_integration.py')

        with open(example_file_path, 'r') as f:
            content = f.read()
        
        # Check for common credential patterns
        credential_patterns = [
            'AKIA',  # AWS Access Key ID prefix
            'aws_access_key_id',
            'aws_secret_access_key',
            'aws_session_token',
            '="AKIA',
            "='AKIA",
            'access_key',
            'secret_key'
        ]
        
        content_lower = content.lower()
        for pattern in credential_patterns:
            # Allow environment variable references but not actual credentials
            if pattern in content_lower:
                # Make sure it's only in environment variable context or documentation
                lines_with_pattern = [line.strip() for line in content.split('\n') if pattern.lower() in line.lower()]
                for line in lines_with_pattern:
                    # Allowed: comments, environment variable references, documentation
                    assert (
                        line.startswith('#') or  # Comment
                        'environ' in line.lower() or  # Environment variable
                        'aws_access_key_id' in line and 'export' in line or  # Documentation
                        'required iam permissions' in line.lower() or  # Documentation
                        'aws cli profiles' in line.lower()  # Documentation
                    ), f"Potential hardcoded credential found: {line}"
    
    def test_environment_variable_documentation(self):
        """Test that environment variables are properly documented."""
        # Get the project root directory relative to this test file
        project_root = os.path.dirname(os.path.dirname(__file__))
        example_file_path = os.path.join(project_root, 'example_bedrock_integration.py')

        with open(example_file_path, 'r') as f:
            content = f.read()
        
        # Should document required environment variables
        assert 'AWS_REGION' in content
        assert 'AWS_DEFAULT_REGION' in content
        assert 'AWS_ACCESS_KEY_ID' in content
        assert 'AWS_SECRET_ACCESS_KEY' in content
        
        # Should provide setup instructions
        assert 'aws configure' in content
        assert 'export AWS_REGION' in content
        
        # Should document required IAM permissions
        assert 'bedrock:InvokeModel' in content
        assert 'sts:GetCallerIdentity' in content


@pytest.mark.security
class TestSecurityRegression:
    """Test that security fixes don't break existing functionality."""
    
    def setup_method(self):
        """Set up clean state for regression tests."""
        self.cot = ChainOfThought()
        ThreadAwareChainOfThought._instances.clear()
    
    def teardown_method(self):
        """Clean up after regression tests."""
        ThreadAwareChainOfThought._instances.clear()
    
    def test_basic_functionality_still_works(self):
        """Test that basic ChainOfThought functionality still works after security fixes."""
        # Test normal step addition
        result = self.cot.add_step(
            thought="Normal thinking step",
            step_number=1,
            total_steps=3,
            next_step_needed=True,
            reasoning_stage="Analysis",
            confidence=0.8,
            dependencies=[],
            contradicts=[],
            evidence=["Market research", "User feedback"],
            assumptions=["Stable market conditions"]
        )
        
        assert result["status"] == "success"
        assert result["step_processed"] == 1
        assert result["progress"] == "1/3"
        assert result["confidence"] == 0.8
        assert result["next_step_needed"] is True
        
        # Test summary generation
        summary = self.cot.generate_summary()
        assert summary["status"] == "success"
        assert summary["total_steps"] == 1
        assert "Analysis" in summary["stages_covered"]
        assert summary["overall_confidence"] == 0.8
        
        # Test additional steps
        self.cot.add_step("Second step", 2, 3, True, "Synthesis", 0.9)
        self.cot.add_step("Final step", 3, 3, False, "Conclusion", 0.85)
        
        final_summary = self.cot.generate_summary()
        assert final_summary["total_steps"] == 3
        assert len(final_summary["stages_covered"]) >= 2
        
        # Test chain clearing
        clear_result = self.cot.clear_chain()
        assert clear_result["status"] == "success"
        assert len(self.cot.steps) == 0
    
    def test_thread_aware_functionality_still_works(self):
        """Test that ThreadAwareChainOfThought functionality still works."""
        # Test basic instance creation
        instance1 = ThreadAwareChainOfThought("test_conv_1")
        instance2 = ThreadAwareChainOfThought("test_conv_2")
        instance3 = ThreadAwareChainOfThought("test_conv_1")  # Same as instance1
        
        # Verify proper isolation
        assert instance1.chain is instance3.chain
        assert instance1.chain is not instance2.chain
        
        # Test step addition through instances
        handlers1 = instance1.get_handlers()
        handlers2 = instance2.get_handlers()
        
        result1 = json.loads(handlers1["chain_of_thought_step"](
            thought="Conversation 1 step",
            step_number=1,
            total_steps=2,
            next_step_needed=True
        ))
        
        result2 = json.loads(handlers2["chain_of_thought_step"](
            thought="Conversation 2 step", 
            step_number=1,
            total_steps=1,
            next_step_needed=False
        ))
        
        assert result1["status"] == "success"
        assert result2["status"] == "success"
        
        # Verify isolation
        assert len(instance1.chain.steps) == 1
        assert len(instance2.chain.steps) == 1
        assert instance1.chain.steps[0].thought == "Conversation 1 step"
        assert instance2.chain.steps[0].thought == "Conversation 2 step"
    
    def test_tool_handlers_still_work(self):
        """Test that all tool handlers still function correctly."""
        # Test chain_of_thought_step_handler
        result = chain_of_thought_step_handler(
            thought="Handler test",
            step_number=1,
            total_steps=1,
            next_step_needed=False,
            confidence=0.7
        )
        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert parsed["confidence"] == 0.7
        
        # Test get_chain_summary_handler
        summary_result = get_chain_summary_handler()
        summary_parsed = json.loads(summary_result)
        assert summary_parsed["status"] == "success"
        assert summary_parsed["total_steps"] == 1
        
        # Test clear_chain_handler
        clear_result = clear_chain_handler()
        clear_parsed = json.loads(clear_result)
        assert clear_parsed["status"] == "success"
        
        # Test generate_hypotheses_handler
        hyp_result = generate_hypotheses_handler(
            observation="Sales are declining",
            hypothesis_count=3
        )
        hyp_parsed = json.loads(hyp_result)
        assert hyp_parsed["status"] == "success"
        assert hyp_parsed["hypotheses_generated"] == 3
        
        # Test map_assumptions_handler
        assump_result = map_assumptions_handler(
            statement="Our product will succeed because customers want it",
            depth="surface"
        )
        assump_parsed = json.loads(assump_result)
        assert assump_parsed["status"] == "success"
        
        # Test calibrate_confidence_handler
        calib_result = calibrate_confidence_handler(
            prediction="The launch will be successful",
            initial_confidence=0.9
        )
        calib_parsed = json.loads(calib_result)
        assert calib_parsed["status"] == "success"
        assert "calibrated_confidence" in calib_parsed
    
    def test_edge_cases_still_handled(self):
        """Test that edge cases are still properly handled."""
        # Test step revision (existing functionality)
        self.cot.add_step("Original step", 1, 2, True)
        revision_result = self.cot.add_step("Revised step", 1, 2, True)  # Same step number
        
        assert revision_result["is_revision"] is True
        assert len(self.cot.steps) == 1  # Should replace, not add
        assert self.cot.steps[0].thought == "Revised step"
        
        # Test with minimal parameters
        minimal_result = self.cot.add_step("Minimal", 2, 2, False)
        assert minimal_result["status"] == "success"
        
        # Test with maximum valid parameters
        max_result = self.cot.add_step(
            "A" * 10000,  # Maximum length
            9999999,      # High step number
            10000000,     # High total steps  
            True,
            "A" * 100,    # Maximum stage length
            100.0,        # Maximum confidence
            list(range(1, 51)),  # Maximum dependencies
            list(range(51, 101)), # Maximum contradicts  
            ["A" * 500] * 50,     # Maximum evidence
            ["A" * 500] * 50      # Maximum assumptions
        )
        assert max_result["status"] == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])