#!/usr/bin/env python3
"""
Comprehensive security tests to achieve 100% coverage for chain_of_thought.security module.
Targets specific uncovered lines and edge cases identified in coverage report.
"""

import pytest
import re
from chain_of_thought.security import (
    RequestValidator, SecurityConfig, SecurityValidationError,
    default_validator
)


class TestRequestValidationCoverage:
    """Tests for request validation uncovered lines."""

    def test_validate_and_sanitize_request_non_dict_input(self):
        """Test line 84: Request type validation (non-dict)."""
        validator = RequestValidator()

        with pytest.raises(SecurityValidationError, match="Request must be a dictionary"):
            validator.validate_and_sanitize_request("not a dict")

        with pytest.raises(SecurityValidationError, match="Request must be a dictionary"):
            validator.validate_and_sanitize_request(None)

        with pytest.raises(SecurityValidationError, match="Request must be a dictionary"):
            validator.validate_and_sanitize_request(123)

        with pytest.raises(SecurityValidationError, match="Request must be a dictionary"):
            validator.validate_and_sanitize_request(["list", "instead", "of", "dict"])

    def test_validate_parameter_aws_specific_params(self):
        """Test lines 118-125: guardrailConfig and additionalModelRequestFields validation."""
        validator = RequestValidator()

        # Test guardrailConfig validation
        with pytest.raises(SecurityValidationError, match="guardrailConfig must be a dictionary"):
            validator._validate_parameter("guardrailConfig", "not a dict")

        with pytest.raises(SecurityValidationError, match="guardrailConfig must be a dictionary"):
            validator._validate_parameter("guardrailConfig", ["list", "instead", "of", "dict"])

        # Valid guardrailConfig
        valid_guardrail = {"trace": "enabled", "streamFilteringThreshold": 0}
        result = validator._validate_parameter("guardrailConfig", valid_guardrail)
        assert result == valid_guardrail

        # Test additionalModelRequestFields validation
        with pytest.raises(SecurityValidationError, match="additionalModelRequestFields must be a dictionary"):
            validator._validate_parameter("additionalModelRequestFields", "not a dict")

        with pytest.raises(SecurityValidationError, match="additionalModelRequestFields must be a dictionary"):
            validator._validate_parameter("additionalModelRequestFields", ["list", "instead", "of", "dict"])

        # Valid additionalModelRequestFields
        valid_fields = {"top_k": 200}
        result = validator._validate_parameter("additionalModelRequestFields", valid_fields)
        assert result == valid_fields

    def test_validate_model_id_type_validation(self):
        """Test line 130: Model ID type validation (non-string)."""
        validator = RequestValidator()

        with pytest.raises(SecurityValidationError, match="modelId must be a string"):
            validator._validate_model_id(123)

        with pytest.raises(SecurityValidationError, match="modelId must be a string"):
            validator._validate_model_id(None)

        with pytest.raises(SecurityValidationError, match="modelId must be a string"):
            validator._validate_model_id(["list", "of", "models"])

        with pytest.raises(SecurityValidationError, match="modelId must be a string"):
            validator._validate_model_id({"model": "anthropic.claude-3-sonnet-20240229-v1:0"})

    def test_validate_messages_type_and_empty_validation(self):
        """Test lines 145, 148: Messages type and empty validation."""
        validator = RequestValidator()

        # Test non-list messages (line 145)
        with pytest.raises(SecurityValidationError, match="messages must be a list"):
            validator._validate_messages("not a list")

        with pytest.raises(SecurityValidationError, match="messages must be a list"):
            validator._validate_messages(123)

        with pytest.raises(SecurityValidationError, match="messages must be a list"):
            validator._validate_messages(None)

        # Test empty messages (line 148)
        with pytest.raises(SecurityValidationError, match="messages cannot be empty"):
            validator._validate_messages([])

        # Test non-dict message (line 153)
        with pytest.raises(SecurityValidationError, match="Message 0 must be a dictionary"):
            validator._validate_messages(["not a dict"])

    def test_validate_message_role_validation(self):
        """Test lines 157, 160: Message role field and type validation."""
        validator = RequestValidator()

        # Missing role field (line 157)
        with pytest.raises(SecurityValidationError, match="Message 0 missing required 'role' field"):
            validator._validate_messages([{"content": [{"text": "test"}]}])

        # Invalid role type (line 160)
        with pytest.raises(SecurityValidationError, match="Message 0 role must be a string"):
            validator._validate_messages([{"role": 123, "content": [{"text": "test"}]}])

        with pytest.raises(SecurityValidationError, match="Message 0 role must be a string"):
            validator._validate_messages([{"role": None, "content": [{"text": "test"}]}])

        with pytest.raises(SecurityValidationError, match="Message 0 role must be a string"):
            validator._validate_messages([{"role": ["user"], "content": [{"text": "test"}]}])

    def test_validate_message_content_field_validation(self):
        """Test line 164: Message content field validation."""
        validator = RequestValidator()

        with pytest.raises(SecurityValidationError, match="Message 0 missing required 'content' field"):
            validator._validate_messages([{"role": "user"}])

    def test_validate_message_content_injection_patterns(self):
        """Test lines 178-182: Injection pattern detection in content."""
        validator = RequestValidator()

        # Test injection patterns
        malicious_contents = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "onclick=alert('xss')",
            "__proto__.pollution",
            "constructor.prototype.attack",
            "../../../etc/passwd",
            "path=/etc/passwd",
            "eval('malicious code')",
            "exec('malicious code')",
            "import os",
            "from subprocess import call"
        ]

        for malicious_content in malicious_contents:
            with pytest.raises(SecurityValidationError, match="contains potential injection patterns"):
                validator._validate_message_content(malicious_content, 0)

    def test_validate_message_content_list_validation(self):
        """Test lines 188, 194, 205-208, 215: Content list validation."""
        validator = RequestValidator()

        # Test non-dict content item (line 188)
        with pytest.raises(SecurityValidationError, match="Message 0 content item 0 must be a dictionary"):
            validator._validate_message_content(["not a dict"], 0)

        # Test empty content item (line 194)
        with pytest.raises(SecurityValidationError, match="Message 0 content item 0 cannot be empty"):
            validator._validate_message_content([{}], 0)

        # Test tool content validation (lines 205-208)
        with pytest.raises(SecurityValidationError, match="Message 0 toolUse must be a dictionary"):
            validator._validate_message_content([{"toolUse": "not a dict"}], 0)

        with pytest.raises(SecurityValidationError, match="Message 0 toolResult must be a dictionary"):
            validator._validate_message_content([{"toolResult": ["not", "a", "dict"]}], 0)

        # Test invalid content type (line 215)
        with pytest.raises(SecurityValidationError, match="Message 0 content must be string or list"):
            validator._validate_message_content(123, 0)

    def test_validate_system_parameter_validation(self):
        """Test lines 222, 226-228, 234, 237, 240, 243, 250: System parameter validation."""
        validator = RequestValidator()

        # Test None handling (line 222)
        result = validator._validate_system(None)
        assert result is None

        # Test injection in string (lines 226-228)
        with pytest.raises(SecurityValidationError, match="System prompt contains potential injection patterns"):
            validator._validate_system("javascript:alert('xss')")

        # Test system item structure validation (line 234)
        with pytest.raises(SecurityValidationError, match="System item 0 must be a dictionary"):
            validator._validate_system(["not a dict"])

        # Test missing text field (line 237)
        with pytest.raises(SecurityValidationError, match="System item 0 missing required 'text' field"):
            validator._validate_system([{"not_text": "test"}])

        # Test invalid text type (line 240)
        with pytest.raises(SecurityValidationError, match="System item 0 text must be a string"):
            validator._validate_system([{"text": 123}])

        # Test injection in system item (line 243)
        with pytest.raises(SecurityValidationError, match="contains potential injection patterns"):
            validator._validate_system([{"text": "<script>alert('xss')</script>"}])

        # Test invalid system type (line 250)
        with pytest.raises(SecurityValidationError, match="System parameter must be string, list, or None"):
            validator._validate_system(123)

    def test_validate_tool_config_validation(self):
        """Test lines 255, 258, 261, 265, 270, 274, 278, 281, 284: Tool config validation."""
        validator = RequestValidator()

        # Test None handling (line 255)
        result = validator._validate_tool_config(None)
        assert result is None

        # Test invalid type (line 258)
        with pytest.raises(SecurityValidationError, match="toolConfig must be a dictionary"):
            validator._validate_tool_config("not a dict")

        # Test missing tools field (line 261)
        result = validator._validate_tool_config({"otherField": "value"})
        assert result == {"otherField": "value"}

        # Test tools not a list (line 265)
        with pytest.raises(SecurityValidationError, match="toolConfig.tools must be a list"):
            validator._validate_tool_config({"tools": "not a list"})

        # Test tool structure validation (line 270)
        with pytest.raises(SecurityValidationError, match="Tool 0 must be a dictionary"):
            validator._validate_tool_config({"tools": ["not a dict"]})

        # Test missing toolSpec (line 274)
        with pytest.raises(SecurityValidationError, match="Tool 0 missing required 'toolSpec' field"):
            validator._validate_tool_config({"tools": [{"not_toolSpec": "value"}]})

        # Test toolSpec not a dict (line 278)
        with pytest.raises(SecurityValidationError, match="Tool 0 toolSpec must be a dictionary"):
            validator._validate_tool_config({"tools": [{"toolSpec": "not a dict"}]})

        # Test missing name field (line 281)
        with pytest.raises(SecurityValidationError, match="Tool 0 toolSpec missing required 'name' field"):
            validator._validate_tool_config({"tools": [{"toolSpec": {"not_name": "value"}}]})

        # Test name not a string (line 284)
        with pytest.raises(SecurityValidationError, match="Tool 0 name must be a string"):
            validator._validate_tool_config({"tools": [{"toolSpec": {"name": 123}}]})

    def test_validate_inference_config_validation(self):
        """Test lines 297, 300, 306, 315, 318-321: Inference config validation."""
        validator = RequestValidator()

        # Test None handling (line 297)
        result = validator._validate_inference_config(None)
        assert result is None

        # Test invalid type (line 300)
        with pytest.raises(SecurityValidationError, match="inferenceConfig must be a dictionary"):
            validator._validate_inference_config("not a dict")

        # Test unauthorized parameter (line 306)
        with pytest.raises(SecurityValidationError, match="Unauthorized inference parameter 'unauthorizedParam'"):
            validator._validate_inference_config({"unauthorizedParam": "value"})

        # Test temperature validation call (line 315) - indirect test via full parameter validation
        config = {"temperature": 0.7, "topP": 0.9, "maxTokens": 100, "stopSequences": ["END"]}
        result = validator._validate_inference_config(config)
        assert "temperature" in result

        # Test parameter validation loop (lines 318-321) - indirect test
        config = {"unknown": "value"}  # This will be handled by the else clause
        with pytest.raises(SecurityValidationError):
            validator._validate_inference_config(config)

    def test_temperature_validation(self):
        """Test line 328: Temperature type validation."""
        validator = RequestValidator()

        with pytest.raises(SecurityValidationError, match="temperature must be a number"):
            validator._validate_temperature("not a number")

        with pytest.raises(SecurityValidationError, match="temperature must be a number"):
            validator._validate_temperature(None)

        with pytest.raises(SecurityValidationError, match="temperature must be a number"):
            validator._validate_temperature(["not", "a", "number"])

    def test_top_p_validation_comprehensive(self):
        """Test lines 341-351: TopP validation including edge cases."""
        validator = RequestValidator()

        # Test type validation
        with pytest.raises(SecurityValidationError, match="topP must be a number"):
            validator._validate_top_p("not a number")

        with pytest.raises(SecurityValidationError, match="topP must be a number"):
            validator._validate_top_p(None)

        # Test range validation - below minimum
        with pytest.raises(SecurityValidationError, match="topP -0.1 out of range"):
            validator._validate_top_p(-0.1)

        # 0.0 should be valid since min_top_p is 0.0
        result = validator._validate_top_p(0.0)
        assert result == 0.0

        # Test range validation - above maximum
        with pytest.raises(SecurityValidationError, match="topP 1.1 out of range"):
            validator._validate_top_p(1.1)

        with pytest.raises(SecurityValidationError, match="topP 2.0 out of range"):
            validator._validate_top_p(2.0)

        # Test valid values
        assert validator._validate_top_p(0.5) == 0.5
        assert validator._validate_top_p(1.0) == 1.0

    def test_max_tokens_validation(self):
        """Test lines 356, 359: Max tokens validation."""
        validator = RequestValidator()

        # Test type validation (line 356)
        with pytest.raises(SecurityValidationError, match="maxTokens must be an integer"):
            validator._validate_max_tokens("not an integer")

        with pytest.raises(SecurityValidationError, match="maxTokens must be an integer"):
            validator._validate_max_tokens(123.5)

        with pytest.raises(SecurityValidationError, match="maxTokens must be an integer"):
            validator._validate_max_tokens(None)

        # Test range validation (line 359)
        with pytest.raises(SecurityValidationError, match="maxTokens 0 out of range"):
            validator._validate_max_tokens(0)

        with pytest.raises(SecurityValidationError, match="maxTokens -1 out of range"):
            validator._validate_max_tokens(-1)

        with pytest.raises(SecurityValidationError, match="maxTokens 100001 out of range"):
            validator._validate_max_tokens(100001)

    def test_stop_sequences_validation_comprehensive(self):
        """Test lines 368-384: Stop sequences validation including injection checks."""
        validator = RequestValidator()

        # Test type validation (line 368)
        with pytest.raises(SecurityValidationError, match="stopSequences must be a list"):
            validator._validate_stop_sequences("not a list")

        with pytest.raises(SecurityValidationError, match="stopSequences must be a list"):
            validator._validate_stop_sequences(123)

        # Test individual sequence validation
        with pytest.raises(SecurityValidationError, match="stopSequence 0 must be a string"):
            validator._validate_stop_sequences([123])

        with pytest.raises(SecurityValidationError, match="stopSequence 1 must be a string"):
            validator._validate_stop_sequences(["valid", 123])

        # Test injection patterns in stop sequences
        with pytest.raises(SecurityValidationError, match="stopSequence 0 '.*' contains potential injection"):
            validator._validate_stop_sequences(["<script>alert('xss')</script>"])

        with pytest.raises(SecurityValidationError, match="stopSequence 1 '.*' contains potential injection"):
            validator._validate_stop_sequences(["valid", "javascript:alert('xss')"])

        # Test valid sequences
        valid_sequences = ["END", "STOP", "###"]
        result = validator._validate_stop_sequences(valid_sequences)
        assert result == valid_sequences

    def test_validate_required_parameters(self):
        """Test line 392: Required parameters validation."""
        validator = RequestValidator()

        # Test missing messages
        with pytest.raises(SecurityValidationError, match="Missing required parameter: messages"):
            validator._validate_required_parameters({"modelId": "test"})

        # Test missing modelId
        with pytest.raises(SecurityValidationError, match="Missing required parameter: modelId"):
            validator._validate_required_parameters({"messages": []})

        # Test missing both
        with pytest.raises(SecurityValidationError, match="Missing required parameter: messages"):
            validator._validate_required_parameters({})

        # Test valid parameters
        validator._validate_required_parameters({
            "messages": [{"role": "user", "content": [{"text": "test"}]}],
            "modelId": "anthropic.claude-3-sonnet-20240229-v1:0"
        })

    def test_contains_injection_patterns_type_validation(self):
        """Test line 397: Injection pattern type validation."""
        validator = RequestValidator()

        # Test non-string inputs
        assert not validator._contains_injection_patterns(123)
        assert not validator._contains_injection_patterns(None)
        assert not validator._contains_injection_patterns([])
        assert not validator._contains_injection_patterns({})
        assert not validator._contains_injection_patterns(True)


class TestSuspiciousToolNames:
    """Test suspicious tool name detection."""

    def test_is_suspicious_tool_name_patterns(self):
        """Test various suspicious tool name patterns."""
        validator = RequestValidator()

        suspicious_names = [
            "malicious_tool",
            "injection_attack",
            "exploit_system",
            "attack_network",
            "hack_database",
            "bypass_security",
            "escalate_privileges",
            "exfiltrate_data",
            "steal_tokens",
            "leak_info",
            "dump_memory",
            "access_token",
            "admin_panel",
            "root_access",
            "system_shell",
            "execute_code",
            "evaluate_input",
            "filesystem_access",
            "network_scan",
            "http_request",
            "sql_inject"
        ]

        for name in suspicious_names:
            assert validator._is_suspicious_tool_name(name), f"Tool name '{name}' should be detected as suspicious"

        # Test valid names
        valid_names = [
            "analyze_data",
            "generate_summary",
            "process_text",
            "validate_input",
            "calculate_result"
        ]

        for name in valid_names:
            assert not validator._is_suspicious_tool_name(name), f"Tool name '{name}' should not be detected as suspicious"


class TestDefaultValidator:
    """Test the default validator instance."""

    def test_default_validator_instance(self):
        """Test that default_validator is a proper RequestValidator instance."""
        assert isinstance(default_validator, RequestValidator)
        assert isinstance(default_validator.config, SecurityConfig)


class TestCustomSecurityConfig:
    """Test custom security configuration."""

    def test_custom_config_validation(self):
        """Test validation with custom security configuration."""
        # Custom config with restricted patterns
        custom_config = SecurityConfig(
            allowed_model_patterns=[r'^custom\.model-v\d+$'],
            min_temperature=0.1,
            max_temperature=0.9,
            min_max_tokens=10,
            max_max_tokens=1000
        )

        validator = RequestValidator(custom_config)

        # Test custom model validation
        with pytest.raises(SecurityValidationError, match="does not match any allowed pattern"):
            validator._validate_model_id("anthropic.claude-3-sonnet-20240229-v1:0")

        # Test custom temperature validation
        with pytest.raises(SecurityValidationError, match="temperature 0.05 out of range"):
            validator._validate_temperature(0.05)

        with pytest.raises(SecurityValidationError, match="temperature 0.95 out of range"):
            validator._validate_temperature(0.95)

        # Test custom max tokens validation
        with pytest.raises(SecurityValidationError, match="maxTokens 5 out of range"):
            validator._validate_max_tokens(5)

        with pytest.raises(SecurityValidationError, match="maxTokens 1001 out of range"):
            validator._validate_max_tokens(1001)