#!/usr/bin/env python3
"""
Focused coverage tests that target specific uncovered lines without making assumptions
about the complete API. These tests are designed to be robust and achieve coverage goals.
"""

import pytest
from unittest.mock import Mock
from chain_of_thought.security import RequestValidator, SecurityValidationError
from chain_of_thought.validators import ParameterValidator


class TestSecurityFocusedCoverage:
    """Focused tests for security.py uncovered lines."""

    def test_line_84_request_type_validation(self):
        """Test line 84: Request type validation (non-dict)."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="Request must be a dictionary"):
            validator.validate_and_sanitize_request("not a dict")

    def test_lines_118_125_aws_params_validation(self):
        """Test lines 118-125: AWS-specific parameters validation."""
        validator = RequestValidator()

        # Test guardrailConfig
        with pytest.raises(SecurityValidationError, match="guardrailConfig must be a dictionary"):
            validator._validate_parameter("guardrailConfig", "not a dict")

        # Test additionalModelRequestFields
        with pytest.raises(SecurityValidationError, match="additionalModelRequestFields must be a dictionary"):
            validator._validate_parameter("additionalModelRequestFields", ["not", "a", "dict"])

    def test_line_130_model_id_type_validation(self):
        """Test line 130: Model ID type validation (non-string)."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="modelId must be a string"):
            validator._validate_model_id(123)

    def test_line_145_messages_type_validation(self):
        """Test line 145: Messages type validation (non-list)."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="messages must be a list"):
            validator._validate_messages("not a list")

    def test_line_148_empty_messages_validation(self):
        """Test line 148: Empty messages validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="messages cannot be empty"):
            validator._validate_messages([])

    def test_line_153_message_structure_validation(self):
        """Test line 153: Message structure validation (non-dict)."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="Message 0 must be a dictionary"):
            validator._validate_messages(["not a dict"])

    def test_line_157_message_role_field_validation(self):
        """Test line 157: Message role field validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="Message 0 missing required 'role' field"):
            validator._validate_messages([{"content": [{"text": "test"}]}])

    def test_line_160_message_role_type_validation(self):
        """Test line 160: Message role type validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="Message 0 role must be a string"):
            validator._validate_messages([{"role": 123, "content": [{"text": "test"}]}])

    def test_line_164_message_content_field_validation(self):
        """Test line 164: Message content field validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="Message 0 missing required 'content' field"):
            validator._validate_messages([{"role": "user"}])

    def test_lines_178_182_injection_detection(self):
        """Test lines 178-182: Injection pattern detection."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="contains potential injection patterns"):
            validator._validate_message_content("<script>alert('xss')</script>", 0)

    def test_line_188_content_item_structure_validation(self):
        """Test line 188: Content item structure validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="Message 0 content item 0 must be a dictionary"):
            validator._validate_message_content(["not a dict"], 0)

    def test_line_194_empty_content_item_validation(self):
        """Test line 194: Empty content item validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="Message 0 content item 0 cannot be empty"):
            validator._validate_message_content([{}], 0)

    def test_lines_205_208_tool_content_validation(self):
        """Test lines 205-208: Tool content validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="Message 0 toolUse must be a dictionary"):
            validator._validate_message_content([{"toolUse": "not a dict"}], 0)

    def test_line_215_message_content_type_validation(self):
        """Test line 215: Message content type validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="Message 0 content must be string or list"):
            validator._validate_message_content(123, 0)

    def test_line_222_system_none_handling(self):
        """Test line 222: System parameter None handling."""
        validator = RequestValidator()
        result = validator._validate_system(None)
        assert result is None

    def test_lines_226_228_system_injection_validation(self):
        """Test lines 226-228: System injection validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="System prompt contains potential injection patterns"):
            validator._validate_system("javascript:alert('xss')")

    def test_line_234_system_item_structure_validation(self):
        """Test line 234: System item structure validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="System item 0 must be a dictionary"):
            validator._validate_system(["not a dict"])

    def test_line_237_system_item_text_field_validation(self):
        """Test line 237: System item text field validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="System item 0 missing required 'text' field"):
            validator._validate_system([{"not_text": "test"}])

    def test_line_240_system_item_text_type_validation(self):
        """Test line 240: System item text type validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="System item 0 text must be a string"):
            validator._validate_system([{"text": 123}])

    def test_line_243_system_item_injection_validation(self):
        """Test line 243: System item injection validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="contains potential injection patterns"):
            validator._validate_system([{"text": "<script>alert('xss')</script>"}])

    def test_line_250_system_parameter_type_validation(self):
        """Test line 250: System parameter type validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="System parameter must be string, list, or None"):
            validator._validate_system(123)

    def test_line_255_tool_config_none_handling(self):
        """Test line 255: Tool config None handling."""
        validator = RequestValidator()
        result = validator._validate_tool_config(None)
        assert result is None

    def test_line_258_tool_config_type_validation(self):
        """Test line 258: Tool config type validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="toolConfig must be a dictionary"):
            validator._validate_tool_config("not a dict")

    def test_line_261_tool_config_missing_tools(self):
        """Test line 261: Tool config missing tools."""
        validator = RequestValidator()
        result = validator._validate_tool_config({"otherField": "value"})
        assert "otherField" in result

    def test_line_265_tool_config_tools_type_validation(self):
        """Test line 265: Tool config tools type validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="toolConfig.tools must be a list"):
            validator._validate_tool_config({"tools": "not a list"})

    def test_line_270_tool_structure_validation(self):
        """Test line 270: Tool structure validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="Tool 0 must be a dictionary"):
            validator._validate_tool_config({"tools": ["not a dict"]})

    def test_line_274_tool_spec_field_validation(self):
        """Test line 274: Tool spec field validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="Tool 0 missing required 'toolSpec' field"):
            validator._validate_tool_config({"tools": [{"not_toolSpec": "value"}]})

    def test_line_278_tool_spec_type_validation(self):
        """Test line 278: Tool spec type validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="Tool 0 toolSpec must be a dictionary"):
            validator._validate_tool_config({"tools": [{"toolSpec": "not a dict"}]})

    def test_line_281_tool_spec_name_field_validation(self):
        """Test line 281: Tool spec name field validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="Tool 0 toolSpec missing required 'name' field"):
            validator._validate_tool_config({"tools": [{"toolSpec": {"not_name": "value"}}]})

    def test_line_284_tool_spec_name_type_validation(self):
        """Test line 284: Tool spec name type validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="Tool 0 name must be a string"):
            validator._validate_tool_config({"tools": [{"toolSpec": {"name": 123}}]})

    def test_line_297_inference_config_none_handling(self):
        """Test line 297: Inference config None handling."""
        validator = RequestValidator()
        result = validator._validate_inference_config(None)
        assert result is None

    def test_line_300_inference_config_type_validation(self):
        """Test line 300: Inference config type validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="inferenceConfig must be a dictionary"):
            validator._validate_inference_config("not a dict")

    def test_line_306_unauthorized_inference_parameter_validation(self):
        """Test line 306: Unauthorized inference parameter validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="Unauthorized inference parameter"):
            validator._validate_inference_config({"unauthorizedParam": "value"})

    def test_line_328_temperature_type_validation(self):
        """Test line 328: Temperature type validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="temperature must be a number"):
            validator._validate_temperature("not a number")

    def test_lines_341_351_top_p_validation(self):
        """Test lines 341-351: TopP validation."""
        validator = RequestValidator()

        # Type validation
        with pytest.raises(SecurityValidationError, match="topP must be a number"):
            validator._validate_top_p("not a number")

        # Range validation
        with pytest.raises(SecurityValidationError, match="topP -0.1 out of range"):
            validator._validate_top_p(-0.1)

        with pytest.raises(SecurityValidationError, match="topP 1.1 out of range"):
            validator._validate_top_p(1.1)

    def test_line_356_max_tokens_type_validation(self):
        """Test line 356: Max tokens type validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="maxTokens must be an integer"):
            validator._validate_max_tokens("not an integer")

    def test_line_359_max_tokens_range_validation(self):
        """Test line 359: Max tokens range validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="maxTokens 0 out of range"):
            validator._validate_max_tokens(0)

    def test_lines_368_384_stop_sequences_validation(self):
        """Test lines 368-384: Stop sequences validation."""
        validator = RequestValidator()

        # Type validation
        with pytest.raises(SecurityValidationError, match="stopSequences must be a list"):
            validator._validate_stop_sequences("not a list")

        # Individual sequence validation
        with pytest.raises(SecurityValidationError, match="stopSequence 0 must be a string"):
            validator._validate_stop_sequences([123])

    def test_line_392_required_parameters_validation(self):
        """Test line 392: Required parameters validation."""
        validator = RequestValidator()
        with pytest.raises(SecurityValidationError, match="Missing required parameter: messages"):
            validator._validate_required_parameters({"modelId": "test"})

    def test_line_397_injection_pattern_type_validation(self):
        """Test line 397: Injection pattern type validation."""
        validator = RequestValidator()
        result = validator._contains_injection_patterns(123)
        assert result is False


class TestValidatorsFocusedCoverage:
    """Focused tests for validators.py uncovered lines."""

    def test_line_523_total_steps_type_validation(self):
        """Test line 523: Total steps type validation."""
        validator = ParameterValidator()

        # This tests the validate_step_parameters method which should validate total_steps
        with pytest.raises(ValueError, match="total_steps must be an integer"):
            validator.validate_step_parameters(1, "not an integer")

    def test_lines_547_553_length_limit_validation(self):
        """Test lines 547-553: Length limit validation."""
        validator = ParameterValidator()

        with pytest.raises(ValueError, match="text must be a string"):
            validator.validate_length_limit(123, 100)

        long_text = "x" * 101
        with pytest.raises(ValueError, match="text cannot exceed 100 characters"):
            validator.validate_length_limit(long_text, 100)

    def test_lines_570_573_type_validation(self):
        """Test lines 570-573: Type validation."""
        validator = ParameterValidator()

        with pytest.raises(ValueError, match="value must be a str"):
            validator.validate_type(123, str)

    def test_line_592_boolean_validation(self):
        """Test line 592: Boolean validation."""
        validator = ParameterValidator()

        with pytest.raises(ValueError, match="parameter must be a boolean"):
            validator.validate_boolean_param(123)

    def test_line_632_integer_list_range_validation(self):
        """Test line 632: Integer list range validation."""
        validator = ParameterValidator()

        with pytest.raises(ValueError, match="parameter\\[0\\] must be between 1 and 1000"):
            validator.validate_integer_list_param([0], min_value=1, max_value=1000)

    def test_line_636_return_validated_items(self):
        """Test line 636: Return validated items."""
        validator = ParameterValidator()

        result = validator.validate_integer_list_param([1, 2, 3])
        assert result == [1, 2, 3]
        assert isinstance(result, list)


class TestBasicFunctionalityCoverage:
    """Tests for basic functionality that may be missing coverage."""

    def test_security_config_defaults(self):
        """Test SecurityConfig default initialization."""
        from chain_of_thought.security import SecurityConfig

        config = SecurityConfig()
        assert config.allowed_model_patterns is not None
        assert config.allowed_top_level_params is not None
        assert config.allowed_inference_params is not None
        assert config.min_temperature == 0.0
        assert config.max_temperature == 1.0

    def test_parameter_validator_basic_validation(self):
        """Test ParameterValidator basic validation methods."""
        validator = ParameterValidator()

        # Test basic thought validation
        result = validator.validate_thought_param("Valid thought process")
        assert result == "Valid thought process"

        # Test basic type validation
        result = validator.validate_type("test", str)
        assert result == "test"

        # Test basic boolean validation
        result = validator.validate_boolean_param(True)
        assert result is True