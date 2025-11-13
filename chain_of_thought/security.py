"""
Security utilities for chain_of_thought module.
Input validation, parameter sanitization, and injection prevention.
"""

import re
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass


@dataclass
class SecurityConfig:
    """Security configuration for parameter validation."""

    # Allowed model ID patterns (whitelist approach)
    allowed_model_patterns: List[str] = None

    # Parameter whitelists
    allowed_top_level_params: Set[str] = None
    allowed_inference_params: Set[str] = None

    # Parameter value validation ranges
    min_temperature: float = 0.0
    max_temperature: float = 1.0
    min_top_p: float = 0.0
    max_top_p: float = 1.0
    min_max_tokens: int = 1
    max_max_tokens: int = 100000

    def __post_init__(self):
        """Initialize default values if not provided."""
        if self.allowed_model_patterns is None:
            # Default to Anthropic Claude 3 models
            self.allowed_model_patterns = [
                r'^anthropic\.claude-3-(sonnet|haiku|opus)-\d{8}-v\d:\d+$',
                r'^anthropic\.claude-3-5-sonnet-\d{8}-v\d:\d+$'
            ]

        if self.allowed_top_level_params is None:
            self.allowed_top_level_params = {
                'messages',           # Required
                'modelId',           # Required
                'system',            # Optional
                'toolConfig',        # Optional
                'inferenceConfig',   # Optional
                'guardrailConfig',   # Optional (AWS specific)
                'additionalModelRequestFields'  # Optional (AWS specific)
            }

        if self.allowed_inference_params is None:
            self.allowed_inference_params = {
                'temperature',
                'topP',
                'maxTokens',
                'stopSequences'
            }


class SecurityValidationError(Exception):
    """Raised when security validation fails."""
    pass


class RequestValidator:
    """Validates and sanitizes Bedrock API requests to prevent injection attacks."""

    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()

    def validate_and_sanitize_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize a Bedrock API request.

        Args:
            request: Raw request dictionary

        Returns:
            Sanitized request dictionary

        Raises:
            SecurityValidationError: If request contains malicious content
        """
        if not isinstance(request, dict):
            raise SecurityValidationError("Request must be a dictionary")

        # Create a copy to avoid modifying the original
        sanitized = {}

        # Validate and sanitize each top-level parameter
        for param_name, param_value in request.items():
            if param_name not in self.config.allowed_top_level_params:
                raise SecurityValidationError(
                    f"Unauthorized parameter '{param_name}'. "
                    f"Allowed parameters: {sorted(self.config.allowed_top_level_params)}"
                )

            # Apply specific validation based on parameter type
            sanitized[param_name] = self._validate_parameter(param_name, param_value)

        # Ensure required parameters are present
        self._validate_required_parameters(sanitized)

        return sanitized

    def _validate_parameter(self, param_name: str, param_value: Any) -> Any:
        """Validate a specific parameter based on its type and name."""

        if param_name == 'modelId':
            return self._validate_model_id(param_value)
        elif param_name == 'messages':
            return self._validate_messages(param_value)
        elif param_name == 'system':
            return self._validate_system(param_value)
        elif param_name == 'toolConfig':
            return self._validate_tool_config(param_value)
        elif param_name == 'inferenceConfig':
            return self._validate_inference_config(param_value)
        elif param_name in ['guardrailConfig', 'additionalModelRequestFields']:
            # AWS-specific parameters - pass through but ensure they're dictionaries
            if not isinstance(param_value, dict):
                raise SecurityValidationError(f"{param_name} must be a dictionary")
            return param_value
        else:
            # Unknown but allowed parameter - return as-is
            return param_value

    def _validate_model_id(self, model_id: str) -> str:
        """Validate model ID against allowed patterns."""
        if not isinstance(model_id, str):
            raise SecurityValidationError("modelId must be a string")

        # Check against allowed patterns
        for pattern in self.config.allowed_model_patterns:
            if re.match(pattern, model_id):
                return model_id

        raise SecurityValidationError(
            f"modelId '{model_id}' does not match any allowed pattern. "
            f"Allowed patterns: {self.config.allowed_model_patterns}"
        )

    def _validate_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate messages array."""
        if not isinstance(messages, list):
            raise SecurityValidationError("messages must be a list")

        if not messages:
            raise SecurityValidationError("messages cannot be empty")

        sanitized_messages = []
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise SecurityValidationError(f"Message {i} must be a dictionary")

            # Validate required 'role' field
            if 'role' not in message:
                raise SecurityValidationError(f"Message {i} missing required 'role' field")

            if not isinstance(message['role'], str):
                raise SecurityValidationError(f"Message {i} role must be a string")

            # Validate required 'content' field
            if 'content' not in message:
                raise SecurityValidationError(f"Message {i} missing required 'content' field")

            # Sanitize content based on its type
            sanitized_content = self._validate_message_content(message['content'], i)
            message['content'] = sanitized_content

            sanitized_messages.append(message)

        return sanitized_messages

    def _validate_message_content(self, content: Any, message_index: int) -> Any:
        """Validate message content."""
        if isinstance(content, str):
            # Simple string content - check for injection patterns
            if self._contains_injection_patterns(content):
                raise SecurityValidationError(
                    f"Message {message_index} content contains potential injection patterns"
                )
            return content
        elif isinstance(content, list):
            # Array of content objects
            sanitized_content = []
            for i, content_item in enumerate(content):
                if not isinstance(content_item, dict):
                    raise SecurityValidationError(
                        f"Message {message_index} content item {i} must be a dictionary"
                    )

                # Validate content item structure
                if not content_item:
                    raise SecurityValidationError(
                        f"Message {message_index} content item {i} cannot be empty"
                    )

                # Check for malicious content types
                for content_type, value in content_item.items():
                    if content_type == 'text' and isinstance(value, str):
                        if self._contains_injection_patterns(value):
                            raise SecurityValidationError(
                                f"Message {message_index} content item {i} contains potential injection"
                            )
                    elif content_type in ['toolUse', 'toolResult']:
                        # Tool-related content - validate structure
                        if not isinstance(value, dict):
                            raise SecurityValidationError(
                                f"Message {message_index} {content_type} must be a dictionary"
                            )

                sanitized_content.append(content_item)
            return sanitized_content
        else:
            raise SecurityValidationError(
                f"Message {message_index} content must be string or list"
            )

    def _validate_system(self, system: Any) -> Any:
        """Validate system parameter."""
        if system is None:
            return None

        if isinstance(system, str):
            # Simple string system prompt
            if self._contains_injection_patterns(system):
                raise SecurityValidationError("System prompt contains potential injection patterns")
            return system
        elif isinstance(system, list):
            # Array of system messages
            sanitized_system = []
            for i, system_item in enumerate(system):
                if not isinstance(system_item, dict):
                    raise SecurityValidationError(f"System item {i} must be a dictionary")

                if 'text' not in system_item:
                    raise SecurityValidationError(f"System item {i} missing required 'text' field")

                if not isinstance(system_item['text'], str):
                    raise SecurityValidationError(f"System item {i} text must be a string")

                if self._contains_injection_patterns(system_item['text']):
                    raise SecurityValidationError(
                        f"System item {i} text contains potential injection patterns"
                    )

                sanitized_system.append(system_item)
            return sanitized_system
        else:
            raise SecurityValidationError("System parameter must be string, list, or None")

    def _validate_tool_config(self, tool_config: Any) -> Any:
        """Validate tool configuration."""
        if tool_config is None:
            return None

        if not isinstance(tool_config, dict):
            raise SecurityValidationError("toolConfig must be a dictionary")

        if 'tools' not in tool_config:
            return tool_config

        tools = tool_config['tools']
        if not isinstance(tools, list):
            raise SecurityValidationError("toolConfig.tools must be a list")

        # Validate each tool specification
        for i, tool in enumerate(tools):
            if not isinstance(tool, dict):
                raise SecurityValidationError(f"Tool {i} must be a dictionary")

            # Basic structure validation - more specific validation could be added
            if 'toolSpec' not in tool:
                raise SecurityValidationError(f"Tool {i} missing required 'toolSpec' field")

            tool_spec = tool['toolSpec']
            if not isinstance(tool_spec, dict):
                raise SecurityValidationError(f"Tool {i} toolSpec must be a dictionary")

            if 'name' not in tool_spec:
                raise SecurityValidationError(f"Tool {i} toolSpec missing required 'name' field")

            if not isinstance(tool_spec['name'], str):
                raise SecurityValidationError(f"Tool {i} name must be a string")

            # Check for suspicious tool names
            if self._is_suspicious_tool_name(tool_spec['name']):
                raise SecurityValidationError(
                    f"Tool {i} name '{tool_spec['name']}' appears suspicious"
                )

        return tool_config

    def _validate_inference_config(self, inference_config: Any) -> Any:
        """Validate inference configuration parameters."""
        if inference_config is None:
            return None

        if not isinstance(inference_config, dict):
            raise SecurityValidationError("inferenceConfig must be a dictionary")

        sanitized_config = {}

        for param_name, param_value in inference_config.items():
            if param_name not in self.config.allowed_inference_params:
                raise SecurityValidationError(
                    f"Unauthorized inference parameter '{param_name}'. "
                    f"Allowed parameters: {sorted(self.config.allowed_inference_params)}"
                )

            # Validate specific inference parameters
            if param_name == 'temperature':
                sanitized_config[param_name] = self._validate_temperature(param_value)
            elif param_name == 'topP':
                sanitized_config[param_name] = self._validate_top_p(param_value)
            elif param_name == 'maxTokens':
                sanitized_config[param_name] = self._validate_max_tokens(param_value)
            elif param_name == 'stopSequences':
                sanitized_config[param_name] = self._validate_stop_sequences(param_value)
            else:
                sanitized_config[param_name] = param_value

        return sanitized_config

    def _validate_temperature(self, temperature: Any) -> float:
        """Validate temperature parameter."""
        if not isinstance(temperature, (int, float)):
            raise SecurityValidationError("temperature must be a number")

        temp_float = float(temperature)
        if not (self.config.min_temperature <= temp_float <= self.config.max_temperature):
            raise SecurityValidationError(
                f"temperature {temp_float} out of range. "
                f"Must be between {self.config.min_temperature} and {self.config.max_temperature}"
            )

        return temp_float

    def _validate_top_p(self, top_p: Any) -> float:
        """Validate topP parameter."""
        if not isinstance(top_p, (int, float)):
            raise SecurityValidationError("topP must be a number")

        top_p_float = float(top_p)
        if not (self.config.min_top_p <= top_p_float <= self.config.max_top_p):
            raise SecurityValidationError(
                f"topP {top_p_float} out of range. "
                f"Must be between {self.config.min_top_p} and {self.config.max_top_p}"
            )

        return top_p_float

    def _validate_max_tokens(self, max_tokens: Any) -> int:
        """Validate maxTokens parameter."""
        if not isinstance(max_tokens, int):
            raise SecurityValidationError("maxTokens must be an integer")

        if not (self.config.min_max_tokens <= max_tokens <= self.config.max_max_tokens):
            raise SecurityValidationError(
                f"maxTokens {max_tokens} out of range. "
                f"Must be between {self.config.min_max_tokens} and {self.config.max_max_tokens}"
            )

        return max_tokens

    def _validate_stop_sequences(self, stop_sequences: Any) -> List[str]:
        """Validate stopSequences parameter."""
        if not isinstance(stop_sequences, list):
            raise SecurityValidationError("stopSequences must be a list")

        sanitized_sequences = []
        for i, sequence in enumerate(stop_sequences):
            if not isinstance(sequence, str):
                raise SecurityValidationError(f"stopSequence {i} must be a string")

            # Check for injection patterns in stop sequences
            if self._contains_injection_patterns(sequence):
                raise SecurityValidationError(
                    f"stopSequence {i} '{sequence}' contains potential injection patterns"
                )

            sanitized_sequences.append(sequence)

        return sanitized_sequences

    def _validate_required_parameters(self, sanitized_request: Dict[str, Any]) -> None:
        """Validate that required parameters are present."""
        required_params = ['messages', 'modelId']  # Use list for deterministic order

        for param in required_params:
            if param not in sanitized_request:
                raise SecurityValidationError(f"Missing required parameter: {param}")

    def _contains_injection_patterns(self, text: str) -> bool:
        """Check if text contains potential injection patterns."""
        if not isinstance(text, str):
            return False

        # Define suspicious patterns
        injection_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',                # JavaScript protocol
            r'on\w+\s*=',                 # Event handlers
            r'__proto__',                 # Prototype pollution
            r'constructor',               # Constructor pollution
            r'\.\./.*\.\.',              # Directory traversal
            r'path\s*=',                  # Path manipulation
            r'eval\s*\(',                 # eval usage
            r'exec\s*\(',                 # exec usage
            r'import\s+\w+',              # Import statements
            r'from\s+\w+\s+import',       # From import statements
        ]

        # Check each pattern
        for pattern in injection_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                return True

        return False

    def _is_suspicious_tool_name(self, tool_name: str) -> bool:
        """Check if a tool name appears suspicious."""
        suspicious_patterns = [
            r'^malicious',
            r'^injection',
            r'^exploit',
            r'^attack',
            r'^hack',
            r'^bypass',
            r'^escalate',
            r'^exfiltrate',
            r'^steal',
            r'^leak',
            r'^dump',
            r'^access.*token',
            r'^admin',
            r'^root',
            r'^system',
            r'^exec',
            r'^eval',
            r'^file.*system',
            r'^network',
            r'http.*request',
            r'^sql.*inject',
        ]

        for pattern in suspicious_patterns:
            if re.match(pattern, tool_name, re.IGNORECASE):
                return True

        return False


# Default validator instance
default_validator = RequestValidator()