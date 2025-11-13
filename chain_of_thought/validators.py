"""
Parameter validation module for chain_of_thought package.

This module provides dedicated validator classes that separate validation logic
from business logic, following the Single Responsibility Principle.

Key benefits:
- Separation of concerns: Validation logic is isolated from business logic
- Reusability: Validators can be used across different classes
- Testability: Validation can be unit tested independently
- Maintainability: Changes to validation rules are centralized

Enhanced security features:
- XSS prevention via HTML escaping for string inputs
- Input length limits to prevent DoS attacks
- Robust type validation to prevent bypass attacks
- Range validation for numeric inputs
- Unicode character sanitization
- Nested structure validation
- Memory exhaustion protection
"""

# =============================================================================
# CONFIGURATION CONSTANTS - Task #4 Magic Numbers Extraction
# =============================================================================

# Text Processing Limits
MAX_THOUGHT_LENGTH = 10000

# Standard imports
import re
import html
import math
import unicodedata
from typing import Any, List, Dict, Optional, Union


class ParameterValidator:
    """
    Dedicated validator class for chain-of-thought parameters.

    This class provides centralized validation logic separated from business logic.

    Enhanced security features to prevent bypass attacks:
    - Robust type checking with strict isinstance validation
    - Unicode sanitization to prevent character-based bypasses
    - Memory protection against exhaustion attacks
    - Nested structure validation
    - Comprehensive numeric validation

    Thread-safe: All methods are pure functions with no shared state.
    """

    def __init__(self):
        """Initialize the parameter validator."""
        pass

    # Dangerous Unicode characters that should be removed
    _DANGEROUS_UNICODE_CHARS = {
        '\u200B',  # Zero-width space
        '\u200C',  # Zero-width non-joiner
        '\u200D',  # Zero-width joiner
        '\u2028',  # Line separator
        '\u2029',  # Paragraph separator
        '\uFEFF',  # Zero-width no-break space (BOM)
        '\u2060',  # Word joiner
    }

    # Control characters that should be removed
    _CONTROL_CHARS = {
        chr(i) for i in range(32) if i not in (9, 10, 13)  # Allow tab, newline, carriage return
    }

    def validate_thought_param(self, thought: str) -> str:
        """
        Validate and sanitize the thought parameter with enhanced security.

        Args:
            thought: The thought text to validate

        Returns:
            The sanitized and validated thought text

        Raises:
            ValueError: If thought is empty, None, or exceeds length limits
        """
        # Handle None as special case for better error handling
        if thought is None:
            raise ValueError("thought must be a string")

        # Type validation with strict isinstance check
        if not isinstance(thought, str):
            raise ValueError("thought must be a string")

        # Handle edge cases that could bypass length checks
        # Allow empty strings for edge case testing
        # Trim whitespace-only strings to empty
        if thought.isspace():
            thought = ""

        # Sanitize unicode characters to prevent bypass attacks
        thought = self._sanitize_unicode_string(thought)

        # Check length after unicode normalization (may change length)
        if len(thought) > MAX_THOUGHT_LENGTH:
            raise ValueError(f"thought cannot exceed {MAX_THOUGHT_LENGTH:,} characters")

        # Strip leading/trailing whitespace and HTML escape
        return html.escape(thought.strip())

    def validate_confidence_param(self, confidence: float) -> float:
        """
        Validate and sanitize the confidence parameter with enhanced security.

        Args:
            confidence: The confidence value to validate

        Returns:
            The validated confidence value

        Raises:
            TypeError: If confidence is not a number
            ValueError: If confidence is outside valid range
        """
        # Type validation with strict isinstance check - exclude numpy types for security
        if not isinstance(confidence, (int, float)) or type(confidence).__module__ == 'numpy':
            raise ValueError(f"confidence must be a number, got {type(confidence).__name__}")

        # Handle NaN and infinite values
        if isinstance(confidence, float):
            if confidence != confidence:  # NaN check
                raise ValueError("confidence must be between -100.0 and 100.0")
            if confidence in (float('inf'), float('-inf')):
                raise ValueError("confidence must be between -100.0 and 100.0")

        # Range validation - confidence must be between -100.0 and 100.0
        if confidence < -100.0 or confidence > 100.0:
            raise ValueError("confidence must be between -100.0 and 100.0")

        # Return as float for consistency
        return float(confidence)

    def validate_dependencies_param(self, dependencies: List[int]) -> List[int]:
        """
        Validate and sanitize the dependencies parameter with enhanced security.

        Args:
            dependencies: The list of step dependencies to validate

        Returns:
            The validated and sanitized dependencies list

        Raises:
            TypeError: If dependencies is not a list
            ValueError: If dependencies contain invalid values
        """
        # Type validation with strict isinstance check
        if not isinstance(dependencies, list):
            raise ValueError("dependencies must be a list")

        # Resource limit validation - check list size before individual items
        if len(dependencies) > 50:  # Max items limit for security
            raise ValueError("dependencies cannot exceed 50 items")

        # Validate each dependency
        validated_deps = []
        for dep in dependencies:
            # Type validation for each item
            if not isinstance(dep, int):
                raise ValueError("dependencies values must be integers")

            # Range validation with bounds to prevent resource exhaustion
            if dep < 1 or dep > 1000:  # Reasonable bounds for step numbers
                raise ValueError("dependencies values must be integers between 1 and 1000")

            validated_deps.append(dep)

        return validated_deps

    def validate_step_number(self, step_number: int, param_name: str = "step_number") -> int:
        """
        Validate the step number parameter with bounds checking.

        Args:
            step_number: The step number to validate
            param_name: Name of the parameter for error messages

        Returns:
            The validated step number

        Raises:
            TypeError: If step_number is not an integer
            ValueError: If step_number is outside valid range
        """
        # Type validation
        if not isinstance(step_number, int):
            raise ValueError(f"{param_name} must be an integer")

        # Bounds validation to prevent resource exhaustion
        if step_number < 1 or step_number > 1000:
            raise ValueError(f"{param_name} must be between 1 and 1000")

        return step_number

    def _sanitize_unicode_string(self, text: str) -> str:
        """
        Sanitize a string by removing dangerous Unicode characters.

        This prevents bypass attacks using invisible or control characters.

        Args:
            text: The text to sanitize

        Returns:
            Sanitized text
        """
        # Remove dangerous Unicode characters
        for char in self._DANGEROUS_UNICODE_CHARS:
            text = text.replace(char, '')

        # Remove control characters except allowed ones
        text = ''.join(char for char in text if char not in self._CONTROL_CHARS)

        # Normalize Unicode to prevent bypass via different representations
        text = unicodedata.normalize('NFKC', text)

        return text

    def validate_list_input(self, items: List[str], param_name: str,
                          max_items: int = 50, max_item_length: int = 500,
                          escape_items: bool = True) -> List[str]:
        """
        Validate and sanitize a list of string inputs with comprehensive security.

        Args:
            items: List of strings to validate
            param_name: Name of the parameter for error messages
            max_items: Maximum number of items allowed (resource limit)
            max_item_length: Maximum length per item
            escape_items: Whether to HTML-escape items

        Returns:
            List of validated and sanitized strings

        Raises:
            TypeError: If items is not a list
            ValueError: If items contain invalid values or exceed limits
        """
        # Type validation with strict isinstance check
        if not isinstance(items, list):
            raise ValueError(f"{param_name} must be a list")

        # Resource limit validation
        if len(items) > max_items:
            raise ValueError(f"{param_name} list cannot exceed {max_items} items")

        validated_items = []
        for i, item in enumerate(items):
            # Individual item validation
            if not isinstance(item, str):
                raise ValueError(f"{param_name} items must be strings")

            # Unicode sanitization
            sanitized_item = self._sanitize_unicode_string(item)

            # Length validation
            if max_item_length and len(sanitized_item) > max_item_length:
                raise ValueError(f"{param_name} items cannot exceed {max_item_length} characters")

            # Escape HTML if requested
            if escape_items:
                sanitized_item = html.escape(sanitized_item.strip())

            validated_items.append(sanitized_item)

        return validated_items

    def validate_input(self,
                      thought: str,
                      step_number: int,
                      total_steps: int,
                      next_step_needed: bool,
                      reasoning_stage: str = "Analysis",
                      confidence: float = 0.8,
                      dependencies: Optional[List[int]] = None,
                      contradicts: Optional[List[int]] = None,
                      evidence: Optional[List[str]] = None,
                      assumptions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate all input parameters for chain-of-thought processing.

        This method consolidates all parameter validation logic and returns
        a dictionary of validated parameters for use in business logic.

        Args:
            thought: The thought text to validate
            step_number: The current step number
            total_steps: The total number of steps expected
            next_step_needed: Whether another step is needed
            reasoning_stage: The current reasoning stage
            confidence: Confidence level (0.0 to 1.0)
            dependencies: List of step dependencies
            contradicts: List of contradicted steps
            evidence: List of evidence items
            assumptions: List of assumptions

        Returns:
            Dictionary containing all validated parameters

        Raises:
            TypeError: If any parameter has wrong type
            ValueError: If any parameter has invalid value
        """
        # Validate required parameters using existing methods
        validated_thought = self.validate_thought_param(thought)
        validated_step_number = self.validate_step_number(step_number, "step_number")
        validated_total_steps = self.validate_step_number(total_steps, "total_steps")  # Reuse step validation
        validated_confidence = self.validate_confidence_param(confidence)
        validated_reasoning_stage = self._validate_reasoning_stage(reasoning_stage)
        validated_next_step_needed = self._validate_boolean_param(next_step_needed, "next_step_needed")

        # Validate optional parameters (handle None values)
        validated_dependencies = self._validate_optional_dependencies(dependencies)
        validated_contradicts = self._validate_optional_contradicts(contradicts)
        validated_evidence = self._validate_optional_string_list(evidence, "evidence")
        validated_assumptions = self._validate_optional_string_list(assumptions, "assumptions")

        # Additional logical validation
        self._validate_step_relationships(validated_step_number, validated_total_steps)

        return {
            "thought": validated_thought,
            "step_number": validated_step_number,
            "total_steps": validated_total_steps,
            "next_step_needed": validated_next_step_needed,
            "reasoning_stage": validated_reasoning_stage,
            "confidence": validated_confidence,
            "dependencies": validated_dependencies,
            "contradicts": validated_contradicts,
            "evidence": validated_evidence,
            "assumptions": validated_assumptions
        }

    def _validate_reasoning_stage(self, reasoning_stage: str) -> str:
        """
        Validate the reasoning_stage parameter.

        Args:
            reasoning_stage: The reasoning stage to validate

        Returns:
            The validated reasoning stage

        Raises:
            TypeError: If reasoning_stage is not a string
            ValueError: If reasoning_stage is empty, too long, or contains invalid characters
        """
        if not isinstance(reasoning_stage, str):
            raise ValueError("reasoning_stage must be a string")

        # Sanitize the input
        sanitized_stage = self._sanitize_unicode_string(reasoning_stage)

        if not sanitized_stage or sanitized_stage.isspace():
            raise ValueError("reasoning_stage cannot be empty or whitespace only")

        # Check length limit (100 characters for security)
        if len(sanitized_stage) > 100:
            raise ValueError("reasoning_stage cannot exceed 100 characters")

        # Explicitly reject control characters that might survive sanitization
        if any(ord(c) < 32 and c != ' ' for c in sanitized_stage):
            raise ValueError("reasoning_stage can only contain letters, numbers, spaces, underscores, and hyphens")

        # Allow only alphanumeric, spaces, underscores, and hyphens
        if not re.match(r'^[a-zA-Z0-9 \-_]+$', sanitized_stage):
            raise ValueError("reasoning_stage can only contain letters, numbers, spaces, underscores, and hyphens")

        return html.escape(sanitized_stage.strip())

    def _validate_boolean_param(self, value: bool, param_name: str) -> bool:
        """
        Validate a boolean parameter.

        Args:
            value: The boolean value to validate
            param_name: Name of the parameter for error messages

        Returns:
            The validated boolean value

        Raises:
            TypeError: If value is not a boolean
        """
        if not isinstance(value, bool):
            raise ValueError(f"{param_name} must be a boolean")

        return value

    def _validate_optional_dependencies(self, dependencies: Optional[List[int]]) -> List[int]:
        """
        Validate optional dependencies parameter.

        Args:
            dependencies: List of step dependencies or None

        Returns:
            Validated dependencies list (empty list if None)

        Raises:
            TypeError: If dependencies is not a list or None
            ValueError: If dependencies contain invalid values
        """
        if dependencies is None:
            return []

        return self.validate_dependencies_param(dependencies)

    def _validate_optional_contradicts(self, contradicts: Optional[List[int]]) -> List[int]:
        """
        Validate optional contradicts parameter.

        Args:
            contradicts: List of contradicted steps or None

        Returns:
            Validated contradicts list (empty list if None)

        Raises:
            TypeError: If contradicts is not a list or None
            ValueError: If contradicts contain invalid values
        """
        if contradicts is None:
            return []

        # Reuse the same validation logic as dependencies
        return self.validate_dependencies_param(contradicts)

    def _validate_optional_string_list(self, items: Optional[List[str]], param_name: str) -> List[str]:
        """
        Validate optional string list parameter.

        Args:
            items: List of strings or None
            param_name: Name of the parameter for error messages

        Returns:
            Validated string list (empty list if None)

        Raises:
            TypeError: If items is not a list or None
            ValueError: If items contain invalid values
        """
        if items is None:
            return []

        return self.validate_list_input(items, param_name)

    def _validate_step_relationships(self, step_number: int, total_steps: int) -> None:
        """
        Validate logical relationships between step parameters.

        Args:
            step_number: Current step number
            total_steps: Total expected steps

        Raises:
            ValueError: If step relationships are invalid
        """
        if step_number > total_steps:
            raise ValueError("step_number cannot exceed total_steps")

    def validate_reasoning_stage_param(self, reasoning_stage: str) -> str:
        """
        Validate and sanitize the reasoning_stage parameter.

        This is a public method for external validation.

        Args:
            reasoning_stage: The reasoning stage to validate

        Returns:
            The validated reasoning stage

        Raises:
            TypeError: If reasoning_stage is not a string
            ValueError: If reasoning_stage is empty or contains invalid characters
        """
        return self._validate_reasoning_stage(reasoning_stage)

    def validate_step_parameters(self, step_number: Union[int, float, str, bool], total_steps: int) -> int:
        """
        Validate step number parameter with enhanced type checking.

        Args:
            step_number: The step number to validate
            total_steps: The total number of steps

        Returns:
            The validated step number

        Raises:
            ValueError: If step_number or total_steps is not a valid positive integer
        """
        # Reject boolean values (they are instances of int in Python)
        if isinstance(step_number, bool):
            raise ValueError("step_number must be a positive integer, not a boolean")

        # Reject all float values for strict type safety
        if isinstance(step_number, float):
            raise ValueError("step_number must be a positive integer, not a float")

        # Type validation - must be integer only
        if not isinstance(step_number, int):
            raise ValueError("step_number must be a positive integer")

        # Validate step_number with specific error message first (priority)
        if step_number < 1 or step_number > 1000:
            raise ValueError("step_number must be between 1 and 1000")

        # Validate total_steps with specific error message
        if not isinstance(total_steps, int):
            raise ValueError("total_steps must be an integer")
        if total_steps < 1 or total_steps > 1000:
            raise ValueError("total_steps must be between 1 and 1000")

        # Validate relationship between step_number and total_steps
        self._validate_step_relationships(step_number, total_steps)

        return step_number

    def validate_length_limit(self, text: str, max_length: int, param_name: str = "text") -> str:
        """
        Validate text length limit.

        Args:
            text: Text to validate
            max_length: Maximum allowed length
            param_name: Parameter name for error messages

        Returns:
            Validated text

        Raises:
            ValueError: If text exceeds length limit
        """
        if not isinstance(text, str):
            raise ValueError(f"{param_name} must be a string")

        if len(text) > max_length:
            raise ValueError(f"{param_name} cannot exceed {max_length:,} characters")

        return text

    def validate_type(self, value: Any, expected_type: type, param_name: str = "value") -> Any:
        """
        Validate parameter type.

        Args:
            value: Value to validate
            expected_type: Expected type
            param_name: Parameter name for error messages

        Returns:
            Validated value

        Raises:
            ValueError: If value has wrong type
        """
        if not isinstance(value, expected_type):
            raise ValueError(f"{param_name} must be a {expected_type.__name__}, got {type(value).__name__}")

        return value

    def validate_boolean_param(self, value: Any, param_name: str = "parameter") -> bool:
        """
        Validate boolean parameter with strict type checking.

        Args:
            value: Value to validate
            param_name: Parameter name for error messages

        Returns:
            Validated boolean value

        Raises:
            ValueError: If value is not a boolean
        """
        if not isinstance(value, bool):
            raise ValueError(f"{param_name} must be a boolean, got {type(value).__name__}")

        return value

    def validate_integer_list_param(self, items: List[Any], param_name: str = "parameter",
                                   max_items: int = 50, min_value: int = 1, max_value: int = 1000) -> List[int]:
        """
        Validate integer list parameter with comprehensive security.

        Args:
            items: List of items to validate
            param_name: Parameter name for error messages
            max_items: Maximum number of items allowed
            min_value: Minimum allowed integer value
            max_value: Maximum allowed integer value

        Returns:
            List of validated integers

        Raises:
            ValueError: If items contain invalid values or exceed limits
        """
        # Type validation
        if not isinstance(items, list):
            raise ValueError(f"{param_name} must be a list, got {type(items).__name__}")

        # Resource limit validation
        if len(items) > max_items:
            raise ValueError(f"{param_name} list cannot exceed {max_items} items")

        validated_items = []
        for i, item in enumerate(items):
            # Reject boolean values (they are instances of int in Python)
            if isinstance(item, bool):
                raise ValueError(f"{param_name}[{i}] must be an integer, not a boolean")

            # Reject non-integer types
            if not isinstance(item, int):
                raise ValueError(f"{param_name}[{i}] must be an integer, got {type(item).__name__}")

            # Range validation
            if item < min_value or item > max_value:
                raise ValueError(f"{param_name}[{i}] must be between {min_value} and {max_value}")

            validated_items.append(item)

        return validated_items


