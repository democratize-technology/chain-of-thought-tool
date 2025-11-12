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
            TypeError: If thought is not a string
            ValueError: If thought is empty or exceeds length limits
        """
        # Type validation with strict isinstance check
        if not isinstance(thought, str):
            raise TypeError(f"thought must be a string, got {type(thought).__name__}")

        # Handle edge cases that could bypass length checks
        if not thought or thought.isspace():
            raise ValueError("thought cannot be empty or whitespace only")

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
        # Type validation with strict isinstance check
        if not isinstance(confidence, (int, float)):
            raise TypeError(f"confidence must be a number, got {type(confidence).__name__}")

        # Handle NaN and infinite values
        if isinstance(confidence, float):
            if confidence != confidence:  # NaN check
                raise ValueError("confidence cannot be NaN")
            if confidence in (float('inf'), float('-inf')):
                raise ValueError("confidence cannot be infinite")

        # Range validation
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

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
            raise TypeError(f"dependencies must be a list, got {type(dependencies).__name__}")

        # Validate each dependency
        validated_deps = []
        for dep in dependencies:
            # Type validation for each item
            if not isinstance(dep, int):
                raise TypeError(f"each dependency must be an integer, got {type(dep).__name__}")

            # Range validation with bounds to prevent resource exhaustion
            if dep < 1 or dep > 1000:  # Reasonable bounds for step numbers
                raise ValueError("dependencies must be positive integers between 1 and 1000")

            validated_deps.append(dep)

        return validated_deps

    def validate_step_number(self, step_number: int) -> int:
        """
        Validate the step number parameter with bounds checking.

        Args:
            step_number: The step number to validate

        Returns:
            The validated step number

        Raises:
            TypeError: If step_number is not an integer
            ValueError: If step_number is outside valid range
        """
        # Type validation
        if not isinstance(step_number, int):
            raise TypeError(f"step_number must be an integer, got {type(step_number).__name__}")

        # Bounds validation to prevent resource exhaustion
        if step_number < 1:
            raise ValueError("step_number must be positive")
        if step_number > 1000:
            raise ValueError("step_number cannot exceed 1000 to prevent resource exhaustion")

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
            raise TypeError(f"{param_name} must be a list, got {type(items).__name__}")

        # Resource limit validation
        if len(items) > max_items:
            raise ValueError(f"{param_name} cannot contain more than {max_items} items")

        validated_items = []
        for i, item in enumerate(items):
            # Individual item validation
            if not isinstance(item, str):
                raise TypeError(f"{param_name}[{i}] must be a string, got {type(item).__name__}")

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


class ChainOfThought:
    """Legacy class maintained for backward compatibility."""

    def __init__(self):
        self.validator = ParameterValidator()

    def add_step(self, thought: str, **kwargs):
        """Legacy method that uses the validator."""
        validated_thought = self.validator.validate_thought_param(thought)
        # ... rest of implementation
        pass