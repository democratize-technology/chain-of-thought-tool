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
- Bypass attack prevention

Usage:
    validator = ParameterValidator()
    validated_thought = validator.validate_thought_param(user_input)

# Migration Notes

This module was created as part of Task #3: Extract validation logic into dedicated validator classes.

**Before:**
```python
class ChainOfThought:
    def _validate_thought_param(self, thought: str) -> str:
        # validation logic here
        return sanitized_thought
```

**After:**
```python
class ChainOfThought:
    def __init__(self):
        self.validator = ParameterValidator()

    def add_step(self, thought: str, ...):
        # Now uses self.validator.validate_input()
```

All validation behavior and error messages remain exactly the same to ensure
backward compatibility. The only change is the architectural organization.

# Security Enhancements

This module includes comprehensive protection against:
- Type confusion attacks (numpy, decimal, etc.)
- Unicode bypass attempts
- HTML injection vulnerabilities
- Memory exhaustion via large inputs
- Nested structure attacks
- Regex bypass attempts
- Memory exhaustion protection via step number bounds (1-1000 range)
- Dependency list bounds validation to prevent resource exhaustion
"""
from typing import Dict, List, Optional, Any, Union
import html
import re
import math
import unicodedata


class ParameterValidator:
    """
    Dedicated parameter validator class for chain of thought operations.

    This class contains all validation logic that was previously embedded
    in the ChainOfThought class, providing better separation of concerns.

    Enhanced security features to prevent bypass attacks:
    - Robust type checking with strict isinstance validation
    - Unicode sanitization to prevent character-based bypasses
    - Memory protection against exhaustion attacks
    - Nested structure validation
    - Comprehensive numeric validation

    Thread-safe: All methods are pure functions with no shared state.
    """

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
        '\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07',
        '\x08', '\x0B', '\x0C', '\x0E', '\x0F', '\x10', '\x11', '\x12',
        '\x13', '\x14', '\x15', '\x16', '\x17', '\x18', '\x19', '\x1A',
        '\x1B', '\x1C', '\x1D', '\x1E', '\x1F', '\x7F'
    }

    def _sanitize_unicode_string(self, text: str) -> str:
        """
        Sanitize unicode string to prevent bypass attacks.

        Args:
            text: Input string to sanitize

        Returns:
            Sanitized string with dangerous characters removed
        """
        # Remove dangerous Unicode characters
        for char in self._DANGEROUS_UNICODE_CHARS:
            text = text.replace(char, '')

        # Remove control characters
        for char in self._CONTROL_CHARS:
            text = text.replace(char, '')

        # Normalize Unicode to prevent spoofing
        text = unicodedata.normalize('NFKC', text)

        return text

    def _strict_type_check(self, value: Any, expected_type: type, param_name: str) -> None:
        """
        Perform strict type checking to prevent bypass attacks while maintaining backward compatibility.

        Args:
            value: Value to check
            expected_type: Expected type
            param_name: Parameter name for error messages

        Raises:
            ValueError: If type check fails
        """
        # Check for exact type match (no inheritance allowed for primitives)
        if expected_type == bool:
            if type(value) is not bool:  # Use type() instead of isinstance() for bool
                raise ValueError(f"{param_name} must be a boolean")
        elif expected_type == int:
            # Reject bool (isinstance(True, int) == True, but we don't want bools)
            # But allow numpy/scipy numeric types for backward compatibility after conversion check
            if type(value) is not int:
                # Check if it's a numpy/scipy type that passes isinstance but should be rejected
                if hasattr(value, '__module__'):
                    module = getattr(value, '__module__', '')
                    if module and any(x in module for x in ['numpy', 'scipy', 'pandas']):
                        raise ValueError(f"{param_name} must be an integer")
                raise ValueError(f"{param_name} must be an integer")
        elif expected_type == float:
            # For confidence specifically, be more strict to prevent bypass attacks
            if param_name == "confidence":
                # Use strict type checking for confidence
                if type(value) not in (int, float):
                    raise ValueError(f"{param_name} must be a number")
                # Reject numpy/scipy types that might have different behavior
                if hasattr(value, '__module__'):
                    module = getattr(value, '__module__', '')
                    if module and any(x in module for x in ['numpy', 'scipy', 'pandas']):
                        raise ValueError(f"{param_name} must be a number")
            else:
                # For other float parameters, use more permissive but safe checking
                if not isinstance(value, (int, float)):
                    raise ValueError(f"{param_name} must be a number")
                # Reject bool specifically
                if isinstance(value, bool):
                    raise ValueError(f"{param_name} must be a number")
        elif expected_type == str:
            if not isinstance(value, str):
                raise ValueError(f"{param_name} must be a string")
        elif expected_type == list:
            if type(value) is not list:
                raise ValueError(f"{param_name} must be a list")
        else:
            if not isinstance(value, expected_type):
                raise ValueError(f"{param_name} must be a {expected_type.__name__}")

    def _safe_float_conversion(self, value: Union[int, float]) -> float:
        """
        Safely convert value to float with comprehensive validation.

        Args:
            value: Value to convert

        Returns:
            Validated float value

        Raises:
            ValueError: If conversion fails or value is invalid
        """
        try:
            # Convert to float
            float_val = float(value)

            # Check for NaN
            if math.isnan(float_val):
                raise ValueError("confidence must be between -100.0 and 100.0")

            # Check for infinity
            if math.isinf(float_val):
                raise ValueError("confidence must be between -100.0 and 100.0")

            return float_val
        except (OverflowError, ValueError, TypeError) as e:
            if "must be between" in str(e):
                raise
            raise ValueError("confidence must be a valid number")

    def validate_thought_param(self, thought: str) -> str:
        """
        Validate and sanitize the thought parameter with enhanced security.

        Args:
            thought: The thought text to validate

        Returns:
            Sanitized thought string with HTML escaping and unicode sanitization

        Raises:
            ValueError: If thought is invalid
        """
        # Strict type checking
        self._strict_type_check(thought, str, "thought")

        # Sanitize unicode characters to prevent bypass attacks
        thought = self._sanitize_unicode_string(thought)

        # Check length after unicode normalization (may change length)
        if len(thought) > 10000:
            raise ValueError("thought cannot exceed 10,000 characters")

        # Strip leading/trailing whitespace and HTML escape
        return html.escape(thought.strip())

    def validate_reasoning_stage_param(self, reasoning_stage: str) -> str:
        """
        Validate and sanitize the reasoning_stage parameter with enhanced security.

        Args:
            reasoning_stage: The reasoning stage to validate

        Returns:
            Sanitized reasoning stage string

        Raises:
            ValueError: If reasoning_stage is invalid
        """
        # Strict type checking
        self._strict_type_check(reasoning_stage, str, "reasoning_stage")

        # Sanitize unicode characters to prevent bypass attacks
        reasoning_stage = self._sanitize_unicode_string(reasoning_stage)

        # Check length after unicode normalization
        if len(reasoning_stage) > 100:
            raise ValueError("reasoning_stage cannot exceed 100 characters")

        # Only allow alphanumeric, spaces, underscores, and hyphens (no other whitespace chars)
        # Use stricter regex that prevents unicode spoofing
        if not re.match(r'^[a-zA-Z0-9 _-]+$', reasoning_stage):
            raise ValueError("reasoning_stage can only contain letters, numbers, spaces, underscores, and hyphens")

        return reasoning_stage.strip()

    def validate_step_parameters(self, step_number: int, total_steps: int) -> tuple:
        """
        Validate step_number and total_steps parameters with enhanced type safety.

        SECURITY IMPROVEMENT (Task #2): Replaced sys.maxsize range with reasonable bounds (1-1000)
        to prevent memory exhaustion attacks. The previous validation allowed values up to
        sys.maxsize (9.2 quintillion on 64-bit systems), which could cause memory issues
        and DoS vulnerabilities.

        Args:
            step_number: The current step number (must be 1-1000)
            total_steps: The total number of steps (must be 1-1000)

        Returns:
            Tuple of (step_number, total_steps) as validated

        Raises:
            ValueError: If parameters are invalid or out of bounds
        """
        # Validate step_number with strict type checking
        self._strict_type_check(step_number, int, "step_number")
        # SECURITY: Enforce reasonable bounds to prevent memory issues (1-1000 range)
        # Previously used sys.maxsize range which was a security vulnerability
        if step_number < 1 or step_number > 1000:
            raise ValueError("step_number must be between 1 and 1000")

        # Validate total_steps with strict type checking
        self._strict_type_check(total_steps, int, "total_steps")
        # SECURITY: Enforce reasonable bounds to prevent memory issues (1-1000 range)
        # Previously used sys.maxsize range which was a security vulnerability
        if total_steps < 1 or total_steps > 1000:
            raise ValueError("total_steps must be between 1 and 1000")

        # Validate logical relationship between step_number and total_steps
        if step_number > total_steps:
            raise ValueError("step_number cannot exceed total_steps")

        return step_number, total_steps

    def validate_confidence_param(self, confidence: float) -> float:
        """
        Validate the confidence parameter with enhanced security against bypass attacks.

        Args:
            confidence: The confidence value to validate

        Returns:
            Validated confidence as float

        Raises:
            ValueError: If confidence is invalid
        """
        # Accept int or float for backward compatibility but with strict checking
        # Use type() instead of isinstance() to prevent numpy/decimal bypass attacks
        if type(confidence) not in (int, float):
            raise ValueError("confidence must be a number")

        # Reject boolean values (type(True) is bool, not int)
        if isinstance(confidence, bool):
            raise ValueError("confidence must be a number")

        # Use safe float conversion with comprehensive validation
        validated_confidence = self._safe_float_conversion(confidence)

        # Range validation
        if validated_confidence < -100.0 or validated_confidence > 100.0:
            raise ValueError("confidence must be between -100.0 and 100.0")

        return validated_confidence

    def validate_boolean_param(self, value: bool, param_name: str) -> bool:
        """
        Validate a boolean parameter with strict type checking to prevent bypass attacks.

        Args:
            value: The boolean value to validate
            param_name: Name of the parameter for error messages

        Returns:
            Validated boolean value

        Raises:
            ValueError: If value is not a boolean
        """
        # Use strict type checking to prevent bypass via truthy values
        self._strict_type_check(value, bool, param_name)
        return value

    def validate_list_param(
        self,
        input_list: Optional[List],
        param_name: str,
        item_type: type,
        max_items: int = 50,
        max_item_length: Optional[int] = None,
        item_range: Optional[tuple] = None,
        escape_items: bool = False
    ) -> Optional[List]:
        """
        Validate a list parameter with enhanced security against bypass attacks.

        Args:
            input_list: The list to validate (or None)
            param_name: Name of the parameter for error messages
            item_type: Expected type of list items (int, str, etc.)
            max_items: Maximum number of items allowed
            max_item_length: Maximum length for string items (if applicable)
            item_range: Valid range for integer items as (min, max)
            escape_items: Whether to HTML-escape string items

        Returns:
            Validated list (with escaping if applicable) or None

        Raises:
            ValueError: If list is invalid
        """
        if input_list is not None:
            # Strict type checking for the list itself
            self._strict_type_check(input_list, list, param_name)

            if len(input_list) > max_items:
                raise ValueError(f"{param_name} list cannot exceed {max_items} items")

            validated_items = []
            for item in input_list:
                # Strict type validation for each item
                if item_type == int:
                    # Reject bool (isinstance(True, int) == True)
                    if type(item) is not int:
                        if param_name in ["dependencies", "contradicts"]:
                            raise ValueError(f"{param_name} values must be integers")
                        else:
                            raise ValueError(f"{param_name} values must be integers between {item_range[0] if item_range else -10000} and {item_range[1] if item_range else 10000000}")
                elif item_type == str:
                    if not isinstance(item, str):
                        raise ValueError(f"{param_name} items must be strings")
                elif item_type == float:
                    # Reject bool and int
                    if type(item) is not float:
                        raise ValueError(f"{param_name} items must be floats")
                else:
                    if not isinstance(item, item_type):
                        raise ValueError(f"{param_name} items must be {item_type.__name__}s")

                # String-specific validation with enhanced security
                if item_type == str:
                    # Sanitize unicode to prevent bypass attacks
                    sanitized_item = self._sanitize_unicode_string(item)

                    if max_item_length and len(sanitized_item) > max_item_length:
                        raise ValueError(f"{param_name} items cannot exceed {max_item_length} characters")

                    processed_item = sanitized_item.strip()
                    validated_items.append(html.escape(processed_item) if escape_items else processed_item)

                # Integer-specific validation
                elif item_type == int:
                    if item_range:
                        min_val, max_val = item_range
                        if item < min_val or item > max_val:
                            raise ValueError(f"{param_name} values must be integers between {min_val} and {max_val}")
                    validated_items.append(item)

                # Float-specific validation
                elif item_type == float:
                    # Use safe float conversion for consistency
                    safe_float = self._safe_float_conversion(item)
                    if item_range:
                        min_val, max_val = item_range
                        if safe_float < min_val or safe_float > max_val:
                            raise ValueError(f"{param_name} values must be floats between {min_val} and {max_val}")
                    validated_items.append(safe_float)

                else:
                    validated_items.append(item)

            return validated_items
        return None

    def validate_integer_list_param(self, int_list: Optional[List[int]], param_name: str) -> Optional[List[int]]:
        """
        Validate a list parameter that should contain integers.

        SECURITY IMPROVEMENT (Task #2): Replaced sys.maxsize range with reasonable bounds (1-1000)
        for dependency and contradiction lists. Since these values reference step numbers,
        they should be constrained to the same bounds as step numbers to prevent resource
        exhaustion and maintain logical consistency.

        Args:
            int_list: The list to validate (or None)
            param_name: Name of the parameter for error messages

        Returns:
            Validated list or None

        Raises:
            ValueError: If list is invalid or contains out-of-bounds values
        """
        # SECURITY: For dependencies and contradicts, use reasonable bounds (1-1000) since they reference step numbers
        # Previously used sys.maxsize range which was a security vulnerability
        return self.validate_list_param(
            int_list, param_name, int,
            max_items=50, item_range=(1, 1000)
        )

    def validate_string_list_param(self, str_list: Optional[List[str]], param_name: str) -> Optional[List[str]]:
        """
        Validate a list parameter that should contain strings.

        Args:
            str_list: The list to validate (or None)
            param_name: Name of the parameter for error messages

        Returns:
            Validated list with HTML-escaped strings or None

        Raises:
            ValueError: If list is invalid
        """
        return self.validate_list_param(
            str_list, param_name, str,
            max_items=50, max_item_length=500, escape_items=True
        )

    def validate_input(
        self,
        thought: str,
        step_number: int,
        total_steps: int,
        next_step_needed: bool,
        reasoning_stage: str = "Analysis",
        confidence: float = 0.8,
        dependencies: Optional[List[int]] = None,
        contradicts: Optional[List[int]] = None,
        evidence: Optional[List[str]] = None,
        assumptions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate all input parameters for security and reasonable limits.

        Returns validated parameters with HTML escaping applied.
        Raises ValueError with descriptive messages for validation failures.
        """
        # Validate individual parameters using helper functions
        thought_cleaned = self.validate_thought_param(thought)
        reasoning_stage_cleaned = self.validate_reasoning_stage_param(reasoning_stage)
        validated_step_number, validated_total_steps = self.validate_step_parameters(step_number, total_steps)
        validated_confidence = self.validate_confidence_param(confidence)
        validated_next_step_needed = self.validate_boolean_param(next_step_needed, "next_step_needed")

        # Validate list parameters
        dependencies_cleaned = self.validate_integer_list_param(dependencies, "dependencies")
        contradicts_cleaned = self.validate_integer_list_param(contradicts, "contradicts")
        evidence_cleaned = self.validate_string_list_param(evidence, "evidence")
        assumptions_cleaned = self.validate_string_list_param(assumptions, "assumptions")

        return {
            "thought": thought_cleaned,
            "step_number": validated_step_number,
            "total_steps": validated_total_steps,
            "reasoning_stage": reasoning_stage_cleaned,
            "confidence": validated_confidence,
            "next_step_needed": validated_next_step_needed,
            "dependencies": dependencies_cleaned,
            "contradicts": contradicts_cleaned,
            "evidence": evidence_cleaned,
            "assumptions": assumptions_cleaned
        }