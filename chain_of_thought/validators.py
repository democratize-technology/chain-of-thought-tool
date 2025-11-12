"""
Parameter validation module for chain_of_thought package.

This module provides dedicated validator classes that separate validation logic
from business logic, following the Single Responsibility Principle.

Key benefits:
- Separation of concerns: Validation logic is isolated from business logic
- Reusability: Validators can be used across different classes
- Testability: Validation can be unit tested independently
- Maintainability: Changes to validation rules are centralized

Security features maintained:
- XSS prevention via HTML escaping for string inputs
- Input length limits to prevent DoS attacks
- Type validation for robust input handling
- Range validation for numeric inputs

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
"""
from typing import Dict, List, Optional, Any
import html
import re


class ParameterValidator:
    """
    Dedicated parameter validator class for chain of thought operations.

    This class contains all validation logic that was previously embedded
    in the ChainOfThought class, providing better separation of concerns.

    Thread-safe: All methods are pure functions with no shared state.
    """

    def validate_thought_param(self, thought: str) -> str:
        """
        Validate and sanitize the thought parameter.

        Args:
            thought: The thought text to validate

        Returns:
            Sanitized thought string with HTML escaping

        Raises:
            ValueError: If thought is invalid
        """
        if not isinstance(thought, str):
            raise ValueError("thought must be a string")
        # Allow empty thoughts for backward compatibility, but limit length for security
        if len(thought) > 10000:
            raise ValueError("thought cannot exceed 10,000 characters")

        # Strip leading/trailing whitespace and HTML escape
        return html.escape(thought.strip())

    def validate_reasoning_stage_param(self, reasoning_stage: str) -> str:
        """
        Validate and sanitize the reasoning_stage parameter.

        Args:
            reasoning_stage: The reasoning stage to validate

        Returns:
            Sanitized reasoning stage string

        Raises:
            ValueError: If reasoning_stage is invalid
        """
        if not isinstance(reasoning_stage, str):
            raise ValueError("reasoning_stage must be a string")
        if len(reasoning_stage) > 100:
            raise ValueError("reasoning_stage cannot exceed 100 characters")

        # Only allow alphanumeric, spaces, underscores, and hyphens (no other whitespace chars)
        if not re.match(r'^[a-zA-Z0-9 _-]+$', reasoning_stage):
            raise ValueError("reasoning_stage can only contain letters, numbers, spaces, underscores, and hyphens")

        return reasoning_stage.strip()

    def validate_step_parameters(self, step_number: int, total_steps: int) -> tuple:
        """
        Validate step_number and total_steps parameters.

        Args:
            step_number: The current step number
            total_steps: The total number of steps

        Returns:
            Tuple of (step_number, total_steps) as validated

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate step_number
        if not isinstance(step_number, int):
            raise ValueError("step_number must be an integer")
        # Allow reasonable range for step numbers (including negative for edge cases)
        # Increased limit to support existing test cases but still prevent DoS attacks
        if step_number < -10000 or step_number > 10000000:
            raise ValueError("step_number must be between -10000 and 10000000")

        # Validate total_steps
        if not isinstance(total_steps, int):
            raise ValueError("total_steps must be an integer")
        # Allow reasonable range for total_steps
        if total_steps < -10000 or total_steps > 10000000:
            raise ValueError("total_steps must be between -10000 and 10000000")

        # Allow flexibility in step_number vs total_steps for backward compatibility
        # (Only validate this for positive numbers where it makes logical sense)
        if step_number > 0 and total_steps > 0 and step_number > total_steps:
            raise ValueError("step_number cannot exceed total_steps")

        return step_number, total_steps

    def validate_confidence_param(self, confidence: float) -> float:
        """
        Validate the confidence parameter.

        Args:
            confidence: The confidence value to validate

        Returns:
            Validated confidence as float

        Raises:
            ValueError: If confidence is invalid
        """
        # Validate confidence - allow wider range for backward compatibility
        # But still prevent extreme values that could cause issues
        if not isinstance(confidence, (int, float)):
            raise ValueError("confidence must be a number")

        # Check for NaN and infinity values which can cause issues
        import math
        if math.isnan(confidence):
            raise ValueError("confidence must be between -100.0 and 100.0")
        if math.isinf(confidence):
            raise ValueError("confidence must be between -100.0 and 100.0")

        if confidence < -100.0 or confidence > 100.0:
            raise ValueError("confidence must be between -100.0 and 100.0")

        return float(confidence)

    def validate_boolean_param(self, value: bool, param_name: str) -> bool:
        """
        Validate a boolean parameter.

        Args:
            value: The boolean value to validate
            param_name: Name of the parameter for error messages

        Returns:
            Validated boolean value

        Raises:
            ValueError: If value is not a boolean
        """
        if not isinstance(value, bool):
            raise ValueError(f"{param_name} must be a boolean")
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
        Validate a list parameter with common validation logic.

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
            if not isinstance(input_list, list):
                raise ValueError(f"{param_name} must be a list")
            if len(input_list) > max_items:
                raise ValueError(f"{param_name} list cannot exceed {max_items} items")

            validated_items = []
            for item in input_list:
                # Type validation
                if not isinstance(item, item_type):
                    if item_type == int:
                        if param_name in ["dependencies", "contradicts"]:
                            raise ValueError(f"{param_name} values must be integers")
                        else:
                            raise ValueError(f"{param_name} values must be integers between {item_range[0] if item_range else -10000} and {item_range[1] if item_range else 10000000}")
                    elif item_type == str:
                        raise ValueError(f"{param_name} items must be strings")
                    else:
                        raise ValueError(f"{param_name} items must be {item_type.__name__}s")

                # String-specific validation
                if item_type == str:
                    if max_item_length and len(item) > max_item_length:
                        raise ValueError(f"{param_name} items cannot exceed {max_item_length} characters")
                    validated_items.append(html.escape(item.strip()) if escape_items else item.strip())

                # Integer-specific validation
                elif item_type == int:
                    if item_range:
                        min_val, max_val = item_range
                        if item < min_val or item > max_val:
                            raise ValueError(f"{param_name} values must be integers between {min_val} and {max_val}")
                    validated_items.append(item)

                else:
                    validated_items.append(item)

            return validated_items
        return None

    def validate_integer_list_param(self, int_list: Optional[List[int]], param_name: str) -> Optional[List[int]]:
        """
        Validate a list parameter that should contain integers.

        Args:
            int_list: The list to validate (or None)
            param_name: Name of the parameter for error messages

        Returns:
            Validated list or None

        Raises:
            ValueError: If list is invalid
        """
        return self.validate_list_param(
            int_list, param_name, int,
            max_items=50, item_range=(-10000, 10000000)
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