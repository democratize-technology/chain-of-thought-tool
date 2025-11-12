# Configuration Constants Documentation

This document describes the configuration constants that were extracted from magic numbers throughout the codebase as part of **Task #4: Magic Numbers Extraction**. These constants improve code maintainability and make it easier to understand system behavior and adjust parameters.

## Core Module Constants (`chain_of_thought/core.py`)

### Rate Limiter Configuration

```python
DEFAULT_MAX_REQUESTS_PER_MINUTE = 60      # Maximum requests per minute per client
DEFAULT_MAX_REQUESTS_PER_HOUR = 1000       # Maximum requests per hour per client
DEFAULT_MAX_BURST_SIZE = 10                # Maximum consecutive immediate requests per client
```

**Usage**: These constants control the default rate limiting behavior for API protection against DoS attacks.

**Location**: Used in `RateLimiter.__init__()` method as default parameter values.

### Sanitization and Security Limits

```python
MAX_RECURSION_DEPTH = 50                   # Maximum recursion depth for sanitization
MAX_LIST_SIZE = 100                        # Maximum list size before truncation
MAX_STRING_LENGTH = 1000                   # Maximum string length after sanitization
MAX_JSON_SIZE = 100000                     # Maximum JSON string size (100KB limit)
```

**Usage**: These limits prevent DoS attacks through recursive data structures, oversized lists, and large JSON payloads.

**Location**: Used in the `_safe_json_dumps()` function's internal `sanitize()` method.

### Confidence Calibration Thresholds

```python
HIGH_CONFIDENCE_THRESHOLD = 0.15           # Threshold for significant confidence reduction
MEDIUM_CONFIDENCE_THRESHOLD = 0.05         # Threshold for moderate confidence adjustment
```

**Usage**: These thresholds determine the level of confidence adjustment applied in the `ConfidenceCalibrator._generate_calibration_reasoning()` method.

**Location**: Used to categorize adjustment magnitude for confidence calibration messages.

### Text Processing Limits

```python
MAX_PREDICTION_WORDS = 20                  # Maximum word count for complexity uncertainty detection
```

**Usage**: This threshold identifies overly complex predictions that may require additional uncertainty factors.

**Location**: Used in `ConfidenceCalibrator._identify_uncertainty_factors()` method.

## Validators Module Constants (`chain_of_thought/validators.py`)

### Text Processing Limits

```python
MAX_THOUGHT_LENGTH = 10000                 # Maximum allowed length for thought parameters
```

**Usage**: Prevents memory exhaustion and DoS attacks by limiting the size of thought input parameters.

**Location**: Used in `ParameterValidator.validate_thought_param()` method.

## Benefits of Configuration Constants

### 1. Improved Maintainability
- **Before**: `if len(thought) > 10000:` (magic number with unknown context)
- **After**: `if len(thought) > MAX_THOUGHT_LENGTH:` (self-documenting with clear purpose)

### 2. Easy Configuration
System behavior can be adjusted by modifying constant values in one central location rather than searching through code for scattered magic numbers.

### 3. Enhanced Code Readability
Developers can immediately understand the purpose and meaning of numeric limits through descriptive constant names.

### 4. Consistent Behavior
Constants ensure the same limits are applied consistently across the codebase, preventing accidental inconsistencies.

### 5. Easier Testing
Constants make it possible to write tests that verify configuration values are applied correctly throughout the system.

## Migration Details

### What Was Changed

1. **Rate Limiter Defaults**: Replaced hardcoded values (60, 1000, 10) with named constants
2. **Security Limits**: Extracted sanitization limits (50, 100, 1000, 100000) to configuration constants
3. **Confidence Thresholds**: Replaced magic numbers (0.15, 0.05) with named thresholds
4. **Text Processing Limits**: Extracted word count limit (20) and thought length limit (10000)

### Backward Compatibility

All magic number replacements maintain the exact same values and behavior to ensure complete backward compatibility. No functional changes were made - only improved code organization and readability.

### Testing

The `tests/test_configuration_constants.py` test suite validates that:
- All configuration constants are defined and accessible
- Constants have the expected values matching the original magic numbers
- Classes and methods properly use the constants instead of magic numbers
- System behavior remains unchanged

## Future Enhancements

Potential improvements to the configuration system:

1. **Environment-based Configuration**: Allow override of constants via environment variables
2. **Runtime Configuration**: Make some constants configurable at runtime for different deployment scenarios
3. **Validation**: Add validation to ensure constant values are within reasonable ranges
4. **Documentation**: Generate automatic documentation from constant definitions and comments

## Usage Examples

### Accessing Constants

```python
from chain_of_thought.core import (
    DEFAULT_MAX_REQUESTS_PER_MINUTE,
    MAX_RECURSION_DEPTH,
    HIGH_CONFIDENCE_THRESHOLD
)
from chain_of_thought.validators import MAX_THOUGHT_LENGTH

print(f"Rate limit: {DEFAULT_MAX_REQUESTS_PER_MINUTE} requests/minute")
print(f"Max recursion depth: {MAX_RECURSION_DEPTH}")
print(f"High confidence threshold: {HIGH_CONFIDENCE_THRESHOLD}")
print(f"Max thought length: {MAX_THOUGHT_LENGTH} characters")
```

### Custom Rate Limiter with Modified Limits

```python
from chain_of_thought.core import RateLimiter, DEFAULT_MAX_REQUESTS_PER_MINUTE

# Use default limits
limiter = RateLimiter()

# Or use custom limits while still referencing defaults
limiter = RateLimiter(
    max_requests_per_minute=DEFAULT_MAX_REQUESTS_PER_MINUTE * 2,  # Double the default
    max_requests_per_hour=2000,
    max_burst_size=20
)
```

This configuration system provides a solid foundation for maintainable, readable, and easily configurable chain-of-thought processing.