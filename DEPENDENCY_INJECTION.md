# Dependency Injection Implementation

## Overview

This document describes the dependency injection (DI) implementation that replaces global singleton patterns while maintaining full backward compatibility.

## Problem Statement

The original codebase used global singleton instances:
- `_chain_processor` - Global ChainOfThought instance
- `_hypothesis_generator` - Global HypothesisGenerator instance
- `_assumption_mapper` - Global AssumptionMapper instance
- `_confidence_calibrator` - Global ConfidenceCalibrator instance

This approach had several issues:
1. **Testing difficulties** - Global state makes unit testing challenging
2. **Thread safety concerns** - Shared state across conversations
3. **Flexibility limitations** - Cannot easily swap implementations
4. **Configuration challenges** - Cannot customize service instances per use case

## Solution: ServiceRegistry Pattern

### Core Components

#### ServiceRegistry
A thread-safe dependency injection container that manages service lifecycles:

```python
from chain_of_thought import ServiceRegistry, get_service_registry

# Create a custom registry
registry = ServiceRegistry()

# Register custom service instances
custom_chain = ChainOfThought()
registry.register_service('chain_of_thought', custom_chain)

# Register custom factories
registry.register_factory('hypothesis_generator', lambda: CustomHypothesisGenerator())

# Get service instances
chain = registry.get_service('chain_of_thought')
```

#### Service Registry Features
- **Thread-safe operations** with RLock
- **Lazy initialization** - Services created on first access
- **Factory support** - Register creation functions
- **Instance management** - Clear and recreate services
- **Default factories** - Built-in service creation

### Handler Factory Functions

Instead of global handlers, we now provide factory functions:

```python
from chain_of_thought import create_chain_of_thought_step_handler

# Create handler with custom registry
handler = create_chain_of_thought_step_handler(registry)
result = handler(thought="test", step_number=1, total_steps=1, next_step_needed=False)

# Create handler with default registry (backward compatibility)
default_handler = create_chain_of_thought_step_handler()
```

## Usage Patterns

### 1. Backward Compatibility (Existing Code)

Existing code continues to work unchanged:

```python
from chain_of_thought import HANDLERS

# This still works exactly as before
result = HANDLERS['chain_of_thought_step'](
    thought="analyze problem",
    step_number=1,
    total_steps=3,
    next_step_needed=True
)
```

### 2. Simple Dependency Injection

```python
from chain_of_thought import ServiceRegistry, create_chain_of_thought_step_handler

# Create isolated service registry
registry = ServiceRegistry()

# Create handlers with DI
step_handler = create_chain_of_thought_step_handler(registry)
summary_handler = create_get_chain_summary_handler(registry)
```

### 3. ThreadSafe Multi-Tenant Usage

```python
from chain_of_thought import ThreadAwareChainOfThought

# Each conversation gets isolated services
conversation1 = ThreadAwareChainOfThought("conv-123")
conversation2 = ThreadAwareChainOfThought("conv-456")

# Get handlers bound to specific conversation
handlers1 = conversation1.get_handlers()
handlers2 = conversation2.get_handlers()

# These operate on completely separate chains
result1 = handlers1['chain_of_thought_step'](...)
result2 = handlers2['chain_of_thought_step'](...)
```

### 4. Custom Service Implementation

```python
from chain_of_thought import ServiceRegistry, ChainOfThought

class CustomChainOfThought(ChainOfThought):
    def add_step(self, **kwargs):
        # Custom logic before/after adding step
        result = super().add_step(**kwargs)
        # Add custom behavior
        return result

# Register custom implementation
registry = ServiceRegistry()
registry.register_factory('chain_of_thought', lambda: CustomChainOfThought())

# Create handlers that use custom implementation
handler = create_chain_of_thought_step_handler(registry)
```

### 5. Test Isolation

```python
def test_my_logic():
    # Create isolated registry for test
    test_registry = ServiceRegistry()

    # Mock services for testing
    mock_chain = Mock()
    mock_chain.add_step.return_value = {"status": "success"}
    test_registry.register_service('chain_of_thought', mock_chain)

    # Test with mocked dependencies
    handler = create_chain_of_thought_step_handler(test_registry)
    result = handler(thought="test", step_number=1, total_steps=1, next_step_needed=False)

    # Verify mock was called
    mock_chain.add_step.assert_called_once()
```

## Service Registry API

### Core Methods
- `register_service(name, service)` - Register a service instance
- `register_factory(name, factory)` - Register a factory function
- `get_service(name)` - Get a service instance (creates if needed)
- `has_service(name)` - Check if service is registered
- `clear_service(name)` - Clear a specific service instance
- `clear_all_services()` - Clear all service instances
- `initialize_default_services()` - Register default service factories

### Default Services
The registry automatically provides these services:
- `'chain_of_thought'` - ChainOfThought instance
- `'hypothesis_generator'` - HypothesisGenerator instance
- `'assumption_mapper'` - AssumptionMapper instance
- `'confidence_calibrator'` - ConfidenceCalibrator instance

## Thread Safety

### ThreadAwareChainOfThought Enhancements
The ThreadAwareChainOfThought class now supports DI:

```python
# Use custom service registry
custom_registry = ServiceRegistry()
thread_aware = ThreadAwareChainOfThought("conversation-1", registry=custom_registry)

# Each instance gets its own ChainOfThought but shares other services
handlers = thread_aware.get_handlers()
```

### Service Registry Thread Safety
- All operations protected by `threading.RLock()`
- Lazy initialization is thread-safe
- Service instances can be safely shared across threads

## Migration Guide

### For Library Users
No changes required - all existing code continues to work.

### For Advanced Users
Consider migrating to DI for better testability:

**Before:**
```python
from chain_of_thought import HANDLERS
result = HANDLERS['chain_of_thought_step'](...)
```

**After:**
```python
from chain_of_thought import ServiceRegistry, create_chain_of_thought_step_handler

registry = ServiceRegistry()
handler = create_chain_of_thought_step_handler(registry)
result = handler(...)
```

### For Testing
Use dependency injection for better test isolation:

```python
# Test with mock services
test_registry = ServiceRegistry()
test_registry.register_service('chain_of_thought', MockChainOfThought())

handler = create_chain_of_thought_step_handler(test_registry)
```

## Benefits

1. **Testability** - Easy to mock services for unit testing
2. **Flexibility** - Can swap implementations at runtime
3. **Thread Safety** - Each conversation can have isolated services
4. **Configuration** - Different environments can use different service configurations
5. **Maintainability** - Clear separation of concerns and dependencies
6. **Performance** - Lazy service creation reduces startup overhead

## Backward Compatibility

The implementation maintains 100% backward compatibility:
- All existing global handlers still work
- `HANDLERS` dictionary unchanged
- All existing imports continue to work
- No breaking changes to public APIs

## Implementation Details

### Global Registry
A default global registry (`_default_registry`) provides backward compatibility:
- Global singletons now use this registry
- Default handlers delegate to DI factory functions
- Thread-safe lazy initialization

### Handler Factory Pattern
Instead of global handlers using global singletons:
- Factory functions create handlers with specific registries
- Default behavior uses global registry
- Custom behavior uses custom registries

### Service Lifecycle
- Services created lazily on first access
- Singletons per registry (not global)
- Can be cleared and recreated
- Thread-safe creation and access

This implementation provides a clean migration path to dependency injection while preserving all existing functionality.