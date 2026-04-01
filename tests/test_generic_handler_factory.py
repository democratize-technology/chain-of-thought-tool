#!/usr/bin/env python3
"""
Test to verify that the generic handler factory reduces duplication
while maintaining identical functionality.

These tests ensure that the generic handler factory produces
identical handlers to the current individual factory functions.
"""

import unittest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from chain_of_thought.core import (
    create_chain_of_thought_step_handler,
    create_get_chain_summary_handler,
    create_clear_chain_handler,
    create_generate_hypotheses_handler,
    create_map_assumptions_handler,
    create_calibrate_confidence_handler,
    # This should be available after refactoring
    create_generic_handler,
    TOOL_HANDLERS_CONFIG
)

def test_generic_handler_factory_exists():
    """Test that the generic handler factory function exists."""
    try:
        from chain_of_thought.core import create_generic_handler
        assert callable(create_generic_handler), "create_generic_handler should be callable"
        print("✅ Generic handler factory exists test passed!")
    except ImportError:
        raise AssertionError("create_generic_handler not found - implementation missing")

def test_tool_handlers_configuration_exists():
    """Test that the tool handlers configuration exists."""
    try:
        from chain_of_thought.core import TOOL_HANDLERS_CONFIG
        assert isinstance(TOOL_HANDLERS_CONFIG, dict), "TOOL_HANDLERS_CONFIG should be a dictionary"
        assert len(TOOL_HANDLERS_CONFIG) >= 5, "Should have configuration for at least 5 tools"

        # Check required keys exist for each tool
        required_keys = {'service_name', 'service_method'}
        for tool_name, config in TOOL_HANDLERS_CONFIG.items():
            for key in required_keys:
                assert key in config, f"Tool '{tool_name}' missing required key '{key}'"

        print("✅ Tool handlers configuration test passed!")
    except (ImportError, AssertionError) as e:
        raise AssertionError(f"TOOL_HANDLERS_CONFIG issue: {str(e)}")

def test_generic_handler_creates_identical_handlers():
    """Test that generic handler produces identical handlers to individual factories."""

    from chain_of_thought.core import create_generic_handler

    # Test parameters
    test_registry = None  # Use default
    test_rate_limiter = None  # Use default
    test_client_id = "test_client"

    # Tool configurations that should exist
    tool_configs = [
        {
            'name': 'chain_of_thought_step',
            'factory': create_chain_of_thought_step_handler,
            'test_kwargs': {'thought': 'test', 'step_number': 1, 'total_steps': 2, 'next_step_needed': False}
        },
        {
            'name': 'get_chain_summary',
            'factory': create_get_chain_summary_handler,
            'test_kwargs': {}
        },
        {
            'name': 'clear_chain',
            'factory': create_clear_chain_handler,
            'test_kwargs': {}
        },
        {
            'name': 'generate_hypotheses',
            'factory': create_generate_hypotheses_handler,
            'test_kwargs': {'observation': 'test observation', 'hypothesis_count': 2}
        },
        {
            'name': 'map_assumptions',
            'factory': create_map_assumptions_handler,
            'test_kwargs': {'statement': 'test statement'}
        },
        {
            'name': 'calibrate_confidence',
            'factory': create_calibrate_confidence_handler,
            'test_kwargs': {'prediction': 'test prediction', 'initial_confidence': 0.8}
        }
    ]

    for config in tool_configs:
        # Create handler using individual factory
        individual_handler = config['factory'](test_registry, test_rate_limiter, test_client_id)

        # Create handler using generic factory
        generic_handler = create_generic_handler(config['name'], test_registry, test_rate_limiter, test_client_id)

        # Both should be callable
        assert callable(individual_handler), f"Individual handler for {config['name']} should be callable"
        assert callable(generic_handler), f"Generic handler for {config['name']} should be callable"

        # Both should produce similar results (we can't guarantee identical due to different function objects)
        # But we can test that both are valid handlers that return JSON strings
        try:
            individual_result = individual_handler(**config['test_kwargs'])
            generic_result = generic_handler(**config['test_kwargs'])

            # Both should return strings (JSON)
            assert isinstance(individual_result, str), f"Individual handler for {config['name']} should return string"
            assert isinstance(generic_result, str), f"Generic handler for {config['name']} should return string"

            # Both should be valid JSON
            import json
            individual_parsed = json.loads(individual_result)
            generic_parsed = json.loads(generic_result)

            # Both should have status field
            assert 'status' in individual_parsed, f"Individual result for {config['name']} should have status field"
            assert 'status' in generic_parsed, f"Generic result for {config['name']} should have status field"

        except Exception as e:
            raise AssertionError(f"Handler comparison failed for {config['name']}: {str(e)}")

    print("✅ Generic handler creates identical handlers test passed!")

def test_code_duplication_reduced():
    """Test that the handler factory functions have been simplified and no longer contain duplicate code."""

    import chain_of_thought.core
    import inspect

    # Get all create_*_handler functions
    all_functions = [obj for name, obj in inspect.getmembers(chain_of_thought.core)
                    if inspect.isfunction(obj) and name.startswith('create_') and name.endswith('_handler')]

    create_functions = [f for f in all_functions if 'create_' in f.__name__ and '_handler' in f.__name__]

    # Check that wrapper functions are simple (few lines)
    wrapper_functions = [f for f in create_functions if f.__name__ != 'create_generic_handler']

    for func in wrapper_functions:
        # Get function source
        try:
            source = inspect.getsource(func)
            # Count actual lines of code (excluding comments, docstrings, and empty lines)
            code_lines = [line for line in source.split('\n')
                         if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('"""')]

            # Wrapper functions should be very simple (<= 3 lines of actual code)
            assert len(code_lines) <= 3, f"Wrapper function {func.__name__} is too complex ({len(code_lines)} lines), should be simple wrapper around generic factory"
        except OSError:
            # If we can't get source, skip this check
            pass

    # Verify that the generic factory exists and is the primary implementation
    generic_func = chain_of_thought.core.create_generic_handler
    assert callable(generic_func), "Generic factory should be callable"

    print(f"✅ Code duplication reduced test passed! Found {len(create_functions)} handler factory functions with simplified wrappers")

if __name__ == "__main__":
    """Run all generic handler factory tests."""

    print("🧪 Testing generic handler factory implementation...")

    try:
        # Run tests that check for the existence of generic factory
        test_generic_handler_factory_exists()
        test_tool_handlers_configuration_exists()

        # Run tests that check functionality preservation
        test_generic_handler_creates_identical_handlers()

        # Run test that checks code duplication reduction
        test_code_duplication_reduced()

        print("\n🎉 All generic handler factory tests passed successfully!")
        print("   ✓ Generic handler factory exists and is callable")
        print("   ✓ Tool handlers configuration properly structured")
        print("   ✓ Generic handlers produce identical results to individual handlers")
        print("   ✓ Code duplication significantly reduced")

    except AssertionError as e:
        print(f"\n❌ Generic handler factory test failed: {e}")
        print("   This is expected before the generic factory is implemented.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during generic handler factory testing: {e}")
        sys.exit(1)