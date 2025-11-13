#!/usr/bin/env python3
"""
Example demonstrating the new dependency injection capabilities.

This file shows how to use the ServiceRegistry for better testability,
configuration, and thread safety while maintaining backward compatibility.
"""

from chain_of_thought import (
    # Legacy API (still works)
    HANDLERS,

    # New DI API
    ServiceRegistry,
    create_chain_of_thought_step_handler,
    create_get_chain_summary_handler,
    create_clear_chain_handler,
    create_generate_hypotheses_handler,
    create_map_assumptions_handler,
    create_calibrate_confidence_handler,

    # Core classes
    ChainOfThought,
    ThreadAwareChainOfThought,
)


def example_backward_compatibility():
    print("=== Backward Compatibility Example ===")

    # This is exactly how it worked before DI implementation
    result = HANDLERS['chain_of_thought_step'](
        thought="Analyze the user's requirements for the new feature",
        step_number=1,
        total_steps=3,
        next_step_needed=True,
        reasoning_stage="Problem Definition",
        confidence=0.8
    )

    print(f"✅ Legacy API still works: {result}")
    return result


def example_simple_dependency_injection():
    print("\n=== Simple Dependency Injection Example ===")

    # Create an isolated service registry and initialize it
    registry = ServiceRegistry()
    registry.initialize_default_services()

    # Create handlers using DI
    step_handler = create_chain_of_thought_step_handler(registry)
    summary_handler = create_get_chain_summary_handler(registry)

    # Use handlers
    result1 = step_handler(
        thought="Design the database schema for user management",
        step_number=1,
        total_steps=2,
        next_step_needed=True,
        reasoning_stage="Analysis",
        evidence=["User requirements doc", "Existing database patterns"]
    )

    result2 = step_handler(
        thought="Implement the user authentication system",
        step_number=2,
        total_steps=2,
        next_step_needed=False,
        reasoning_stage="Implementation",
        dependencies=[1],
        evidence=["Security best practices", "JWT token approach"]
    )

    # Get summary
    summary = summary_handler()

    print(f"✅ DI step 1: {result1}")
    print(f"✅ DI step 2: {result2}")
    print(f"✅ DI summary: {summary}")

    return result1, result2, summary


def example_thread_safe_multi_tenant():
    print("\n=== Thread-Safe Multi-Tenant Example ===")

    # Simulate multiple concurrent conversations
    conversation1 = ThreadAwareChainOfThought("user-123-session-1")
    conversation2 = ThreadAwareChainOfThought("user-456-session-2")

    # Each conversation gets its own handlers
    handlers1 = conversation1.get_handlers()
    handlers2 = conversation2.get_handlers()

    # Simulate different reasoning processes
    conv1_step1 = handlers1['chain_of_thought_step'](
        thought="Analyze quarterly sales data trends",
        step_number=1,
        total_steps=2,
        next_step_needed=True,
        reasoning_stage="Analysis"
    )

    conv2_step1 = handlers2['chain_of_thought_step'](
        thought="Plan marketing strategy for Q1",
        step_number=1,
        total_steps=3,
        next_step_needed=True,
        reasoning_stage="Planning"
    )

    conv1_step2 = handlers1['chain_of_thought_step'](
        thought="Identify top performing products",
        step_number=2,
        total_steps=2,
        next_step_needed=False,
        reasoning_stage="Synthesis",
        dependencies=[1]
    )

    # Get summaries - they should be different
    summary1_json = handlers1['get_chain_summary']()
    summary2_json = handlers2['get_chain_summary']()

    # Parse JSON results since handlers return JSON strings
    import json
    summary1 = json.loads(summary1_json)
    summary2 = json.loads(summary2_json)

    print(f"✅ Conversation 1 step 1: {conv1_step1}")
    print(f"✅ Conversation 1 step 2: {conv1_step2}")
    print(f"✅ Conversation 1 summary steps: {summary1_json}")
    print(f"✅ Conversation 2 step 1: {conv2_step1}")
    print(f"✅ Conversation 2 summary steps: {summary2_json}")

    # Verify isolation
    steps1 = len(summary1.get('chain', []))
    steps2 = len(summary2.get('chain', []))
    print(f"✅ Conversation isolation: Conv1 has {steps1} steps, Conv2 has {steps2} steps")


class CustomChainOfThought(ChainOfThought):

    def add_step(self, **kwargs):
        print(f"🔧 Custom handler processing step {kwargs.get('step_number')}")
        result = super().add_step(**kwargs)
        # Add custom behavior here
        return result


def example_custom_service_implementation():
    print("\n=== Custom Service Implementation Example ===")

    # Create registry with custom service and initialize defaults
    registry = ServiceRegistry()
    registry.initialize_default_services()
    registry.register_factory('chain_of_thought', lambda: CustomChainOfThought())

    # Create handler that uses custom implementation
    handler = create_chain_of_thought_step_handler(registry)

    result = handler(
        thought="This will use the custom ChainOfThought implementation",
        step_number=1,
        total_steps=1,
        next_step_needed=False,
        reasoning_stage="Implementation"
    )

    print(f"✅ Custom service result: {result}")


def example_service_lifecycle_management():
    print("\n=== Service Lifecycle Management Example ===")

    registry = ServiceRegistry()
    registry.initialize_default_services()

    # Get service (creates lazily)
    chain1 = registry.get_service('chain_of_thought')
    print(f"✅ Service created: {type(chain1).__name__}")

    # Get same service instance (singleton per registry)
    chain2 = registry.get_service('chain_of_thought')
    print(f"✅ Same instance returned: {chain1 is chain2}")

    # Clear service instance
    registry.clear_service('chain_of_thought')
    print("✅ Service cleared from registry")

    # Get fresh instance
    chain3 = registry.get_service('chain_of_thought')
    print(f"✅ Fresh instance created: {chain1 is not chain3}")


def example_all_handlers_with_di():
    print("\n=== All Handlers with Dependency Injection ===")

    registry = ServiceRegistry()
    registry.initialize_default_services()

    # Create all handlers
    step_handler = create_chain_of_thought_step_handler(registry)
    summary_handler = create_get_chain_summary_handler(registry)
    clear_handler = create_clear_chain_handler(registry)
    hypothesis_handler = create_generate_hypotheses_handler(registry)
    assumption_handler = create_map_assumptions_handler(registry)
    confidence_handler = create_calibrate_confidence_handler(registry)

    # Add a reasoning step
    step_handler(
        thought="User adoption has increased by 25% this quarter",
        step_number=1,
        total_steps=1,
        next_step_needed=False,
        reasoning_stage="Conclusion",
        confidence=0.9
    )

    # Generate hypotheses for observation
    hypotheses = hypothesis_handler(
        observation="Website traffic spike on Tuesday afternoons",
        hypothesis_count=3
    )
    print(f"✅ Hypotheses generated: {hypotheses}")

    # Map assumptions
    assumptions = assumption_handler(
        statement="If we launch the new feature, user engagement will increase by 30%",
        depth="deep"
    )
    print(f"✅ Assumptions mapped: {assumptions}")

    # Calibrate confidence
    confidence = confidence_handler(
        prediction="AI will solve this optimization problem within 6 months",
        initial_confidence=0.8
    )
    print(f"✅ Confidence calibrated: {confidence}")

    # Get summary of reasoning chain
    summary = summary_handler()
    print(f"✅ Chain summary: {summary}")

    # Clear the chain
    cleared = clear_handler()
    print(f"✅ Chain cleared: {cleared}")


def main():
    print("🚀 Chain of Thought - Dependency Injection Examples")
    print("=" * 60)

    try:
        # Run all examples
        example_backward_compatibility()
        example_simple_dependency_injection()
        example_thread_safe_multi_tenant()
        example_custom_service_implementation()
        example_service_lifecycle_management()
        example_all_handlers_with_di()

        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print("\nKey Benefits Demonstrated:")
        print("✅ Backward compatibility - existing code unchanged")
        print("✅ Service isolation - each conversation has separate state")
        print("✅ Custom implementations - easily swap service classes")
        print("✅ Lifecycle management - control service creation and clearing")
        print("✅ Thread safety - safe concurrent usage")
        print("✅ Testability - easy to mock services for testing")

    except Exception as e:
        print(f"\n❌ Error in examples: {e}")
        raise


if __name__ == "__main__":
    main()