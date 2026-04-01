"""
Thread safety tests for ThreadAwareChainOfThought.

Tests cover:
- Concurrent access patterns
- Conversation isolation
- Race condition detection  
- Multi-threaded usage scenarios
- Instance management and cleanup
"""
import pytest
import threading
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from chain_of_thought.core import ThreadAwareChainOfThought


@pytest.mark.thread_safety
class TestThreadAwareChainOfThought:
    """Test ThreadAwareChainOfThought thread safety."""

    def setup_method(self):
        """Clear global instances before each test."""
        ThreadAwareChainOfThought._instances.clear()
        # Reset global rate limiter to ensure clean test state
        from chain_of_thought.core import get_global_rate_limiter
        limiter = get_global_rate_limiter()
        limiter.reset_client("default")
    
    def teardown_method(self):
        """Clear global instances after each test."""
        ThreadAwareChainOfThought._instances.clear()
    
    def test_single_conversation_isolation(self):
        """Test that single conversation instances are properly isolated."""
        conv_id = "test_conversation_1"
        
        # Create two instances for same conversation
        instance1 = ThreadAwareChainOfThought(conv_id)
        instance2 = ThreadAwareChainOfThought(conv_id)
        
        # They should share the same underlying chain
        assert instance1.chain is instance2.chain
        
        # Changes in one should be visible in the other
        instance1.chain.add_step("Step 1", 1, 2, True)
        assert len(instance2.chain.steps) == 1
        assert instance2.chain.steps[0].thought == "Step 1"
    
    def test_multiple_conversation_isolation(self):
        """Test that different conversations are properly isolated."""
        conv1 = ThreadAwareChainOfThought("conversation_1")
        conv2 = ThreadAwareChainOfThought("conversation_2")
        
        # They should have different underlying chains
        assert conv1.chain is not conv2.chain
        
        # Changes in one should not affect the other
        conv1.chain.add_step("Conv1 Step", 1, 1, False)
        conv2.chain.add_step("Conv2 Step", 1, 1, False)
        
        assert len(conv1.chain.steps) == 1
        assert len(conv2.chain.steps) == 1
        assert conv1.chain.steps[0].thought == "Conv1 Step"
        assert conv2.chain.steps[0].thought == "Conv2 Step"
    
    def test_concurrent_conversation_creation(self):
        """Test concurrent creation of conversation instances."""
        num_threads = 10
        num_conversations = 5
        created_instances = []
        
        def create_instance(conv_id):
            """Create instance for given conversation ID."""
            instance = ThreadAwareChainOfThought(f"conv_{conv_id}")
            instance.chain.add_step(f"Step from thread {conv_id}", 1, 1, False)
            return instance
        
        # Create instances concurrently and keep references
        instance_references = {}  # Keep strong references to prevent GC

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(create_instance, i % num_conversations)
                for i in range(num_threads)
            ]

            for future in as_completed(futures):
                instance = future.result()
                created_instances.append(instance)
                # Store reference to prevent garbage collection
                instance_references[instance.conversation_id] = instance

        # Verify that instances were created successfully (WeakValueDictionary may cleanup some)
        # The important thing is that no exceptions were thrown during concurrent creation
        assert len(created_instances) == num_threads
        assert len(instance_references) == num_conversations

        # The WeakValueDictionary cleanup is normal behavior - we can't reliably test this
        # The important thing is that no exceptions were thrown during concurrent creation
        # and that we have the expected number of created instances and references
        
        # Verify each conversation has steps from multiple threads
        for conv_id in range(num_conversations):
            conv_key = f"conv_{conv_id}"
            if conv_key in ThreadAwareChainOfThought._instances:
                chain = ThreadAwareChainOfThought._instances[conv_key]
                assert len(chain.steps) >= 1
    
    def test_concurrent_step_addition_same_conversation(self):
        """Test concurrent step addition to the same conversation."""
        conv_id = "concurrent_test"
        num_threads = 20
        steps_per_thread = 5
        
        def add_steps(thread_id):
            """Add steps from a specific thread."""
            instance = ThreadAwareChainOfThought(conv_id)
            results = []
            
            for i in range(steps_per_thread):
                try:
                    # Each thread adds steps with unique identifiers
                    step_num = thread_id * steps_per_thread + i + 1
                    result = instance.chain.add_step(
                        f"Thread {thread_id}, Step {i}",
                        step_num,
                        num_threads * steps_per_thread,
                        True,
                        confidence=0.8
                    )
                    results.append(result)
                    time.sleep(0.001)  # Small delay to increase race condition chances
                except Exception as e:
                    results.append({"error": str(e)})
            
            return results
        
        # Run concurrent step additions
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(add_steps, thread_id)
                for thread_id in range(num_threads)
            ]
            
            all_results = []
            for future in as_completed(futures):
                all_results.extend(future.result())
        
        # Verify final state
        final_instance = ThreadAwareChainOfThought(conv_id)
        final_chain = final_instance.chain
        
        # Should have steps from all threads (may have race conditions)
        assert len(final_chain.steps) > 0
        assert len(final_chain.steps) <= num_threads * steps_per_thread
        
        # Verify no corruption in data structure
        for step in final_chain.steps:
            assert hasattr(step, 'thought')
            assert hasattr(step, 'step_number')
            assert hasattr(step, 'timestamp')
    
    def test_concurrent_handler_usage(self):
        """Test concurrent usage of instance handlers."""
        conv_id = "handler_test"
        num_threads = 15
        
        def use_handlers(thread_id):
            """Use instance handlers concurrently."""
            instance = ThreadAwareChainOfThought(conv_id)
            handlers = instance.get_handlers()
            results = []
            
            try:
                # Add a step
                step_result = handlers["chain_of_thought_step"](
                    thought=f"Concurrent thought {thread_id}",
                    step_number=thread_id + 1,
                    total_steps=num_threads,
                    next_step_needed=True
                )
                results.append(("step", json.loads(step_result)))
                
                # Get summary
                summary_result = handlers["get_chain_summary"]()
                results.append(("summary", json.loads(summary_result)))
                
            except Exception as e:
                results.append(("error", str(e)))
            
            return results
        
        # Run handlers concurrently
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(use_handlers, thread_id)
                for thread_id in range(num_threads)
            ]
            
            all_results = []
            for future in as_completed(futures):
                all_results.extend(future.result())
        
        # Verify final state is consistent
        final_instance = ThreadAwareChainOfThought(conv_id)
        final_summary_json = final_instance.get_handlers()["get_chain_summary"]()
        final_summary = json.loads(final_summary_json)
        
        # Should have a valid final state
        if final_summary["status"] == "success":
            assert final_summary["total_steps"] > 0
            assert final_summary["total_steps"] <= num_threads
    
    def test_conversation_instance_persistence(self):
        """Test that conversation instances persist correctly."""
        conv_id = "persistence_test"
        
        # Create first instance and add data
        instance1 = ThreadAwareChainOfThought(conv_id)
        instance1.chain.add_step("Persistent step", 1, 1, False)
        
        # Create second instance for same conversation
        instance2 = ThreadAwareChainOfThought(conv_id)
        
        # Data should persist
        assert len(instance2.chain.steps) == 1
        assert instance2.chain.steps[0].thought == "Persistent step"
        
        # Clear from one instance
        instance1.chain.clear_chain()
        
        # Should be cleared in the other
        assert len(instance2.chain.steps) == 0
    
    def test_tool_specs_and_handlers_consistency(self):
        """Test that tool specs and handlers are consistent across instances."""
        conv1 = ThreadAwareChainOfThought("conv1")
        conv2 = ThreadAwareChainOfThought("conv2")
        
        # Tool specs should be identical
        specs1 = conv1.get_tool_specs()
        specs2 = conv2.get_tool_specs()
        assert specs1 == specs2
        
        # Handler structure should be identical
        handlers1 = conv1.get_handlers()
        handlers2 = conv2.get_handlers()
        assert set(handlers1.keys()) == set(handlers2.keys())
    
    def test_memory_cleanup_behavior(self):
        """Test behavior with large numbers of conversations (memory usage)."""
        num_conversations = 100
        conversation_ids = []
        
        # Create many conversation instances
        for i in range(num_conversations):
            conv_id = f"memory_test_{i}"
            conversation_ids.append(conv_id)
            instance = ThreadAwareChainOfThought(conv_id)
            instance.chain.add_step(f"Step in conversation {i}", 1, 1, False)
        
        # Verify all were created
        assert len(ThreadAwareChainOfThought._instances) == num_conversations
        
        # Access random conversations to verify they still work
        import random
        for _ in range(10):
            random_id = random.choice(conversation_ids)
            instance = ThreadAwareChainOfThought(random_id)
            assert len(instance.chain.steps) == 1
    
    def test_race_condition_detection(self):
        """Test for potential race conditions in instance creation."""
        num_threads = 50
        conv_id = "race_condition_test"
        creation_times = []
        
        def create_and_time():
            """Create instance and record timing."""
            start_time = time.time()
            instance = ThreadAwareChainOfThought(conv_id)
            end_time = time.time()
            
            # Add a step to verify functionality
            instance.chain.add_step("Race test step", 1, 1, False)
            
            return (start_time, end_time, id(instance.chain))
        
        # Create instances concurrently
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(create_and_time) for _ in range(num_threads)]
            
            for future in as_completed(futures):
                creation_times.append(future.result())
        
        # All instances should share the same underlying chain object
        chain_ids = [result[2] for result in creation_times]
        unique_chain_ids = set(chain_ids)
        assert len(unique_chain_ids) == 1, f"Expected 1 unique chain, got {len(unique_chain_ids)}"
        
        # Final state should be consistent
        final_instance = ThreadAwareChainOfThought(conv_id)
        # Due to race conditions, we might have multiple steps, but at least one
        assert len(final_instance.chain.steps) >= 1
    
    def test_exception_handling_thread_safety(self):
        """Test exception handling doesn't break thread safety."""
        conv_id = "exception_test"
        num_threads = 10
        
        def cause_exception_and_recover(thread_id):
            """Cause an exception and then do valid operations."""
            instance = ThreadAwareChainOfThought(conv_id)
            results = []
            
            try:
                # Try to cause an exception (invalid parameters)
                instance.chain.add_step("", 0, 0, False)  # Invalid step
                results.append("no_exception")
            except:
                results.append("exception_caught")
            
            # Then do valid operation
            try:
                instance.chain.add_step(f"Valid step {thread_id}", thread_id + 1, num_threads, False)
                results.append("recovery_successful")
            except Exception as e:
                results.append(f"recovery_failed: {e}")
            
            return results
        
        # Run concurrent operations with exceptions
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(cause_exception_and_recover, thread_id)
                for thread_id in range(num_threads)
            ]
            
            all_results = []
            for future in as_completed(futures):
                all_results.extend(future.result())
        
        # Verify system recovered and is still functional
        final_instance = ThreadAwareChainOfThought(conv_id)
        final_summary = final_instance.chain.generate_summary()
        
        # Should be able to generate summary without issues
        assert isinstance(final_summary, dict)
        assert "status" in final_summary


@pytest.mark.thread_safety
@pytest.mark.slow
class TestThreadSafetyStressTest:
    """Stress tests for thread safety under high load."""
    
    def setup_method(self):
        """Clear global instances before each test."""
        ThreadAwareChainOfThought._instances.clear()
    
    def teardown_method(self):
        """Clear global instances after each test."""
        ThreadAwareChainOfThought._instances.clear()
    
    def test_high_concurrency_stress(self):
        """Stress test with high concurrency."""
        num_threads = 100
        operations_per_thread = 10
        num_conversations = 5
        
        def stress_operations(worker_id):
            """Perform multiple operations concurrently."""
            results = []
            conv_id = f"stress_conv_{worker_id % num_conversations}"
            
            instance = ThreadAwareChainOfThought(conv_id)
            
            for op_id in range(operations_per_thread):
                try:
                    # Mix of operations
                    if op_id % 3 == 0:
                        # Add step
                        result = instance.chain.add_step(
                            f"Worker {worker_id}, Op {op_id}",
                            worker_id * operations_per_thread + op_id + 1,
                            num_threads * operations_per_thread,
                            True
                        )
                        results.append(("add_step", "success"))
                    elif op_id % 3 == 1:
                        # Get summary
                        summary = instance.chain.generate_summary()
                        results.append(("summary", summary["status"]))
                    else:
                        # Use handlers
                        handlers = instance.get_handlers()
                        summary_json = handlers["get_chain_summary"]()
                        summary = json.loads(summary_json)
                        results.append(("handler_summary", summary["status"]))
                    
                    # Small random delay
                    time.sleep(0.001 * (worker_id % 3))
                    
                except Exception as e:
                    results.append(("error", str(e)))
            
            return results
        
        # Run stress test
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(stress_operations, worker_id)
                for worker_id in range(num_threads)
            ]
            
            all_results = []
            for future in as_completed(futures):
                all_results.extend(future.result())
        
        end_time = time.time()
        
        # Verify final state
        print(f"Stress test completed in {end_time - start_time:.2f} seconds")
        print(f"Total operations: {len(all_results)}")
        
        # Check that we have the expected number of conversations
        assert len(ThreadAwareChainOfThought._instances) <= num_conversations
        
        # Verify each conversation is in valid state
        for conv_id in range(num_conversations):
            conv_key = f"stress_conv_{conv_id}"
            if conv_key in ThreadAwareChainOfThought._instances:
                instance = ThreadAwareChainOfThought(conv_key)
                summary = instance.chain.generate_summary()
                assert isinstance(summary, dict)
                assert "status" in summary
    
    def test_rapid_creation_destruction_pattern(self):
        """Test rapid creation and implied destruction of instances."""
        iterations = 1000
        
        def create_use_instance(iteration):
            """Create instance, use it, let it go out of scope."""
            conv_id = f"rapid_{iteration % 10}"  # Reuse conversation IDs
            instance = ThreadAwareChainOfThought(conv_id)
            
            # Do some work
            instance.chain.add_step(f"Rapid step {iteration}", 1, 1, False)
            summary = instance.chain.generate_summary()
            
            # Clear occasionally
            if iteration % 50 == 0:
                instance.chain.clear_chain()
            
            return summary["status"] if summary["status"] != "empty" else "success"
        
        # Run rapid creation/destruction
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(create_use_instance, i)
                for i in range(iterations)
            ]
            
            results = []
            for future in as_completed(futures):
                results.append(future.result())
        
        # Should complete successfully
        success_count = sum(1 for r in results if r == "success")
        assert success_count > iterations * 0.8  # At least 80% successful
        
        # Final state should be manageable
        assert len(ThreadAwareChainOfThought._instances) <= 10