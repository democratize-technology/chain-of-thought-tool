"""
Test for service registry error handling vulnerability.

This test demonstrates the missing error handling in ServiceRegistry.get_service()
where factory function exceptions are not caught, potentially crashing the application.

Fails BEFORE fix, passes AFTER fix.
"""
import pytest
from chain_of_thought.core import ServiceRegistry


@pytest.mark.security
@pytest.mark.service_registry
class TestServiceRegistryErrorHandling:
    """Test service registry error handling vulnerabilities."""

    def setup_method(self):
        """Set up test service registry."""
        self.registry = ServiceRegistry()

    def test_factory_function_exception_not_caught_vulnerability(self):
        """
        VULNERABILITY: Factory function exceptions are not caught

        This demonstrates how a faulty factory function can crash the entire application.
        """
        # Register a factory that will raise an exception
        def failing_factory():
            raise RuntimeError("Factory initialization failed")

        self.registry.register_factory("failing_service", failing_factory)

        # This should NOT crash the application
        with pytest.raises(Exception) as exc_info:
            self.registry.get_service("failing_service")

        # BEFORE FIX: Raw RuntimeError propagates
        # AFTER FIX: Should be wrapped in ServiceCreationError
        assert "ServiceCreationError" in str(type(exc_info.value))
        assert "failing_service" in str(exc_info.value)
        assert "Factory initialization failed" in str(exc_info.value)

    def test_factory_function_import_error_vulnerability(self):
        """
        VULNERABILITY: Import errors in factory functions are not handled
        """
        def import_error_factory():
            raise ImportError("Cannot import required module")

        self.registry.register_factory("import_error_service", import_error_factory)

        with pytest.raises(Exception) as exc_info:
            self.registry.get_service("import_error_service")

        # Should be properly handled and wrapped
        assert "ServiceCreationError" in str(type(exc_info.value))
        assert "Cannot import required module" in str(exc_info.value)

    def test_factory_function_memory_error_vulnerability(self):
        """
        VULNERABILITY: Memory errors in factory functions are not handled
        """
        def memory_error_factory():
            raise MemoryError("Out of memory during initialization")

        self.registry.register_factory("memory_error_service", memory_error_factory)

        with pytest.raises(Exception) as exc_info:
            self.registry.get_service("memory_error_service")

        # Should be properly handled and wrapped
        assert "ServiceCreationError" in str(type(exc_info.value))
        assert "Out of memory during initialization" in str(exc_info.value)

    def test_factory_function_attribute_error_vulnerability(self):
        """
        VULNERABILITY: Attribute errors in factory functions are not handled
        """
        def attribute_error_factory():
            # Simulate missing dependency
            return NonExistentClass()  # This will raise AttributeError

        self.registry.register_factory("attribute_error_service", attribute_error_factory)

        with pytest.raises(Exception) as exc_info:
            self.registry.get_service("attribute_error_service")

        # Should be properly handled and wrapped
        assert "ServiceCreationError" in str(type(exc_info.value))
        assert "NonExistentClass" in str(exc_info.value)

    def test_factory_function_returns_none_vulnerability(self):
        """
        VULNERABILITY: Factory functions returning None are not validated
        """
        def none_returning_factory():
            return None  # Bad factory that returns None

        self.registry.register_factory("none_service", none_returning_factory)

        # Should detect and handle None return values
        with pytest.raises(Exception) as exc_info:
            self.registry.get_service("none_service")

        assert "ServiceCreationError" in str(type(exc_info.value))
        assert "returned None" in str(exc_info.value)

    def test_successful_factory_after_failure(self):
        """
        Test that service registry recovers properly after factory failures.
        """
        # First register a failing factory
        def failing_factory():
            raise ValueError("Initial failure")

        self.registry.register_factory("recoverable_service", failing_factory)

        # First call should fail gracefully
        with pytest.raises(Exception):
            self.registry.get_service("recoverable_service")

        # Register a working factory
        def working_factory():
            return "Working service instance"

        self.registry.register_factory("recoverable_service", working_factory)

        # Should now work correctly
        result = self.registry.get_service("recoverable_service")
        assert result == "Working service instance"

    def test_concurrent_factory_failures(self):
        """
        Test error handling under concurrent access to factory failures.
        """
        import threading
        import time

        # Create a factory that fails initially but works after delay
        call_count = [0]

        def intermittent_factory():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise RuntimeError(f"Factory failure #{call_count[0]}")
            return "Success after retries"

        self.registry.register_factory("intermittent_service", intermittent_factory)

        results = []
        exceptions = []

        def worker():
            try:
                result = self.registry.get_service("intermittent_service")
                results.append(result)
            except Exception as e:
                exceptions.append(e)

        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should have some failures and some successes
        assert len(exceptions) > 0, "Should have handled some factory failures"
        assert len(results) > 0, "Should have some successful creations"
        assert all(r == "Success after retries" for r in results), "All results should be successful"

    def test_error_logging_and_diagnostics(self):
        """
        Test that service creation errors are properly logged for debugging.
        """
        def diagnostic_error_factory():
            raise ValueError("Diagnostic error for testing")

        self.registry.register_factory("diagnostic_service", diagnostic_error_factory)

        with pytest.raises(Exception) as exc_info:
            self.registry.get_service("diagnostic_service")

        # Error message should contain useful diagnostic information
        error_message = str(exc_info.value)
        assert "diagnostic_service" in error_message
        assert "Diagnostic error for testing" in error_message
        assert "Failed to create service" in error_message


if __name__ == "__main__":
    # Run standalone test
    test_instance = TestServiceRegistryErrorHandling()
    test_instance.setup_method()

    try:
        test_instance.test_factory_function_exception_not_caught_vulnerability()
        print("✅ Service registry error handling test PASSED")
    except AssertionError as e:
        print(f"❌ Service registry error handling test FAILED: {e}")
    except Exception as e:
        print(f"❌ Test error: {e}")