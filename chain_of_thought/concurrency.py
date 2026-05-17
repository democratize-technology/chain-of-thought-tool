"""
Concurrency utilities for the Chain of Thought library.

Provides thread-safe multi-conversation management and rate limiting.
"""
from typing import Dict, List, Optional, Any
import threading
import time
import weakref


DEFAULT_MAX_REQUESTS_PER_MINUTE = 60
DEFAULT_MAX_REQUESTS_PER_HOUR = 1000
DEFAULT_MAX_BURST_SIZE = 10


class RateLimiter:
    """
    Thread-safe rate limiting to prevent DoS attacks on handler functions.

    Implements token bucket algorithm with multiple time windows:
    - Burst limit: Immediate consecutive requests
    - Per-minute limit: Requests within 1-minute window
    - Per-hour limit: Requests within 1-hour window

    Each client is tracked separately to ensure isolation.
    """

    def __init__(self, max_requests_per_minute: int = DEFAULT_MAX_REQUESTS_PER_MINUTE, max_requests_per_hour: int = DEFAULT_MAX_REQUESTS_PER_HOUR, max_burst_size: int = DEFAULT_MAX_BURST_SIZE):
        """
        Initialize rate limiter with configurable limits.

        Args:
            max_requests_per_minute: Maximum requests per minute per client
            max_requests_per_hour: Maximum requests per hour per client
            max_burst_size: Maximum consecutive immediate requests per client
        """
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour
        self.max_burst_size = max_burst_size

        self._request_counts: Dict[str, int] = {}
        self._request_timestamps: Dict[str, List[float]] = {}
        self._lock = threading.RLock()

    def _cleanup_old_timestamps(self, client_id: str, current_time: float) -> None:
        """Remove timestamps older than 1 hour from tracking."""
        if client_id not in self._request_timestamps:
            return

        one_hour_ago = current_time - 3600.0
        timestamps = self._request_timestamps[client_id]
        self._request_timestamps[client_id] = [
            ts for ts in timestamps if ts > one_hour_ago
        ]

        if not self._request_timestamps[client_id]:
            del self._request_timestamps[client_id]

    def _get_minute_count(self, client_id: str, current_time: float) -> int:
        """Count requests in the last minute for a client."""
        if client_id not in self._request_timestamps:
            return 0

        one_minute_ago = current_time - 60.0
        return sum(1 for ts in self._request_timestamps[client_id] if ts > one_minute_ago)

    def _get_hour_count(self, client_id: str, current_time: float) -> int:
        """Count requests in the last hour for a client."""
        if client_id not in self._request_timestamps:
            return 0

        one_hour_ago = current_time - 3600.0
        return sum(1 for ts in self._request_timestamps[client_id] if ts > one_hour_ago)

    def check_rate_limit(self, client_id: str = "default") -> bool:
        """
        Check if a request from the client should be allowed.

        Args:
            client_id: Unique identifier for the client (IP address, session ID, etc.)

        Returns:
            True if request should be allowed, False if rate limited
        """
        current_time = time.time()

        with self._lock:
            self._cleanup_old_timestamps(client_id, current_time)

            current_burst = self._request_counts.get(client_id, 0)
            if current_burst >= self.max_burst_size:
                return False

            minute_count = self._get_minute_count(client_id, current_time)
            if minute_count >= self.max_requests_per_minute:
                return False

            hour_count = self._get_hour_count(client_id, current_time)
            if hour_count >= self.max_requests_per_hour:
                return False

            self._request_counts[client_id] = current_burst + 1

            if client_id not in self._request_timestamps:
                self._request_timestamps[client_id] = []
            self._request_timestamps[client_id].append(current_time)

            return True

    def get_retry_after(self, client_id: str = "default") -> Optional[int]:
        """
        Get suggested retry-after seconds for a rate-limited client.

        Args:
            client_id: Unique identifier for the client

        Returns:
            Seconds to wait before retry, or None if not rate limited
        """
        current_time = time.time()

        with self._lock:
            current_burst = self._request_counts.get(client_id, 0)
            if current_burst >= self.max_burst_size:
                return 1

            minute_count = self._get_minute_count(client_id, current_time)
            if minute_count >= self.max_requests_per_minute:
                if client_id in self._request_timestamps and self._request_timestamps[client_id]:
                    oldest_timestamp = min(self._request_timestamps[client_id])
                    retry_after = int(60 - (current_time - oldest_timestamp)) + 1
                    return max(retry_after, 1)

            hour_count = self._get_hour_count(client_id, current_time)
            if hour_count >= self.max_requests_per_hour:
                if client_id in self._request_timestamps and self._request_timestamps[client_id]:
                    oldest_timestamp = min(self._request_timestamps[client_id])
                    retry_after = int(3600 - (current_time - oldest_timestamp)) + 1
                    return max(retry_after, 60)

            return None

    def reset_client(self, client_id: str = "default") -> None:
        """Reset rate limiting tracking for a specific client."""
        with self._lock:
            if client_id in self._request_counts:
                del self._request_counts[client_id]
            if client_id in self._request_timestamps:
                del self._request_timestamps[client_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get current rate limiting statistics."""
        with self._lock:
            return {
                "active_clients": len(self._request_counts),
                "total_tracked_timestamps": sum(len(timestamps) for timestamps in self._request_timestamps.values()),
                "max_requests_per_minute": self.max_requests_per_minute,
                "max_requests_per_hour": self.max_requests_per_hour,
                "max_burst_size": self.max_burst_size
            }


_global_rate_limiter: Optional[RateLimiter] = None
_rate_limiter_lock = threading.Lock()


def get_global_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter instance."""
    global _global_rate_limiter

    if _global_rate_limiter is None:
        with _rate_limiter_lock:
            if _global_rate_limiter is None:
                _global_rate_limiter = RateLimiter()

    return _global_rate_limiter


def set_global_rate_limiter(limiter: RateLimiter) -> None:
    """Set a custom global rate limiter instance."""
    global _global_rate_limiter

    with _rate_limiter_lock:
        _global_rate_limiter = limiter


class ThreadAwareChainOfThought:
    """Thread-safe version for production use with dependency injection support."""

    _instances: weakref.WeakValueDictionary[str, Any] = weakref.WeakValueDictionary()
    _strong_refs: Dict[str, Any] = {}
    _lock = threading.RLock()

    @classmethod
    def for_conversation(cls, conversation_id: str, registry: Optional[Any] = None):
        """Get or create a ChainOfThought instance for a conversation."""
        from .core import ChainOfThought, get_service_registry

        with cls._lock:
            if conversation_id in cls._strong_refs:
                return cls._strong_refs[conversation_id]

            try:
                weak_instance = cls._instances[conversation_id]
                if weak_instance is not None:
                    cls._strong_refs[conversation_id] = weak_instance
                    return weak_instance
            except KeyError:
                pass

            service_registry = registry or get_service_registry()
            new_instance = ChainOfThought()

            cls._instances[conversation_id] = new_instance
            cls._strong_refs[conversation_id] = new_instance
            return new_instance

    @classmethod
    def clear_conversation(cls, conversation_id: str) -> bool:
        """Explicitly clear a conversation from the cache.

        Args:
            conversation_id: The conversation ID to clear

        Returns:
            True if conversation was removed, False if not found
        """
        with cls._lock:
            removed_from_weak = cls._instances.pop(conversation_id, None) is not None
            removed_from_strong = cls._strong_refs.pop(conversation_id, None) is not None
            return removed_from_weak or removed_from_strong

    @classmethod
    def clear_all_conversations(cls) -> int:
        """Clear all conversations and return count cleared.

        Returns:
            Number of conversations that were cleared
        """
        with cls._lock:
            count = max(len(cls._instances), len(cls._strong_refs))
            cls._instances.clear()
            cls._strong_refs.clear()
            return count

    @classmethod
    def get_cached_conversation_count(cls) -> int:
        """Get the current number of cached conversations.

        Returns:
            Number of conversations currently cached
        """
        with cls._lock:
            return max(len(cls._instances), len(cls._strong_refs))

    @classmethod
    def release_conversation(cls, conversation_id: str) -> bool:
        """Release strong reference for a conversation, allowing weak reference cleanup.

        This is the key method for memory management - call this when conversation
        is no longer actively needed but should remain available for weak reference GC.

        Args:
            conversation_id: The conversation ID to release

        Returns:
            True if conversation was released, False if not found
        """
        with cls._lock:
            return cls._strong_refs.pop(conversation_id, None) is not None

    def __init__(self, conversation_id: str, registry: Optional[Any] = None):
        from .core import get_service_registry

        self.conversation_id = conversation_id
        self.registry = registry or get_service_registry()
        self.chain = self.for_conversation(conversation_id, self.registry)

    def get_tool_specs(self):
        """Get tool specs for this instance."""
        from . import TOOL_SPECS
        return TOOL_SPECS

    def get_handlers(self):
        from .core import (
            ServiceRegistry as _ServiceRegistry,
            _safe_json_dumps,
        )
        from .handlers import (
            create_chain_of_thought_step_handler,
            create_get_chain_summary_handler,
            create_clear_chain_handler,
            create_generate_hypotheses_handler,
            create_map_assumptions_handler,
            create_calibrate_confidence_handler,
        )

        instance_registry = _ServiceRegistry()

        for service_name in ['hypothesis_generator', 'assumption_mapper', 'confidence_calibrator']:
            if self.registry.has_service(service_name):
                instance_registry.register_factory(service_name, lambda name=service_name: self.registry.get_service(name))

        instance_registry.register_service('chain_of_thought', self.chain)

        chain = self.chain

        handlers = {
            "chain_of_thought_step": create_chain_of_thought_step_handler(instance_registry),
            "get_chain_summary": create_get_chain_summary_handler(instance_registry),
            "clear_chain": create_clear_chain_handler(instance_registry),
            "generate_hypotheses": create_generate_hypotheses_handler(instance_registry),
            "map_assumptions": create_map_assumptions_handler(instance_registry),
            "calibrate_confidence": create_calibrate_confidence_handler(instance_registry),
            "export_chain": lambda **kwargs: _safe_json_dumps(chain.export_chain(**kwargs), indent=2),
            "import_chain": lambda **kwargs: _safe_json_dumps(chain.import_chain(**kwargs), indent=2)
        }
        return handlers
