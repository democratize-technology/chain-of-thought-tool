"""
AWS Bedrock integration for the Chain of Thought library.

Provides stopReason handling and async tool loop orchestration.
"""
from typing import Dict, Optional, Any, Callable
from abc import ABC, abstractmethod
import asyncio
import json

from .security import RequestValidator, SecurityValidationError, default_validator


class StopReasonHandler(ABC):
    """Abstract base for handling stopReason integration with CoT."""

    @abstractmethod
    async def should_continue_reasoning(self, chain: Any) -> bool:
        """Return True if reasoning should continue, False if end_turn."""
        pass

    @abstractmethod
    async def execute_tool_call(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call and return the result."""
        pass


class BedrockStopReasonHandler(StopReasonHandler):
    """Bedrock-specific stop reason handler that integrates with CoT flow."""

    def __init__(self, handlers: Optional[Dict[str, Callable]] = None, chain: Optional[Any] = None):
        self.chain = chain  # If provided, use this chain instead of global
        if self.chain is not None:
            # Create instance-specific handlers
            self.handlers = handlers or {
                "chain_of_thought_step": self._create_chain_step_handler(),
                "get_chain_summary": self._create_summary_handler(),
                "clear_chain": self._create_clear_handler(),
                "export_chain": self._create_handler_factory("export_chain", takes_kwargs=True),
                "import_chain": self._create_handler_factory("import_chain", takes_kwargs=True),
                "generate_hypotheses": self._create_handler_factory("generate_hypotheses", takes_kwargs=True),
                "map_assumptions": self._create_handler_factory("map_assumptions", takes_kwargs=True),
                "calibrate_confidence": self._create_handler_factory("calibrate_confidence", takes_kwargs=True),
            }
        else:
            # Use global handlers
            from .handlers import (
                chain_of_thought_step_handler,
                get_chain_summary_handler,
                clear_chain_handler,
                export_chain_handler,
                import_chain_handler,
                generate_hypotheses_handler,
                map_assumptions_handler,
                calibrate_confidence_handler,
            )
            self.handlers = handlers or {
                "chain_of_thought_step": chain_of_thought_step_handler,
                "get_chain_summary": get_chain_summary_handler,
                "clear_chain": clear_chain_handler,
                "export_chain": export_chain_handler,
                "import_chain": import_chain_handler,
                "generate_hypotheses": generate_hypotheses_handler,
                "map_assumptions": map_assumptions_handler,
                "calibrate_confidence": calibrate_confidence_handler,
            }

    def _create_handler_factory(self, method_name: str, takes_kwargs: bool = False):
        """
        Create a generic handler factory for any method on this instance's chain.

        Args:
            method_name: Name of the method to call on self.chain
            takes_kwargs: Whether the method accepts keyword arguments

        Returns:
            A handler function bound to this instance's chain
        """
        from .core import _safe_json_dumps

        def handler(**kwargs):
            try:
                method = getattr(self.chain, method_name)
                if takes_kwargs:
                    result = method(**kwargs)
                else:
                    result = method()
                return _safe_json_dumps(result, indent=2)
            except Exception as e:
                return _safe_json_dumps({"status": "error", "message": str(e)}, indent=2)
        return handler

    def _create_chain_step_handler(self):
        """Create a chain step handler bound to this instance's chain."""
        return self._create_handler_factory("add_step", takes_kwargs=True)

    def _create_summary_handler(self):
        """Create a summary handler bound to this instance's chain."""
        return self._create_handler_factory("generate_summary", takes_kwargs=False)

    def _create_clear_handler(self):
        """Create a clear handler bound to this instance's chain."""
        return self._create_handler_factory("clear_chain", takes_kwargs=False)

    async def should_continue_reasoning(self, chain: Any) -> bool:
        """Check if CoT indicates more steps needed."""
        if not chain.steps:
            return True  # No steps yet, continue

        last_step = chain.steps[-1]
        return last_step.next_step_needed

    async def execute_tool_call(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CoT tool call asynchronously."""
        if tool_name not in self.handlers:
            raise ValueError(f"Unknown tool: {tool_name}")

        handler = self.handlers[tool_name]

        # Run handler in executor if it's synchronous
        if asyncio.iscoroutinefunction(handler):
            result = await handler(**tool_args)
        else:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: handler(**tool_args))

        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                result = {"status": "error", "message": "Invalid JSON response"}

        return result


class AsyncChainOfThoughtProcessor:
    """Async wrapper for CoT that integrates with Bedrock tool loops."""

    def __init__(self, conversation_id: str, stop_handler: Optional[StopReasonHandler] = None,
                 request_validator: Optional[RequestValidator] = None,
                 aws_call_timeout: float = 30.0, tool_call_timeout: float = 10.0):
        """
        Initialize AsyncChainOfThoughtProcessor with configurable timeouts.

        Args:
            conversation_id: Unique identifier for the conversation
            stop_handler: Handler for stopReason logic
            request_validator: Security request validator
            aws_call_timeout: Timeout in seconds for AWS API calls
            tool_call_timeout: Timeout in seconds for tool handler calls
        """
        from .core import ChainOfThought

        self.conversation_id = conversation_id
        self.chain = ChainOfThought()
        # Pass the chain instance to the handler so it uses this specific chain
        self.stop_handler = stop_handler or BedrockStopReasonHandler(chain=self.chain)
        self.request_validator = request_validator or default_validator
        self._tool_use_count = 0
        self._max_iterations = 20

        # Timeout configuration
        self.aws_call_timeout = aws_call_timeout
        self.tool_call_timeout = tool_call_timeout

    async def _safe_aws_call(self, bedrock_client, **kwargs) -> Dict[str, Any]:
        """
        Execute AWS Bedrock call with proper timeout handling.

        Args:
            bedrock_client: AWS Bedrock client
            **kwargs: Parameters for converse call

        Returns:
            AWS response

        Raises:
            TimeoutError: If AWS call exceeds timeout
        """
        try:
            loop = asyncio.get_running_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: bedrock_client.converse(**kwargs)
                ),
                timeout=self.aws_call_timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"AWS Bedrock call timed out after {self.aws_call_timeout} seconds")

    async def process_tool_loop(self,
                              bedrock_client,
                              initial_request: Dict[str, Any],
                              max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """Process Bedrock tool loop with CoT integration."""
        from .core import _safe_json_dumps

        # Validate and sanitize the initial request to prevent injection attacks
        try:
            sanitized_request = self.request_validator.validate_and_sanitize_request(initial_request)
        except SecurityValidationError as e:
            raise SecurityValidationError(f"Security validation failed: {str(e)}")

        max_iter = max_iterations or self._max_iterations
        messages = sanitized_request.get("messages", []).copy()

        for iteration in range(max_iter):
            # Use safe AWS call with timeout protection
            response = await self._safe_aws_call(
                bedrock_client,
                **{**sanitized_request, "messages": messages}
            )

            stop_reason = response.get("stopReason")

            if stop_reason == "end_turn":
                # Check if CoT actually wants to continue
                should_continue = await self.stop_handler.should_continue_reasoning(self.chain)
                if not should_continue:
                    return response
                # If CoT wants to continue but Bedrock says end_turn, we're done
                return response

            elif stop_reason == "tool_use":
                message_content = response.get("output", {}).get("message", {}).get("content", [])
                tool_results = []

                for content_item in message_content:
                    if "toolUse" in content_item:
                        tool_use = content_item["toolUse"]
                        tool_name = tool_use["name"]
                        tool_input = tool_use["input"]
                        tool_use_id = tool_use["toolUseId"]

                        try:
                            # Use safe tool call with timeout protection
                            result = await asyncio.wait_for(
                                self.stop_handler.execute_tool_call(tool_name, tool_input),
                                timeout=self.tool_call_timeout
                            )
                            tool_results.append({
                                "toolResult": {
                                    "toolUseId": tool_use_id,
                                    "content": [{"text": _safe_json_dumps(result)}]
                                }
                            })
                        except asyncio.TimeoutError:
                            tool_results.append({
                                "toolResult": {
                                    "toolUseId": tool_use_id,
                                    "content": [{"text": _safe_json_dumps({"status": "error", "message": f"Tool call timed out after {self.tool_call_timeout} seconds"})}],
                                    "status": "error"
                                }
                            })
                        except Exception as e:
                            tool_results.append({
                                "toolResult": {
                                    "toolUseId": tool_use_id,
                                    "content": [{"text": _safe_json_dumps({"status": "error", "message": str(e)})}],
                                    "status": "error"
                                }
                            })

                messages.append(response["output"]["message"])
                if tool_results:
                    messages.append({
                        "role": "user",
                        "content": tool_results
                    })

                self._tool_use_count += len(tool_results)

            else:
                # Unexpected stop reason
                return response

        return {
            "stopReason": "max_tokens",
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Maximum reasoning iterations reached."}]
                }
            }
        }

    async def process_tool_loop_with_timeout(self,
                                           bedrock_client,
                                           initial_request: Dict[str, Any],
                                           max_iterations: Optional[int] = None,
                                           overall_timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Process Bedrock tool loop with overall timeout protection.

        Args:
            bedrock_client: AWS Bedrock client
            initial_request: Initial Bedrock request
            max_iterations: Maximum number of tool loop iterations
            overall_timeout: Overall timeout for the entire process

        Returns:
            Bedrock response or timeout error response
        """
        timeout = overall_timeout or (self.aws_call_timeout * 2)  # Default to 2x AWS timeout

        try:
            return await asyncio.wait_for(
                self.process_tool_loop(bedrock_client, initial_request, max_iterations),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return {
                "stopReason": "timeout",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": f"Request timeout after {timeout} seconds"}]
                    }
                }
            }

    async def get_reasoning_summary(self) -> Dict[str, Any]:
        """Get summary of the reasoning process."""
        return self.chain.generate_summary()

    def clear_reasoning(self) -> Dict[str, Any]:
        """Clear the reasoning chain."""
        self._tool_use_count = 0
        return self.chain.clear_chain()
