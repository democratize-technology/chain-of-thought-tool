"""
Async tests for AsyncChainOfThoughtProcessor and AWS Bedrock integration patterns.

Tests cover:
- Async tool loop processing
- StopReason handling (end_turn vs tool_use)
- Mock AWS Bedrock client interactions
- Error handling in async contexts
- Tool execution and result formatting
- Iteration limits and edge cases
"""
import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from chain_of_thought.core import (
    AsyncChainOfThoughtProcessor,
    BedrockStopReasonHandler,
    StopReasonHandler,
    ChainOfThought
)
from chain_of_thought.security import SecurityConfig, RequestValidator


class MockBedrockClient:
    """Mock Bedrock client for testing."""
    
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        self.call_history = []
    
    def converse(self, **kwargs):
        """Mock converse method."""
        # Store a deep copy to prevent reference issues with mutable parameters
        import copy
        self.call_history.append(copy.deepcopy(kwargs))
        
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            # Default response
            response = {
                "stopReason": "end_turn",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "Default response"}]
                    }
                }
            }
        
        self.call_count += 1
        return response


class MockStopReasonHandler(StopReasonHandler):
    """Mock stop reason handler for testing."""
    
    def __init__(self):
        self.should_continue_calls = []
        self.execute_tool_calls = []
        self.continue_reasoning = True
        self.tool_results = {}
    
    async def should_continue_reasoning(self, chain: ChainOfThought) -> bool:
        """Mock should continue reasoning."""
        self.should_continue_calls.append(len(chain.steps))
        return self.continue_reasoning
    
    async def execute_tool_call(self, tool_name: str, tool_args: dict) -> dict:
        """Mock tool execution."""
        self.execute_tool_calls.append((tool_name, tool_args))
        
        if tool_name in self.tool_results:
            return self.tool_results[tool_name]
        
        # Default successful response
        return {
            "status": "success",
            "result": f"Executed {tool_name} with args {tool_args}"
        }


@pytest.mark.async_test
class TestAsyncChainOfThoughtProcessor:
    """Test AsyncChainOfThoughtProcessor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.conversation_id = "test_async_conversation"

        # Create permissive security config for testing
        test_security_config = SecurityConfig(
            allowed_model_patterns=[r'^test-model$', r'^.*$'],  # Allow test models
            allowed_top_level_params={
                'messages', 'modelId', 'system', 'toolConfig', 'inferenceConfig',
                'guardrailConfig', 'additionalModelRequestFields', 'temperature',
                'maxTokens', 'topP', 'stopSequences'  # Allow test parameters
            }
        )
        test_validator = RequestValidator(test_security_config)

        self.processor = AsyncChainOfThoughtProcessor(
            self.conversation_id,
            request_validator=test_validator
        )
    
    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor.conversation_id == self.conversation_id
        assert isinstance(self.processor.chain, ChainOfThought)
        assert isinstance(self.processor.stop_handler, BedrockStopReasonHandler)
        assert self.processor._tool_use_count == 0
        assert self.processor._max_iterations == 20
    
    def test_initialization_with_custom_handler(self):
        """Test initialization with custom stop handler."""
        custom_handler = MockStopReasonHandler()
        processor = AsyncChainOfThoughtProcessor("test", custom_handler)
        
        assert processor.stop_handler is custom_handler
    
    @pytest.mark.asyncio
    async def test_simple_end_turn_response(self):
        """Test simple end_turn response handling."""
        mock_client = MockBedrockClient([
            {
                "stopReason": "end_turn",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "Simple response"}]
                    }
                }
            }
        ])
        
        initial_request = {
            "messages": [
                {"role": "user", "content": [{"text": "Test message"}]}
            ],
            "modelId": "test-model"
        }
        
        result = await self.processor.process_tool_loop(mock_client, initial_request)
        
        assert result["stopReason"] == "end_turn"
        assert mock_client.call_count == 1
        assert self.processor._tool_use_count == 0
    
    @pytest.mark.asyncio
    async def test_tool_use_single_iteration(self):
        """Test single tool use iteration."""
        mock_client = MockBedrockClient([
            {
                "stopReason": "tool_use",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"text": "I need to use a tool"},
                            {
                                "toolUse": {
                                    "toolUseId": "test-tool-id-1",
                                    "name": "chain_of_thought_step",
                                    "input": {
                                        "thought": "First reasoning step",
                                        "step_number": 1,
                                        "total_steps": 2,
                                        "next_step_needed": True
                                    }
                                }
                            }
                        ]
                    }
                }
            },
            {
                "stopReason": "end_turn",
                "output": {
                    "message": {
                        "role": "assistant", 
                        "content": [{"text": "Reasoning complete"}]
                    }
                }
            }
        ])
        
        initial_request = {
            "messages": [
                {"role": "user", "content": [{"text": "Start reasoning"}]}
            ],
            "modelId": "test-model"
        }
        
        result = await self.processor.process_tool_loop(mock_client, initial_request)
        
        assert result["stopReason"] == "end_turn"
        assert mock_client.call_count == 2
        assert self.processor._tool_use_count == 1
        
        # Verify reasoning chain has the step
        assert len(self.processor.chain.steps) == 1
        assert self.processor.chain.steps[0].thought == "First reasoning step"
    
    @pytest.mark.asyncio
    async def test_multiple_tool_uses_in_single_response(self):
        """Test multiple tool uses in a single response."""
        mock_client = MockBedrockClient([
            {
                "stopReason": "tool_use",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "tool-1",
                                    "name": "chain_of_thought_step",
                                    "input": {
                                        "thought": "Step 1",
                                        "step_number": 1,
                                        "total_steps": 3,
                                        "next_step_needed": True
                                    }
                                }
                            },
                            {
                                "toolUse": {
                                    "toolUseId": "tool-2", 
                                    "name": "chain_of_thought_step",
                                    "input": {
                                        "thought": "Step 2",
                                        "step_number": 2,
                                        "total_steps": 3,
                                        "next_step_needed": True
                                    }
                                }
                            }
                        ]
                    }
                }
            },
            {
                "stopReason": "end_turn",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "Multiple steps added"}]
                    }
                }
            }
        ])
        
        initial_request = {
            "messages": [{"role": "user", "content": [{"text": "Multi-step reasoning"}]}],
            "modelId": "test-model"
        }
        
        result = await self.processor.process_tool_loop(mock_client, initial_request)
        
        assert result["stopReason"] == "end_turn"
        assert self.processor._tool_use_count == 2
        assert len(self.processor.chain.steps) == 2
    
    @pytest.mark.asyncio
    async def test_tool_execution_error_handling(self):
        """Test error handling during tool execution."""
        # Create a handler that will throw an error
        error_handler = MockStopReasonHandler()
        error_handler.tool_results["chain_of_thought_step"] = None  # Will cause KeyError

        # Create permissive security config for testing
        test_security_config = SecurityConfig(
            allowed_model_patterns=[r'^test-model$', r'^.*$'],  # Allow test models
            allowed_top_level_params={
                'messages', 'modelId', 'system', 'toolConfig', 'inferenceConfig',
                'guardrailConfig', 'additionalModelRequestFields', 'temperature',
                'maxTokens', 'topP', 'stopSequences'  # Allow test parameters
            }
        )
        test_validator = RequestValidator(test_security_config)

        processor = AsyncChainOfThoughtProcessor("test", error_handler, request_validator=test_validator)
        
        # Override execute_tool_call to raise an exception
        async def failing_execute(tool_name, tool_args):
            raise ValueError("Tool execution failed")
        
        error_handler.execute_tool_call = failing_execute
        
        mock_client = MockBedrockClient([
            {
                "stopReason": "tool_use",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "failing-tool",
                                    "name": "chain_of_thought_step",
                                    "input": {"thought": "This will fail"}
                                }
                            }
                        ]
                    }
                }
            },
            {
                "stopReason": "end_turn",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "Error handled"}]
                    }
                }
            }
        ])
        
        initial_request = {
            "messages": [{"role": "user", "content": [{"text": "Test error"}]}],
            "modelId": "test-model"
        }
        
        result = await processor.process_tool_loop(mock_client, initial_request)
        
        # Should complete despite tool error
        assert result["stopReason"] == "end_turn"
        assert mock_client.call_count == 2
        
        # Check that error was included in tool results
        call_history = mock_client.call_history
        assert len(call_history) == 2
        
        # Second call should include error response
        second_call_messages = call_history[1]["messages"]
        tool_result_message = second_call_messages[-1]
        assert tool_result_message["role"] == "user"
        
        tool_result_content = tool_result_message["content"][0]["toolResult"]["content"][0]["text"]
        error_response = json.loads(tool_result_content)
        assert "error" in error_response
    
    @pytest.mark.asyncio
    async def test_max_iterations_limit(self):
        """Test maximum iterations limit."""
        # Create responses that always continue with tool use
        responses = []
        for i in range(25):  # More than default max_iterations (20)
            responses.append({
                "stopReason": "tool_use",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": f"tool-{i}",
                                    "name": "chain_of_thought_step",
                                    "input": {
                                        "thought": f"Iteration {i}",
                                        "step_number": i + 1,
                                        "total_steps": 100,
                                        "next_step_needed": True
                                    }
                                }
                            }
                        ]
                    }
                }
            })
        
        mock_client = MockBedrockClient(responses)
        
        initial_request = {
            "messages": [{"role": "user", "content": [{"text": "Long reasoning"}]}],
            "modelId": "test-model"
        }
        
        result = await self.processor.process_tool_loop(mock_client, initial_request, max_iterations=5)
        
        # Should stop at max iterations
        assert result["stopReason"] == "max_tokens"
        assert mock_client.call_count == 5
        assert self.processor._tool_use_count == 5
    
    @pytest.mark.asyncio
    async def test_custom_max_iterations(self):
        """Test custom max iterations parameter."""
        responses = []
        for i in range(10):
            responses.append({
                "stopReason": "tool_use",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": f"tool-{i}",
                                    "name": "chain_of_thought_step",
                                    "input": {
                                        "thought": f"Step {i}",
                                        "step_number": i + 1,
                                        "total_steps": 20,
                                        "next_step_needed": True
                                    }
                                }
                            }
                        ]
                    }
                }
            })
        
        mock_client = MockBedrockClient(responses)
        
        initial_request = {
            "messages": [{"role": "user", "content": [{"text": "Custom limit"}]}],
            "modelId": "test-model"
        }
        
        result = await self.processor.process_tool_loop(mock_client, initial_request, max_iterations=3)
        
        assert result["stopReason"] == "max_tokens"
        assert mock_client.call_count == 3
    
    @pytest.mark.asyncio
    async def test_unexpected_stop_reason(self):
        """Test handling of unexpected stop reasons."""
        mock_client = MockBedrockClient([
            {
                "stopReason": "content_filter",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "Content filtered"}]
                    }
                }
            }
        ])
        
        initial_request = {
            "messages": [{"role": "user", "content": [{"text": "Test unexpected"}]}],
            "modelId": "test-model"
        }
        
        result = await self.processor.process_tool_loop(mock_client, initial_request)
        
        # Should return immediately with unexpected stop reason
        assert result["stopReason"] == "content_filter"
        assert mock_client.call_count == 1
    
    @pytest.mark.asyncio
    async def test_get_reasoning_summary(self):
        """Test getting reasoning summary."""
        # Add some steps to the chain
        self.processor.chain.add_step("Test step 1", 1, 2, True)
        self.processor.chain.add_step("Test step 2", 2, 2, False)
        
        summary = await self.processor.get_reasoning_summary()
        
        assert summary["status"] == "success"
        assert summary["total_steps"] == 2
        assert len(summary["chain"]) == 2
    
    def test_clear_reasoning(self):
        """Test clearing reasoning."""
        # Add some steps and tool use count
        self.processor.chain.add_step("Test step", 1, 1, False)
        self.processor._tool_use_count = 5
        
        result = self.processor.clear_reasoning()
        
        assert result["status"] == "success"
        assert len(self.processor.chain.steps) == 0
        assert self.processor._tool_use_count == 0
    
    @pytest.mark.asyncio
    async def test_message_history_preservation(self):
        """Test that message history is properly preserved and extended."""
        mock_client = MockBedrockClient([
            {
                "stopReason": "tool_use",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"text": "Using tool"},
                            {
                                "toolUse": {
                                    "toolUseId": "test-id",
                                    "name": "chain_of_thought_step",
                                    "input": {"thought": "Step 1", "step_number": 1, "total_steps": 1, "next_step_needed": False}
                                }
                            }
                        ]
                    }
                }
            },
            {
                "stopReason": "end_turn",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "Complete"}]
                    }
                }
            }
        ])
        
        initial_request = {
            "messages": [
                {"role": "user", "content": [{"text": "Initial message"}]},
                {"role": "assistant", "content": [{"text": "Previous response"}]},
                {"role": "user", "content": [{"text": "Follow up"}]}
            ],
            "modelId": "test-model"
        }
        
        await self.processor.process_tool_loop(mock_client, initial_request)
        
        # Check message history in calls
        call_history = mock_client.call_history

        # First call should have original messages
        first_call_messages = call_history[0]["messages"]
        assert len(first_call_messages) == 3

        # Second call should have additional messages (original + assistant response + tool result)
        second_call_messages = call_history[1]["messages"]
        assert len(second_call_messages) == 5  # Original 3 + assistant response + tool result


@pytest.mark.async_test
class TestBedrockStopReasonHandler:
    """Test BedrockStopReasonHandler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = BedrockStopReasonHandler()
        self.chain = ChainOfThought()
    
    @pytest.mark.asyncio
    async def test_should_continue_reasoning_empty_chain(self):
        """Test should continue reasoning with empty chain."""
        result = await self.handler.should_continue_reasoning(self.chain)
        assert result is True  # Should continue if no steps yet
    
    @pytest.mark.asyncio
    async def test_should_continue_reasoning_with_steps(self):
        """Test should continue reasoning with steps."""
        # Add step that needs continuation
        self.chain.add_step("Step 1", 1, 2, True)  # next_step_needed=True
        
        result = await self.handler.should_continue_reasoning(self.chain)
        assert result is True
        
        # Add final step
        self.chain.add_step("Final step", 2, 2, False)  # next_step_needed=False
        
        result = await self.handler.should_continue_reasoning(self.chain)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_valid_tools(self):
        """Test executing valid tool calls."""
        # Test chain_of_thought_step
        result = await self.handler.execute_tool_call(
            "chain_of_thought_step",
            {
                "thought": "Test thought",
                "step_number": 1,
                "total_steps": 1,
                "next_step_needed": False
            }
        )
        
        assert isinstance(result, dict)
        # Result should be JSON-serializable string parsed back to dict
        if isinstance(result, str):
            result = json.loads(result)
        assert "status" in result
        
        # Test get_chain_summary
        result = await self.handler.execute_tool_call("get_chain_summary", {})
        assert isinstance(result, dict) or isinstance(result, str)
        
        # Test clear_chain
        result = await self.handler.execute_tool_call("clear_chain", {})
        assert isinstance(result, dict) or isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_invalid_tool(self):
        """Test executing invalid tool call."""
        with pytest.raises(ValueError) as exc_info:
            await self.handler.execute_tool_call("invalid_tool", {})
        
        assert "Unknown tool" in str(exc_info.value)
    
    def test_custom_handlers(self):
        """Test BedrockStopReasonHandler with custom handlers."""
        custom_handlers = {
            "custom_tool": lambda **kwargs: {"result": "custom"}
        }
        
        handler = BedrockStopReasonHandler(custom_handlers)
        assert "custom_tool" in handler.handlers
        assert handler.handlers["custom_tool"]() == {"result": "custom"}


@pytest.mark.async_test
class TestAsyncIntegrationScenarios:
    """Test realistic async integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_reasoning_workflow(self):
        """Test complete multi-step reasoning workflow."""
        responses = [
            # First iteration - start reasoning
            {
                "stopReason": "tool_use",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"text": "I'll analyze this step by step."},
                            {
                                "toolUse": {
                                    "toolUseId": "step-1",
                                    "name": "chain_of_thought_step",
                                    "input": {
                                        "thought": "First, let me define the problem clearly",
                                        "step_number": 1,
                                        "total_steps": 3,
                                        "next_step_needed": True,
                                        "reasoning_stage": "Problem Definition",
                                        "confidence": 0.9
                                    }
                                }
                            }
                        ]
                    }
                }
            },
            # Second iteration - continue reasoning
            {
                "stopReason": "tool_use",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"text": "Now let me analyze the data."},
                            {
                                "toolUse": {
                                    "toolUseId": "step-2",
                                    "name": "chain_of_thought_step",
                                    "input": {
                                        "thought": "Based on the available data, I can see several patterns",
                                        "step_number": 2,
                                        "total_steps": 3,
                                        "next_step_needed": True,
                                        "reasoning_stage": "Analysis",
                                        "confidence": 0.7,
                                        "evidence": ["Pattern A", "Pattern B"],
                                        "dependencies": [1]
                                    }
                                }
                            }
                        ]
                    }
                }
            },
            # Third iteration - conclude
            {
                "stopReason": "tool_use",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"text": "Let me conclude my analysis."},
                            {
                                "toolUse": {
                                    "toolUseId": "step-3",
                                    "name": "chain_of_thought_step",
                                    "input": {
                                        "thought": "Synthesizing all information, the recommendation is clear",
                                        "step_number": 3,
                                        "total_steps": 3,
                                        "next_step_needed": False,
                                        "reasoning_stage": "Conclusion",
                                        "confidence": 0.85,
                                        "dependencies": [1, 2]
                                    }
                                }
                            }
                        ]
                    }
                }
            },
            # Fourth iteration - get summary
            {
                "stopReason": "tool_use",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"text": "Let me provide a summary."},
                            {
                                "toolUse": {
                                    "toolUseId": "summary",
                                    "name": "get_chain_summary",
                                    "input": {}
                                }
                            }
                        ]
                    }
                }
            },
            # Final response
            {
                "stopReason": "end_turn",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "Analysis complete. Here's my final recommendation based on the reasoning above."}]
                    }
                }
            }
        ]
        
        mock_client = MockBedrockClient(responses)
        processor = AsyncChainOfThoughtProcessor("complete_workflow")
        
        initial_request = {
            "messages": [
                {"role": "user", "content": [{"text": "Please analyze this complex problem systematically."}]}
            ],
            "modelId": "anthropic.claude-3-sonnet-20240229-v1:0"
        }
        
        result = await processor.process_tool_loop(mock_client, initial_request)
        
        # Verify complete workflow
        assert result["stopReason"] == "end_turn"
        assert mock_client.call_count == 5
        assert processor._tool_use_count == 4  # 3 steps + 1 summary
        
        # Verify reasoning chain
        chain = processor.chain
        assert len(chain.steps) == 3
        assert chain.steps[0].reasoning_stage == "Problem Definition"
        assert chain.steps[1].reasoning_stage == "Analysis"
        assert chain.steps[2].reasoning_stage == "Conclusion"
        
        # Verify dependencies
        assert chain.steps[1].dependencies == [1]
        assert chain.steps[2].dependencies == [1, 2]
        
        # Verify evidence was captured
        assert chain.steps[1].evidence == ["Pattern A", "Pattern B"]
        
        # Get final summary
        final_summary = await processor.get_reasoning_summary()
        assert final_summary["status"] == "success"
        assert final_summary["total_steps"] == 3
        assert len(final_summary["stages_covered"]) == 3
    
    @pytest.mark.asyncio
    async def test_error_recovery_scenario(self):
        """Test error recovery during multi-step reasoning."""
        responses = [
            # First step succeeds
            {
                "stopReason": "tool_use",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "step-1",
                                    "name": "chain_of_thought_step",
                                    "input": {
                                        "thought": "Valid step",
                                        "step_number": 1,
                                        "total_steps": 3,
                                        "next_step_needed": True
                                    }
                                }
                            }
                        ]
                    }
                }
            },
            # Second step has tool error (invalid tool name)
            {
                "stopReason": "tool_use",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "step-2",
                                    "name": "invalid_tool_name",
                                    "input": {"invalid": "parameters"}
                                }
                            }
                        ]
                    }
                }
            },
            # Third step continues after error
            {
                "stopReason": "tool_use",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "step-3",
                                    "name": "chain_of_thought_step",
                                    "input": {
                                        "thought": "Continuing after error",
                                        "step_number": 2,
                                        "total_steps": 3,
                                        "next_step_needed": False
                                    }
                                }
                            }
                        ]
                    }
                }
            },
            # Final response
            {
                "stopReason": "end_turn",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "Completed despite error"}]
                    }
                }
            }
        ]
        
        mock_client = MockBedrockClient(responses)
        processor = AsyncChainOfThoughtProcessor("error_recovery")
        
        initial_request = {
            "messages": [{"role": "user", "content": [{"text": "Test error recovery"}]}],
            "modelId": "anthropic.claude-3-sonnet-20240229-v1:0"
        }
        
        result = await processor.process_tool_loop(mock_client, initial_request)
        
        # Should complete despite tool error
        assert result["stopReason"] == "end_turn"
        assert mock_client.call_count == 4
        
        # Should have 2 successful steps (first and third)
        assert len(processor.chain.steps) == 2
        assert processor.chain.steps[0].thought == "Valid step"
        assert processor.chain.steps[1].thought == "Continuing after error"
        
        # Tool use count should include the failed attempt
        assert processor._tool_use_count == 3