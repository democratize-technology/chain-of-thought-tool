#!/usr/bin/env python3
"""
Example: Bedrock CoT Integration using stopReason pattern

This shows how to integrate the Chain of Thought tool with AWS Bedrock
using the native stopReason detection pattern, similar to the XState example.
"""

import asyncio
import boto3
from chain_of_thought import TOOL_SPECS, AsyncChainOfThoughtProcessor

async def main():
    """Example of running CoT with Bedrock using stopReason pattern."""
    
    client = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    cot_processor = AsyncChainOfThoughtProcessor(
        conversation_id="example-session-123"
    )
    
    request = {
        "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": "Please analyze the pros and cons of remote work vs office work. Use chain of thought reasoning to structure your analysis."
                    }
                ]
            }
        ],
        "system": [
            {
                "text": "You are an analytical assistant. Use the chain_of_thought_step tool to structure your reasoning process step by step."
            }
        ],
        "toolConfig": {
            "tools": TOOL_SPECS
        },
        "inferenceConfig": {
            "temperature": 0.7,
            "maxTokens": 4096
        }
    }
    
    print("🧠 Starting Chain of Thought analysis with Bedrock...")
    print("=" * 60)
    
    try:
        result = await cot_processor.process_tool_loop(
            bedrock_client=client,
            initial_request=request,
            max_iterations=20
        )
        
        print("✅ Analysis complete!")
        print(f"Stop reason: {result.get('stopReason')}")
        
        if "output" in result and "message" in result["output"]:
            final_content = result["output"]["message"].get("content", [])
            for item in final_content:
                if "text" in item:
                    print("\n📝 Final Response:")
                    print("-" * 40)
                    print(item["text"])
        
        summary = await cot_processor.get_reasoning_summary()
        print(f"\n🧠 Reasoning Summary:")
        print("-" * 40)
        print(f"Total steps: {summary.get('total_steps', 0)}")
        print(f"Stages covered: {', '.join(summary.get('stages_covered', []))}")
        print(f"Overall confidence: {summary.get('overall_confidence', 'N/A')}")
        
        if summary.get('chain'):
            print(f"\n📋 Step-by-step breakdown:")
            for step in summary['chain']:
                print(f"  {step['step']}. [{step['stage']}] {step['thought_preview']} (confidence: {step['confidence']})")
    
    except Exception as e:
        print(f"❌ Error: {e}")


async def simple_integration_example():
    """
    Simpler example showing just the stopReason pattern without Bedrock.
    This demonstrates the core concept.
    """
    
    print("\n🔄 Simple stopReason pattern example:")
    print("=" * 50)
    
    # Create processor
    processor = AsyncChainOfThoughtProcessor("simple-example")
    
    # Simulate the pattern from your XState example
    class MockResponse:
        def __init__(self, stop_reason, has_tool_use=False):
            self.stop_reason = stop_reason
            self.has_tool_use = has_tool_use
            
        def get(self, key):
            if key == "stopReason":
                return self.stop_reason
            return {}
    
    # Simulate responses
    responses = [
        MockResponse("tool_use", True),   # LLM wants to use CoT tool
        MockResponse("tool_use", True),   # More reasoning steps
        MockResponse("tool_use", True),   # Even more steps
        MockResponse("end_turn", False),  # LLM finished reasoning
    ]
    
    for i, mock_response in enumerate(responses, 1):
        stop_reason = mock_response.get("stopReason")
        
        if stop_reason == "tool_use":
            print(f"Step {i}: stopReason = 'tool_use' → Continue reasoning loop")
            
            # Simulate tool execution
            await processor.stop_handler.execute_tool_call(
                "chain_of_thought_step",
                {
                    "thought": f"This is reasoning step {i}",
                    "step_number": i,
                    "total_steps": 4,
                    "next_step_needed": i < 3,
                    "reasoning_stage": "Analysis",
                    "confidence": 0.8
                }
            )
            
        elif stop_reason == "end_turn":
            print(f"Step {i}: stopReason = 'end_turn' → Check if CoT wants to continue")
            
            should_continue = await processor.stop_handler.should_continue_reasoning(processor.chain)
            print(f"  CoT says continue: {should_continue}")
            
            if not should_continue:
                print("  ✅ Both Bedrock and CoT agree: reasoning complete!")
                break
    
    # Show final summary
    summary = await processor.get_reasoning_summary()
    print(f"\n📊 Final summary: {summary['total_steps']} steps completed")


if __name__ == "__main__":
    print("🚀 Chain of Thought + Bedrock Integration Examples")
    print("=" * 60)
    
    # Run simple pattern example first
    asyncio.run(simple_integration_example())
    
    # Uncomment to run full Bedrock example (requires AWS credentials)
    # asyncio.run(main())
    
    print("\n✨ Integration examples complete!")
    print("\nKey takeaways:")
    print("1. Use stopReason to control tool loops naturally")
    print("2. CoT 'next_step_needed' maps to stopReason flow")
    print("3. AsyncChainOfThoughtProcessor handles the complexity")
    print("4. Your XState pattern works perfectly with this approach")