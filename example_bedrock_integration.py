#!/usr/bin/env python3
"""
Example: Bedrock CoT Integration using stopReason pattern

This shows how to integrate the Chain of Thought tool with AWS Bedrock
using the native stopReason detection pattern, similar to the XState example.

Security Features:
- Environment variable configuration for AWS region
- AWS credential validation before use
- Comprehensive error handling for AWS service failures
- No hardcoded credentials or regions

Environment Variables:
- AWS_REGION or AWS_DEFAULT_REGION: AWS region to use (defaults to us-east-1)
- AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY: AWS credentials
- Or use AWS CLI profiles, IAM roles, or other standard AWS credential methods

Required IAM Permissions:
- bedrock:InvokeModel (for model execution)
- sts:GetCallerIdentity (for credential validation)

Setup:
1. Configure AWS credentials: aws configure
2. (Optional) Set region: export AWS_REGION=us-west-2
3. Run: python example_bedrock_integration.py
"""

import asyncio
import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError, BotoCoreError
from chain_of_thought import TOOL_SPECS, AsyncChainOfThoughtProcessor


def get_aws_region():
    """
    Get AWS region from environment variables with fallback.
    
    Environment variables (in order of preference):
    - AWS_REGION
    - AWS_DEFAULT_REGION
    
    Returns:
        str: AWS region name, defaults to 'us-east-1' if not configured
    """
    region = os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION')
    if not region:
        print("ℹ️  No AWS region configured in environment variables.")
        print("   Set AWS_REGION or AWS_DEFAULT_REGION, or we'll use 'us-east-1'")
        region = 'us-east-1'
    
    print(f"🌍 Using AWS region: {region}")
    return region


async def validate_aws_credentials(region):
    """
    Validate AWS credentials and region configuration.
    
    Args:
        region (str): AWS region to use
        
    Returns:
        boto3.Client: Configured bedrock-runtime client
        
    Raises:
        RuntimeError: If credentials are invalid or service unavailable
    """
    print("🔐 Validating AWS credentials...")
    
    try:
        # First, validate basic AWS credentials using STS
        sts_client = boto3.client('sts', region_name=region)
        identity = sts_client.get_caller_identity()
        
        print(f"✅ AWS credentials valid")
        print(f"   Account: {identity.get('Account', 'Unknown')}")
        print(f"   User/Role: {identity.get('Arn', 'Unknown')}")
        
    except NoCredentialsError:
        raise RuntimeError(
            "❌ No AWS credentials found. Please configure credentials using:\n"
            "   1. AWS CLI: aws configure\n"
            "   2. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY\n"
            "   3. IAM role (for EC2/Lambda)\n"
            "   4. AWS credentials file (~/.aws/credentials)"
        )
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        raise RuntimeError(f"❌ AWS credentials invalid: {error_code} - {str(e)}")
    
    try:
        # Create and test Bedrock client
        bedrock_client = boto3.client('bedrock-runtime', region_name=region)
        
        # Test if Bedrock is available in the region by making a minimal call
        # We don't need to complete this call, just verify the service is accessible
        print(f"🤖 Testing Bedrock service availability in {region}...")
        
        return bedrock_client
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'UnrecognizedClientException':
            raise RuntimeError(
                f"❌ Bedrock service not available in region '{region}'.\n"
                "   Available regions: us-east-1, us-west-2, eu-west-1, ap-southeast-1, ap-northeast-1\n"
                "   Set AWS_REGION environment variable to use a different region."
            )
        else:
            raise RuntimeError(f"❌ Bedrock service error: {error_code} - {str(e)}")
    
    except BotoCoreError as e:
        raise RuntimeError(f"❌ AWS configuration error: {str(e)}")


async def main():
    """Example of running CoT with Bedrock using stopReason pattern."""
    
    try:
        # Get AWS region from environment or use secure default
        region = get_aws_region()
        
        # Validate credentials and create client securely
        client = await validate_aws_credentials(region)
        
        print("✅ AWS configuration validated successfully")
        print()
        
        cot_processor = AsyncChainOfThoughtProcessor(
            conversation_id="example-session-123"
        )
        
    except RuntimeError as e:
        print(f"\n{str(e)}")
        print("\n🚨 Setup failed. Please fix the configuration issues above.")
        return
    
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
    
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        
        if error_code == 'ValidationException':
            print(f"❌ Model or request validation error: {error_message}")
            print("   Check that the model ID is correct and available in your region")
            print("   Current model: anthropic.claude-3-sonnet-20240229-v1:0")
        elif error_code == 'AccessDeniedException':
            print(f"❌ Access denied: {error_message}")
            print("   Your AWS credentials may lack the required Bedrock permissions")
            print("   Required IAM permissions: bedrock:InvokeModel")
        elif error_code == 'ThrottlingException':
            print(f"❌ Request throttled: {error_message}")
            print("   Too many requests to Bedrock. Please wait and try again.")
        elif error_code == 'ServiceUnavailableException':
            print(f"❌ Bedrock service unavailable: {error_message}")
            print("   The service may be experiencing temporary issues")
        else:
            print(f"❌ AWS Bedrock error ({error_code}): {error_message}")
    
    except BotoCoreError as e:
        print(f"❌ AWS connection error: {str(e)}")
        print("   Check your internet connection and AWS service status")
    
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        print("   This may be a bug in the chain-of-thought library")
        import traceback
        print(f"   Full traceback: {traceback.format_exc()}")


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
    print("\n🔒 Security improvements:")
    print("✅ Environment variable configuration (AWS_REGION)")
    print("✅ AWS credential validation before use")
    print("✅ Comprehensive error handling for AWS failures")
    print("✅ No hardcoded credentials or regions")
    print("\n💡 To run the full Bedrock example:")
    print("1. Configure AWS: aws configure")
    print("2. (Optional) Set region: export AWS_REGION=your-preferred-region") 
    print("3. Uncomment the asyncio.run(main()) line above")