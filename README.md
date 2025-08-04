# Chain of Thought Tool

A lightweight Python package that provides structured Chain of Thought reasoning capabilities for LLMs through function calling.

## Installation

```bash
pip install chain-of-thought-tool
```

Or install from source:
```bash
cd chain-of-thought-tool
pip install -e .
```

## Quick Start

```python
from chain_of_thought import TOOL_SPECS, HANDLERS

# Add to your LLM tools array
tools = [
    *TOOL_SPECS,  # Adds chain_of_thought_step, get_chain_summary, clear_chain
]

# In your tool handling logic
def handle_tool_call(tool_name, tool_args):
    if tool_name in HANDLERS:
        return HANDLERS[tool_name](**tool_args)
    # ... handle other tools
```

## Usage with AWS Bedrock Converse API

```python
import boto3
from chain_of_thought import TOOL_SPECS, HANDLERS

bedrock = boto3.client('bedrock-runtime')

# Your conversation with tools
response = bedrock.converse(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[
        {
            "role": "user", 
            "content": [{"text": "Help me think through whether I should buy a house or keep renting."}]
        }
    ],
    toolConfig={
        "tools": TOOL_SPECS  # Just drop it in!
    }
)

# Handle tool calls
for content in response['output']['message']['content']:
    if content.get('toolUse'):
        tool_use = content['toolUse']
        tool_name = tool_use['name']
        tool_args = tool_use['input']
        
        # Execute the tool
        result = HANDLERS[tool_name](**tool_args)
        print(f"Tool {tool_name} result: {result}")
```

## Usage with OpenAI

```python
import openai
from chain_of_thought import TOOL_SPECS, HANDLERS

# Convert to OpenAI format
openai_tools = []
for tool in TOOL_SPECS:
    openai_tools.append({
        "type": "function",
        "function": {
            "name": tool["toolSpec"]["name"],
            "description": tool["toolSpec"]["description"],
            "parameters": tool["toolSpec"]["inputSchema"]["json"]
        }
    })

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Help me think through a complex decision."}],
    tools=openai_tools
)

# Handle tool calls
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        result = HANDLERS[tool_call.function.name](**eval(tool_call.function.arguments))
```

## How It Works

The Chain of Thought tool provides three main functions:

### 1. `chain_of_thought_step`
Process individual thoughts in a structured sequence with confidence tracking:

```python
{
    "thought": "I need to consider the financial implications of buying vs renting",
    "step_number": 1,
    "total_steps": 5,
    "next_step_needed": true,
    "reasoning_stage": "Problem Definition",
    "confidence": 0.8,
    "evidence": ["Current market conditions", "Personal financial situation"],
    "assumptions": ["Interest rates will remain stable"]
}
```

### 2. `get_chain_summary`
Get a comprehensive summary of the thinking process:

```python
# No arguments needed
{}
```

### 3. `clear_chain`
Reset the thinking process:

```python
# No arguments needed  
{}
```

## Advanced Features

### Confidence Tracking
Each step can include a confidence level (0.0-1.0) to indicate certainty:

```python
{
    "thought": "Based on my analysis, renting is more flexible",
    "confidence": 0.85,
    ...
}
```

### Dependencies and Contradictions
Track relationships between thoughts:

```python
{
    "thought": "This contradicts my earlier assumption",
    "dependencies": [1, 2],  # Depends on steps 1 and 2
    "contradicts": [3],      # Contradicts step 3
    ...
}
```

### Evidence and Assumptions
Make reasoning transparent:

```python
{
    "evidence": ["Market data shows 5% annual appreciation"],
    "assumptions": ["My job will remain stable"],
    ...
}
```

### Structured Stages
Guide thinking through defined stages:
- `Problem Definition`
- `Research` 
- `Analysis`
- `Synthesis`
- `Conclusion`

## Why This Approach?

**Traditional Problems:**
- ❌ MCP tools require separate server processes
- ❌ Framework-specific tools (LangChain, etc.)
- ❌ Complex infrastructure for simple functions

**Our Solution:**
- ✅ Simple `pip install` and import
- ✅ Works with any LLM API (OpenAI, Anthropic, etc.)
- ✅ Self-contained tool specs and implementations
- ✅ Zero infrastructure - just Python functions
- ✅ Structured reasoning with confidence tracking

## Thread Safety

For production use with multiple concurrent conversations:

```python
from chain_of_thought import ThreadAwareChainOfThought

# Create isolated instance per conversation
cot = ThreadAwareChainOfThought(conversation_id="user-123")
tools = cot.get_tool_specs()
handlers = cot.get_handlers()

# Use in your conversation
response = bedrock.converse(
    toolConfig={"tools": tools},
    # ...
)

# Handle with thread-specific handlers
result = handlers[tool_name](**tool_args)
```

## Contributing

This project demonstrates pluggable LLM tools. Contributions welcome for:
- Improved reasoning capabilities
- Additional metadata tracking
- Better summarization algorithms
- Integration helpers for more platforms

## License

MIT License
