# Chain of Thought Tool

A lightweight Python package that provides structured Chain of Thought reasoning capabilities for LLMs through function calling.

## What This Library Is (and Isn't)

This library provides a **stateful reasoning tracker** for LLM function-calling APIs. It is named for Wei et al. 2022 "Chain of Thought Prompting Elicits Reasoning in Large Language Models" but is structurally descended from the MCP `sequential-thinking` server pattern — a tool-call-based step tracker where the LLM calls tools to record each reasoning step.

**What it does:** Track multi-step reasoning with confidence scoring, evidence collection, assumption mapping, and contradiction detection.

**What it doesn't do:** Perform LLM inference, prompt engineering, or chain-of-thought prompting. The LLM is the reasoner; this library is the notebook.

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
    *TOOL_SPECS,  # Adds all 8 tools
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
import json
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
        result = HANDLERS[tool_call.function.name](**json.loads(tool_call.function.arguments))
```

## How It Works

The library provides 8 tools in two categories:

### Core Reasoning Tools

#### `chain_of_thought_step`
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

### `get_chain_summary`
Get a comprehensive summary of the thinking process:

```python
# No arguments needed
{}
```

### `clear_chain`
Reset the thinking process:

```python
# No arguments needed  
{}
```

### `export_chain` / `import_chain`

Persist and restore reasoning chains:

```python
HANDLERS["export_chain"](file_path="analysis.json")
HANDLERS["import_chain"](file_path="analysis.json")
```

### Auxiliary Reasoning Tools

Three additional tools provide structured scaffolding for common reasoning patterns. These use template and heuristic-based implementations — they produce structured output shapes that guide an LLM's reasoning, not AI-driven analysis.

#### `generate_hypotheses`
Returns a framework of hypothesis types (scientific, intuitive, contrarian, systematic) for an observation. Templates structure divergent thinking — the LLM fills in the substance.

```python
HANDLERS["generate_hypotheses"](observation="Why did user engagement drop 30%?", hypothesis_count=4)
```

#### `map_assumptions`
Identifies linguistic indicators of assumptions (e.g., "clearly", "obviously", "must") using keyword-based heuristics. Returns a structured framework for critical thinking.

```python
HANDLERS["map_assumptions"](statement="Clearly the migration will improve performance", depth="deep")
```

#### `calibrate_confidence`
Applies heuristic pattern matching (absolute language detection, domain complexity) to produce adjusted confidence with uncertainty bands. A calibration rubric for the LLM.

```python
HANDLERS["calibrate_confidence"](prediction="Revenue will grow 20%", initial_confidence=0.9, context="Q4 forecast")
```

## Capability Assessment

Honest evaluation of what each tool category delivers:

| Category | Tools | Implementation | What They Actually Do |
|----------|-------|----------------|----------------------|
| **Core** | `chain_of_thought_step`, `get_chain_summary`, `clear_chain`, `export_chain`, `import_chain` | Full stateful reasoning chain | Store, retrieve, summarize, persist, and restore structured reasoning steps with confidence tracking, evidence, assumptions, and stage classification. These are the real deal. |
| **Auxiliary** | `generate_hypotheses`, `map_assumptions`, `calibrate_confidence` | Template/heuristic scaffolding | Produce structured output shapes using keyword matching and fixed templates. The LLM consuming these outputs performs the actual analysis. Zero-cost, zero-latency, synchronous. |

**Key distinction**: Core tools manage stateful reasoning data. Auxiliary tools provide reasoning frameworks — structured containers that prompt more careful thinking. The LLM is the analyst; the library provides the scaffolding.

### Topology and Revision Capabilities

| Capability | Status | Notes |
|------------|--------|-------|
| Branching | Not supported | Steps are a flat sequence; no forking or merge. See ADR-0012. |
| Revision | Implicit via step_number collision | Re-submitting an existing `step_number` replaces that step in place. See ADR-0013. |
| Self-consistency sampling | Not supported | Could be implemented externally by running multiple chains and comparing summaries. See ADR-0014 for a recipe. |
| Topology | Primarily linear | Optional DAG structure via `dependencies` and `contradicts` fields, but no cycle detection or traversal. See ADR-0015. |

### Intellectual Genealogy

The auxiliary reasoning tools trace to different traditions than Wei et al. 2022:
- **Hypothesis Generation** (abductive reasoning): Peirce's framework for generating explanatory hypotheses
- **Assumption Mapping** (critical thinking): Systematic identification of explicit and implicit assumptions
- **Confidence Calibration** (forecasting): Tetlock's superforecasting research on overconfidence correction

See ADR-0016 for full genealogy.

## Advanced Features

### Parameter Validation

The package includes dedicated parameter validators that ensure secure and robust input handling:

```python
from chain_of_thought import ParameterValidator

validator = ParameterValidator()

# Validate individual parameters
safe_thought = validator.validate_thought_param(user_input)
safe_confidence = validator.validate_confidence_param(0.85)

# Or validate all parameters at once
validated = validator.validate_input(
    thought="My thinking process",
    step_number=1,
    total_steps=5,
    next_step_needed=True,
    reasoning_stage="Analysis",
    confidence=0.8
)
```

**Security Features:**
- **XSS Prevention**: HTML escaping for all string inputs
- **Input Length Limits**: Prevents DoS attacks with size restrictions
- **Type Validation**: Ensures correct data types for all parameters
- **Range Validation**: Numeric inputs are bounded within reasonable limits

**Architecture Benefits:**
- **Separation of Concerns**: Validation logic is isolated from business logic
- **Reusability**: Validators can be used across different classes
- **Testability**: Validation rules can be unit tested independently
- **Maintainability**: Changes to validation rules are centralized

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
