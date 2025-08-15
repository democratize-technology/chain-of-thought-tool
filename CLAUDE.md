# sequential-thinking-tool - Claude Code Configuration

## 🧠 PROJECT OVERVIEW
**Type**: Python Library - LLM Function Calling Tools  
**Purpose**: Lightweight Chain of Thought reasoning capabilities for any LLM API  
**Architecture**: Tool-based function calling with async AWS Bedrock integration  
**Maturity**: Alpha (v0.1.0) - Well-architected but needs testing infrastructure

## 🏗️ CORE ARCHITECTURE

### Design Patterns
* **Tool-based Function Calling**: Drop-in compatibility with LLM APIs
* **Zero Dependencies**: Pure Python approach for maximum compatibility  
* **Multi-tenant Thread Safety**: Production-ready conversation isolation
* **stopReason Integration**: Native AWS Bedrock tool loop handling

### Key Components
```
chain_of_thought/
├── __init__.py          # Tool specs (TOOL_SPECS, HANDLERS) + exports
├── core.py             # Main logic (461 lines, dataclasses + async)
example_bedrock_integration.py  # Complete AWS integration example
setup.py               # Clean packaging, no external deps
```

### Three Usage Patterns
1. **Simple**: Global singleton for basic usage
2. **Production**: `ThreadAwareChainOfThought` for multi-conversation apps  
3. **AWS Bedrock**: `AsyncChainOfThoughtProcessor` for stopReason patterns

## 🔧 DEVELOPMENT WORKFLOW

### Quick Start
```bash
# Install in development mode
pip install -e .

# Install dev dependencies
pip install -e ".[dev]"

# Test the library (currently missing - HIGH PRIORITY)
# pytest

# Format and lint (configurations missing)
# black .
# flake8
```

### What Actually Works Right Now
```bash
# Syntax and import validation
python -m py_compile chain_of_thought/*.py
python -c "import chain_of_thought; print('Import successful')"

# Manual integration testing
python example_bedrock_integration.py

# Basic AWS credential check
aws sts get-caller-identity

# Package build verification
python setup.py check
```

### Build & Distribution
```bash
# Build package
python setup.py sdist bdist_wheel

# Test locally
pip install dist/chain-of-thought-tool-*.whl
```

## 🚨 CRITICAL GAPS (Immediate Priorities)

### 1. Testing Infrastructure (URGENT)
```bash
# Missing: Complete test suite
# Need: pytest configuration, unit tests, integration tests
# Priority: CRITICAL - This is a tool library, testing is essential
```

### 2. Development Automation (HIGH)
```bash
# Missing: .github/workflows/, pre-commit hooks
# Need: CI/CD pipeline, automated testing, code quality checks
# Files needed: .github/workflows/test.yml, .pre-commit-config.yaml
```

### 3. Code Quality Setup (MEDIUM)
```bash
# Missing: Configuration files for declared dev dependencies
# Need: pytest.ini, pyproject.toml (black config), .flake8
# Currently: setup.py declares pytest/black/flake8 but no configs exist
```

## 🎯 SPECIALIZED AGENT RECOMMENDATIONS

### Immediate Delegation Tasks
1. **tester** → Create comprehensive test suite (unit + integration tests)
2. **oss-readiness** → Setup CI/CD pipeline and PyPI publishing workflow
3. **architect** → Review async patterns and thread safety implementation
4. **code-reviewer** → Assess code quality and suggest improvements

### Agent Workflow
```bash
# Start with testing foundation
Task(tester) → "Create pytest suite for chain_of_thought module"

# Then setup automation  
Task(oss-readiness) → "Setup GitHub Actions CI/CD and PyPI workflow"

# Code quality pass
Task(code-reviewer) → "Review architecture and suggest improvements"

# Architecture validation
Task(architect) → "Validate async patterns and stopReason integration"
```

## 🔍 UNIQUE PROJECT FEATURES

### stopReason Integration Innovation
```python
# This library's key innovation: Native Bedrock tool loop handling
async def process_tool_loop(self, bedrock_client, initial_request):
    # Automatically handles stopReason="tool_use" vs "end_turn"
    # Maps CoT "next_step_needed" to Bedrock flow control
```

### Confidence & Evidence Tracking
```python
# Built-in reasoning metadata
{
    "confidence": 0.8,
    "evidence": ["Market data", "User research"],
    "assumptions": ["Stable interest rates"],
    "contradicts": [2, 3]  # References to other steps
}
```

### Zero Dependency Philosophy
* No external libraries required
* Pure Python 3.8+ compatibility
* Drop-in integration with any LLM API

## 📊 INTEGRATION PATTERNS

### AWS Bedrock (Primary)
```python
from chain_of_thought import TOOL_SPECS, AsyncChainOfThoughtProcessor

# Direct integration with Converse API
bedrock.converse(toolConfig={"tools": TOOL_SPECS})
```

### OpenAI/Anthropic
```python
# Convert format for other providers
openai_tools = [{
    "type": "function", 
    "function": {
        "name": tool["toolSpec"]["name"],
        "description": tool["toolSpec"]["description"],
        "parameters": tool["toolSpec"]["inputSchema"]["json"]
    }
} for tool in TOOL_SPECS]
```

## ⚡ COMMON OPERATIONS

### Testing New Features
```bash
# Currently manual - needs automation
python example_bedrock_integration.py

# Should be:
# pytest tests/
# pytest tests/integration/
```

### Adding New Tool Functions  
1. Add tool spec to `__init__.py` TOOL_SPECS
2. Implement handler in `core.py`
3. Add to HANDLERS mapping
4. **Missing**: Add tests for new functionality

### Debugging Integration Issues
```bash
# Use the example file for testing
python example_bedrock_integration.py

# Check AWS credentials
aws sts get-caller-identity

# Verify tool loop behavior
# (Currently requires manual inspection)
```

## 🧠 CLAUDE CODE MEMORY INTEGRATION

### Pattern Storage & Retrieval
```bash
# Before implementing new features, check for existing patterns
graphiti:search_nodes → query="sequential thinking tool patterns"
graphiti:search_facts → query="AWS Bedrock stopReason integration"

# After successful implementation, store for future reference
graphiti:add_episode → name="Feature: [description]"
                    → episode_body="Implementation approach, gotchas, performance"
                    → group_id="sequential-thinking-tool"
```

### Common Memory Queries
```bash
# Architecture patterns
graphiti:search_nodes → "tool calling patterns", "async processing"

# Integration solutions  
graphiti:search_facts → "Bedrock integration", "multi-provider support"

# Development solutions
graphiti:search_nodes → "testing setup", "CI/CD pipeline"
```

### Storing Successful Solutions
```python
# After completing major features
graphiti:add_episode(
    name="AWS Bedrock Tool Loop Implementation",
    episode_body="""
    Pattern: AsyncChainOfThoughtProcessor with stopReason handling
    Key insight: Map 'next_step_needed' boolean to Bedrock flow control
    Performance: Reduces API calls by 40% vs polling approach
    Gotchas: Requires careful async context management
    """,
    group_id="sequential-thinking-tool"
)
```

## 🔒 SECURITY CONSIDERATIONS
* **Input Validation**: Tool inputs should be validated (currently basic)
* **Resource Limits**: No limits on reasoning steps or memory usage
* **Thread Safety**: Properly implemented via ThreadAwareChainOfThought
* **AWS Credentials**: Relies on standard AWS credential chain

## 🎯 SUCCESS METRICS
* **Adoption**: Integration simplicity (currently excellent)
* **Reliability**: Test coverage (currently 0% - critical gap)
* **Performance**: Async efficiency (good architecture, needs benchmarks)
* **Documentation**: Usage clarity (excellent README, needs API docs)

## 📈 GROWTH OPPORTUNITIES
1. **Testing Foundation**: Essential for library credibility
2. **PyPI Publishing**: Setup automated releases
3. **Performance Benchmarks**: Measure async efficiency
4. **Type Safety**: Add mypy support for better DX
   ```bash
   # Setup mypy configuration (missing but declared in setup.py)
   cat > mypy.ini << EOF
   [mypy]
   python_version = 3.8
   warn_return_any = True
   warn_unused_configs = True
   disallow_untyped_defs = True
   
   [mypy-chain_of_thought.*]
   strict = True
   EOF
   
   # Run type checking
   mypy chain_of_thought/
   ```
5. **Integration Helpers**: More LLM provider adapters

---

**BOTTOM LINE**: Excellent architecture and innovative stopReason integration, but critically missing testing infrastructure. Priority #1 is comprehensive test suite, then CI/CD automation.