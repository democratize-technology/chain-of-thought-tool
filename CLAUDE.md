# chain-of-thought-tool - Claude Code Configuration

## PROJECT OVERVIEW
**Type**: Python Library - LLM Function Calling Tools
**Purpose**: Lightweight Chain of Thought reasoning capabilities for any LLM API
**Version**: 0.3.0
**Architecture**: Tool-based function calling with async AWS Bedrock integration
**Maturity**: Alpha - Well-architected with comprehensive test infrastructure (321 tests, 80% coverage)
**Identity**: Named for Wei et al. 2022 "Chain of Thought" prompting; structurally descended from MCP `sequential-thinking` server pattern. See ADR-0010/0011 for dual-anchor model.

## CORE ARCHITECTURE

### Design Patterns
* **Tool-based Function Calling**: Drop-in compatibility with LLM APIs
* **Zero Dependencies**: Pure Python approach for maximum compatibility (ADR-0002)
* **Multi-tenant Thread Safety**: Production-ready conversation isolation (ADR-0004)
* **stopReason Integration**: Native AWS Bedrock tool loop handling (ADR-0009)

### Module Layout (7 modules, ~3,756 LOC)
```
chain_of_thought/
├── __init__.py       # Tool specs (TOOL_SPECS, HANDLERS) + exports (300 lines)
├── core.py           # ChainOfThought, ThoughtStep, ServiceRegistry (938 lines)
├── handlers.py       # Handler wrapper functions (84 lines)
├── bedrock.py        # AsyncChainOfThoughtProcessor, StopReasonHandler (345 lines)
├── concurrency.py    # ThreadAwareChainOfThought, RateLimiter (347 lines)
├── validators.py     # ParameterValidator - input validation + XSS prevention (596 lines)
├── security.py       # RequestValidator - Bedrock request sanitization (445 lines)
├── auxiliary.py      # HypothesisGenerator, AssumptionMapper, ConfidenceCalibrator (701 lines)
└── py.typed          # PEP 561 marker
```

### Three Usage Patterns
1. **Simple**: Global singleton for basic usage
2. **Production**: `ThreadAwareChainOfThought` for multi-conversation apps
3. **AWS Bedrock**: `AsyncChainOfThoughtProcessor` for stopReason patterns

### API Stability Tiers (ADR-0006)
* **Tier 1 (Stable)**: Tool names, TOOL_SPECS schemas - major version bump for breaks
* **Tier 2 (Evolving)**: Python class API, public methods - minor version bump
* **Tier 3 (Internal)**: Implementation details - no stability guarantee

## DEVELOPMENT WORKFLOW

### Quick Start
```bash
pip install -e .
pip install -e ".[dev]"
pytest                           # 321 tests, 80% coverage enforced
black .                          # 88 char line length
flake8                           # Configured in pyproject.toml and .flake8
mypy chain_of_thought/           # Strict mode
```

### Build & Distribution
```bash
python3 -m build                 # Uses pyproject.toml (canonical)
pip install dist/chain-of-thought-tool-*.whl
```

## TEST INFRASTRUCTURE

### Test Suite
* 321 tests across 19 test files
* Security vulnerability tests (XSS, injection, JSON serialization)
* Thread safety and concurrency tests
* Async timeout and edge case tests
* Coverage: 80% minimum enforced by pyproject.toml

### Test Categories (markers)
`unit`, `integration`, `async_test`, `thread_safety`, `mock`, `edge_case`, `slow`, `security`, `service_registry`, `memory_leak`, `async_timeout`, `json_vulnerability`

## INTEGRATION PATTERNS

### AWS Bedrock (Primary)
```python
from chain_of_thought import TOOL_SPECS, AsyncChainOfThoughtProcessor
bedrock.converse(toolConfig={"tools": TOOL_SPECS})
```

### OpenAI/Anthropic
```python
openai_tools = [{
    "type": "function",
    "function": {
        "name": tool["toolSpec"]["name"],
        "description": tool["toolSpec"]["description"],
        "parameters": tool["toolSpec"]["inputSchema"]["json"]
    }
} for tool in TOOL_SPECS]
```

## SECURITY
* **Input Validation**: Comprehensive via ParameterValidator (XSS, Unicode, injection)
* **Request Validation**: Bedrock request sanitization via RequestValidator
* **Resource Limits**: MAX_RECURSION_DEPTH=50, MAX_IMPORT_STEPS=10000, rate limiting
* **Thread Safety**: RLock-protected dictionaries and chains

## ARCHITECTURE DECISIONS

16 ADRs in `docs/adr/` covering:
- ADR-0002: Zero external dependencies
- ADR-0003: Bedrock Converse API as primary format
- ADR-0004: WeakValueDictionary hybrid for multi-tenant isolation
- ADR-0005: Handler factory with cross-cutting concerns
- ADR-0006: 3-tier API stability contract
- ADR-0007: Auxiliary tools as structural scaffolding (honest capability docs)
- ADR-0008: 5 canonical reasoning stages
- ADR-0009: Async Bedrock tool loop orchestration
- ADR-0010-0016: Drift analysis vs canonical references

## SPECIALIZED AGENT RECOMMENDATIONS

* **architect** -> Review async patterns and thread safety
* **code-reviewer** -> Assess code quality
* **security-engineer** -> Validate input sanitization and injection prevention
* **oss-readiness** -> PyPI publishing readiness review
