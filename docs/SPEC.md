# Specification: chain-of-thought-tool

**Version:** 0.3.0
**Status:** Active
**Date:** 2026-05-14

---

## 1. Problem Statement

LLM function-calling APIs (AWS Bedrock, OpenAI, Anthropic) provide tool-use capabilities but no built-in mechanism for structured, multi-step reasoning. Developers who want chain-of-thought tracking — with confidence scoring, evidence collection, assumption mapping, and contradiction detection — must build it ad-hoc in every integration.

This library provides that mechanism as a zero-dependency Python package with drop-in tool specifications and handler functions compatible with any LLM function-calling API.

**What this library is:** A stateful reasoning tracker — a function-calling tool surface where the LLM calls tools to record each reasoning step. Named for Wei et al. 2022 "Chain of Thought Prompting" (the goal-lineage) but structurally descended from the MCP `sequential-thinking` server pattern (the shape-lineage). The LLM is the reasoner; this library is the notebook.

## 2. Scope

### In Scope

- Tool specifications conforming to the AWS Bedrock Converse API `toolConfig.tools` schema
- Handler functions that accept tool arguments and return JSON-string results
- A core reasoning engine (`ChainOfThought`) that stores, validates, and summarizes thought steps
- Additional reasoning tools: hypothesis generation, assumption mapping, confidence calibration
- Thread-safe multi-conversation isolation (`ThreadAwareChainOfThought`)
- Async integration with AWS Bedrock tool loops (`AsyncChainOfThoughtProcessor`)
- Input validation with XSS prevention, type checking, and DoS protection
- Serialization: export/import chains to/from JSON files
- Rate limiting on handler invocations

### Out of Scope

- LLM inference — this library does not call any model API directly
- Persistence beyond process lifetime (no database, no network storage)
- UI or visualization of reasoning chains
- Authentication or authorization of end users
- Provider-specific adapters beyond AWS Bedrock (OpenAI/Anthropic are format-compatible but not wrapped)
- Self-consistency sampling (Wang et al. 2022) — this library is not in the LLM-calling layer where sampling would naturally live. Callers can implement it externally by running multiple chains and comparing summaries (see ADR-0014 for a recipe)

## 3. Architecture

### 3.1 Module Structure

```
chain_of_thought/
  __init__.py       # Public API: TOOL_SPECS, HANDLERS, class exports, version
  core.py           # ChainOfThought, ThoughtStep, ServiceRegistry, handler factory
  handlers.py       # Handler wrapper functions
  bedrock.py        # AsyncChainOfThoughtProcessor, StopReasonHandler
  concurrency.py    # ThreadAwareChainOfThought, RateLimiter
  validators.py     # ParameterValidator - input validation
  security.py       # RequestValidator - Bedrock request sanitization
  auxiliary.py      # HypothesisGenerator, AssumptionMapper, ConfidenceCalibrator
```

### Identity

This library is named for Wei et al. 2022 but structurally descended from the MCP `sequential-thinking` server. See ADR-0010 for the dual-anchor model and ADR-0011 for identity analysis.

### 3.2 Core Data Model

**`ThoughtStep`** (dataclass) — the fundamental unit of reasoning:

| Field | Type | Required | Default | Constraints |
|-------|------|----------|---------|-------------|
| `thought` | `str` | yes | — | Non-empty after strip; max 10,000 chars; HTML-escaped on storage |
| `step_number` | `int` | yes | — | 1–1000; must be ≤ `total_steps` |
| `total_steps` | `int` | yes | — | 1–1000 |
| `next_step_needed` | `bool` | yes | — | — |
| `reasoning_stage` | `str` | no | `"Analysis"` | Enum: Problem Definition, Research, Analysis, Synthesis, Conclusion |
| `confidence` | `float` | no | `0.8` | 0.0–1.0, finite, non-NaN |
| `dependencies` | `List[int]` | no | `[]` | Step references; max 50 items; values 1–1000 |
| `contradicts` | `List[int]` | no | `[]` | Step references; max 50 items; values 1–1000 |
| `evidence` | `List[str]` | no | `[]` | Max 50 items; 500 chars each; HTML-escaped |
| `assumptions` | `List[str]` | no | `[]` | Max 50 items; 500 chars each; HTML-escaped |
| `timestamp` | `str` | no | ISO 8601 | Auto-generated if not provided |

**Revision Semantics.** When `add_step()` receives a `step_number` that already exists, the prior step is replaced in-place via `_handle_step_revision`. The response includes `"is_revision": true` to signal this occurred. This is implicit revision — there is no separate `is_revision` input field or `revises_step` pointer. This is a known divergence from the MCP `sequential-thinking` reference, which uses explicit `isRevision` and `revisesThought` fields (see ADR-0013).

**Storage Topology.** Steps are stored as a flat `List[ThoughtStep]` indexed by position. The `dependencies` and `contradicts` fields create an optional primarily-linear DAG structure, but the library performs no cycle detection, topological sort, or graph traversal. For non-linear reasoning topologies, see the sibling `graph-of-thought` library (ADR-0015).

### 3.3 Tool Specifications (8 Tools)

The library exposes 8 tools via `TOOL_SPECS`:

#### 3.3.1 `chain_of_thought_step`
Add or revise a reasoning step. If `step_number` matches an existing step, it replaces it (revision). Returns progress, confidence, and contextual feedback.

**Branching is not supported.** Steps form a flat sequence; there is no `branch_from_step` or `branch_id` field. This is a known gap relative to the MCP `sequential-thinking` reference, which supports branching via `branchFromThought` and `branchId` fields (see ADR-0012).

#### 3.3.2 `get_chain_summary`
Returns: total steps, stages covered, overall confidence, confidence by stage, chain (with thought previews), insights (evidence, assumptions, contradictions), content synthesis (full thoughts grouped by stage), completion status (% of 5 canonical stages present), and metadata.

#### 3.3.3 `clear_chain`
Resets the chain. All steps discarded.

#### 3.3.4 `generate_hypotheses`
Accepts an `observation` string and optional `hypothesis_count` (1–4). Generates scientific, intuitive, contrarian, and systematic hypotheses. Ranks by testability score.

**Genealogy:** Divergent thinking rubrics (de Bono's lateral thinking, Osborn's brainstorming) and abductive reasoning (Peirce's framework for generating explanatory hypotheses). Not descended from Wei et al. 2022 CoT lineage (see ADR-0016).

#### 3.3.5 `map_assumptions`
Accepts a `statement` string and optional `depth` (`"surface"` | `"deep"`). Extracts explicit and implicit assumptions with criticality assessment and dependency graph.

**Genealogy:** Critical thinking and informal logic pedagogy (systematic identification of explicit/implicit premises) and design thinking (assumption surfacing during ideation). Not descended from Wei et al. 2022 CoT lineage (see ADR-0016).

#### 3.3.6 `calibrate_confidence`
Accepts `prediction`, `initial_confidence` (0.0–1.0), and optional `context`. Detects overconfidence patterns, applies calibration adjustment, returns uncertainty bands.

**Genealogy:** Calibration research (Lichtenstein, Fischhoff & Phillips, 1982) on overconfidence in judgment, and forecasting literature (Tetlock's superforecasting research on debiasing confidence). Not descended from Wei et al. 2022 CoT lineage (see ADR-0016).

#### 3.3.7 `export_chain`
Accepts `file_path`. Serializes all steps + metadata to JSON file.

#### 3.3.8 `import_chain`
Accepts `file_path`. Restores chain from JSON file. Validates structure, types, and constraints before restoring. Max 10,000 steps (DoS prevention).

### 3.4 Handler Architecture

All tool handlers follow the pattern:
1. Accept `**kwargs` matching the tool's input schema
2. Return a JSON string (`str`)
3. Response always contains a `"status"` key: `"success"` or `"error"`

Handlers are wired through:
- **`HANDLERS`** dict — maps tool name to handler function
- **`create_generic_handler()`** — factory that wraps service calls with rate limiting and error handling
- **`ServiceRegistry`** — dependency injection container with lazy factory-based service creation

### 3.5 Concurrency Model

| Class | Thread Safety | Use Case |
|-------|--------------|----------|
| `ChainOfThought` | Per-instance `RLock` | Single conversation |
| `ThreadAwareChainOfThought` | Class-level `RLock` + `WeakValueDictionary` + strong refs | Multi-conversation server |
| `AsyncChainOfThoughtProcessor` | Own `ChainOfThought` instance per conversation | AWS Bedrock tool loops |

`ThreadAwareChainOfThought` uses a hybrid memory model:
- Strong references for active conversations (prevents GC)
- Weak references for automatic cleanup of abandoned conversations
- Explicit `release_conversation()` to drop strong refs

### 3.6 Security Layers

1. **Input validation** (`ParameterValidator`): Type checking, XSS prevention (HTML escaping), Unicode sanitization (NFKC normalization, zero-width char removal), length limits, range validation
2. **Request validation** (`RequestValidator`): Bedrock-specific — parameter allowlisting, inference config bounds, injection pattern detection
3. **Safe JSON serialization** (`_safe_json_dumps`): Type whitelisting, sensitive key redaction, dangerous content filtering, recursion depth limiting (50), list size limiting (100), string length limiting (1000), output size limiting (100KB)
4. **Rate limiting** (`RateLimiter`): Token bucket with burst (10), per-minute (60), per-hour (1000) windows, per-client isolation

### 3.7 Async Bedrock Integration

`AsyncChainOfThoughtProcessor` orchestrates the Bedrock tool loop:
1. Validates and sanitizes the initial request via `RequestValidator`
2. Sends request to Bedrock
3. On `stopReason: "tool_use"` — executes each tool call via `StopReasonHandler`, collects results, appends to messages, loops
4. On `stopReason: "end_turn"` — returns response
5. Guards: max 20 iterations, configurable timeouts (AWS call: 30s, tool call: 10s), overall timeout wrapper

## 4. Public API

### Package-Level Exports

```python
# Tool specifications and handlers
TOOL_SPECS: List[Dict]
HANDLERS: Dict[str, Callable]

# Core classes
ChainOfThought
ThoughtStep
ThreadAwareChainOfThought
StopReasonHandler           # ABC
BedrockStopReasonHandler    # Concrete
AsyncChainOfThoughtProcessor

# Dependency injection
ServiceRegistry
get_service_registry() -> ServiceRegistry

# Handler factory
create_generic_handler(tool_name, ...) -> Callable
TOOL_HANDLERS_CONFIG: Dict

# Individual handler functions
chain_of_thought_step_handler(**kwargs) -> str
get_chain_summary_handler(**kwargs) -> str
clear_chain_handler(**kwargs) -> str
generate_hypotheses_handler(**kwargs) -> str
map_assumptions_handler(**kwargs) -> str
calibrate_confidence_handler(**kwargs) -> str
export_chain_handler(**kwargs) -> str
import_chain_handler(**kwargs) -> str

# Utilities
ParameterValidator
```

### Minimum Python Version

3.8+

### Dependencies

None. Zero external dependencies for runtime.

## 5. Configuration Constants

All limits are defined as module-level constants in `core.py`:

| Constant | Value | Purpose |
|----------|-------|---------|
| `DEFAULT_MAX_REQUESTS_PER_MINUTE` | 60 | Rate limit per client |
| `DEFAULT_MAX_REQUESTS_PER_HOUR` | 1000 | Rate limit per client |
| `DEFAULT_MAX_BURST_SIZE` | 10 | Burst limit per client |
| `MAX_RECURSION_DEPTH` | 50 | JSON serialization safety |
| `MAX_LIST_SIZE` | 100 | JSON serialization safety |
| `MAX_STRING_LENGTH` | 1000 | JSON serialization safety |
| `MAX_JSON_SIZE` | 100000 | JSON output size cap (100KB) |
| `MAX_IMPORT_STEPS` | 10000 | Import chain step cap |
| `HIGH_CONFIDENCE_THRESHOLD` | 0.15 | Calibration adjustment magnitude |
| `MEDIUM_CONFIDENCE_THRESHOLD` | 0.05 | Calibration adjustment magnitude |
| `MAX_PREDICTION_WORDS` | 20 | Confidence calibration complexity threshold |
| `MAX_THOUGHT_LENGTH` | 10000 | Thought text length cap |

## 6. Non-Functional Requirements

| Requirement | Specification |
|-------------|--------------|
| Zero dependencies | `pip install chain-of-thought-tool` works in any Python 3.8+ env with no network fetch at runtime |
| Thread safety | All shared state protected by `RLock`; per-conversation isolation guaranteed |
| Test coverage | ≥80% line coverage enforced by `pyproject.toml` |
| Package size | Single namespace package (`chain_of_thought/`), no data files |
| API stability | Tool schema (`TOOL_SPECS`) is the stability contract; internal classes may evolve |

## 7. Current State Assessment

### What Exists and Works

- 8 tool specifications in Bedrock Converse format
- 8 handler functions with rate limiting, validation, and error handling
- `ChainOfThought` core class with add/revise/summarize/clear/export/import
- `HypothesisGenerator`, `AssumptionMapper`, `ConfidenceCalibrator` auxiliary services
- `ServiceRegistry` with lazy factory-based DI
- `ThreadAwareChainOfThought` with hybrid weak/strong ref memory model
- `AsyncChainOfThoughtProcessor` with Bedrock tool loop orchestration
- `ParameterValidator` with comprehensive input sanitization
- `RequestValidator` for Bedrock-specific request validation
- `RateLimiter` with token bucket algorithm
- `_safe_json_dumps` with defense-in-depth serialization
- 321 tests, all passing
- 80% coverage floor enforced
- pyproject.toml with pytest, black, flake8, mypy configs

### Known Gaps

- No CI/CD pipeline
- No PyPI publishing automation
- No performance benchmarks
- `mypy strict` is configured but not enforced in CI
- Hypothesis/assumption/confidence tools generate template responses, not LLM-driven analysis
- No async test coverage for the Bedrock tool loop (tests mock the AWS client)

## 8. Revision History

| Date | Version | Change |
|------|---------|--------|
| 2026-05-14 | 0.3.0 | Spec written from codebase analysis |
| 2026-05-12 | — | ADR-0001: Removed model ID allowlist from security validation |
