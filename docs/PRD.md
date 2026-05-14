# Product Requirements Document: chain-of-thought-tool

**Version:** 0.3.0
**Status:** Active
**Date:** 2026-05-14
**Supersedes:** N/A (first PRD)

---

## 1. Product Vision

**One sentence:** A zero-dependency Python library that gives any LLM function-calling API structured chain-of-thought reasoning with confidence scoring, evidence tracking, and contradiction detection — no server, no framework, no inference.

**Target user:** Python developer integrating LLM function calling (AWS Bedrock, OpenAI, Anthropic) who needs the model to reason in auditable, structured steps rather than unstructured free-text.

**Differentiator:** Tool specs + handler functions as a drop-in import. No MCP server, no LangChain dependency, no separate process. The library is the tool, not the infrastructure.

## 2. Problem

LLM function-calling APIs provide tool-use but no built-in mechanism for structured multi-step reasoning. Developers who want:

- Confidence scores per reasoning step
- Evidence collection and assumption tracking
- Contradiction detection between steps
- Stage-aware reasoning (Problem Definition → Research → Analysis → Synthesis → Conclusion)
- Export/import of reasoning chains

must build it ad-hoc in every integration. This library provides it as a `pip install`.

## 3. User Stories

### Primary (Must Ship)

| ID | Story | Acceptance Criteria |
|----|-------|-------------------|
| US-1 | As a developer, I want to add CoT tools to my Bedrock Converse request so the model reasons in structured steps. | `TOOL_SPECS` drops into `toolConfig.tools`. Model calls `chain_of_thought_step` and receives structured feedback. |
| US-2 | As a developer, I want to retrieve a full reasoning summary so I can audit the model's chain. | `get_chain_summary` returns: total steps, stages covered, confidence by stage, evidence/assumptions/contradictions collected, completion status against 5 canonical stages, and full thought synthesis grouped by stage. |
| US-3 | As a developer, I want to reset the chain between conversations. | `clear_chain` discards all steps and resets metadata. |
| US-4 | As a developer, I want to persist and restore reasoning chains. | `export_chain` writes JSON to disk; `import_chain` reads and validates it, restoring the full chain with all metadata. |
| US-5 | As a developer running a server, I want per-conversation isolation so multiple users don't share state. | `ThreadAwareChainOfThought` provides isolated `ChainOfThought` instances per conversation ID with thread-safe access and automatic cleanup via weak references. |
| US-6 | As a developer, I want the model's tool calls rate-limited so one conversation can't starve others. | `RateLimiter` enforces burst (10), per-minute (60), per-hour (1000) limits per client with configurable retry-after. |

### Secondary (Ships but acknowledged as template-level)

| ID | Story | Acceptance Criteria |
|----|-------|-------------------|
| US-7 | As a developer, I want the model to generate diverse hypotheses for an observation. | `generate_hypotheses` returns 1–4 hypotheses (scientific, intuitive, contrarian, systematic) ranked by testability score. |
| US-8 | As a developer, I want the model to surface hidden assumptions in a statement. | `map_assumptions` returns explicit and implicit assumptions with criticality assessment and dependency graph. Surface and deep analysis modes. |
| US-9 | As a developer, I want the model to calibrate its confidence estimates. | `calibrate_confidence` detects overconfidence patterns, applies adjustment, and returns uncertainty bands. |

### Tertiary (Integration)

| ID | Story | Acceptance Criteria |
|----|-------|-------------------|
| US-10 | As a developer using AWS Bedrock, I want an async processor that handles the full tool loop. | `AsyncChainOfThoughtProcessor` manages the stopReason loop: sends request, handles tool_use by executing handlers, appends results, loops until end_turn or max iterations (20). Timeouts on AWS calls (30s) and tool calls (10s). |
| US-11 | As a developer, I want tool specs compatible with OpenAI/Anthropic function calling. | `TOOL_SPECS` are in Bedrock Converse format but trivially convertible to OpenAI function format (documented in README). No provider-specific code in the library. |

## 4. Functional Requirements

### 4.1 Core Reasoning Engine

| Req ID | Requirement | Priority | Status |
|--------|------------|----------|--------|
| FR-1 | Accept a thought step with: thought text, step number, total steps, next_step_needed, reasoning_stage, confidence, dependencies, contradicts, evidence, assumptions | P0 | Implemented |
| FR-2 | Revise existing steps when step_number matches | P0 | Implemented |
| FR-3 | Generate contextual feedback per step (stage guidance, confidence warnings, dependency/contradiction notes, progress alerts) | P1 | Implemented |
| FR-4 | Track 5 canonical reasoning stages: Problem Definition, Research, Analysis, Synthesis, Conclusion | P0 | Implemented |
| FR-5 | Calculate average confidence across all steps | P1 | Implemented |
| FR-6 | Calculate per-stage average confidence | P1 | Implemented |
| FR-7 | Detect contradiction pairs between steps | P1 | Implemented |
| FR-8 | Compute completion status: percentage of 5 canonical stages present, list of missing stages | P1 | Implemented |
| FR-9 | Provide full content synthesis: thought text grouped by stage | P1 | Implemented |

### 4.2 Auxiliary Reasoning Tools

| Req ID | Requirement | Priority | Status |
|--------|------------|----------|--------|
| FR-10 | Generate 1–4 hypotheses of types: scientific, intuitive, contrarian, systematic | P2 | Template-level |
| FR-11 | Rank hypotheses by testability score | P2 | Template-level |
| FR-12 | Extract explicit assumptions via linguistic pattern matching | P2 | Template-level |
| FR-13 | Identify implicit assumptions via keyword-based heuristics | P2 | Template-level |
| FR-14 | Flag critical assumptions based on confidence and causal reasoning | P2 | Template-level |
| FR-15 | Build dependency graph between assumptions | P2 | Template-level |
| FR-16 | Detect overconfidence via absolute language, future prediction, complexity, and domain-specific indicators | P2 | Implemented (heuristic) |
| FR-17 | Apply confidence calibration adjustment (max 30% reduction) | P2 | Implemented |
| FR-18 | Calculate uncertainty bands around calibrated confidence | P2 | Implemented |

### 4.3 Concurrency and Isolation

| Req ID | Requirement | Priority | Status |
|--------|------------|----------|--------|
| FR-19 | Per-instance `RLock` on `ChainOfThought` for thread-safe step operations | P0 | Implemented |
| FR-20 | Class-level conversation isolation via `ThreadAwareChainOfThought` with weak + strong reference hybrid | P0 | Implemented |
| FR-21 | Automatic GC of abandoned conversations via `WeakValueDictionary` | P1 | Implemented |
| FR-22 | Explicit `release_conversation()` to drop strong refs without destroying data | P1 | Implemented |
| FR-23 | Per-conversation `ChainOfThought` instance in `AsyncChainOfThoughtProcessor` | P0 | Implemented |

### 4.4 Security

| Req ID | Requirement | Priority | Status |
|--------|------------|----------|--------|
| FR-24 | Input validation: type checking, XSS prevention (HTML escaping), Unicode sanitization (NFKC + zero-width char removal), length limits | P0 | Implemented |
| FR-25 | Range validation: confidence 0.0–1.0, step_number 1–1000, list sizes max 50 items | P0 | Implemented |
| FR-26 | Safe JSON serialization: type whitelisting, sensitive key redaction, dangerous content filtering, recursion depth limit (50), output size limit (100KB) | P0 | Implemented |
| FR-27 | Rate limiting: token bucket with burst (10), per-minute (60), per-hour (1000) per client | P0 | Implemented |
| FR-28 | Bedrock request validation: parameter allowlisting, inference config bounds, injection pattern detection | P1 | Implemented |
| FR-29 | No model ID policing (ADR-0001: wrong-layer enforcement removed) | P0 | Implemented |

### 4.5 Serialization

| Req ID | Requirement | Priority | Status |
|--------|------------|----------|--------|
| FR-30 | Export chain to JSON file with all steps and metadata | P1 | Implemented |
| FR-31 | Import chain from JSON file with full validation (structure, types, constraints) | P1 | Implemented |
| FR-32 | DoS prevention on import: max 10,000 steps | P1 | Implemented |

### 4.6 Async Bedrock Integration

| Req ID | Requirement | Priority | Status |
|--------|------------|----------|--------|
| FR-33 | Orchestrate full Bedrock tool loop: validate request → send → handle stopReason → execute tools → loop | P1 | Implemented |
| FR-34 | Max 20 iterations per tool loop | P1 | Implemented |
| FR-35 | Configurable timeouts: AWS call (30s), tool call (10s), overall (2x AWS timeout) | P1 | Implemented |
| FR-36 | Timeout-protected tool results returned as error status (not exception) | P1 | Implemented |

## 5. Non-Functional Requirements

| NFR ID | Requirement | Target | Current |
|--------|------------|--------|---------|
| NFR-1 | Zero external dependencies | 0 runtime deps | Met |
| NFR-2 | Python version compatibility | 3.8+ | Met |
| NFR-3 | Test coverage | ≥80% line coverage | Met (321 tests, enforced by pyproject.toml) |
| NFR-4 | Thread safety | All shared state under RLock | Met |
| NFR-5 | Package size | Single namespace package, no data files | Met |
| NFR-6 | API stability contract | TOOL_SPECS schema is stable; internal classes may evolve | Documented |
| NFR-7 | Code quality enforcement | black, flake8, mypy strict configured | Configured but not enforced in CI |
| NFR-8 | No `python` command dependency | Must work with `python3` | Met |

## 6. Product Boundaries

### What This Product IS

- A library of tool specifications and handler functions
- A reasoning state machine that tracks steps, confidence, and relationships
- A thread-safe container for multi-conversation isolation
- An async orchestrator for the Bedrock tool loop
- A security layer that validates and sanitizes all inputs

### What This Product IS NOT

- An LLM inference engine (never calls a model API directly)
- A persistence layer (no database, no network storage)
- A visualization tool
- An authentication/authorization system
- A framework adapter (no LangChain/LlamaIndex wrappers)

### Honest Capability Assessment

| Capability | Claimed | Actual |
|-----------|---------|--------|
| Structured step tracking | Yes | Yes — fully functional |
| Confidence scoring | Yes | Yes — numeric tracking works |
| Evidence/assumption collection | Yes | Yes — string list tracking works |
| Contradiction detection | Yes | Partial — tracks cross-references but doesn't detect semantic contradictions |
| Hypothesis generation | "Diverse hypotheses" | Template strings — observation interpolated into canned responses |
| Assumption mapping | "Surface hidden assumptions" | Keyword matching heuristics — finds indicators in text but not semantic understanding |
| Confidence calibration | "Detect overconfidence" | Heuristic pattern matching — flags absolute language, future predictions, domain keywords |

The auxiliary tools (hypotheses, assumptions, calibration) are useful as scaffolding for LLM reasoning but do not perform genuine analysis. They return structured responses that give the model a framework to reason within. This is a feature, not a bug — the LLM is the analyst; the library is the scaffolding.

## 7. Tool Inventory

| Tool | Input Schema | Output Contract | Purpose |
|------|-------------|-----------------|---------|
| `chain_of_thought_step` | thought (required), step_number, total_steps, next_step_needed, reasoning_stage, confidence, dependencies, contradicts, evidence, assumptions | status, step_processed, progress, confidence, feedback, next_step_needed, total_steps_recorded, is_revision | Core reasoning unit |
| `get_chain_summary` | (none) | total_steps, stages_covered, overall_confidence, confidence_by_stage, chain (previews), insights, content_synthesis, completion_status, metadata | Audit and review |
| `clear_chain` | (none) | status, message | Reset state |
| `generate_hypotheses` | observation (required), hypothesis_count (1–4) | hypotheses (ranked), insights | Divergent thinking scaffold |
| `map_assumptions` | statement (required), depth (surface/deep) | explicit, implicit, critical, graph, insights | Critical thinking scaffold |
| `calibrate_confidence` | prediction (required), initial_confidence (required), context | calibrated_confidence, confidence_band, adjustment, overconfidence_analysis, uncertainty_factors | Metacognition scaffold |
| `export_chain` | file_path (required) | status, steps_exported | Persistence |
| `import_chain` | file_path (required) | status, steps_imported | Persistence |

## 8. Public API Contract

### Stability Tiers

**Tier 1 — Stable (breaking changes require major version bump):**
- `TOOL_SPECS` structure and tool names
- `HANDLERS` dict interface (name → callable)
- Tool input/output schemas as documented in TOOL_SPECS

**Tier 2 — Evolving (breaking changes require minor version bump):**
- `ChainOfThought` public methods
- `ThoughtStep` dataclass fields
- `ThreadAwareChainOfThought` class methods
- `AsyncChainOfThoughtProcessor` interface
- `ParameterValidator` public methods

**Tier 3 — Internal (may change without notice):**
- `ServiceRegistry` internals
- `RateLimiter` implementation details
- `_safe_json_dumps` internals
- Handler factory internals
- Module-level constants (may change between minor versions)

### Minimum Viable API Surface

```python
# This must always work:
from chain_of_thought import TOOL_SPECS, HANDLERS

result = HANDLERS["chain_of_thought_step"](
    thought="Thinking about X",
    step_number=1,
    total_steps=3,
    next_step_needed=True
)
# result is a JSON string with {"status": "success", ...}
```

## 9. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Install simplicity | `pip install chain-of-thought-tool` works with zero config | Manual verification per release |
| Import simplicity | `from chain_of_thought import TOOL_SPECS, HANDLERS` works | CI test |
| Zero dependencies | `pip install` downloads exactly one package | `pip check` passes |
| Test coverage | ≥80% line coverage | pytest --cov enforced |
| Thread safety | No data corruption under concurrent access | Thread safety test suite |
| API compatibility | TOOL_SPECS drops into Bedrock Converse `toolConfig.tools` | Integration test |
| Security | No XSS, injection, or DoS vectors in handler inputs | Security test suite |

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Auxiliary tools perceived as AI-driven analysis | High | Reputation | Document honestly that they are template/heuristic scaffolding, not LLM inference |
| `core.py` becomes a god module (2288 lines) | High | Maintainability | Future ADR should address module decomposition |
| README only documents 3 of 8 tools | Medium | Adoption | Update README to cover all 8 tools |
| No CI/CD pipeline | High | Quality | GitHub Actions workflow needed |
| No PyPI publishing automation | Medium | Distribution | CI/CD pipeline should include publish step |
| `mypy strict` configured but not enforced | Low | Type safety | Add to CI pipeline |
| Generic handler factory creates new handler per call | Medium | Performance | Cache handlers or refactor call pattern |

## 11. Out of Scope (Explicitly)

These are intentionally excluded from the product:

- **LLM inference** — The library never calls a model. It provides tools for models to call.
- **Persistence** — No database, no Redis, no network storage. Export/import to local files only.
- **Visualization** — No UI, no graphs, no dashboards.
- **Auth/authz** — No user management. Caller is responsible for access control.
- **Framework adapters** — No LangChain, LlamaIndex, or Semantic Kernel wrappers. The conversion is trivial (documented in README).
- **Semantic analysis** — Contradiction detection, hypothesis generation, and assumption mapping are structural/heuristic, not NLP-driven.
- **Provider-specific code beyond Bedrock** — OpenAI/Anthropic format conversion is documented but not wrapped.

## 12. Dependencies

### Runtime
None. Zero external dependencies. Python 3.8+ stdlib only.

### Development
- pytest ≥7.0 (test runner)
- pytest-cov ≥4.0 (coverage reporting)
- pytest-mock ≥3.10 (mocking)
- black ≥23.0 (formatting)
- flake8 ≥6.0 (linting)
- mypy ≥1.0 (type checking)
- build ≥0.10 (packaging)
- twine ≥4.0 (PyPI upload)

## 13. Revision History

| Date | Version | Change |
|------|---------|--------|
| 2026-05-14 | 0.3.0 | PRD written from codebase analysis |
