---
id: ADR-0002
title: Zero External Dependency Philosophy
status: proposed
date: 2026-05-14
decision_makers:
  - Engineering
category:
  - architecture
supersedes: null
superseded_by: null
related: []
tags: [dependencies, compatibility, philosophy]
---

# ADR 0002: Zero External Dependency Philosophy

## Context

### The Problem
`chain-of-thought-tool` is a library that plugs into LLM function-calling APIs -- systems that already carry heavy dependency trees (AWS SDK, Anthropic SDK, OpenAI SDK, etc.). Adding runtime dependencies like pydantic, dataclasses-json, or aiohttp on top of those stacks introduces version conflict risk and enlarges the supply chain attack surface for every downstream consumer.

The library's core job is structured reasoning metadata (thought steps, confidence scores, evidence tracking, assumption mapping). This domain does not require any capability beyond what Python's standard library provides: `dataclasses` for modeling, `json` for serialization, `threading` for concurrency, `asyncio` for async patterns, `html` for sanitization, `re` for validation, `math` for numeric operations, `unicodedata` for normalization.

### Constraints
- Must support Python 3.8+ (widest practical compatibility).
- Must install without resolving any external packages.
- Must work in locked-down environments (air-gapped, Lambda layers, Docker scratch images) where `pip install` from PyPI is restricted.
- Dev dependencies (pytest, black, flake8, mypy) are allowed but must never appear in `install_requires`.

---

## Decision

The library declares **zero runtime dependencies**. All functionality is implemented using Python standard library modules only. The `install_requires` list in `setup.py` is empty.

Where third-party libraries would normally provide convenience (e.g., pydantic for validation, dataclasses-json for serialization), we write the equivalent logic by hand. Concrete examples in this codebase:

- **`_safe_json_dumps`** in `core.py` -- manual recursive JSON serialization with security sanitization (sensitive key filtering, dangerous content detection, recursion depth limiting). Without this, we would need pydantic or a custom serializer.
- **`ParameterValidator`** in `validators.py` -- manual type checking, range validation, Unicode sanitization, and XSS prevention via `html.escape()`. Without this, we would need pydantic validators.
- **`dataclasses.asdict`** for serialization of `ThoughtStep`, `Hypothesis`, `Assumption`, `ConfidenceAssessment` -- used instead of dataclasses-json or pydantic's `.dict()`.
- **`threading.RLock`** and `weakref.WeakValueDictionary` for thread-safe conversation isolation in `ThreadAwareChainOfThought` -- used instead of concurrency libraries.
- **`asyncio.to_thread`** and `asyncio.wait_for` for async wrappers in `AsyncChainOfThoughtProcessor` -- used instead of aiohttp or anyio.

### Requirements
<!-- adr:requirements -->
requirements:
  - id: REQ-0002-1
    category: architecture
    description: "install_requires must be empty in setup.py"
    verification:
      type: grep
      pattern: "install_requires"
      paths:
        - "pyproject.toml"
      expect: present
  - id: REQ-0002-2
    category: architecture
    description: "All runtime imports must be from Python stdlib"
    verification:
      type: grep
      pattern: "^(import |from )(json|html|threading|asyncio|dataclasses|typing|os|re|time|weakref|copy|abc|io)"
      paths:
        - "chain_of_thought/__init__.py"
        - "chain_of_thought/core.py"
        - "chain_of_thought/validators.py"
        - "chain_of_thought/security.py"
      expect: present
  - id: REQ-0002-3
    category: architecture
    description: "Dev dependencies declared in extras only"
    verification:
      type: grep
      pattern: "dev"
      paths:
        - "pyproject.toml"
      expect: present
  - id: REQ-0002-4
    category: architecture
    description: "Must support Python 3.8+"
    verification:
      type: grep
      pattern: "python_requires"
      paths:
        - "pyproject.toml"
      expect: present
<!-- /adr:requirements -->

---

## Alternatives Considered

### Alternative 1: pydantic for Validation and Serialization

**Approach:** Replace `ParameterValidator` and `_safe_json_dumps` with pydantic models. `ThoughtStep` would become a `BaseModel` subclass with automatic validation, schema generation, and `.model_dump_json()`.

**Pros:**
- Automatic type coercion and validation (no manual `isinstance` chains)
- JSON Schema generation for free (useful for tool spec validation)
- Less boilerplate code in validators and serialization
- Widely understood by Python developers

**Cons:**
- Adds pydantic (and its transitive dependencies: pydantic-core, annotated-types, typing-extensions) to every consumer's dependency tree
- Version conflicts with consumers already pinning pydantic v1 vs v2
- Increases install time and package size
- Breaks air-gapped deployment scenarios

**Decision:** Rejected. The dependency cost outweighs the developer convenience for a library whose value proposition is zero-friction integration.

### Alternative 2: dataclasses-json for Dataclass Serialization

**Approach:** Use `dataclasses_json` to add `.to_json()` / `.from_json()` to `ThoughtStep`, `Hypothesis`, `Assumption`, and `ConfidenceAssessment`.

**Pros:**
- Clean serialization API on dataclasses
- Handles Optional and nested types automatically
- Less manual code than `asdict()` + `json.dumps()`

**Cons:**
- Adds dataclasses-json + marshmallow + typing-extensions as dependencies
- We only need basic dict-to-JSON, not full marshmallow schema power
- `asdict()` from stdlib already handles our use case

**Decision:** Rejected. `dataclasses.asdict()` plus `json.dumps()` covers all serialization needs in this library. The added dependency provides negligible value.

### Alternative 3: aiohttp for Async HTTP

**Approach:** Use aiohttp for async HTTP in `AsyncChainOfThoughtProcessor` instead of `asyncio.to_thread` + synchronous boto3 calls.

**Pros:**
- True async HTTP without blocking thread pool
- Better throughput under high concurrency
- More idiomatic async patterns

**Cons:**
- Adds aiohttp (and its C extensions) as a dependency
- The library does not make HTTP calls itself -- it accepts a `bedrock_client` from the caller
- `asyncio.to_thread` is sufficient since the blocking I/O happens in boto3, which this library does not own

**Decision:** Rejected. This library delegates HTTP to the caller's client. Adding an HTTP dependency for a library that never makes HTTP requests would be architectural nonsense.

### Alternative 4: tenacity for Retry Logic

**Approach:** Use tenacity for retry with exponential backoff in `AsyncChainOfThoughtProcessor.process_tool_loop`.

**Pros:**
- Battle-tested retry patterns with jitter, stop conditions
- Declarative API via decorators

**Cons:**
- Adds tenacity as a dependency
- Current implementation has no retry logic -- the tool loop delegates retry semantics to the caller
- If retry is needed later, a simple stdlib loop with `asyncio.sleep` suffices for our use case

**Decision:** Rejected. No retry logic exists to justify the dependency. If retry is added in the future, stdlib `asyncio.sleep` with a loop is adequate for the bounded retry scenarios this library encounters.

---

## Consequences

### Positive
1. **Universal install** -- `pip install chain-of-thought-tool` resolves instantly with zero dependency resolution. No version conflicts with any consumer's existing stack.
2. **Minimal attack surface** -- zero third-party packages means zero third-party supply chain vectors. No risk from compromised pydantic releases, typosquatted packages, or transitive dependency vulnerabilities.
3. **Deterministic deployments** -- works in Lambda layers, Docker minimal images, air-gapped environments, and any Python 3.8+ runtime without network access.
4. **Fast CI** -- test environments install in seconds since only dev dependencies are fetched.
5. **No deprecation churn** -- stdlib modules are stable across Python versions. No risk of a dependency shipping a breaking major version.

### Negative
1. **More manual code** -- `_safe_json_dumps` is ~120 lines of manual recursive serialization with security controls that pydantic would handle in ~5 lines of model definition.
2. **No auto-schema generation** -- tool specs in `__init__.py` are hand-maintained JSON Schema objects. pydantic would generate these automatically from model type hints.
3. **No type coercion** -- `ParameterValidator` enforces strict types (`isinstance` checks) without the ergonomic coercion pydantic provides (e.g., accepting `"0.8"` as a float).
4. **More boilerplate** -- each new dataclass requires manual `asdict()` calls for serialization, manual validation in `ParameterValidator`, and manual JSON Schema in `TOOL_SPECS`.
5. **Validation is less composable** -- pydantic validators compose via model inheritance; our `ParameterValidator` is a single class that must be updated for each new field.

### Tradeoffs
<!-- adr:tradeoffs -->
```yaml
tradeoffs:
  - gain: Zero dependency resolution failures across all consumer environments
    cost: ~200 lines of manual validation and serialization code
    acceptable: true
    rationale: "The manual code is finite, testable, and rarely changes. Dependency conflicts are unbounded and unpredictable."
  - gain: Zero supply chain attack surface from third-party packages
    cost: No ecosystem features like auto-schema generation or type coercion
    acceptable: true
    rationale: "This is a tool specification library, not an application framework. Schema generation is a convenience, not a requirement."
  - gain: Works in any Python 3.8+ environment without network access
    cost: Developer ergonomics are lower than pydantic-based approaches
    acceptable: true
    rationale: "Library consumers are LLM integration engineers who value reliability over ergonomics. They chose this library specifically because it does not add to their dependency tree."
```
<!-- /adr:tradeoffs -->

---

## Approval
<!-- adr:approval -->
```yaml
approval:
  required_approvers:
    - role: Engineering
      approved: false
      date: null
  review_schedule: annually
  next_review: null
```
<!-- /adr:approval -->

---

## References
- [Python Standard Library - dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [Python Standard Library - asyncio](https://docs.python.org/3/library/asyncio.html)
- [PEP 558 -- Hooks for the PEP 558 Standard Library](https://peps.python.org/pep-0558/)
- [setup.py - install_requires: empty list](../setup.py)
- [core.py - _safe_json_dumps manual serialization](../chain_of_thought/core.py)
- [validators.py - ParameterValidator manual validation](../chain_of_thought/validators.py)
