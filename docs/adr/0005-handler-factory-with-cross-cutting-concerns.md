---
id: ADR-0005
title: Handler Factory with Cross-Cutting Concerns
status: accepted
date: 2026-05-14
decision_makers:
  - Engineering
category:
  - architecture
supersedes: null
superseded_by: null
related: []
tags: [handlers, patterns, cross-cutting]
---

# ADR 0005: Handler Factory with Cross-Cutting Concerns

## Context

### The Problem

The library exposes 8 tool handlers (`chain_of_thought_step`, `get_chain_summary`, `clear_chain`, `generate_hypotheses`, `map_assumptions`, `calibrate_confidence`, `export_chain`, `import_chain`) that all require the same cross-cutting behavior: rate limiting, input validation, service invocation, and error wrapping. Without a shared mechanism, each handler would duplicate the same try/except/rate-limit/service-lookup boilerplate. Six of the eight handlers follow an identical shape (get service from registry, call a method, return JSON). The remaining two (`export_chain`, `import_chain`) call methods directly on the `ChainOfThought` instance. Duplicating this logic across 8 callables violates DRY and guarantees drift when any cross-cutting concern changes.

### Constraints

- Handlers must return JSON strings (`{"status": "success", ...}` or `{"status": "error", ...}`) for LLM API compatibility.
- Rate limiting must be per-client and optional (injectable `RateLimiter` instance).
- The `ServiceRegistry` must be injectable for testing and per-conversation isolation.
- No external dependencies (ADR-0002).
- Thread-safe operation across multi-tenant conversations (ADR-0004).

---

## Decision

Use a single `create_generic_handler(tool_name, registry, rate_limiter, client_id)` factory function that produces handler callables. Each handler is configured via the `TOOL_HANDLERS_CONFIG` dictionary, which maps tool names to their `service_name` and `service_method`. The factory:

1. Resolves the rate limiter (injected or global singleton).
2. Returns a closure that checks rate limits first, then resolves the service from the registry, calls the configured method via `getattr`, and wraps the result in `_safe_json_dumps`.
3. Catches all exceptions and returns a uniform `{"status": "error", "message": str(e)}` JSON string.

The two file I/O handlers (`export_chain`, `import_chain`) remain standalone functions because they bypass the service registry and call `_chain_processor` directly. The convenience wrapper functions (`chain_of_thought_step_handler`, etc.) delegate to `create_generic_handler` for backward-compatible top-level callables.

### Requirements

<!-- adr:requirements -->
requirements:
  - id: REQ-0005-1
    category: architecture
    description: "create_generic_handler MUST validate tool_name against TOOL_HANDLERS_CONFIG"
    verification:
      type: grep
      pattern: "TOOL_HANDLERS_CONFIG"
      paths:
        - "chain_of_thought/core.py"
      expect: present
  - id: REQ-0005-2
    category: architecture
    description: "All handlers MUST return JSON strings with a top-level status key"
    verification:
      type: grep
      pattern: "status.*success|status.*error"
      paths:
        - "chain_of_thought/core.py"
      expect: present
  - id: REQ-0005-3
    category: architecture
    description: "Rate limiting MUST return error JSON with rate_limit_exceeded and retry_after"
    verification:
      type: grep
      pattern: "rate_limit_exceeded"
      paths:
        - "chain_of_thought/core.py"
      expect: present
  - id: REQ-0005-4
    category: architecture
    description: "Factory MUST accept optional registry and rate_limiter for dependency injection"
    verification:
      type: grep
      pattern: "create_generic_handler"
      paths:
        - "chain_of_thought/core.py"
      expect: present
  - id: REQ-0005-5
    category: architecture
    description: "Exceptions during service invocation MUST be caught and returned as error JSON"
    verification:
      type: grep
      pattern: "status.*error"
      paths:
        - "chain_of_thought/core.py"
      expect: present
<!-- /adr:requirements -->

---

## Alternatives Considered

### Alternative 1: Per-Handler Implementation

**Approach:** Write each of the 6 (or 8) handlers as a standalone function containing its own rate-limit check, registry lookup, method call, and error handling.

**Pros:**
- Each handler is self-contained and readable in isolation.
- No indirection; stack traces point directly to the handler.

**Cons:**
- 6 copies of identical rate-limit + try/except + JSON serialization logic.
- Any change to cross-cutting behavior requires editing all 6 functions.
- High risk of drift: one handler gets a fix, another does not.

**Decision:** Rejected. DRY violation is unacceptable for cross-cutting concerns that must stay in lockstep.

### Alternative 2: Decorator Chain

**Approach:** Stack decorators: `@rate_limit @error_handler @validate` over a bare service call function.

**Pros:**
- Pythonic; each concern is a composable decorator.
- Ordering is explicit in the decorator stack.

**Cons:**
- Debugging requires unwinding the decorator chain to find where an error originates.
- `functools.wraps` metadata can mask the real function in tracebacks.
- Harder to inject per-handler configuration (which service, which method) through decorators alone.
- Less readable for contributors unfamiliar with decorator composition patterns.

**Decision:** Rejected. The decorator pattern is elegant but obscures the control flow. A single closure is easier to step through.

### Alternative 3: Middleware Pipeline

**Approach:** Build a request/response pipeline with middleware objects (rate limit middleware, validation middleware, error middleware, etc.).

**Pros:**
- Extensible: add new middleware without touching existing code.
- Industry-standard pattern (Django, Express, WSGI).

**Cons:**
- Vastly overengineered for 8 handlers in a zero-dependency library.
- Introduces abstract base classes, registration, and ordering concerns.
- Violates YAGNI given the current and foreseeable handler count.

**Decision:** Rejected. The complexity budget does not justify a middleware framework for a library with fewer than 10 handlers.

### Alternative 4: Base Class with Template Method

**Approach:** Abstract `BaseHandler` with `execute()` template method; subclasses override `get_service()`, `validate()`, and `run()`.

**Pros:**
- Familiar OOP pattern; clear extension points.
- Inheritance provides shared state naturally.

**Cons:**
- Composition is preferred over inheritance in Python.
- Each handler would need its own class file for a single method override.
- The factory function already provides composition-based reuse without class hierarchies.

**Decision:** Rejected. Composition via closures is simpler and avoids class proliferation.

---

## Consequences

### Positive

1. **Single source of truth** for cross-cutting behavior. Rate limiting, error handling, and JSON serialization change in exactly one place.
2. **Consistent error format.** Every handler returns `{"status": "error", "message": "..."}` on failure, which LLM consumers can parse uniformly.
3. **Easy handler addition.** Adding a new tool requires one entry in `TOOL_HANDLERS_CONFIG` and zero new functions.
4. **Testability via injection.** The `registry` and `rate_limiter` parameters allow full unit testing without global state.
5. **Backward compatibility.** Existing top-level handler functions (`chain_of_thought_step_handler`, etc.) are thin wrappers that delegate to the factory.

### Negative

1. **Handler callable not cached.** Each call to `create_generic_handler` creates a new closure. The convenience wrappers call the factory on every invocation, not at module load. This is a minor allocation cost but could be optimized with `functools.lru_cache` or module-level instantiation if profiling shows it matters.
2. **Uniform error format.** All errors surface as `{"status": "error", "message": str(e)}`. This loses exception type information. Callers who need to distinguish `KeyError` from `ValueError` must parse the message string.
3. **Indirection layer.** Tracing a handler call requires understanding the factory, `TOOL_HANDLERS_CONFIG`, and `getattr` dispatch. Contributors must look up the config to find which service and method a tool name maps to.

### Tradeoffs

<!-- adr:tradeoffs -->
```yaml
tradeoffs:
  - gain: Single place to modify rate limiting, error handling, and serialization
    cost: One level of indirection between tool name and service method
    acceptable: true
    rationale: >
      The indirection is a named config dictionary, not hidden magic.
      Contributors can grep TOOL_HANDLERS_CONFIG to find any mapping.
  - gain: Zero code duplication across 6+ handlers
    cost: All handlers share the same error response shape
    acceptable: true
    rationale: >
      Uniform error format is a feature for LLM consumers, not a limitation.
      If a handler needs custom error structure, it can bypass the factory.
  - gain: Dependency injection for testing and isolation
    cost: Convenience wrappers allocate a new closure per call
    acceptable: true
    rationale: >
      Closure allocation is cheap. If profiling reveals it as a bottleneck,
      caching is a localized optimization that does not change the API.
```
<!-- /adr:tradeoffs -->

---

## Approval

<!-- adr:approval -->
```yaml
approval:
  required_approvers:
    - role: Engineering
      approved: true
      date: "2026-05-14"
  review_schedule: annually
  next_review: null
```
<!-- /adr:approval -->

---

## References

- `chain_of_thought/core.py`: `create_generic_handler`, `TOOL_HANDLERS_CONFIG`
- `chain_of_thought/handlers.py`: Convenience wrapper functions
- ADR-0002: Zero External Dependency Philosophy (no third-party DI frameworks)
- ADR-0004: WeakValueDictionary Hybrid for Multi-Tenant Isolation (per-conversation handler injection)
