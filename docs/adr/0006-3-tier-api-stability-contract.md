---
id: ADR-0006
title: 3-Tier API Stability Contract
status: proposed
date: 2026-05-14
decision_makers:
  - Engineering
category:
  - architecture
supersedes: null
superseded_by: null
related: []
tags: [api, versioning, stability]
---

# ADR 0006: 3-Tier API Stability Contract

## Context

### The Problem

This library has two distinct consumers: LLMs that call tools through structured schemas, and Python code that instantiates classes and calls methods directly. These consumers have different stability needs. LLM-facing tool schemas must remain stable across versions because changing a tool name or input field breaks every integration that passes `TOOL_SPECS` to a model. Python-facing class APIs need room to evolve as the library matures through alpha.

Applying full semver rigor to every symbol would freeze development velocity. Treating everything as internal except `TOOL_SPECS` would alienate consumers who legitimately use `ChainOfThought` or `ThreadAwareChainOfThought` directly. An ad-hoc approach leads to accidental breakage and unclear upgrade expectations.

### Constraints

- Current version is 0.3.0 (alpha). Breaking changes are expected but should be predictable.
- `TOOL_SPECS` defines the LLM contract. This is the primary integration surface.
- Some consumers use the Python class API directly (`ChainOfThought`, `ThreadAwareChainOfThought`, `AsyncChainOfThoughtProcessor`).
- Internal plumbing (service registry, rate limiter, sanitization) must be free to change without deprecation cycles.

---

## Decision

Establish a three-tier API stability contract. Each tier defines what breaking changes require and which symbols it covers.

**Tier 1 -- Stable (major version bump required for breaks)**

The LLM-facing contract. These symbols define what models interact with and how tool results flow back:

- `TOOL_SPECS` list structure and each `toolSpec` shape
- Tool names: `chain_of_thought_step`, `get_chain_summary`, `clear_chain`, `generate_hypotheses`, `map_assumptions`, `calibrate_confidence`, `export_chain`, `import_chain`
- `HANDLERS` dict interface: `str -> Callable`, keyed by tool name
- Tool input schemas (required fields, field types, enum values)
- Tool output structure: JSON with `"status"` key at minimum

Breaking changes to any of these require a major version bump.

**Tier 2 -- Evolving (minor version bump required for breaks)**

The Python class API used by direct consumers:

- `ChainOfThought` public methods: `add_step`, `generate_summary`, `clear_chain`, `export_chain`, `import_chain`
- `ThoughtStep` dataclass fields (existing fields will not be removed; new fields may be added with defaults)
- `ThreadAwareChainOfThought` methods: `for_conversation`, `clear_conversation`, `clear_all_conversations`, `release_conversation`, `get_cached_conversation_count`
- `AsyncChainOfThoughtProcessor` interface: `process_tool_loop`, `process_tool_loop_with_timeout`, `get_reasoning_summary`, `clear_reasoning`, constructor signature
- `ParameterValidator` public methods
- `BedrockStopReasonHandler` constructor and public methods
- Handler functions: `chain_of_thought_step_handler`, `get_chain_summary_handler`, `clear_chain_handler`, `generate_hypotheses_handler`, `map_assumptions_handler`, `calibrate_confidence_handler`, `export_chain_handler`, `import_chain_handler`
- `create_generic_handler` function signature
- `ServiceRegistry` public methods: `register_service`, `register_factory`, `get_service`, `has_service`, `clear_service`, `clear_all_services`

Breaking changes require a minor version bump. Additive changes (new methods, new fields with defaults) are non-breaking.

**Tier 3 -- Internal (may change without notice)**

Implementation details that consumers should not depend on:

- `ServiceRegistry` internals: `_services`, `_factories`, `_lock`, `_register_default_factories`
- `RateLimiter` implementation: internal data structures, algorithm details, constructor defaults
- `_safe_json_dumps` internals: sanitization logic, `SAFE_TYPES`, `SENSITIVE_KEYS`, `DANGEROUS_PATTERNS`
- Handler factory internals: `TOOL_HANDLERS_CONFIG` dict structure and values
- Module-level constants: `DEFAULT_MAX_REQUESTS_PER_MINUTE`, `DEFAULT_MAX_REQUESTS_PER_HOUR`, `DEFAULT_MAX_BURST_SIZE`, `MAX_RECURSION_DEPTH`, `MAX_LIST_SIZE`, `MAX_STRING_LENGTH`, `MAX_JSON_SIZE`, `MAX_IMPORT_STEPS`, `HIGH_CONFIDENCE_THRESHOLD`, `MEDIUM_CONFIDENCE_THRESHOLD`, `MAX_PREDICTION_WORDS`
- `HypothesisGenerator`, `AssumptionMapper`, `ConfidenceCalibrator` internal classes and their private methods
- Global instance variables: `_chain_processor`, `_hypothesis_generator`, `_assumption_mapper`, `_confidence_calibrator`, `_default_registry`
- `get_global_rate_limiter`, `set_global_rate_limiter` functions
- Individual `create_*_handler` convenience wrappers (the underlying `create_generic_handler` is Tier 2)

No version bump obligation for changes to Tier 3 symbols.

### Requirements

<!-- adr:requirements -->
requirements:
  - id: REQ-001
    category: architecture
    description: "TOOL_SPECS tool names must not change within a major version"
    verification:
      type: grep
      pattern: "chain_of_thought_step|get_chain_summary|clear_chain|generate_hypotheses|map_assumptions|calibrate_confidence|export_chain|import_chain"
      paths:
        - "chain_of_thought/__init__.py"
      expect: present
  - id: REQ-002
    category: architecture
    description: "TOOL_SPECS input schemas must remain stable"
    verification:
      type: grep
      pattern: "inputSchema"
      paths:
        - "chain_of_thought/__init__.py"
      expect: present
  - id: REQ-003
    category: architecture
    description: "HANDLERS dict must map tool name strings to callables"
    verification:
      type: grep
      pattern: "HANDLERS"
      paths:
        - "chain_of_thought/__init__.py"
      expect: present
  - id: REQ-004
    category: architecture
    description: "Tier 2 public method signatures must not remove parameters"
    verification:
      type: grep
      pattern: "def add_step|def get_summary|def clear|def export_chain|def import_chain"
      paths:
        - "chain_of_thought/core.py"
      expect: present
  - id: REQ-005
    category: architecture
    description: "ThoughtStep fields must not be removed; new fields must have defaults"
    verification:
      type: grep
      pattern: "class ThoughtStep"
      paths:
        - "chain_of_thought/core.py"
      expect: present
  - id: REQ-006
    category: architecture
    description: "Tier 3 symbols carry no stability guarantee"
    verification:
      type: grep
      pattern: "ServiceRegistry|RateLimiter|_safe_json_dumps"
      paths:
        - "chain_of_thought/core.py"
      expect: present
<!-- /adr:requirements -->

---

## Alternatives Considered

### Alternative 1: Full semver for every public symbol

**Approach:** Every public class, method, function, and constant follows strict semver. Any breaking change to any symbol requires a major version bump.

**Pros:**
- Maximum consumer safety
- No ambiguity about what might break

**Cons:**
- Freezes internal refactoring behind major version bumps
- During 0.x alpha, this effectively prevents any API evolution
- High maintenance burden for symbols few consumers actually depend on

**Decision:** Rejected. Overly rigid for an alpha-stage library. Would force either premature 1.0 or perpetual version 0.x with no stability promises.

### Alternative 2: Everything internal except TOOL_SPECS

**Approach:** Only `TOOL_SPECS` and `HANDLERS` are stable. All classes, methods, and functions are internal and may change at any time.

**Pros:**
- Maximum implementation freedom
- Simple mental model

**Cons:**
- Consumers using `ChainOfThought` or `ThreadAwareChainOfThought` directly have zero predictability
- Forces all consumers to go through the handler layer even for programmatic use
- Unnecessarily hostile to legitimate direct usage patterns

**Decision:** Rejected. Too restrictive. Consumers have legitimate reasons to instantiate classes directly, and they deserve predictable upgrade paths.

### Alternative 3: Ad-hoc versioning ("we'll figure it out")

**Approach:** No formal tier system. Use judgment on a case-by-case basis for each change.

**Pros:**
- No upfront commitment
- Maximum flexibility

**Cons:**
- Accidental breaking changes in tool specs
- No way for consumers to reason about upgrade risk
- Inconsistent decisions depending on who makes the change

**Decision:** Rejected. The whole point of this ADR is to replace ad-hoc judgment with a written contract.

### Alternative 4: Full backward compatibility for all public symbols

**Approach:** Every symbol in `__all__` is guaranteed stable. Deprecation cycles required for any removal.

**Pros:**
- Strongest possible consumer guarantee
- Industry standard for mature libraries

**Cons:**
- `__all__` currently exports 25 symbols including internals like `TOOL_HANDLERS_CONFIG`
- Would lock down implementation details that should evolve freely
- Deprecation cycles slow development during alpha

**Decision:** Rejected. Appropriate for 1.0+, but premature for 0.x. The tier system lets us graduate symbols to stronger guarantees as the API stabilizes.

---

## Consequences

### Positive

1. LLM integrations are safe by default. `TOOL_SPECS` stability means no integration breaks when upgrading within a major version.
2. Clear upgrade expectations. Consumers know exactly which symbols are safe to depend on and what version bumps mean for each tier.
3. Internals can evolve freely. Service registry, rate limiter, sanitization, and constants can be refactored without deprecation overhead.
4. Path to 1.0 is clear. As the library matures, Tier 2 symbols can be promoted to Tier 1, and Tier 3 symbols can be promoted to Tier 2 or removed from `__all__`.
5. The tier system documents itself. New contributors can look at this ADR to understand which symbols require version bump coordination.

### Negative

1. Consumers using Tier 2 APIs face possible breakage on minor version bumps during 0.x. They must pin their version or accept occasional migration work.
2. Tier boundary judgments are subjective. A symbol's tier assignment requires engineering judgment about how many consumers depend on it. Misclassification risk exists.
3. Tier 3 symbols appear in `__all__` today (e.g., `TOOL_HANDLERS_CONFIG`, `get_service_registry`). This creates a tension between the export list and the stability contract. Consumers may reasonably assume `__all__` means "public and stable."

### Tradeoffs

<!-- adr:tradeoffs -->
```yaml
tradeoffs:
  - gain: "TOOL_SPECS never break within a major version"
    cost: "Tier 2/3 APIs may break on minor versions"
    acceptable: true
    rationale: "TOOL_SPECS is the actual LLM contract. Python class APIs are secondary integration paths used by fewer consumers, who can pin versions."

  - gain: "Internals can refactor freely"
    cost: "Consumers who reach into internals may break on any version"
    acceptable: true
    rationale: "Depending on implementation details is always at-your-own-risk. Tier 3 symbols should not be in __all__ long-term."

  - gain: "Clear versioning semantics for a 0.x library"
    cost: "More complex than 'everything is unstable until 1.0'"
    acceptable: true
    rationale: "The LLM-facing contract is stable enough to promise even during alpha. The Python API gets reasonable but not ironclad guarantees."
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

- [ADR-0003: Bedrock Converse API as Primary Tool Spec Format](0003-bedrock-converse-api-as-primary-tool-spec-format.md) -- defines why TOOL_SPECS uses the Converse API shape
- [ADR-0005: Handler Factory with Cross-Cutting Concerns](0005-handler-factory-with-cross-cutting-concerns.md) -- defines the handler factory that sits behind HANDLERS
- `chain_of_thought/__init__.py` -- `TOOL_SPECS`, `HANDLERS`, `__all__`, `__version__`
- `chain_of_thought/core.py` -- class implementations and public method signatures
