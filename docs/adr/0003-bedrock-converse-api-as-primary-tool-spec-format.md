---
id: ADR-0003
title: Bedrock Converse API as Primary Tool Spec Format
status: accepted
date: 2026-05-14
decision_makers:
  - Engineering
category:
  - architecture
supersedes: null
superseded_by: null
related:
  - ADR-0002
  - ADR-0009
tags: [api, bedrock, tool-specs, interoperability]
---

# ADR 0003: Bedrock Converse API as Primary Tool Spec Format

## Context

### The Problem

Tool specifications must be consumable by multiple LLM providers (AWS Bedrock, OpenAI, Anthropic), but each uses a slightly different JSON schema for function calling. The library needs a single canonical representation for its `TOOL_SPECS` that avoids duplicated definitions, format drift, and maintenance overhead from maintaining parallel schemas.

The core tension: picking any provider's native format privileges that provider, while inventing a provider-agnostic intermediate format adds an abstraction layer the library must maintain and document.

### Constraints

- The library ships zero external dependencies (ADR-0002). No format conversion libraries.
- `TOOL_SPECS` is the single source of truth for tool schema definitions. No parallel arrays or generated variants.
- The primary integration path is AWS Bedrock via `AsyncChainOfThoughtProcessor` (ADR-0009), which uses the Converse API's `toolConfig.tools` shape directly.
- OpenAI and Anthropic use JSON Schema directly in their function calling payloads, which is a subset of what Bedrock's `inputSchema.json` already contains.

---

## Decision

`TOOL_SPECS` uses the AWS Bedrock Converse API tool format as its canonical representation. Each tool is a dict with the structure `{"toolSpec": {"name": str, "description": str, "inputSchema": {"json": <JSON Schema>}}}`.

No provider adapter code ships in the library. Conversion to OpenAI or Anthropic format is documented in README as a trivial inline mapping (three-field extraction: `toolSpec.name`, `toolSpec.description`, `toolSpec.inputSchema.json`). Consumers copy the mapping, not a dependency.

### Requirements

<!-- adr:requirements -->
requirements:
  - id: REQ-001
    category: architecture
    description: "TOOL_SPECS must be passable directly to bedrock.converse toolConfig without transformation"
    verification:
      type: grep
      pattern: "toolSpec"
      paths:
        - "chain_of_thought/__init__.py"
      expect: present
  - id: REQ-002
    category: architecture
    description: "Each tool spec must contain toolSpec.name, toolSpec.description, and toolSpec.inputSchema.json"
    verification:
      type: grep
      pattern: "inputSchema"
      paths:
        - "chain_of_thought/__init__.py"
      expect: present
  - id: REQ-003
    category: architecture
    description: "No provider adapter code or format conversion functions in the library"
    verification:
      type: grep_negative
      pattern: "openai|anthropic.*adapter|format.*convert"
      paths:
        - "chain_of_thought/__init__.py"
        - "chain_of_thought/core.py"
      expect: absent
  - id: REQ-004
    category: architecture
    description: "README must document the conversion mapping for non-Bedrock providers"
    verification:
      type: grep
      pattern: "openai|OpenAI"
      paths:
        - "README.md"
      expect: present
<!-- /adr:requirements -->

---

## Alternatives Considered

### Alternative 1: Provider-Agnostic Intermediate Format

**Approach:** Define a neutral tool schema (e.g., `{name, description, parameters}` at top level) and ship adapter functions for each provider.

**Pros:**
- No provider is privileged in the canonical format
- Symmetric conversion between any pair of providers

**Cons:**
- Adds code the library must maintain (violates zero-dependency simplicity)
- Every new provider requires a new adapter
- Bedrock users (the primary audience) must call an adapter on every import
- The "neutral" format would inevitably mirror one provider's schema, making the abstraction dishonest

**Decision:** Rejected. The abstraction adds maintenance burden without proportional benefit. All three provider schemas are close enough that inline conversion is a three-line loop. An adapter layer would be infrastructure for infrastructure's sake.

### Alternative 2: OpenAI Function Calling Format as Primary

**Approach:** Use `{"type": "function", "function": {"name", "description", "parameters"}}` as the canonical `TOOL_SPECS` shape, since OpenAI was first to market with function calling.

**Pros:**
- Largest existing ecosystem of examples and tooling
- Direct compatibility with OpenAI SDK without conversion

**Cons:**
- Bedrock users (the primary integration path) must convert on every call
- The `{"type": "function"}` wrapper is OpenAI-specific ceremony, not a standard
- OpenAI's `parameters` field is plain JSON Schema, identical to Bedrock's `inputSchema.json` -- the only difference is the wrapper structure
- Prioritizing first-to-market over architectural fit is a weak rationale

**Decision:** Rejected. OpenAI's format is a thin wrapper around JSON Schema, same as Bedrock's. The choice between them is which wrapper you prefer, not which schema is richer. Bedrock's wrapper is the better fit because the library's primary processor (`AsyncChainOfThoughtProcessor`) targets Bedrock natively.

### Alternative 3: Multiple Format Exports

**Approach:** Export `TOOL_SPECS_BEDROCK`, `TOOL_SPECS_OPENAI`, `TOOL_SPECS_ANTHROPIC` as separate constants.

**Pros:**
- Zero-conversion import for every supported provider
- Explicit format choice at import time

**Cons:**
- N copies of the same schema that must stay in sync
- Adding a new tool means editing N constants
- Most users only use one provider, so N-1 exports are dead code
- Violates single source of truth

**Decision:** Rejected. Format drift between the exports is inevitable. The sync problem grows linearly with tools and providers. A single canonical format with documented conversion is simpler and more reliable.

---

## Consequences

### Positive

1. **Drop-in Bedrock compatibility.** `TOOL_SPECS` passes directly to `bedrock.converse(toolConfig={"tools": TOOL_SPECS})` with zero transformation. This is the highest-value path because `AsyncChainOfThoughtProcessor` is the library's primary integration pattern.

2. **Lossless conversion to other formats.** Bedrock's `inputSchema.json` is JSON Schema. OpenAI's `parameters` is JSON Schema. Anthropic's `input_schema` is JSON Schema. The wrapper structures differ, but the schema payload is identical. Conversion is a three-field extraction with zero information loss.

3. **No adapter maintenance.** No conversion functions to test, version, or break when a provider changes their API. The README documents the mapping; consumers own it.

4. **Single source of truth.** One `TOOL_SPECS` array. Adding a new tool means one edit. No sync risk across format variants.

### Negative

1. **Non-Bedrock users must write conversion.** OpenAI and Anthropic consumers copy a three-line loop from README. This is friction, even if minimal.

2. **README bears the conversion burden.** If providers diverge their schema shapes in the future, README documentation must track those changes. The library itself does not enforce conversion correctness.

3. **Format tied to AWS schema evolution.** If AWS changes the Bedrock Converse API's `toolSpec` shape in a breaking way, `TOOL_SPECS` must change with it. The library's canonical format is not insulated from upstream API evolution.

### Tradeoffs

<!-- adr:tradeoffs -->
```yaml
tradeoffs:
  - gain: Native Bedrock drop-in and zero adapter code
    cost: Non-Bedrock users must copy a 3-line conversion from README
    acceptable: true
    rationale: >
      The library's primary processor (AsyncChainOfThoughtProcessor) targets Bedrock.
      Non-Bedrock conversion is a trivial field rename, not a semantic transform.
      Documenting the mapping is cheaper than maintaining adapter code.

  - gain: Single source of truth for tool schemas
    cost: Format is coupled to AWS Bedrock's toolSpec shape
    acceptable: true
    rationale: >
      Bedrock's toolSpec is a thin wrapper around standard JSON Schema.
      If AWS changes the wrapper, the JSON Schema payload moves unchanged.
      The coupling is to the container, not the content.

  - gain: No provider-specific code in the library
    cost: README must document conversion for every supported provider
    acceptable: true
    rationale: >
      The conversion is a one-time copy-paste per consumer project.
      It is simpler than depending on a library that might not match
      the consumer's exact provider SDK version.
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

- [AWS Bedrock Converse API - Tool use (toolConfig)](https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use.html)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- ADR-0002: Zero External Dependency Philosophy
- ADR-0009: Async Bedrock Tool Loop Orchestration Pattern
