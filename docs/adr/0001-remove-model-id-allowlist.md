---
id: ADR-0001
title: Remove model ID allowlist from security validation
status: accepted
date: 2026-05-12
decision_makers:
  - Engineering
category:
  - architecture
supersedes: null
superseded_by: null
related: []
tags: [security, validation, bedrock]
---

# ADR-0001: Remove model ID allowlist from security validation

## Context

### The Problem

The `RequestValidator` in `security.py` maintained a hardcoded allowlist of Bedrock model ID patterns that only matched Anthropic Claude 3 and Claude 3.5 Sonnet models. This allowlist rejected:

- Cross-region inference profiles (`us.`, `eu.`, `apac.` prefixed)
- Any Anthropic model newer than Claude 3.5 Sonnet (Opus 4, Sonnet 4.5, Haiku 4.5, Opus 4.7)
- All non-Anthropic Bedrock providers (Meta Llama, DeepSeek, Amazon, Cohere, Mistral, AI21)

The error message ("Security validation failed") implied a security boundary was crossed when the caller simply chose a model not in the static list.

---

## Decision

Remove the model ID allowlist entirely. `_validate_model_id` now accepts any non-empty string and trusts the caller. AWS Bedrock rejects invalid model IDs at the API layer with accurate error messages.

### Requirements

<!-- adr:requirements -->
requirements:
  - id: REQ-0001-1
    category: architecture
    description: "_validate_model_id accepts any non-empty string"
    verification:
      type: grep_negative
      pattern: "allowed_model_patterns"
      paths:
        - "chain_of_thought/security.py"
      expect: absent
  - id: REQ-0001-2
    category: architecture
    description: "allowed_model_patterns field removed from SecurityConfig"
    verification:
      type: grep_negative
      pattern: "allowed_model_patterns"
      paths:
        - "chain_of_thought/security.py"
      expect: absent
<!-- /adr:requirements -->

---

## Alternatives Considered

### Alternative 1: Expand the allowlist to cover more models

**Approach:** Add patterns for cross-region profiles, newer Claude models, and other Bedrock providers.

**Pros:**
- Retains allowlist as a security boundary

**Cons:**
- Treadmill: every new model release requires a library update
- Cannot predict future model naming conventions
- LLM provider landscape changes faster than library release cadence

**Decision:** Rejected. The maintenance burden grows without bound.

### Alternative 2: Configurable allowlist

**Approach:** Let callers provide their own allowed model patterns via `SecurityConfig`.

**Pros:**
- Caller retains control over model authorization
- No library updates needed for new models

**Cons:**
- Pushes configuration burden to every consumer
- Most consumers don't need model-level authorization at the library layer
- AWS IAM already provides this control

**Decision:** Rejected. Wrong layer for this configuration.

---

## Consequences

### Positive

1. Any non-empty string model ID is accepted through the validator
2. AWS Bedrock remains the authoritative rejection point for invalid model IDs
3. Downstream consumers (devil-advocate-mcp, etc.) no longer need custom SecurityConfig overrides to use modern models
4. Library no longer breaks silently on every new Bedrock model release

### Negative

1. The library no longer validates model IDs at all — relies entirely on AWS API rejection
2. Typos in model IDs won't be caught until the AWS API call

### Tradeoffs

<!-- adr:tradeoffs -->
```yaml
tradeoffs:
  - gain: Universal model compatibility without library updates
    cost: Model ID typos caught at AWS API layer instead of locally
    acceptable: true
    rationale: >
      AWS provides clear, accurate error messages for invalid model IDs.
      The library's job is reasoning tools, not model authorization.
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
      date: 2026-05-12
  review_schedule: annually
  next_review: null
```
<!-- /adr:approval -->

---

## References

- AWS Bedrock Converse API: Model ID validation behavior
