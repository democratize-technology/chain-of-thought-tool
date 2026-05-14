---
id: ADR-0007
title: Auxiliary Reasoning Tools as Structural Scaffolding
status: accepted
date: 2026-05-14
decision_makers:
  - Engineering
category:
  - architecture
supersedes: null
superseded_by: null
related: []
tags: [honesty, capability, tools, scaffolding]
---

# ADR 0007: Auxiliary Reasoning Tools as Structural Scaffolding

## Context

### The Problem

The library exposes three auxiliary tools alongside its core chain-of-thought reasoning tools:

1. **`generate_hypotheses`** -- Returns 1-4 hypotheses (scientific, intuitive, contrarian, systematic) ranked by testability score. Implementation: template strings with the caller's observation interpolated into canned responses.

2. **`map_assumptions`** -- Returns explicit and implicit assumptions with criticality assessment and a dependency graph. Implementation: keyword-based heuristics matching linguistic patterns ("obviously", "clearly", "must", "everyone knows", "assuming", "if", "because") to detect assumption indicators.

3. **`calibrate_confidence`** -- Returns a calibrated confidence value with overconfidence detection, uncertainty bands, and adjustment reasoning. Implementation: heuristic pattern matching for absolute language ("definitely", "certainly", "impossible"), future prediction indicators, complexity word counts, and domain-specific keywords.

These tools sound like AI-driven semantic analysis. They are not. `generate_hypotheses` produces the same four structural templates regardless of input. `map_assumptions` finds keywords, not meaning. `calibrate_confidence` applies fixed penalty weights to detected patterns.

This creates a capability honesty problem: users may reasonably expect these tools to perform genuine analytical work, when their actual value is providing structured output shapes that guide an LLM's reasoning process.

### Constraints

- **Zero external dependencies** (ADR-0002): Adding NLP/ML libraries or calling external LLM APIs for genuine analysis would violate the library's core philosophy.
- **Low latency**: These tools run synchronously in the Bedrock tool loop. Adding network calls or heavy computation would break the interaction pattern.
- **No cost overhead**: The library adds zero marginal cost per invocation. External API calls would change this fundamental property.

---

## Decision

Keep the three auxiliary tools as structural scaffolding with honest documentation. They are not analysts -- they are frameworks. The LLM consuming their output is the analyst. The library provides structured response shapes (hypothesis types, assumption categories, confidence bands) that prompt the LLM to reason more carefully within those structures.

Specifically:

1. **Retain all three tools** with their current template/heuristic implementations.
2. **Document honestly** in tool descriptions, README, and any API reference that these are scaffolding tools, not analytical engines. The PRD's "Honest Capability Assessment" table (Section 6) establishes this precedent.
3. **Name them accurately** in tool metadata. The tool specs should describe what they structurally provide (e.g., "Returns a framework of hypothesis types for the observation" rather than "Generates diverse hypotheses").
4. **Accept the limitation** that template responses will feel shallow for complex or unusual inputs.

### Requirements

<!-- adr:requirements -->
requirements:
  - id: REQ-001
    category: architecture
    description: "Tool specs must describe the scaffolding nature of auxiliary tools"
    verification:
      type: grep
      pattern: "generate_hypotheses|map_assumptions|calibrate_confidence"
      paths:
        - "chain_of_thought/__init__.py"
      expect: present
  - id: REQ-002
    category: architecture
    description: "README must include a capability assessment documenting tool limitations"
    verification:
      type: grep
      pattern: "scaffolding|template|heuristic"
      paths:
        - "README.md"
      expect: present
  - id: REQ-003
    category: architecture
    description: "Auxiliary tools must remain synchronous with no external API calls"
    verification:
      type: grep_negative
      pattern: 'import requests|import aiohttp|urllib|http\.client'
      paths:
        - "chain_of_thought/core.py"
      expect: absent
<!-- /adr:requirements -->

---

## Alternatives Considered

### Alternative 1: Remove auxiliary tools entirely

**Approach:** Delete `generate_hypotheses`, `map_assumptions`, and `calibrate_confidence` from the library. Focus exclusively on the core chain-of-thought tools.

**Pros:**
- Eliminates the capability honesty problem entirely
- Smaller API surface, less maintenance burden
- No risk of users expecting AI analysis

**Cons:**
- Loses useful scaffolding that structures LLM reasoning
- The three tools correspond to real reasoning patterns (divergent thinking, critical thinking, metacognition) that benefit from formal structure
- Users who understand the scaffolding model lose access to it

**Decision:** Rejected. The tools provide genuine value as reasoning frameworks even without genuine analysis. Removing them removes useful cognitive scaffolding.

### Alternative 2: Make tools call an LLM for real analysis

**Approach:** Route auxiliary tool inputs through an LLM API call to generate semantically meaningful hypotheses, assumptions, and confidence assessments.

**Pros:**
- Genuine analytical output
- Users get what the tool names imply

**Cons:**
- Violates ADR-0002 (zero external dependencies)
- Adds latency (network round-trip per auxiliary tool call)
- Adds cost (token consumption per invocation)
- Creates recursive dependency: LLM calls tool that calls LLM
- Breaks the synchronous tool loop pattern with Bedrock

**Decision:** Rejected. The cost, latency, and architectural violations are disqualifying. If users want AI-driven analysis, they should use the chain-of-thought tools to guide the LLM through that analysis directly.

### Alternative 3: Rebrand as framework tools with different naming

**Approach:** Rename tools to explicitly signal their scaffolding nature (e.g., `hypothesis_framework`, `assumption_checklist`, `confidence_rubric`).

**Pros:**
- Sets accurate expectations from the tool name alone
- Avoids overclaim in the API surface

**Cons:**
- Breaks the existing API contract (ADR-0006 stability tiers)
- Tool names become verbose and less intuitive
- The current names are fine if descriptions are honest
- Renaming is a breaking change that requires migration

**Decision:** Rejected. Honest descriptions solve the expectation problem without breaking the API. Tool names can remain concise; documentation carries the accuracy burden.

### Alternative 4: Don't document the limitation

**Approach:** Keep tools as-is without explicitly documenting that they are template/heuristic implementations.

**Pros:**
- Avoids drawing attention to the limitation
- Marketing appears stronger

**Cons:**
- Dishonest. Users will discover the limitation and lose trust.
- Reputation damage when users realize "generate hypotheses" produces canned templates
- Violates the library's own PRD, which explicitly calls this out
- The "Honest Capability Assessment" in the PRD already committed to transparency

**Decision:** Rejected. Reputation is the asset. Honest capability boundaries build more trust than inflated claims.

---

## Consequences

### Positive

1. **Zero cost and latency.** Auxiliary tools execute synchronously with no network calls, fitting cleanly into the Bedrock tool loop.
2. **Reasoning structure for LLMs.** The output shapes (hypothesis types, assumption categories, confidence bands with uncertainty ranges) give the consuming LLM a framework to think within, even when the raw content is template-generated.
3. **Honest capability boundary.** Users know exactly what they are getting. No misleading marketing, no disappointed users, no reputation damage.
4. **Consistent with zero-dependency philosophy.** Template/heuristic implementations require only Python stdlib, maintaining ADR-0002 compliance.
5. **Useful even as templates.** The `calibrate_confidence` heuristics (absolute language detection, future prediction penalty, technology domain penalty) capture real calibration patterns from cognitive science literature. The pattern matching is shallow but directionally correct.

### Negative

1. **Risk of user misunderstanding.** Despite documentation, some users will see "generate hypotheses" and expect semantic analysis. Honest docs mitigate but do not eliminate this risk.
2. **Template shallowness.** `generate_hypotheses` produces the same structural responses regardless of input complexity. A one-sentence observation and a 500-word technical analysis receive the same four templates.
3. **Heuristic brittleness.** `map_assumptions` depends on keyword matching. It misses assumptions expressed without indicator words and produces false positives when indicator words appear in non-assumption contexts.
4. **Documentation burden.** Maintaining honest capability descriptions requires ongoing discipline as the tools evolve.

### Tradeoffs

<!-- adr:tradeoffs -->
```yaml
tradeoffs:
  - gain: Structured reasoning scaffolding at zero cost and latency
    cost: Template/heuristic output that lacks semantic depth
    acceptable: true
    rationale: >
      The LLM consuming the output is the analyst. The library's job is to provide
      the framework, not the analysis. This is the same relationship as a worksheet
      and a student -- the worksheet structures thinking but does not do the thinking.
  - gain: Honest capability claims that build user trust
    cost: Users who want genuine analysis must look elsewhere
    acceptable: true
    rationale: >
      Honest boundaries are more valuable than inflated claims. Users who need
      AI-driven hypothesis generation can use the chain-of-thought tools to guide
      the LLM through that process directly.
  - gain: Zero-dependency, zero-cost tool execution
    cost: Cannot upgrade to genuine analysis without violating ADR-0002
    acceptable: true
    rationale: >
      ADR-0002 is a core architectural constraint. If the zero-dependency philosophy
      changes, this decision should be revisited. Until then, the constraint is
      a feature, not a limitation.
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
      date: 2026-05-14
  review_schedule: annually
  next_review: null
```
<!-- /adr:approval -->

---

## References

- [PRD Section 6: Honest Capability Assessment](../PRD.md)
- [PRD Section 4.2: Auxiliary Reasoning Tools](../PRD.md)
- [ADR-0002: Zero External Dependency Philosophy](0002-zero-external-dependency-philosophy.md)
- [ADR-0006: 3-Tier API Stability Contract](0006-3-tier-api-stability-contract.md)
