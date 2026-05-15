---
id: ADR-0008
title: 5 Canonical Reasoning Stages Domain Model
status: accepted
date: 2026-05-14
decision_makers:
  - Engineering
category:
  - architecture
supersedes: null
superseded_by: null
related: []
tags: [domain-model, reasoning, stages]
---

# ADR 0008: 5 Canonical Reasoning Stages Domain Model

## Context

### The Problem

Each thought step submitted to `chain_of_thought_step` needs a stage categorization so the system can track reasoning progress, deliver stage-aware feedback, and compute completion metrics. Without structured stages, `get_chain_summary` has no basis for measuring how far reasoning has progressed or which phases are missing.

### Constraints

- The tool spec must remain compatible with the AWS Bedrock Converse API `toolConfig` format (ADR-0003).
- The `reasoning_stage` parameter is optional with a default value, since not every LLM caller will provide it.
- Stage names appear in JSON output and must be human-readable.
- The completion percentage reported by `get_chain_summary` must be meaningful across all callers.

---

## Decision

Define a fixed set of 5 canonical reasoning stages that cover the full reasoning lifecycle:

| # | Stage | Purpose |
|---|-------|---------|
| 1 | Problem Definition | Scope the problem, identify constraints, establish success criteria |
| 2 | Research | Gather information, consider multiple sources and perspectives |
| 3 | Analysis | Break down components, identify patterns and relationships |
| 4 | Synthesis | Integrate insights, draw connections, assess implications |
| 5 | Conclusion | Finalize reasoning, verify conclusions address the original problem |

These are declared as the `enum` constraint on the `reasoning_stage` parameter in the `chain_of_thought_step` tool spec. The default is `"Analysis"` because most steps naturally fall into the analytical phase.

Custom stage names are accepted by the validator (alphanumeric, spaces, underscores, hyphens, max 100 characters) but are stored as-is without normalization to the canonical set. The `completion_status` in `get_chain_summary` computes coverage exclusively against the 5 canonical stages. A chain containing only custom stages reports 0% completion.

The `_generate_feedback` method provides contextual guidance per canonical stage, telling the model what to focus on next.

### Requirements

<!-- adr:requirements -->
requirements:
  - id: REQ-001
    category: architecture
    description: "reasoning_stage MUST accept 5 canonical values: Problem Definition, Research, Analysis, Synthesis, Conclusion"
    verification:
      type: grep
      pattern: "Problem Definition|Research|Analysis|Synthesis|Conclusion"
      paths:
        - "chain_of_thought/__init__.py"
      expect: present
  - id: REQ-002
    category: architecture
    description: "reasoning_stage MUST be optional with default Analysis"
    verification:
      type: grep
      pattern: "Analysis"
      paths:
        - "chain_of_thought/core.py"
      expect: present
  - id: REQ-003
    category: architecture
    description: "get_chain_summary MUST compute completion_status for canonical stages"
    verification:
      type: grep
      pattern: "completion_status|stages_missing|percent_complete"
      paths:
        - "chain_of_thought/core.py"
      expect: present
  - id: REQ-004
    category: architecture
    description: "Feedback system MUST provide stage-specific guidance for all 5 stages"
    verification:
      type: grep
      pattern: "stage_guidance|_generate_feedback"
      paths:
        - "chain_of_thought/core.py"
      expect: present
  - id: REQ-005
    category: architecture
    description: "Validator MUST accept custom stage names"
    verification:
      type: grep
      pattern: "_validate_reasoning_stage"
      paths:
        - "chain_of_thought/validators.py"
      expect: present
<!-- /adr:requirements -->

---

## Alternatives Considered

### Alternative 1: Free-Form String Stages

**Approach:** Accept any string as a reasoning stage with no canonical set.

**Pros:**
- Maximum flexibility for domain-specific workflows
- No rigidity for non-linear reasoning patterns

**Cons:**
- No structured progress tracking
- `completion_status` cannot compute a meaningful percentage
- Stage-aware feedback becomes impossible without a known set of stages
- No basis for comparing reasoning coverage across sessions

**Decision:** Rejected. Progress tracking and stage-aware feedback are core value propositions of the library.

### Alternative 2: Configurable Stages Per Instance

**Approach:** Allow callers to define their own stage set when constructing a `ChainOfThought` instance.

**Pros:**
- Domain-specific stage names (e.g., "Hypothesis", "Experiment", "Peer Review")
- Theoretically more expressive

**Cons:**
- Breaks summarization across instances (different stage sets are incomparable)
- The `enum` constraint in the tool spec becomes dynamic, requiring tool spec regeneration per instance
- Completion percentage is only meaningful if the caller-defined set is known and stable
- Significantly increases API surface area and testing burden

**Decision:** Rejected. The cost to cross-session comparability and tool spec stability outweighs the flexibility gain.

### Alternative 3: Bloom's Taxonomy

**Approach:** Use Bloom's taxonomy levels (Remember, Understand, Apply, Analyze, Evaluate, Create) as reasoning stages.

**Pros:**
- Well-established cognitive framework
- Education-domain recognition

**Cons:**
- Oriented toward learning assessment, not reasoning workflows
- "Remember" and "Understand" are not reasoning stages
- "Create" is an output, not a reasoning phase
- 6 stages don't map cleanly to typical problem-solving flows

**Decision:** Rejected. Bloom's taxonomy models knowledge acquisition depth, not reasoning process phases.

### Alternative 4: Scientific Method Stages

**Approach:** Use scientific method phases (Observation, Hypothesis, Experiment, Analysis, Conclusion).

**Pros:**
- Familiar and well-defined
- Good fit for empirical reasoning tasks

**Cons:**
- Too narrow for non-empirical reasoning (legal analysis, design decisions, debugging)
- "Experiment" is not a reasoning step in most LLM use cases
- Does not account for synthesis or integration of prior steps

**Decision:** Rejected. Scientific method stages assume empirical verification that most LLM reasoning tasks cannot perform.

---

## Consequences

### Positive

1. **Consistent progress tracking.** `completion_status.percent_complete` is comparable across all sessions because the denominator is always 5.
2. **Stage-aware feedback.** Each canonical stage gets tailored guidance in `_generate_feedback`, nudging the model toward productive next steps.
3. **Missing-stage visibility.** `stages_missing` tells callers exactly which reasoning phases were skipped, enabling quality gates.
4. **Minimal caller burden.** The parameter is optional with a sensible default; callers get value without explicit stage management.
5. **Tool spec stability.** The `enum` constraint is static, so the Bedrock tool spec never changes between sessions.

### Negative

1. **Poor fit for iterative patterns.** Debugging cycles (hypothesize, test, observe, repeat) don't map cleanly to a linear 5-stage progression.
2. **Custom stages are second-class.** A chain using only custom stages reports 0% completion regardless of reasoning depth.
3. **Rigidity.** Some domains naturally have 3 stages or 7; forcing 5 may feel arbitrary.
4. **No stage ordering enforcement.** The system tracks which stages appear but does not validate that they appear in order (e.g., a "Problem Definition" step after "Conclusion" is accepted).

### Tradeoffs

<!-- adr:tradeoffs -->
```yaml
tradeoffs:
  - gain: Cross-session comparable completion metrics
    cost: All reasoning must be categorized into exactly 5 stages
    acceptable: true
    rationale: >
      The 5 stages cover the vast majority of structured reasoning patterns.
      Custom stages provide an escape hatch for unusual domains, and the
      completion metric simply ignores them rather than rejecting them.
  - gain: Stage-aware feedback guidance in every step response
    cost: Custom stage names receive no stage-specific feedback
    acceptable: true
    rationale: >
      Feedback for custom stages falls back to generic confidence and
      dependency guidance, which remains useful. Callers who want feedback
      should use canonical stage names.
  - gain: Fixed enum in tool spec for Bedrock compatibility
    cost: Tool spec cannot represent the full range of accepted custom stages
    acceptable: true
    rationale: >
      The enum documents the recommended stages. The validator accepts
      additional values as an extension, which is standard practice for
      enum-constrained API parameters.
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

- `chain_of_thought/__init__.py`: `reasoning_stage` enum in `chain_of_thought_step` tool spec
- `chain_of_thought/core.py`: `ThoughtStep` dataclass, `ChainOfThought._generate_feedback`, `ChainOfThought.generate_summary`
- `chain_of_thought/validators.py`: `ParameterValidator._validate_reasoning_stage`
