---
id: ADR-0015
title: "Linear Chain vs Emergent DAG via dependencies and contradicts"
status: proposed
date: 2026-05-14
decision_makers:
  - Engineering
category:
  - architecture
supersedes: null
superseded_by: null
related:
  - ADR-0008
  - ADR-0010
  - ADR-0011
  - ADR-0012
tags:
  - topology
  - dag
  - drift
  - data-model
---

# ADR 0015: Linear Chain vs Emergent DAG via dependencies and contradicts

## Context

### The Problem

The `chain_of_thought_step` tool accepts two list-of-integer fields:

```python
"dependencies": {"type": "array", "items": {"type": "integer"},
                 "description": "Step numbers this depends on"},
"contradicts":  {"type": "array", "items": {"type": "integer"},
                 "description": "Step numbers this contradicts"},
```

Used unaware, these are metadata fields. But used fully, they make the chain into a directed graph:

- `dependencies` is a list of incoming edges from prior steps. A step can depend on several earlier steps. Read as a graph, this is a multi-parent relationship.
- `contradicts` is a list of incoming edges with negative polarity. A step can contradict several earlier steps simultaneously.

A chain is, by definition in graph theory, a path where each node has at most one predecessor. The moment a step lists two or more `dependencies` (or one of each in `dependencies` and `contradicts`), the topology is a DAG, not a chain. The library is called `chain-of-thought-tool` and stores `steps: List[ThoughtStep]`, but the *referenced* topology is broader than a chain.

This is not actively broken. Today:

- The storage is linear (`List[ThoughtStep]`).
- The summary computes contradiction *pairs* by walking the `contradicts` field, which is graph-aware.
- The summary's "best path" or "ancestor traversal" is *not* computed — we have nothing analogous to `graph-of-thought`'s `_trace_best_path`.
- `dependencies` and `contradicts` are validated for type and range but the validator does not check that referenced step numbers exist, does not check for cycles, and does not require dependencies be on earlier (lower-numbered) steps.

So we have three layers of inconsistency:

1. **Naming** says chain.
2. **Storage** is a list (compatible with chain or general topology).
3. **References** are graph-shaped, weakly validated.

If `dependencies` and `contradicts` are used in earnest, the library is doing graph-of-thought-lite without the graph mechanics, the cycle protection, or the topology queries. If they are not used in earnest, the fields are documentation-grade metadata and the linear story holds.

Either reading should be a deliberate decision, not the current quiet ambiguity.

### Constraints

- A sibling library, `graph-of-thought`, exists for genuinely DAG-shaped reasoning. Doubling down on graph mechanics here would compete with it.
- ADR-0008 (5 canonical stages) assumes linear progression for the completion metric. A DAG with branches breaks that assumption (see ADR-0012's overlapping concern).
- ADR-0012 proposes adding explicit branching (`branch_from_step`, `branch_id`). If accepted, this further moves the topology toward DAG-shaped — branching plus dependencies plus contradicts is a graph.
- The `dependencies` and `contradicts` fields are already in the Tier 1 stable tool schema (ADR-0006). Removing them is breaking; tightening their validation is non-breaking.
- LLMs operating against this tool today produce both linear chains (most common) and dependency-rich graphs (less common but real). Both patterns work; only one is documented.

---

## Decision

This ADR **identifies** the topology ambiguity; it does not yet resolve it. The recommended path is **Option B (keep `dependencies` and `contradicts` as references, strengthen validation, define the topology explicitly as a "primarily-linear DAG", and document the boundary with `graph-of-thought` sibling library)** — but the decision is open until accepted.

What this ADR commits to immediately:

1. Document in `docs/SPEC.md` §3.2 that the stored structure is a *primarily-linear DAG*: a `List[ThoughtStep]` whose `dependencies` and `contradicts` fields admit graph-shaped reference patterns.
2. Document in `README.md` the relationship to `graph-of-thought`: this library tracks a single primary chain with graph-shaped cross-references; the sibling library is the right choice when reasoning is genuinely a graph from the start.
3. Add a follow-on issue (not in this ADR's scope) to strengthen validation: referenced step numbers must exist; dependencies must reference earlier steps; contradicts may reference any step.
4. Leave the tool surface and storage unchanged in v0.3.x.

### Requirements

<!-- adr:requirements -->
```yaml
requirements: []
```
<!-- /adr:requirements -->

---

## Alternatives Considered

### Alternative A: Status quo — fields exist, semantics undocumented

**Approach:** Don't change anything. Leave `dependencies` and `contradicts` as weakly-documented graph-shaped metadata.

**Pros:**
- Zero work.

**Cons:**
- The naming/storage/reference mismatch stays implicit. Future readers will either over-interpret these fields (treat them as full DAG edges) or under-interpret them (treat them as decorative).
- Validation gaps remain: referenced steps may not exist; cycles are not detected.
- ADR-0012 (branching) and this ADR overlap and the overlap is never named.

### Alternative B: Primarily-linear DAG — recommended

**Approach:** Adopt the explicit framing: **the structure is a linear sequence of `ThoughtStep` records, with `dependencies` and `contradicts` as auxiliary graph-shaped references that do not change the storage shape.** Strengthen validation: dependencies must reference existing earlier steps; contradicts must reference existing steps. Document the relationship to `graph-of-thought` as: "use that library when the topology is graph-shaped from the start; use this library when reasoning is primarily linear with occasional cross-references."

**Pros:**
- Names what we already are without changing the shipped product.
- The validation strengthening is non-breaking (existing valid chains stay valid; invalid chains that referenced non-existent steps fail loudly instead of silently).
- Carves out a clear boundary with `graph-of-thought` so the two libraries do not compete on the same use case.
- Composes with ADR-0012 (branching): branches are an explicit DAG operation; `dependencies` and `contradicts` are the implicit auxiliary edges.

**Cons:**
- The "primarily-linear DAG" framing is jargon that not every reader will absorb on first pass.
- Stricter validation may surface bugs in existing user chains (referencing step 99 when only 5 steps exist used to be silently accepted; now rejected). This is the right behaviour but is a small migration.
- Doesn't add new capability — purely documentary and validation work.

**Decision:** Recommended.

### Alternative C: Drop `dependencies` and `contradicts` — return to pure linear chain

**Approach:** Remove the two fields from `chain_of_thought_step` in v2.0. The chain is strictly linear. Users wanting cross-references use `graph-of-thought`.

**Pros:**
- Honest naming: it really is a chain.
- Simpler tool surface for LLMs.
- Removes the ambiguity completely.

**Cons:**
- Breaking change (Tier 1 fields removed).
- Loses real expressiveness — `contradicts` is genuinely useful for chains that revisit earlier conclusions.
- ADR-0008's contradiction-detection feature depends on `contradicts`.
- Forces users into `graph-of-thought` for what is really a chain with a contradiction note.

**Decision:** Rejected. The fields earn their keep even in primarily-linear reasoning.

### Alternative D: Embrace graph fully — promote to a DAG-first library

**Approach:** Rework storage to be a graph. Add ancestor traversal, cycle detection, best-path computation à la `graph-of-thought`.

**Pros:**
- Resolves the topology ambiguity by going all-in.
- Adds real graph mechanics.

**Cons:**
- Duplicates `graph-of-thought`'s value proposition. The two libraries would compete head-on.
- Significant rewrite. Tier 1 API changes.
- ADR-0008's stage-completion metric is fundamentally linear; would need rework.
- The MCP `sequential-thinking` reference is a step tracker, not a graph engine. We would diverge further from our structural anchor.

**Decision:** Rejected. The sibling library exists for this reason.

---

## Consequences

### Positive (under recommended Option B)

1. The naming/storage/reference inconsistency becomes a documented, defensible architectural position.
2. The boundary with `graph-of-thought` is articulated, so users (and future maintainers) know which library to reach for.
3. Validation strengthening turns silent acceptance of malformed references into loud rejection — a strictly better feedback loop for LLMs.
4. ADR-0012 (branching) and this ADR can coexist: branching is the *primary topology operation*; `dependencies`/`contradicts` are the *cross-cutting reference layer*. Both serve different needs.

### Negative (under recommended Option B)

1. "Primarily-linear DAG" is a phrase that has to be explained every time a new reader encounters it.
2. Stricter validation surfaces silent failures in existing chains as visible errors. Small migration; worth doing.
3. No new capability — purely framing and validation work. Some reviewers will want a stronger move.

### Tradeoffs

<!-- adr:tradeoffs -->
```yaml
tradeoffs:
  - gain: Topology ambiguity becomes a named, defensible position; validation matches the implied semantics
    cost: A jargon phrase ("primarily-linear DAG") and a small validation-tightening migration
    acceptable: true
    rationale: The fields already exist and already create graph-shaped references. Naming the situation is strictly better than continuing to leave it implicit.
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

- `chain_of_thought/__init__.py` — `dependencies` and `contradicts` fields in `chain_of_thought_step` tool spec
- `chain_of_thought/core.py` — `ThoughtStep` dataclass and `generate_summary`'s contradiction-pair computation
- `chain_of_thought/validators.py` — current weak validation of dependency/contradicts references
- Sibling library: `enginez/graph-of-thought` (genuine DAG reasoning)
- ADR-0006 (API Stability) — `dependencies` and `contradicts` are Tier 1
- ADR-0008 (5 Canonical Stages) — linear completion model
- ADR-0010 (Canonical Reference Mapping)
- ADR-0011 (Library Identity) — the naming-vs-shape conversation that frames this one
- ADR-0012 (Branching Parity) — sibling topology drift; together they describe the full DAG-shape
