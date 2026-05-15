---
id: ADR-0012
title: "Branching Parity with MCP Sequential-Thinking"
status: accepted
accepted_date: 2026-05-14
date: 2026-05-14
decision_makers:
  - Engineering
category:
  - architecture
supersedes: null
superseded_by: null
related:
  - ADR-0010
  - ADR-0011
  - ADR-0015
tags:
  - branching
  - mcp-parity
  - drift
  - tool-surface
---

# ADR 0012: Branching Parity with MCP Sequential-Thinking

## Context

### The Problem

The Model Context Protocol `sequential-thinking` reference server exposes four optional fields that this library does not:

| MCP field | Type | Purpose |
|---|---|---|
| `branchFromThought` | integer, optional | The thought number to branch off from |
| `branchId` | string, optional | Identifier for the new branch |
| `isRevision` | boolean, optional | Marks this thought as a revision (see ADR-0013) |
| `revisesThought` | integer, optional | Which thought is being revised (see ADR-0013) |

This ADR covers the first two — branching. ADR-0013 covers the revision pair.

Branching matters because real reasoning often forks. The MCP reference docs are explicit:

> Each thought can build on, question, or revise previous insights as understanding deepens.

> The Sequential Thinking tool is designed for... Analysis that might need course correction... Problems where the full scope might not be clear initially.

Our `chain_of_thought_step` accepts no branching parameters. The chain is a flat `list[ThoughtStep]` indexed by `step_number`. If an LLM wants to explore two alternatives, it can only:

- Pick one and commit (losing the other).
- Use `dependencies` and `contradicts` to imply branching, which is what ADR-0015 examines — the chain becomes a DAG by accident rather than by design.
- Call `clear_chain` and start over (losing all history).

None of these match the canonical MCP reference's affordance. The user-facing cost is that LLMs operating against our tool cannot represent "I want to explore approach A and approach B in parallel and decide later."

The PRD does not list branching as a requirement. The SPEC does not mention it. ADR-0008 (5 canonical stages) makes the chain *linear by stage progression* — branching collides slightly with the "5-stage completion percentage" model because branches multiply the stage space.

### Constraints

- ADR-0006 makes `TOOL_SPECS` Tier 1 stability. Adding optional fields is **not** a breaking change (existing callers ignore new optional fields). Adding required fields would be.
- ADR-0008 computes `completion_status.percent_complete` against five canonical stages. Branches inflate stage coverage if each branch adds its own stage steps — the metric needs a definition.
- The `ChainOfThought` storage is `self.steps: List[ThoughtStep]`. Branches can be stored in the same list (with an extra `branch_id` field on `ThoughtStep`) or in a separate `branches: Dict[str, List[ThoughtStep]]` structure.
- ADR-0015's "DAG via dependencies" already creates an implicit branching mechanism. Adding explicit branching means we now have *two* branching mechanisms — one structural (the new fields) and one emergent (the existing reference lists). They must be reconciled.

---

## Decision

This ADR **identifies** the drift; it does not yet resolve it. The recommended path is **Option B (add `branch_from_step` and `branch_id` as optional fields; store branches in the same `steps` list with a `branch_id` attribute on `ThoughtStep`)** — but the decision is open until accepted.

What this ADR commits to immediately:

1. Document in `docs/SPEC.md` §3.3.1 that branching is not currently supported and that this is a known gap relative to the MCP `sequential-thinking` reference.
2. Note in `README.md`'s capability assessment table that this library has no branching, distinguishing it from the MCP reference.
3. Leave the tool surface unchanged in v0.3.x.

### Requirements

<!-- adr:requirements -->
```yaml
requirements: []
```
<!-- /adr:requirements -->

---

## Alternatives Considered

### Alternative A: Status quo — no branching

**Approach:** Don't add branching. Document the gap. Users who need branching use `graph-of-thought` (the sibling library) instead.

**Pros:**
- Zero code change.
- Keeps the "this is a *chain*, not a graph" identity clean.
- ADR-0008's completion metric stays simple.
- Cross-references nicely with the sibling library — different tool for different need.

**Cons:**
- Permanent gap against the named structural reference (MCP sequential-thinking).
- LLMs operating against our tool still hit the wall when reasoning naturally wants to fork.
- The `dependencies` / `contradicts` mechanism (ADR-0015) already creates implicit branching — pretending we don't have any is inconsistent.

### Alternative B: Add optional `branch_from_step` and `branch_id` fields; store in the same `steps` list — recommended

**Approach:**

- Add two optional fields to `chain_of_thought_step`: `branch_from_step` (int, must reference an existing step) and `branch_id` (string, free-form identifier).
- Add a `branch_id: Optional[str] = None` field to `ThoughtStep`. `None` means "main trunk."
- `step_number` becomes unique within a `branch_id`, not globally. (Or: globally unique with branches storing a list of step numbers — see implementation note below.)
- `get_chain_summary` gains a `branches` field that lists known branch IDs and their root step numbers.
- `completion_status` is computed per branch by default, with an `overall` view that takes union across branches.
- The MCP reference's `branchId` is a string with no global registry. We follow that — branches are caller-named.

Implementation note: keep `step_number` globally unique to avoid breaking existing callers and tests. `branch_id` is metadata. A branch is "all steps with the same `branch_id` plus their ancestors up to `branch_from_step`."

**Pros:**
- MCP-canonical surface.
- Backwards compatible (new fields are optional).
- Resolves the inconsistency with `dependencies` / `contradicts` — branching becomes first-class instead of emergent.
- Lets ADR-0015's DAG behaviour become a *consequence* of explicit branching rather than a workaround for its absence.

**Cons:**
- `ThoughtStep` gains a field. Dataclass deserialization in `import_chain` needs an update (backwards-compatible: default `None`).
- `get_chain_summary` complicates: per-branch confidence, per-branch completion, contradiction pairs across vs within branches.
- The 5-stage completion model has to decide: does a branch with only "Problem Definition" count toward overall coverage, or does it need to cover all 5 stages itself? The decision is policy, not data.
- Two more strings the LLM has to manage (`branch_id`, `branch_from_step`). Tool description has to make their purpose obvious.

**Decision:** Recommended.

### Alternative C: Promote to graph-of-thought as the branching solution

**Approach:** Don't add branching. Cross-link to the sibling `graph-of-thought` library aggressively. Suggest in tool descriptions that users wanting branching use that library.

**Pros:**
- Clean library boundaries.
- Existing graph-of-thought already supports DAG topology.

**Cons:**
- Cross-library dependency push: every branching need now requires two libraries.
- The MCP reference treats branching as a feature of the *same* tool, not a separate tool. Forcing the user to switch is a worse developer experience than adding two fields.
- ADR-0015 will have to confront the DAG question anyway — the two libraries can both have branching without conflict.

**Decision:** Rejected. The libraries can have overlapping but differently-shaped capabilities. CoT branching at the step-tool level and GoT graph operations at the topology level serve different mental models.

### Alternative D: Add branching with global step numbers and branch metadata only

**Approach:** Step numbers stay globally unique. A `branches` registry maps branch IDs to root step numbers. No per-branch step numbering — a branch is identified by its `branch_id` tag.

**Pros:**
- Minimal change to `step_number` semantics. Existing tests don't break.
- Easier to merge branches back (no renumbering).

**Cons:**
- Less natural for LLMs to use. They have to track global step numbers across branches.
- The MCP reference's `branchFromThought` + `branchId` model doesn't quite map — there's no notion of "branch step 1 of branch X" in our version.

**Decision:** This is approximately what Option B's implementation note already does. Treating Option D as a variant of B; the implementation choice between "step_number unique globally" vs "unique per branch" is a sub-question for whoever accepts B.

---

## Consequences

### Positive (under recommended Option B)

1. The library reaches structural parity with its named canonical reference (MCP sequential-thinking) on branching.
2. Users wanting to express "explore both A and B" can do so without abandoning the library or losing history.
3. ADR-0015's DAG-via-references story gets cleaner: explicit branching for forks, `dependencies` / `contradicts` for the genuinely DAG-shaped relationships (cross-branch references, contradictions).
4. Backwards compatible — old chains and old callers keep working.

### Negative (under recommended Option B)

1. `get_chain_summary`'s output gains complexity. The 5-stage completion percentage has to decide its branch semantics.
2. Tool description for `chain_of_thought_step` gets longer. Each added optional field is one more thing the LLM has to learn.
3. `import_chain` and `export_chain` need a version bump to handle the new `branch_id` field gracefully.
4. The "this is a chain" identity (ADR-0011) gets stretched. With branching, it's no longer strictly a chain — it's a tree-shaped sequence. We may want to acknowledge that in the rebrand conversation reserved for v2.0.

### Tradeoffs

<!-- adr:tradeoffs -->
```yaml
tradeoffs:
  - gain: Structural parity with MCP sequential-thinking on branching; honest about the DAG nature of the current implicit branching
    cost: Two more optional fields on ThoughtStep; summary metric complications; identity drifts further from "a chain"
    acceptable: true
    rationale: The MCP reference is what we structurally are (ADR-0010). Closing a documented gap with a backwards-compatible addition is a strong default. The identity drift surfaces a conversation we should have anyway.
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

- MCP `sequential-thinking` server schema: https://github.com/modelcontextprotocol/servers/blob/main/src/sequentialthinking/index.ts
- `chain_of_thought/__init__.py` — `chain_of_thought_step` tool spec (current, no branching fields)
- `chain_of_thought/core.py` — `ThoughtStep` dataclass and `ChainOfThought.steps` list
- ADR-0006 (API Stability) — optional-field additions are non-breaking
- ADR-0008 (5 Canonical Stages) — completion metric must define branch semantics
- ADR-0010 (Canonical Reference Mapping)
- ADR-0011 (Library Identity)
- ADR-0013 (Explicit Revision) — pairs with branching as the other MCP-reference gap
- ADR-0015 (Linear Chain vs Emergent DAG) — depends on which branching mechanism wins
