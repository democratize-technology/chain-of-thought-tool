---
id: ADR-0013
title: "Explicit Revision Semantics — isRevision and revisesThought"
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
  - ADR-0012
tags:
  - revision
  - mcp-parity
  - drift
  - tool-surface
---

# ADR 0013: Explicit Revision Semantics — isRevision and revisesThought

## Context

### The Problem

The Model Context Protocol `sequential-thinking` reference server exposes two optional fields for explicit revision:

| MCP field | Type | Purpose |
|---|---|---|
| `isRevision` | boolean, optional | Marks this thought as a revision |
| `revisesThought` | integer, optional | Which thought is being revised |

Our `chain_of_thought_step` has neither. Revision is implicit: if the caller submits a step with a `step_number` that already exists, `_handle_step_revision` (core.py L341) replaces the existing step in-place. The handler's response includes `"is_revision": true` so the caller can tell after the fact what happened, but the *input* gives no way to signal revision intent.

```python
# core.py L341-352
def _handle_step_revision(self, step_number: int, validated_params):
    for i, step in enumerate(self.steps):
        if step.step_number == step_number:
            self.steps[i] = self._create_thought_step(validated_params)
            self._update_metadata()
            return self._generate_feedback(self.steps[i], is_revision=True)
    return None
```

This has three observable consequences:

1. **Silent overwrites.** An LLM that mistypes a `step_number` (e.g., meant 4, wrote 3) silently destroys step 3. The replacement is treated identically to an intentional revision. There is no way for the caller to say "I'm about to add step 4 *unless* one already exists, in which case warn me."

2. **No revision history.** When step 3 is revised to a new content, the prior content is gone. `export_chain` writes only the current state. The PRD claim that the chain is "auditable" is partly defeated — you can audit the final state, not the revision trail.

3. **Audit trail divergence from the canonical reference.** The MCP server preserves a `thoughtHistoryLength` and treats `isRevision: true` as semantically distinct from an unmarked thought. Downstream tooling that consumes both — e.g., an LLM trained to expect MCP-shaped output — sees a different shape from us.

The implicit-revision design is internally consistent and the validators handle it correctly. The question is whether it's the right design.

### Constraints

- ADR-0006 makes `TOOL_SPECS` Tier 1 stability. Adding optional fields is non-breaking. Changing the *behaviour* of step-number collision (e.g., to reject unless `is_revision=True`) **is** breaking.
- `export_chain` / `import_chain` (ADR-not-yet) is the persistence surface. A revision history would change the on-disk format.
- ADR-0008's 5-stage completion model is unaffected by revision — revisions stay within a stage by default.
- The `_handle_step_revision` path is well-tested. Changing its semantics is a meaningful test surface change.

---

## Decision

This ADR **identifies** the drift; it does not yet resolve it. The recommended path is **Option C (add `is_revision` and `revises_step` as optional fields, default to implicit-revision semantics when neither is supplied, log a warning when a step_number collides without explicit `is_revision=True`, and preserve revision history in a separate `revisions: List[ThoughtStep]` list)** — but the decision is open until accepted.

What this ADR commits to immediately:

1. Document in `docs/SPEC.md` §3.2 the current implicit-revision semantics (collision on `step_number` silently overwrites) and explicitly mark this as a known divergence from the MCP `sequential-thinking` reference.
2. Document in the `chain_of_thought_step` tool description that submitting an existing `step_number` revises the prior step in-place.
3. Leave the tool surface and storage unchanged in v0.3.x.

### Requirements

<!-- adr:requirements -->
```yaml
requirements: []
```
<!-- /adr:requirements -->

---

## Alternatives Considered

### Alternative A: Status quo — implicit revision via step_number collision

**Approach:** Don't add anything. Document the implicit semantics. Note the divergence from MCP.

**Pros:**
- Zero code change.
- The current behaviour is simple to explain in one sentence.
- LLM tool descriptions stay short.

**Cons:**
- Silent overwrites stay silent.
- No audit trail of revisions.
- Permanent divergence from MCP reference on what is a small, additive change.

### Alternative B: Add optional `is_revision` and `revises_step`, no history

**Approach:** Mirror the MCP fields directly. When provided, `is_revision=True` and `revises_step=N` make the revision intent explicit and the validator can sanity-check (e.g., `revises_step` must reference an existing step). When not provided, fall back to the current implicit behaviour (collision on `step_number` overwrites).

**Pros:**
- MCP parity on the surface.
- Backwards compatible.
- Lets the LLM signal intent — and lets the library reject obvious mistakes (e.g., `is_revision=True` but `step_number` doesn't exist yet).

**Cons:**
- Doesn't solve the audit-trail problem. Prior content is still lost on revision.
- Two different revision paths (explicit + implicit) is more code paths than one. Each needs tests.

### Alternative C: Add fields *and* preserve revision history — recommended

**Approach:** Add `is_revision` and `revises_step` as in Option B. Also add a `revisions: List[ThoughtStep]` list to `ChainOfThought` that captures the prior content whenever a step is replaced. `get_chain_summary` gains an optional `include_revision_history` flag. `export_chain` / `import_chain` preserve the revision list.

When a `step_number` collides:

- If `is_revision=True`: explicit revision — push old step to `revisions`, replace.
- If `is_revision` is unspecified: implicit revision — same behaviour as today, but emit a structured warning in the response (`feedback` field gets a "Implicit revision; consider passing is_revision=true for clarity" line). Old step still pushed to `revisions`.
- If `is_revision=False` *and* `step_number` collides: reject with error. The caller asserted "this is a new step" and contradicted themselves.

**Pros:**
- MCP parity.
- Audit trail intact — revisions are preserved, queryable via summary, and survive export/import.
- Backwards compatible for the "submit step 3 again to revise it" pattern; tightens behaviour where the caller explicitly says "this is not a revision."
- The feedback-warning gives LLMs a soft nudge toward explicit revision without rejecting the implicit path.

**Cons:**
- `ChainOfThought.revisions` adds memory cost proportional to revision count. Caps may be desirable (e.g., keep last 10 revisions per step).
- `export_chain` / `import_chain` format version bump.
- More state to think about during concurrency. The `_lock` already protects `self.steps`; it has to extend to `self.revisions`.
- Tool description grows.

**Decision:** Recommended. The audit-trail story is the actual prize; MCP parity is the polish.

### Alternative D: Strict mode — reject all step_number collisions without explicit is_revision

**Approach:** Make `is_revision=True` *required* to overwrite an existing `step_number`. Implicit revision is removed.

**Pros:**
- Cleanest semantics.
- Eliminates silent overwrite class of bug.

**Cons:**
- Breaking change. Existing callers (and tests) rely on implicit revision.
- Tier 1 API stability violation per ADR-0006 — requires major version bump.
- Forces all LLMs to learn the new contract before they can revise anything.

**Decision:** Rejected for v0.x. Reasonable for a v2.0 if Option C runs long enough to gather data on whether implicit revision is genuinely useful.

---

## Consequences

### Positive (under recommended Option C)

1. MCP-reference parity on explicit revision.
2. Audit trail: revisions are preserved, exportable, and queryable.
3. Soft-warning path on implicit revision nudges LLMs toward explicit intent without breaking existing callers.
4. The `is_revision=False` + collision case becomes a detectable mistake instead of a silent footgun.
5. `get_chain_summary` becomes genuinely more useful — "what changed and when" is now answerable.

### Negative (under recommended Option C)

1. Memory cost proportional to revision count. Some chains may revise heavily; cap policy may be needed.
2. `ChainOfThought` state grows; concurrency lock coverage extends.
3. Export format version bump; `import_chain` must handle older format.
4. Two more optional tool fields. Each adds a small amount of LLM-side learning cost.
5. The feedback warning text needs to be useful without being noisy. Tunable.

### Tradeoffs

<!-- adr:tradeoffs -->
```yaml
tradeoffs:
  - gain: MCP parity on revision plus an actual audit trail; mistakes (is_revision=False + collision) become detectable
    cost: Bounded memory growth for revision history; export format version bump; two more optional fields
    acceptable: true
    rationale: The audit trail is real product value, not just canonical conformance. The MCP parity is a side benefit.
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
- `chain_of_thought/core.py` L341–352 — current `_handle_step_revision`
- `chain_of_thought/__init__.py` — `chain_of_thought_step` tool spec (current, no revision fields)
- `docs/SPEC.md` §3.2 — `ThoughtStep` data model (would need a `revisions` companion structure)
- ADR-0006 (API Stability) — optional-field additions are non-breaking; behaviour changes for `is_revision=False` are
- ADR-0010 (Canonical Reference Mapping)
- ADR-0012 (Branching Parity) — sibling MCP gap; the two paired ADRs close the explicit-MCP-fields drift together
