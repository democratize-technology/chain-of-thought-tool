---
id: ADR-0010
title: "Canonical Reference Mapping and Drift Index"
status: proposed
date: 2026-05-14
decision_makers:
  - Engineering
category:
  - architecture
supersedes: null
superseded_by: null
related:
  - ADR-0007
  - ADR-0008
  - ADR-0011
  - ADR-0012
  - ADR-0013
  - ADR-0014
  - ADR-0015
  - ADR-0016
tags:
  - canonical-cot
  - mcp-sequential-thinking
  - drift-analysis
  - meta
---

# ADR 0010: Canonical Reference Mapping and Drift Index

## Context

### The Problem

This library is named `chain-of-thought-tool` and its top-level type label in `CLAUDE.md` reads "Lightweight Chain of Thought reasoning capabilities for any LLM API". But the internal project name in the same file is `sequential-thinking-tool`. That single line is the most honest summary of where we are: **the library carries the name of one canonical reference and the shape of a different one**.

There are two canonical references that bear on what this library is or claims to be:

1. **Wei et al. 2022, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"** (arXiv:2201.11903, NeurIPS 2022). The paper that named CoT. It defines CoT as a **prompting technique**: include a few-shot exemplars in the prompt that show intermediate reasoning steps, and a sufficiently large model will generate intermediate reasoning steps before answering. CoT in Wei et al. is text generation guided by exemplars — no state, no tools, no API.

2. **The Model Context Protocol `sequential-thinking` reference server** (`modelcontextprotocol/servers/src/sequentialthinking`). The canonical reference for "structured step tracker exposed as a function-calling tool." Its tool surface is: `thought`, `thoughtNumber`, `totalThoughts`, `nextThoughtNeeded`, `isRevision`, `revisesThought`, `branchFromThought`, `branchId`, `needsMoreThoughts`.

Our `chain_of_thought_step` tool is structurally a direct descendant of the MCP sequential-thinking surface (`thought`, `step_number`, `total_steps`, `next_step_needed`, plus extras). It is **not** structurally what Wei et al. describe. We added domain extensions (`reasoning_stage`, `confidence`, `dependencies`, `contradicts`, `evidence`, `assumptions`) and dropped two MCP features (`branchFromThought`/`branchId`, explicit `isRevision`/`revisesThought`).

The ADRs 0001–0009 collectively document our concrete decisions but never confront the question: **what canonical artifact are we tracking, and where do we deliberately differ?** Without that anchor, drift accumulates silently. ADR-0007 (auxiliary tools as scaffolding) and ADR-0008 (5 canonical stages) both make decisions with strong genealogies — neither cites a source. This series fixes that.

### Constraints

- The library is alpha-ish (v0.3.0, 321 tests, 80% coverage). Breaking changes are still feasible but cost more than they did at v0.1.
- ADRs 0001–0009 are largely already in place. New ADRs in this series must not relitigate decisions already accepted (e.g., ADR-0007's auxiliary-tool stance, ADR-0008's five stages).
- Some drift is fundamental and not realistically closable (we will never become Wei et al. CoT — that's a prompting technique, not a library). Some drift is a real gap (MCP branching parity is one config change away).
- The library name `chain-of-thought-tool` is part of the public stability contract (ADR-0006 Tier 1). Renaming is a v2.0 move at the earliest.

---

## Decision

Adopt a **dual-anchor** model for reasoning about this library's canonical genealogy:

| Anchor | Role | What it covers |
|---|---|---|
| Wei et al. 2022 (arXiv:2201.11903) | **Name origin** | The phrase "chain of thought" and the goal of step-by-step reasoning. **Not** the API shape, the storage model, or the tool surface. |
| MCP `sequential-thinking` server | **Shape origin** | The tool-call-based step tracker pattern with revision, branching, and a `nextThoughtNeeded` flag. This is what our `chain_of_thought_step` tool structurally is. |

Catalogue the gaps and divergences in ADRs 0011–0016. Each gets its own decision space so the trade-offs are not bundled. The mapping:

| Concept | Wei et al. 2022 | MCP sequential-thinking | This library | ADR |
|---|---|---|---|---|
| Identity / "what is this" | Prompting technique | Stateful tool | Stateful tool, mis-named | ADR-0011 |
| Branching | N/A | `branchFromThought`, `branchId` | Absent | ADR-0012 |
| Explicit revision | N/A | `isRevision`, `revisesThought` | Implicit via `step_number` collision | ADR-0013 |
| Self-consistency (multi-sample voting) | Wang et al. 2022 extension | N/A | Absent | ADR-0014 |
| Linearity of reasoning | Chain (linear) | Tree (branchable) | DAG (via `dependencies` / `contradicts`) — emergent | ADR-0015 |
| Auxiliary tools (hypotheses, assumptions, calibration) | Not in CoT lineage | Not in MCP reference | Present, ADR-0007 documents honesty | ADR-0016 |
| 5 canonical stages | Not in CoT lineage | Not in MCP reference | Hardcoded enum | Already documented (ADR-0008) |
| Confidence, evidence, assumptions per step | Not in CoT lineage | Not in MCP reference | Present | Already documented (ADR-0008, ADR-0007) |

This ADR is the index. ADRs 0011–0016 do the work of naming each drift point and surfacing options.

### Requirements

<!-- adr:requirements -->
```yaml
requirements: []
```
<!-- /adr:requirements -->

---

## Alternatives Considered

### Alternative 1: Single-anchor on Wei et al. 2022 (the name)

**Approach:** Treat Wei et al. 2022 as the only canonical reference. Frame everything else as extension.

**Pros:**
- Matches the library's name.
- One reference is simpler than two.

**Cons:**
- Wei et al. CoT is a prompting technique. We are a tool. Treating it as our anchor forces every ADR to start with "this doesn't really apply because we're not a prompting technique."
- Hides the much closer structural relative (MCP sequential-thinking) entirely.
- The honest gaps (branching, explicit revision) become invisible because Wei et al. never described them in the first place.

**Decision:** Rejected. A canonical reference that doesn't match the library's shape is worse than no reference at all.

### Alternative 2: Single-anchor on MCP sequential-thinking (the shape)

**Approach:** Acknowledge the library is a stateful tool, anchor on the MCP reference, and treat Wei et al. as historical naming.

**Pros:**
- Structurally accurate. The MCP reference is what we actually look like.
- Makes the concrete gaps (branching, explicit revision) first-class instead of buried.
- Aligns with the `CLAUDE.md` internal project name `sequential-thinking-tool`.

**Cons:**
- Conflicts with the package name `chain-of-thought-tool` and the public README. Users arrive expecting CoT in some sense — at minimum, they expect step-by-step reasoning, which Wei et al. is the canonical citation for.
- Loses the (legitimate) lineage claim on "chain of thought" as a goal.

**Decision:** Rejected on its own, but its core observation is correct and feeds the dual-anchor decision.

### Alternative 3: Dual-anchor with explicit role separation — chosen

**Approach:** Wei et al. is the name origin; the MCP reference is the shape origin. Each ADR in this series cites whichever anchor it draws from.

**Pros:**
- Honest about both ancestries.
- Lets the naming-vs-shape mismatch (ADR-0011) be the *first decision* rather than an unstated tension behind everything else.
- Lets concrete gaps (ADRs 0012–0014) cite the right canonical reference instead of stretching one to cover the other.

**Cons:**
- Two anchors take longer to explain.
- Some users will read the name and not the ADRs; the dual-anchor story doesn't reach them.

**Decision:** Chosen.

### Alternative 4: No canonical anchor — document only what we do

**Approach:** Don't tie ADRs to any external reference. Just describe what the library does.

**Pros:**
- Maximally local. No external sources to keep current.

**Cons:**
- Re-invites the implicit drift problem these ADRs exist to solve.
- Loses the value of external benchmarks against which to evaluate our choices.

**Decision:** Rejected. Anchoring is the whole point.

---

## Consequences

### Positive

1. Every drift point in ADRs 0011–0016 can cite the specific canonical artifact it diverges from, instead of waving toward "the CoT literature" generically.
2. The naming-vs-shape mismatch becomes a documented, debatable thing rather than an unstated assumption.
3. ADR-0007 and ADR-0008 (already accepted, already honest) get retroactive context: the stages and the auxiliary tools are *additions* not present in either anchor, which is a stronger claim than ADR-0008 currently makes about itself.
4. Future readers (and contributors) get a starting map: read this ADR, then read whichever drift ADR matches their question.

### Negative

1. Seven new ADRs to maintain. They are `proposed`, so they have no verification burden until accepted.
2. The dual-anchor story is more nuance than single-source. Some readers will skim past it.
3. The mapping table in this ADR must be kept current if either anchor evolves materially. (Wei et al. 2022 is frozen; the MCP reference is not.)

### Tradeoffs

<!-- adr:tradeoffs -->
```yaml
tradeoffs:
  - gain: Every concrete decision in ADRs 0011-0016 has a named canonical anchor
    cost: Two anchors instead of one; the naming-vs-shape mismatch must be confronted in ADR-0011
    acceptable: true
    rationale: One anchor that doesn't fit is worse than two anchors that do. The mismatch is real either way; this ADR names it instead of hiding it.
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

- Wei, J., et al. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* arXiv:2201.11903, NeurIPS 2022. https://arxiv.org/abs/2201.11903
- Model Context Protocol `sequential-thinking` server: https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking
- Kojima, T., et al. *Large Language Models are Zero-Shot Reasoners.* arXiv:2205.11916 (Zero-shot CoT).
- Wang, X., et al. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* arXiv:2203.11171.
- `CLAUDE.md` project type line: "sequential-thinking-tool"
- ADR-0007 (Auxiliary Tools as Scaffolding) — already takes a related honest stance
- ADR-0008 (5 Canonical Reasoning Stages) — establishes a domain model not present in either anchor
- ADRs 0011–0016 — individual drift points
