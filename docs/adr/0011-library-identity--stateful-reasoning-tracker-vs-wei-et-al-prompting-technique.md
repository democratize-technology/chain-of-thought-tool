---
id: ADR-0011
title: "Library Identity — Stateful Reasoning Tracker vs Wei et al. Prompting Technique"
status: proposed
date: 2026-05-14
decision_makers:
  - Engineering
category:
  - architecture
supersedes: null
superseded_by: null
related:
  - ADR-0006
  - ADR-0010
tags:
  - identity
  - naming
  - canonical-cot
  - drift
---

# ADR 0011: Library Identity — Stateful Reasoning Tracker vs Wei et al. Prompting Technique

## Context

### The Problem

The library is named `chain-of-thought-tool`. The README opens with "Chain of Thought reasoning capabilities for LLMs." The package, the tools, and the docs all use the phrase "chain of thought" with no qualifier. The reference for that phrase is Wei et al. 2022 (arXiv:2201.11903), which defines Chain-of-Thought (CoT) as:

> A series of intermediate reasoning steps... a simple method called chain of thought prompting, where a few chain of thought demonstrations are provided as exemplars in prompting.

In Wei et al., CoT is:

- A **prompting technique** — exemplars in the input prompt.
- Producing **text output** — the model generates intermediate reasoning steps as part of its natural-language response.
- **Stateless** — no per-step API, no tool calls, no stored thoughts. The chain lives in the model's output tokens for one inference.
- **Emergent at scale** — works in models above roughly 100B parameters.

This library is none of those things. It is:

- A **function-calling tool surface** — eight tools exposed via `TOOL_SPECS`.
- Producing **structured state** — `ThoughtStep` dataclasses stored in a list, with stage, confidence, evidence, assumptions, dependencies, and contradicts.
- **Stateful** — `ChainOfThought` is a stateful object; `ThreadAwareChainOfThought` provides per-conversation isolation; `export_chain` / `import_chain` persist state to JSON.
- **Model-agnostic** — works with any function-calling LLM, regardless of scale or training.

Internally, `CLAUDE.md` already names this honestly: it labels the project `sequential-thinking-tool`. The actual structural ancestor is the Model Context Protocol `sequential-thinking` reference server (see ADR-0010 for the mapping). What we call ourselves and what we are come from different lineages.

This is not just a naming nit. It has three operational consequences:

1. **User expectation drift.** A developer who reads Wei et al. and then `pip install chain-of-thought-tool` reasonably expects this library to be related to that paper. It is — by goal, not by mechanism. They will spend time looking for `few_shot_exemplars=[...]` or `let_s_think_step_by_step=True` and not find them.
2. **Capability claims drift.** When we say "chain of thought reasoning capabilities" we're trading on the paper's empirical reputation (62% improvement on GSM8K, etc.) without delivering its mechanism. ADR-0007 already took an honest stance on auxiliary-tool capability claims; this is the same kind of claim, one level up.
3. **Design discussions drift.** Every architectural debate ("should we add streaming?", "should we add multi-sample?") is muddied by ambiguity about whether we're trying to emulate Wei et al. (where streaming is the natural mode) or extend the MCP tool (where state checkpoints are).

### Constraints

- ADR-0006 makes `TOOL_SPECS`, `HANDLERS`, and tool input schemas Tier 1 stability — renaming `chain_of_thought_step` is a major-version change.
- The package name `chain-of-thought-tool` is on PyPI and in user `requirements.txt` files. Renaming it on PyPI is possible but disruptive.
- The phrase "chain of thought" is not a Wei et al. trademark or copyright. We can keep using it. The question is *whether we should*, given the mismatch.
- The MCP `sequential-thinking` reference predates this library; we are arguably an unattributed re-implementation. Naming the lineage is overdue.

---

## Decision

This ADR **identifies** the identity mismatch; it does not yet resolve the naming question. The recommended path is **Option B (keep the public name, add explicit "what this library actually is" framing throughout the docs, cite both anchors)** — but the decision is open until accepted.

What this ADR commits to immediately:

1. Add a "What this library actually is" section near the top of `README.md` and `docs/SPEC.md` §1 that explicitly states: *this is a stateful function-calling tool inspired by the Wei et al. CoT goal but structurally descended from the MCP sequential-thinking pattern.*
2. Update `CLAUDE.md`'s "Purpose" line from "Lightweight Chain of Thought reasoning capabilities for any LLM API" to one that names both lineages.
3. Leave the public package name, tool names, and `TOOL_SPECS` schema **unchanged** in v0.x.
4. Reserve a v2.0 decision on whether to rebrand. Don't make that call now.

### Requirements

<!-- adr:requirements -->
```yaml
requirements: []
```
<!-- /adr:requirements -->

---

## Alternatives Considered

### Alternative A: Status quo — keep using "Chain of Thought" with no explicit framing

**Approach:** Leave docs as they are. Assume readers will figure out the relationship between the name and the mechanism.

**Pros:**
- Zero work.
- No risk of confusing users who already understood what the library does.

**Cons:**
- The implicit drift problem ADR-0010 documents stays unaddressed.
- Reproduces the failure mode of "auxiliary tools sound like AI analysis but aren't" — the same problem ADR-0007 solved by documenting honestly.
- Users who arrive expecting Wei et al. mechanics keep getting confused.

**Decision:** Rejected.

### Alternative B: Keep the name, add explicit framing in docs — recommended

**Approach:** Don't rename anything. Add prominent "What this library is and isn't" sections to README, SPEC, and CLAUDE.md. Cite Wei et al. for the goal-lineage and the MCP sequential-thinking server for the shape-lineage. Update tool descriptions where they overclaim.

**Pros:**
- Zero breakage. Tier 1 API stability preserved.
- Solves the user-expectation problem directly: anyone reading the docs sees the framing immediately.
- Matches the precedent set by ADR-0007 (Honest Capability Assessment in the PRD).
- Cheap.

**Cons:**
- The README has to argue for the name slightly — "we are called X but we are also Y" is more work to write than either pure description.
- Doesn't fix the *grep-ability* problem: users searching for "MCP sequential thinking" still won't find us.

**Decision:** Recommended.

### Alternative C: Rename to `structured-reasoning-tool` (or similar) in v2.0

**Approach:** Plan a v2.0 with `structured-reasoning-tool` (or `sequential-thinking-tool` to match the internal name) as the new package name. Keep `chain-of-thought-tool` as a thin compatibility shim that re-exports.

**Pros:**
- Long-term honest naming.
- Aligns the public name with the actual mechanism.
- The internal `CLAUDE.md` name already tells us where this would land.

**Cons:**
- Major work: PyPI rename, new package, compatibility shim, migration docs.
- Loses the (modest) SEO value of "chain of thought" in package search.
- The "chain" in our library *is* a chain (sequential `ThoughtStep` list), so the name is not maximally wrong — just incomplete.
- Tier 1 API change at minimum on the import path.

**Decision:** Deferred. This is a v2.0 conversation, not a v0.x one. If the recommended Option B framing reduces user confusion sufficiently, Option C may never become necessary.

### Alternative D: Reposition as a Wei-et-al.-style helper (different library)

**Approach:** Pivot the library toward Wei et al. CoT mechanics — add few-shot exemplar management, "Let's think step by step" prompt prepending, self-consistency voting (see ADR-0014). Make the library actually be CoT in the paper's sense.

**Pros:**
- Resolves the naming-vs-mechanism mismatch by changing the mechanism.
- Opens a path to add canonical CoT extensions (Zero-shot CoT, Self-Consistency, Plan-and-Solve).

**Cons:**
- The existing tool surface and state tracker are already shipped and used. Pivoting means either abandoning them or splitting the library in two.
- Wei et al. CoT requires *prompt assembly*, which puts us in the prompt-engineering layer of someone else's stack. Today we are below that layer.
- Most of the value users currently get (state tracking, summaries, export/import) is orthogonal to Wei et al. mechanics. Pivoting would lose that.

**Decision:** Rejected. The current product is good at what it does; the mismatch is in the *name*, not the *value*.

---

## Consequences

### Positive (under recommended Option B)

1. The implicit drift documented in ADR-0010 gets a concrete first decision: we are honest about the dual lineage in user-facing docs.
2. The READMEs of related projects in `enginez/` (graph-of-thought, rubber-duck-mcp) can cross-reference this library's actual mechanism, not its name.
3. New contributors land in a doc that names what the library *is*, not just what it *does*.
4. Lays the groundwork for ADRs 0012–0016 to cite the right canonical reference for each gap (MCP for branching/revision, Wei extensions for self-consistency).

### Negative (under recommended Option B)

1. The README gets longer. Adding "what this isn't" sections is always a slight cost to brevity.
2. The mismatch is named but not resolved. Anyone who finds the name actively misleading will not be satisfied by framing alone — they will want the rename.
3. Cross-reference burden: every ADR in 0012–0016 has to be careful about whether it's drawing from the CoT lineage or the MCP lineage. The dual-anchor approach in ADR-0010 helps but doesn't eliminate this.

### Tradeoffs

<!-- adr:tradeoffs -->
```yaml
tradeoffs:
  - gain: Honest user-facing framing of what the library actually is; preserved public API
    cost: README and CLAUDE.md grow a "what this is" section; the naming mismatch remains, just named
    acceptable: true
    rationale: The name is on PyPI and in users' requirements.txt. Framing is cheap; renaming is not. The mismatch was implicit anyway; making it explicit is strictly better.
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

- Wei, J., et al. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* arXiv:2201.11903, NeurIPS 2022.
- Model Context Protocol `sequential-thinking` server: https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking
- `CLAUDE.md` — internal project name `sequential-thinking-tool`
- `README.md` §1 — current opening claim
- `docs/SPEC.md` §1 — current problem statement
- ADR-0006 (3-Tier API Stability Contract) — constrains the rename option
- ADR-0007 (Auxiliary Tools as Scaffolding) — sets the precedent for honest capability framing
- ADR-0010 (Canonical Reference Mapping)
