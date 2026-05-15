---
id: ADR-0016
title: "Auxiliary Tool Genealogy — Not From CoT Lineage"
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
  - ADR-0010
  - ADR-0011
tags:
  - genealogy
  - auxiliary-tools
  - drift
  - documentation
---

# ADR 0016: Auxiliary Tool Genealogy — Not From CoT Lineage

## Context

### The Problem

The library exposes three auxiliary tools alongside the core chain-of-thought step tool:

| Tool | What it does | Genealogy |
|---|---|---|
| `generate_hypotheses` | Returns four template hypotheses (scientific, intuitive, contrarian, systematic) ranked by a fixed "testability score" | Not from CoT lineage |
| `map_assumptions` | Returns explicit and implicit assumptions found by keyword matching, plus a "criticality" classification | Not from CoT lineage |
| `calibrate_confidence` | Adjusts a confidence value downward when overconfidence-pattern keywords are present, returns uncertainty bands | Not from CoT lineage |

ADR-0007 already takes a clear stance on *capability*: these tools are template/heuristic scaffolding, not AI-driven analysis, and the documentation must say so. That ADR is excellent. It deliberately scopes itself to capability honesty.

This ADR addresses a different question that ADR-0007 leaves on the table: **where do these tools come from intellectually?** The library is called `chain-of-thought-tool` but none of the three auxiliary tools have any clear lineage in the CoT literature.

What they *do* trace to:

- `generate_hypotheses`: closest to **abductive reasoning** frameworks (Peirce's logic of abduction) and to **divergent thinking** rubrics from creativity research (de Bono's lateral thinking, Osborn's brainstorming). The four hypothesis types (scientific, intuitive, contrarian, systematic) are a stylised version of a divergent-thinking checklist.

- `map_assumptions`: closest to **critical thinking pedagogy** — assumption mapping is a standard exercise in informal logic courses (e.g., Hitchcock's argumentation textbook) and in design thinking ("identify your assumptions" workshops). The linguistic-indicator approach (flagging "clearly", "obviously", "must") is a specific subset.

- `calibrate_confidence`: closest to **calibration research** in psychology (Lichtenstein, Fischhoff & Phillips, "Calibration of probabilities," 1982) and the **forecasting** literature (Tetlock's *Superforecasting*). The "overconfidence indicators" — absolute language, future predictions, technology-domain bias — match well-known overconfidence patterns from that literature, applied as keyword heuristics.

None of these traditions is CoT. None is MCP sequential-thinking. They are useful reasoning scaffolds, but they sit on a different intellectual axis than the core tool.

This matters because:

1. **Discoverability.** A user reading `generate_hypotheses` and looking for the theoretical basis finds nothing. There is no citation to Peirce, to abductive-reasoning textbooks, to de Bono — and there should be, because that's where the scaffolding comes from.

2. **Extension paths.** "Where should we go next with these tools?" is currently a question with no anchor. If we knew the lineage, we'd know which adjacent ideas to consider next (e.g., for `map_assumptions`, the related notion of "load-bearing assumptions" from systems thinking; for `calibrate_confidence`, Brier scoring or Brier-improvement training).

3. **Honesty composition.** ADR-0007 says "these are scaffolding, not analysis." This ADR adds: "and the scaffolding is from these traditions, not from CoT." The two together produce a fully honest position.

### Constraints

- ADR-0007 already locks in *capability* claims. This ADR must not reopen that decision — it adds documentation, not new capability claims.
- The auxiliary tools are Tier 1 (`TOOL_SPECS`) — names and schemas are stable. No renaming.
- These tools' template/heuristic implementations are by design (ADR-0007 Alternative 2 rejected the "make them call an LLM" option). This ADR does not change that.
- The library's `README.md` already includes a "Capability Assessment" table that anchors the honesty story. This ADR proposes adding a "Genealogy" table or paragraph alongside it.

---

## Decision

This ADR **identifies** the missing genealogy; it does not yet resolve documentation placement. The recommended path is **Option A (add a "Genealogy of auxiliary tools" paragraph to README and SPEC, with citations; do not change tool descriptions in `TOOL_SPECS`)** — but the decision is open until accepted.

What this ADR commits to immediately:

1. Document in `docs/SPEC.md` §3.3.4–3.3.6 (the per-auxiliary-tool sections) the closest intellectual lineage for each tool, with a one-line citation.
2. Add a "Genealogy" subsection to `README.md` immediately after the Capability Assessment table.
3. Leave the `TOOL_SPECS` descriptions unchanged (they are Tier 1 stable and serve a different audience — LLMs, not human readers seeking intellectual context).

### Requirements

<!-- adr:requirements -->
```yaml
requirements: []
```
<!-- /adr:requirements -->

---

## Alternatives Considered

### Alternative A: Add genealogy to README and SPEC, leave tool descriptions alone — recommended

**Approach:**

- README's existing "Capability Assessment" table gets a "Genealogy" companion table or paragraph:
  - `generate_hypotheses` → divergent thinking rubrics (de Bono, Osborn); abductive reasoning (Peirce)
  - `map_assumptions` → critical thinking and informal logic pedagogy; design thinking assumption-mapping exercises
  - `calibrate_confidence` → calibration research (Lichtenstein, Fischhoff & Phillips, 1982); forecasting literature (Tetlock)
- SPEC's per-tool subsections (§3.3.4–3.3.6) gain a "Lineage" line each.
- `TOOL_SPECS` descriptions are unchanged — they are sized for LLM consumption, not human research context.

**Pros:**
- Closes the "where does this come from" gap without changing any Tier 1 surface.
- Composes with ADR-0007: capability honesty + genealogy honesty = full picture.
- Opens extension paths (which Brier-scoring extensions? which assumption-mapping methodologies?) that are currently blocked by the absence of an anchor.
- Cheap.

**Cons:**
- Three more citations to keep current.
- Some of these traditions are pre-paper (Peirce, de Bono); the citations point at textbooks or surveys rather than seminal arXiv preprints. Lower SEO value than a clean arXiv link.
- The "Genealogy" framing is unusual for a software library README. Some readers will skim past it.

**Decision:** Recommended.

### Alternative B: Update tool descriptions in `TOOL_SPECS` with lineage hints

**Approach:** Add a short lineage clause to each auxiliary tool's `description` in `TOOL_SPECS`.

**Pros:**
- The lineage information reaches the LLM directly. The model may use the framing when generating outputs.

**Cons:**
- Tool descriptions are sized for LLM context. Adding 30–50 words per tool to cite an academic tradition uses tokens for context that doesn't change behaviour.
- The LLM does not need to know that `generate_hypotheses` is grounded in de Bono. It needs to know how to use the tool.
- ADR-0006 Tier 1 — descriptions are part of the stable contract. Changing them is non-breaking but is a real change.

**Decision:** Rejected. Wrong audience for the information.

### Alternative C: Status quo — leave genealogy implicit

**Approach:** Don't add anything. ADR-0007 covers the honesty story sufficiently.

**Pros:**
- Zero work.

**Cons:**
- The discoverability and extension-path problems remain.
- ADR-0007's honesty story is about capability, not intellectual lineage. They are not the same.
- A user who reads `generate_hypotheses`, finds the four canned templates surprising, and tries to learn more has nowhere obvious to go.

**Decision:** Rejected.

### Alternative D: Add a dedicated `docs/GENEALOGY.md` file

**Approach:** Put the genealogy story in its own document, separate from README and SPEC.

**Pros:**
- Doesn't bloat the main docs.
- Lets the lineage section be as long as it needs to be.

**Cons:**
- One more doc to discover. Most readers will never find it.
- Splits content that should live next to the tool descriptions (in SPEC) and the capability assessment (in README).
- README is where most discovery happens; banishing the genealogy from there reduces its reach.

**Decision:** Rejected. The genealogy benefits most from being adjacent to its kin (capability assessment, per-tool spec).

---

## Consequences

### Positive (under recommended Option A)

1. The auxiliary tools' intellectual context becomes findable in two places (README, SPEC) without bloating the LLM-facing tool descriptions.
2. Extension conversations become anchored: "should we add Brier-scoring to `calibrate_confidence`?" has a clear yes/no story once the forecasting lineage is named.
3. ADR-0007 and this ADR together close the auxiliary-tool honesty story — what they are (scaffolding, not analysis) and where they come from (other traditions, not CoT).
4. Sets a precedent for naming intellectual lineage when adding future tools.

### Negative (under recommended Option A)

1. Three citations and their associated maintenance burden. Citations to seminal pre-paper traditions are harder to keep "current" than arXiv links.
2. The "Genealogy" framing is unusual; some readers will see it as academic affectation rather than useful context.
3. Future tools added to the auxiliary surface now have an expectation of declared lineage. That is the right precedent but is one more discipline to maintain.

### Tradeoffs

<!-- adr:tradeoffs -->
```yaml
tradeoffs:
  - gain: Intellectual lineage of auxiliary tools made discoverable; extension paths anchored
    cost: Three citations to maintain; a "Genealogy" section that some readers will find unusual
    acceptable: true
    rationale: ADR-0007 made capability claims honest. Naming the genealogy closes the parallel gap and costs only documentation.
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

- `chain_of_thought/auxiliary.py` — `HypothesisGenerator`, `AssumptionMapper`, `ConfidenceCalibrator` (the tools whose genealogy this ADR documents)
- ADR-0007 (Auxiliary Tools as Scaffolding) — capability honesty; this ADR adds lineage honesty
- ADR-0010 (Canonical Reference Mapping) — neither CoT nor MCP-sequential-thinking covers these tools
- ADR-0011 (Library Identity) — clarifies that the library's structural anchor does not extend to these tools
- Peirce, C.S. *Collected Papers* (abductive reasoning).
- Lichtenstein, S., Fischhoff, B., & Phillips, L.D. (1982). *Calibration of Probabilities: The State of the Art to 1980.* In Kahneman, Slovic, & Tversky, *Judgment Under Uncertainty: Heuristics and Biases*.
- Tetlock, P. & Gardner, D. (2015). *Superforecasting: The Art and Science of Prediction.*
- de Bono, E. (1970). *Lateral Thinking: Creativity Step by Step.*
- Osborn, A. (1953). *Applied Imagination.*
