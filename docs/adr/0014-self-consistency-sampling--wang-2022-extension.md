---
id: ADR-0014
title: "Self-Consistency Sampling — Wang 2022 Extension"
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
  - ADR-0002
  - ADR-0010
  - ADR-0011
tags:
  - self-consistency
  - canonical-cot
  - drift
  - sampling
---

# ADR 0014: Self-Consistency Sampling — Wang 2022 Extension

## Context

### The Problem

The most influential extension of Wei et al. 2022 CoT is **Self-Consistency** (Wang et al. 2022, "Self-Consistency Improves Chain of Thought Reasoning in Language Models," arXiv:2203.11171). The core idea:

> Sample multiple diverse reasoning paths via temperature sampling, then marginalize over them by taking the majority vote on the final answer.

Self-Consistency reliably improves CoT performance on arithmetic and commonsense reasoning by 5–15 percentage points. The mechanism is simple: instead of trusting a single chain, run the model `k` times at non-zero temperature, extract the final answer from each, and return the most common one.

The follow-up paper landscape (Plan-and-Solve, Tree of Thoughts, Auto-CoT, Active Prompt) almost all assume self-consistency as a baseline. Anyone arriving at this library with knowledge of "modern CoT" will look for self-consistency support and not find it.

This library currently has no sampling abstraction. The `AsyncChainOfThoughtProcessor` runs a single Bedrock Converse loop with whatever temperature the caller configured in `inferenceConfig`. The chain is deterministic in the sense that each call produces one chain. There is no concept of "run the same starting problem N times and vote on the conclusion."

This is a real gap relative to canonical CoT — and it is *specifically the gap* that distinguishes "we are a state tracker" from "we are CoT in the Wei lineage." Self-consistency is the bridge between the two anchors named in ADR-0010: it is a Wei-lineage extension that operates over the kind of structured state we already track.

### Constraints

- ADR-0002 (Zero External Dependency Philosophy) means we cannot pull in `numpy`, `scipy`, or other voting/statistics libraries.
- Self-consistency requires running the LLM multiple times. We don't call the LLM directly — we expose tools. So the implementation question is: *where does the sampling loop live?*
- The MCP `sequential-thinking` reference does not support self-consistency. So this is purely a Wei-lineage gap, not an MCP-parity gap.
- "Majority vote on the final answer" requires extracting an answer. Our `ThoughtStep` has no `final_answer` field. The notion of "the answer" of a chain is implicit in the final step's content.
- Adding a sampling primitive risks pulling the library into the prompt-engineering layer, which ADR-0011 Alternative D rejected. Self-consistency is a *consumer-side* operation more naturally — the caller runs the loop multiple times and uses our `export_chain` to inspect each.

---

## Decision

This ADR **identifies** the drift; it does not yet resolve it. The recommended path is **Option B (add a `vote_on_conclusion` tool that takes a list of exported chain JSON paths, extracts conclusions, returns vote counts; do not run the LLM ourselves)** — but the decision is open until accepted.

What this ADR commits to immediately:

1. Document in `docs/SPEC.md` §2 that self-consistency (Wang et al. 2022) is not supported and that the library is not in the LLM-calling layer where it would naturally live.
2. Document in `README.md` a recipe showing how a caller can implement self-consistency externally by looping their own LLM calls and using `export_chain` / `import_chain` to collect samples.
3. Leave the tool surface unchanged in v0.3.x.

### Requirements

<!-- adr:requirements -->
```yaml
requirements: []
```
<!-- /adr:requirements -->

---

## Alternatives Considered

### Alternative A: Status quo — document the gap, do not implement

**Approach:** Add a paragraph to README/SPEC pointing readers at Wang et al. 2022 and explaining that self-consistency is a caller-side concern.

**Pros:**
- Zero code change.
- Keeps the library out of the LLM-calling layer.
- Aligns with ADR-0011's "we are a state tracker, not a prompting technique" framing.

**Cons:**
- Users who want self-consistency get nothing from us beyond a pointer.
- Loses an opportunity to add real value over the MCP reference.

### Alternative B: Add `vote_on_conclusions` tool over multiple exported chains — recommended

**Approach:**

- Add a tool `vote_on_conclusions(chain_paths: list[str], extract: Literal["last_step", "synthesis_stage"] = "last_step", normalisation: Literal["exact", "case_insensitive", "stripped"] = "stripped")`.
- The tool reads each JSON via `import_chain` semantics (read-only), extracts the conclusion per the `extract` policy, and returns vote counts plus the winning conclusion.
- The library still does not call the LLM. The caller is responsible for producing `k` separate chains (e.g., by running their Bedrock loop `k` times at temperature 0.7).
- Optional follow-on: a `merge_chains_into_consensus` helper that combines all input chains' evidence and assumptions into a single output chain with the winning conclusion.

**Pros:**
- Honest about Wang et al. 2022 — we add the *voting* piece (the part the library can do without becoming an LLM caller), and tell the user to do the *sampling* piece.
- Stays out of the LLM-calling layer (preserves ADR-0011 Alternative D's rejection).
- Composes cleanly with existing export/import.
- Zero dependencies (ADR-0002 preserved) — voting is just counting.

**Cons:**
- "Extract the conclusion" has policy choices. `last_step` and `synthesis_stage` are reasonable defaults but the conclusion-extraction problem is itself non-trivial. Some chains end with a question; others end with the actual answer in the Synthesis stage.
- The caller still has to do the LLM loop themselves. We are providing one of the two pieces, not the whole pattern.
- Tool surface grows.

**Decision:** Recommended.

### Alternative C: Add an LLM-calling self-consistency loop

**Approach:** Add `AsyncSelfConsistentProcessor` that runs `AsyncChainOfThoughtProcessor.process_tool_loop` `k` times in parallel at the caller-configured temperature, then votes.

**Pros:**
- True self-consistency in one library call.
- Matches Wang et al. 2022 mechanics directly.

**Cons:**
- Pulls the library deep into the LLM-calling layer. ADR-0011 Alternative D rejected this stance.
- ADR-0002 (zero dependencies) is preserved only if we keep using boto3 — which we do indirectly through the caller's client, but `AsyncSelfConsistentProcessor` would need to know about temperature, sampling parameters, and parallel call semantics.
- `k` calls means `k×` cost. The library now has opinions about caller spend.
- Bedrock-only (currently). To support OpenAI/Anthropic we'd need provider-specific sampling logic.

**Decision:** Rejected. The cost-and-layer story is wrong.

### Alternative D: Cross-link to a separate `self-consistency` companion library

**Approach:** Don't add anything here. Create a sibling library that depends on `chain-of-thought-tool` and provides self-consistency.

**Pros:**
- Clean layering.
- Keeps this library focused.

**Cons:**
- One more library to maintain for a feature that is genuinely small (Option B is one tool).
- The companion library would mostly just be Option B in a different package. Forcing the user to import two packages for one operation is friction.

**Decision:** Rejected. Option B is small enough to live here.

---

## Consequences

### Positive (under recommended Option B)

1. The library closes one named gap with canonical CoT (Wang et al. 2022) without pulling itself into the LLM-calling layer.
2. The voting primitive composes with the existing export/import surface — incremental, not a new architectural pattern.
3. Users get a story: "to do self-consistency, loop your LLM call `k` times, export each, vote with `vote_on_conclusions`."
4. ADR-0002 preserved.

### Negative (under recommended Option B)

1. Conclusion-extraction has policy choices we will keep getting questions about. Each policy is an opinion baked into the library.
2. The user is still responsible for the `k`-sample loop. Some users will misunderstand and expect `vote_on_conclusions` to do the sampling.
3. One more tool in `TOOL_SPECS`. ADR-0006's stability contract now covers it.
4. Conclusion-extraction logic is non-trivial when chains have heterogeneous final structure. Edge cases (chain has no Synthesis stage; last step is a question; multiple competing final claims) need defined behaviour.

### Tradeoffs

<!-- adr:tradeoffs -->
```yaml
tradeoffs:
  - gain: Wang et al. 2022 voting primitive added without making the library an LLM caller
    cost: Conclusion-extraction policy decisions plus user-side responsibility for the k-sample loop
    acceptable: true
    rationale: Splitting Wang et al.'s mechanism along the right seam — voting belongs in the state-tracker layer; sampling belongs above it.
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

- Wang, X., et al. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* arXiv:2203.11171, ICLR 2023. https://arxiv.org/abs/2203.11171
- Wei, J., et al. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* arXiv:2201.11903, NeurIPS 2022.
- `chain_of_thought/core.py` — `export_chain` / `import_chain` (would be reused read-side)
- `chain_of_thought/__init__.py` — current `TOOL_SPECS` (would gain one entry)
- ADR-0002 (Zero External Dependency Philosophy) — preserved by Option B
- ADR-0010 (Canonical Reference Mapping)
- ADR-0011 (Library Identity) — Alternative D rejected the LLM-caller stance that Option C would reopen
