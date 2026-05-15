# Architecture Decision Records

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0001](0001-remove-model-id-allowlist.md) | Remove model ID allowlist from security validation | Accepted | 2026-05-12 |
| [0002](0002-zero-external-dependency-philosophy.md) | Zero External Dependency Philosophy | Proposed | 2026-05-14 |
| [0003](0003-bedrock-converse-api-as-primary-tool-spec-format.md) | Bedrock Converse API as Primary Tool Spec Format | Proposed | 2026-05-14 |
| [0004](0004-weakvaluedictionary-hybrid-for-multi-tenant-isolation.md) | WeakValueDictionary Hybrid for Multi-Tenant Isolation | Proposed | 2026-05-14 |
| [0005](0005-handler-factory-with-cross-cutting-concerns.md) | Handler Factory with Cross-Cutting Concerns | Proposed | 2026-05-14 |
| [0006](0006-3-tier-api-stability-contract.md) | 3-Tier API Stability Contract | Proposed | 2026-05-14 |
| [0007](0007-auxiliary-reasoning-tools-as-structural-scaffolding.md) | Auxiliary Reasoning Tools as Structural Scaffolding | Accepted | 2026-05-14 |
| [0008](0008-5-canonical-reasoning-stages-domain-model.md) | 5 Canonical Reasoning Stages Domain Model | Proposed | 2026-05-14 |
| [0009](0009-async-bedrock-tool-loop-orchestration-pattern.md) | Async Bedrock Tool Loop Orchestration Pattern | Proposed | 2026-05-14 |
| [0010](0010-canonical-reference-mapping-and-drift-index.md) | Canonical Reference Mapping and Drift Index | Proposed | 2026-05-14 |
| [0011](0011-library-identity--stateful-reasoning-tracker-vs-wei-et-al-prompting-technique.md) | Library Identity — Stateful Reasoning Tracker vs Wei et al. Prompting Technique | Proposed | 2026-05-14 |
| [0012](0012-branching-parity-with-mcp-sequential-thinking.md) | Branching Parity with MCP Sequential-Thinking | Proposed | 2026-05-14 |
| [0013](0013-explicit-revision-semantics--isrevision-and-revisesthought.md) | Explicit Revision Semantics — isRevision and revisesThought | Proposed | 2026-05-14 |
| [0014](0014-self-consistency-sampling--wang-2022-extension.md) | Self-Consistency Sampling — Wang 2022 Extension | Proposed | 2026-05-14 |
| [0015](0015-linear-chain-vs-emergent-dag-via-dependencies-and-contradicts.md) | Linear Chain vs Emergent DAG via dependencies and contradicts | Proposed | 2026-05-14 |
| [0016](0016-auxiliary-tool-genealogy--not-from-cot-lineage.md) | Auxiliary Tool Genealogy — Not From CoT Lineage | Proposed | 2026-05-14 |

## Drift-Analysis Cluster (Canonical CoT and MCP Sequential-Thinking vs. This Library)

ADRs 0010–0016 form a coherent set. They audit this library against two canonical references — Wei et al. 2022 (the name origin) and the Model Context Protocol `sequential-thinking` server (the shape origin) — and surface where we deliberately or accidentally diverge from each.

- **ADR-0010** is the index. Read it first. It establishes the dual-anchor model.
- **ADR-0011** addresses the foundational naming-vs-shape mismatch: we are called "Chain of Thought" but we are structurally a stateful tool, not a prompting technique.
- **ADR-0012** and **ADR-0013** address the two concrete gaps with the MCP `sequential-thinking` reference: branching and explicit revision.
- **ADR-0014** addresses the largest missing canonical-CoT extension: self-consistency sampling (Wang et al. 2022).
- **ADR-0015** names the topology ambiguity created by the `dependencies` and `contradicts` fields (the chain is, in practice, a primarily-linear DAG).
- **ADR-0016** documents the intellectual genealogy of the auxiliary tools, which trace to abductive-reasoning, critical-thinking, and calibration traditions rather than CoT.

## Lifecycle

```
proposed -> accepted -> deprecated
                \-> superseded (by newer ADR)
```

## Numbering

- **ADR-0001**: Pre-existing decision (retroactively documented)
- **ADR-0002 through ADR-0009**: Architecture decisions documented from codebase analysis (spec/PRD phase)
- **ADR-0010 through ADR-0016**: Canonical-reference drift analysis (this cluster)
