---
id: ADR-0004
title: WeakValueDictionary Hybrid for Multi-Tenant Isolation
status: accepted
date: 2026-05-14
decision_makers:
  - Engineering
category:
  - architecture
supersedes: null
superseded_by: null
related: []
tags: [concurrency, memory-management, multi-tenant]
---

# ADR 0004: WeakValueDictionary Hybrid for Multi-Tenant Isolation

## Context

### The Problem

`ThreadAwareChainOfThought` manages per-conversation `ChainOfThought` instances in a long-running server process. Each conversation gets its own isolated chain with its own steps, metadata, and lock. Two opposing forces govern the storage:

1. Active conversations must survive GC -- a mid-reasoning chain cannot disappear between tool calls.
2. Abandoned conversations must not leak memory -- if a client disconnects or a conversation ID is never referenced again, the `ChainOfThought` instance and its step history must eventually be reclaimed.

A single storage strategy cannot satisfy both: pure strong references leak on abandonment; pure weak references risk mid-conversation GC if the caller lets its local reference drop between calls.

### Constraints

- Zero external dependencies (ADR-0002). No Redis, Memcached, or external cache.
- No background threads or timer-based cleanup. The library must not spawn daemon threads.
- Thread-safe access from concurrent request handlers. A class-level `RLock` protects the dictionaries.
- Each `ChainOfThought` has its own per-instance `RLock` for its internal state, independent of the class-level lock.

## Decision

Use a hybrid two-dictionary storage pattern on `ThreadAwareChainOfThought`:

```python
_instances: WeakValueDictionary[str, ChainOfThought] = WeakValueDictionary()
_strong_refs: Dict[str, ChainOfThought] = {}
_lock = threading.RLock()  # class-level, protects both dicts
```

- `_strong_refs` holds active conversations. Entries here prevent GC entirely.
- `_instances` holds weak references. Once an entry is removed from `_strong_refs`, the `ChainOfThought` is eligible for GC the next time Python's collector runs.
- `for_conversation(conv_id)` checks `_strong_refs` first, then `_instances`, then creates a new instance stored in both.
- `release_conversation(conv_id)` removes from `_strong_refs` only. The weak reference persists until GC collects the object.
- `clear_conversation(conv_id)` removes from both dictionaries immediately.

### Requirements

<!-- adr:requirements -->
requirements:
  - id: REQ-0004-1
    category: architecture
    description: "for_conversation MUST return the same ChainOfThought instance for the same conversation_id"
    verification:
      type: grep
      pattern: "WeakValueDictionary"
      paths:
        - "chain_of_thought/concurrency.py"
      expect: present
  - id: REQ-0004-2
    category: architecture
    description: "release_conversation MUST remove the strong reference without destroying the instance"
    verification:
      type: grep
      pattern: "release_conversation"
      paths:
        - "chain_of_thought/concurrency.py"
      expect: present
  - id: REQ-0004-3
    category: architecture
    description: "After release_conversation and no external references, instance MUST become eligible for GC"
    verification:
      type: grep
      pattern: "_strong_refs"
      paths:
        - "chain_of_thought/concurrency.py"
      expect: present
  - id: REQ-0004-4
    category: architecture
    description: "All dictionary mutations MUST occur under the class-level RLock"
    verification:
      type: grep
      pattern: "class.*RLock|_lock.*acquire"
      paths:
        - "chain_of_thought/concurrency.py"
      expect: present
  - id: REQ-0004-5
    category: architecture
    description: "No background threads, timers, or external dependencies for cleanup"
    verification:
      type: grep_negative
      pattern: 'threading\.Timer|threading\.Thread|schedule'
      paths:
        - "chain_of_thought/concurrency.py"
      expect: absent
<!-- /adr:requirements -->

---

## Alternatives Considered

### Alternative 1: Pure Strong References with Explicit Cleanup

**Approach:** A single `Dict[str, ChainOfThought]`. Conversations live until `clear_conversation` is called.

**Pros:**
- Simpler implementation. One dictionary, no weakref import.
- Deterministic lifecycle. Instance exists until explicitly removed.

**Cons:**
- Memory leak if caller forgets to call `clear_conversation` or crashes before doing so.
- In a long-running server with thousands of conversations, this is a slow bomb.

**Decision:** Rejected. Memory safety cannot depend on caller discipline. One forgotten cleanup in a server handling 10k conversations is a production incident.

### Alternative 2: TTL-Based Cleanup with Background Thread

**Approach:** Each entry gets a timestamp. A background daemon thread periodically scans and evicts expired entries.

**Pros:**
- Deterministic eviction timing. Expired conversations cleaned at known intervals.
- Works without caller cooperation.

**Cons:**
- Introduces a background thread, violating the no-daemon-threads constraint.
- Premature cleanup risk: a slow conversation could time out mid-reasoning.
- Thread coordination complexity: the cleanup thread and request handlers contend on the same dictionary.
- Timer tuning becomes an operational burden (what TTL? what scan interval?).

**Decision:** Rejected. Background threads are a significant complexity increase for a library that must stay zero-dependency and thread-pool-transparent.

### Alternative 3: LRU Cache with Size Limit

**Approach:** `functools.lru_cache` or manual ordered-dict eviction. Oldest conversations evicted when capacity is reached.

**Pros:**
- Bounded memory by design.
- No background threads.

**Cons:**
- Evicts by recency, not by abandonment. An active but long-running conversation could be evicted if newer ones fill the cache.
- Fixed capacity is a tuning problem. Too small = premature eviction. Too large = delayed cleanup.
- `lru_cache` is not designed for per-instance threading patterns across a class hierarchy.

**Decision:** Rejected. LRU eviction is the wrong policy. We need abandonment-based cleanup, not recency-based eviction.

### Alternative 4: External Store (Redis, Memcached)

**Approach:** Serialize `ChainOfThought` state to an external cache with TTL.

**Pros:**
- Proven, scalable session management.
- TTL-based cleanup handled by the store.
- Survives process restarts.

**Cons:**
- Violates ADR-0002 (zero external dependencies).
- Serialization/deserialization overhead on every tool call.
- Introduces network latency into a purely in-memory reasoning step.
- Requires callers to provision and manage external infrastructure.

**Decision:** Rejected. Hard dependency violation. The library must work with nothing but Python 3.8+.

---

## Consequences

### Positive

1. **Automatic GC of abandoned conversations.** Once `release_conversation` is called and no external references remain, Python's garbage collector reclaims the `ChainOfThought` instance. No caller discipline required beyond calling `release_conversation` when done.
2. **Explicit control via `release_conversation`.** Callers can release conversations they know are complete, while the GC handles cases they forget or crash before reaching.
3. **No background threads or timers.** Cleanup piggybacks on Python's existing GC cycle. Zero additional infrastructure.
4. **Thread-safe by design.** A single class-level `RLock` serializes all dictionary mutations. Each `ChainOfThought` instance has its own `RLock` for its internal state.
5. **Zero dependencies.** Uses only `weakref.WeakValueDictionary` from the standard library.

### Negative

1. **Non-deterministic GC timing.** After `release_conversation`, the instance survives until the next GC cycle. There is no guarantee of immediate reclamation. Under memory pressure, CPython typically collects quickly, but this is not contractually guaranteed.
2. **Weak reference complexity.** Callers must understand that `for_conversation` can return a new instance if the previous one was collected. Holding a local reference across calls is required for mid-conversation continuity.
3. **Performance overhead.** `WeakValueDictionary` has higher lookup overhead than a plain `dict` due to weak reference indirection and GC callback management. For this library's scale (tens to hundreds of concurrent conversations), this is negligible.
4. **Dual-dictionary bookkeeping.** Both `_instances` and `_strong_refs` must be kept in sync. Every write path must update both. The `clear_conversation` and `clear_all_conversations` methods must clear both. This is a maintenance surface.

### Tradeoffs

<!-- adr:tradeoffs -->
```yaml
tradeoffs:
  - gain: Automatic memory reclamation without caller discipline
    cost: Non-deterministic GC timing (no guaranteed immediate cleanup)
    acceptable: true
    rationale: >
      Chain-of-thought instances are ephemeral reasoning state. A few extra
      seconds of memory retention after release is harmless. The alternative
      (memory leak) is far worse for long-running servers.

  - gain: No background threads or timers for cleanup
    cost: Dual-dictionary bookkeeping (_instances + _strong_refs)
    acceptable: true
    rationale: >
      The bookkeeping is contained in five methods on one class. The
      complexity cost is fixed and auditable, unlike a background thread
      that introduces concurrency bugs.

  - gain: Zero external dependencies
    cost: Cannot persist conversations across process restarts
    acceptable: true
    rationale: >
      Conversations are transient reasoning chains. If the process restarts,
      the reasoning is already interrupted. Persistence is the caller's job.

  - gain: Explicit release_conversation control
    cost: Callers must remember to call release_conversation for optimal cleanup
    acceptable: true
    rationale: >
      Forgetting to call release_conversation does not cause a leak -- it only
      delays GC until the strong ref is the sole remaining reference. This is
      strictly better than pure strong refs where forgetting causes a permanent
      leak.
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

- [ThreadAwareChainOfThought implementation](../../chain_of_thought/concurrency.py)
- [Python weakref.WeakValueDictionary documentation](https://docs.python.org/3/library/weakref.html#weakref.WeakValueDictionary)
- ADR-0002: Zero External Dependency Philosophy
