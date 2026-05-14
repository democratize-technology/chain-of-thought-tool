---
id: ADR-0009
title: Async Bedrock Tool Loop Orchestration Pattern
status: proposed
date: 2026-05-14
decision_makers:
  - Engineering
category:
  - architecture
supersedes: null
superseded_by: null
related:
  - ADR-0003
  - ADR-0004
  - ADR-0005
tags: [async, bedrock, integration, orchestration]
---

# ADR 0009: Async Bedrock Tool Loop Orchestration Pattern

## Context

### The Problem

AWS Bedrock uses the `stopReason` field in its Converse API response to signal whether the model wants to invoke a tool (`"tool_use"`) or has finished generating (`"end_turn"`). When `stopReason` is `"tool_use"`, the caller must execute each tool, collect results, append them to the message history, and call `converse` again. This loop continues until the model signals `"end_turn"` or an error occurs.

Every consumer that integrates Bedrock with chain-of-thought tools must implement this loop. Without a shared implementation, each consumer independently handles:

- Parsing `stopReason` and branching on its value
- Iterating over multiple tool use requests in a single response
- Routing tool names to handler functions
- Appending assistant messages and tool result messages in the correct order
- Guarding against infinite tool loops
- Handling timeouts on both AWS calls and individual tool executions
- Returning error status without crashing the loop

This is repetitive, error-prone work that has nothing to do with the consumer's actual domain logic.

### Constraints

- The library has a zero-external-dependency philosophy (ADR-0002). The orchestration must be pure Python.
- The library targets AWS Bedrock as its primary integration surface (ADR-0003).
- Multi-tenant isolation requires per-conversation state (ADR-0004).
- Handler invocation uses the generic factory pattern with rate limiting (ADR-0005).

---

## Decision

Implement `AsyncChainOfThoughtProcessor` as the single entry point for Bedrock tool loop orchestration. The processor manages the full lifecycle:

1. **Request validation.** The initial request passes through `RequestValidator.validate_and_sanitize_request()` to prevent injection attacks before any AWS call.
2. **AWS call with timeout.** Each `converse` call executes via `_safe_aws_call`, which wraps the synchronous boto3 call in `run_in_executor` with `asyncio.wait_for` and a configurable timeout (default 30s).
3. **Tool execution loop.** On `stopReason: "tool_use"`, the processor iterates over each `toolUse` content item, dispatches to `StopReasonHandler.execute_tool_call()`, collects results, and appends them to the message history.
4. **End turn.** On `stopReason: "end_turn"`, the processor checks `should_continue_reasoning` on the stop handler, then returns the response.
5. **Iteration guard.** The loop hard-stops at 20 iterations. On exhaustion, it returns a synthetic response with `stopReason: "max_tokens"`.
6. **Timeout protection at three levels:**
   - Per AWS call: 30s default (`aws_call_timeout`)
   - Per tool call: 10s default (`tool_call_timeout`)
   - Overall process: 2x AWS timeout default, applied by `process_tool_loop_with_timeout`
7. **Error-as-status.** Timeouts and exceptions during tool execution produce `toolResult` entries with `"status": "error"` rather than raising exceptions. This keeps the loop alive so the model can react to the failure.

The processor creates its own `ChainOfThought` instance per construction, ensuring conversation isolation. The `BedrockStopReasonHandler` receives this instance to bind tool handlers to the correct chain.

`StopReasonHandler` is an abstract base class with two methods: `should_continue_reasoning(chain)` and `execute_tool_call(tool_name, tool_args)`. `BedrockStopReasonHandler` is the concrete implementation that maps tool names to handler functions and runs synchronous handlers via `run_in_executor`.

### Requirements

<!-- adr:requirements -->
requirements:
  - id: REQ-0009-1
    category: architecture
    description: "AsyncChainOfThoughtProcessor must validate requests through RequestValidator"
    verification:
      type: grep
      pattern: "RequestValidator"
      paths:
        - "chain_of_thought/core.py"
      expect: present
  - id: REQ-0009-2
    category: architecture
    description: "AWS converse calls must be wrapped in configurable timeout"
    verification:
      type: grep
      pattern: "aws_call_timeout"
      paths:
        - "chain_of_thought/core.py"
      expect: present
  - id: REQ-0009-3
    category: architecture
    description: "Tool handler invocations must be wrapped in configurable timeout"
    verification:
      type: grep
      pattern: "tool_call_timeout"
      paths:
        - "chain_of_thought/core.py"
      expect: present
  - id: REQ-0009-4
    category: architecture
    description: "Tool execution failures must return error-status results not raise exceptions"
    verification:
      type: grep
      pattern: "error.*tool_result|toolResult.*error"
      paths:
        - "chain_of_thought/core.py"
      expect: present
  - id: REQ-0009-5
    category: architecture
    description: "Loop must hard-stop at configurable maximum iteration count"
    verification:
      type: grep
      pattern: "max_iterations|MAX_ITERATIONS"
      paths:
        - "chain_of_thought/core.py"
      expect: present
  - id: REQ-0009-6
    category: architecture
    description: "Each processor instance must own its own ChainOfThought instance"
    verification:
      type: grep
      pattern: 'ChainOfThought\(\)'
      paths:
        - "chain_of_thought/core.py"
      expect: present
  - id: REQ-0009-7
    category: architecture
    description: "StopReasonHandler must be an abstract base class"
    verification:
      type: grep
      pattern: "class StopReasonHandler|ABC"
      paths:
        - "chain_of_thought/core.py"
      expect: present
<!-- /adr:requirements -->

---

## Alternatives Considered

### Alternative 1: Caller Manages the Loop

**Approach:** Each consumer writes its own while-loop that checks `stopReason`, dispatches tools, and manages timeouts.

**Pros:**
- Maximum flexibility per consumer
- No abstraction mismatch if consumer has unusual requirements

**Cons:**
- Every consumer reinvents timeout handling, iteration guards, error-as-status, and message append ordering
- Bug fixes to the loop pattern must be applied N times across N consumers
- The most error-prone part of Bedrock integration (the loop) gets zero shared testing

**Decision:** Rejected. The loop logic is identical across consumers and contains subtle ordering requirements (assistant message before tool results) that are easy to get wrong.

### Alternative 2: Callback / Event-Driven Architecture

**Approach:** Register callbacks for `on_tool_use`, `on_end_turn`, `on_timeout`, etc. The processor emits events rather than controlling flow directly.

**Pros:**
- Extensible for cross-cutting concerns (logging, metrics) without modifying the processor
- Decouples orchestration from handler dispatch

**Cons:**
- Control flow is harder to follow: execution jumps between the processor and registered callbacks
- Timeout and iteration guard semantics become implicit in callback registration order
- Adds complexity without clear benefit for the single-consumer Bedrock use case
- Debugging requires tracing through callback chains rather than reading a single method

**Decision:** Rejected. The procedural loop in `process_tool_loop` is linear and auditable. Callbacks would obscure the exact sequence of AWS-call, tool-dispatch, message-append without providing enough extensibility upside for a library at this scale.

### Alternative 3: Synchronous Processor

**Approach:** A blocking `process_tool_loop` that calls `converse` synchronously and handles tools in the same thread.

**Pros:**
- Simpler implementation (no `run_in_executor`, no `asyncio.to_thread`)
- No event loop requirement

**Cons:**
- Blocks the calling thread for the entire loop duration (potentially minutes)
- Cannot handle concurrent conversations in a single-threaded server (e.g., ASGI)
- Timeouts require threading rather than asyncio, which is harder to compose
- The library already uses `asyncio` for `StopReasonHandler.execute_tool_call`

**Decision:** Rejected. Production Bedrock integrations run in async servers. A synchronous processor would force every consumer to wrap it in thread management, which defeats the purpose of providing the abstraction.

### Alternative 4: Framework-Level Integration (LangChain / LlamaIndex Wrapper)

**Approach:** Implement the tool loop as a LangChain agent executor or LlamaIndex tool abstraction.

**Pros:**
- Immediate access to ecosystem features (memory, retrieval, tracing)
- Community familiarity with the framework patterns

**Cons:**
- Violates the zero-external-dependency philosophy (ADR-0002)
- Framework abstractions impose their own tool interface, losing the direct Bedrock Converse API alignment (ADR-0003)
- Framework version churn becomes the library's dependency problem
- Consumers not using that framework must still extract the raw orchestration logic

**Decision:** Rejected. The library's value proposition is zero-dependency Bedrock integration. Wrapping a framework contradicts that directly.

---

## Consequences

### Positive

1. **Drop-in async Bedrock integration.** A consumer creates one `AsyncChainOfThoughtProcessor`, calls `process_tool_loop_with_timeout`, and receives the final response. No loop code required.
2. **Timeout protection prevents runaway loops.** Three timeout levels (AWS call, tool call, overall) ensure no single request can hang indefinitely.
3. **Error-as-status keeps the loop alive.** When a tool times out or raises, the error is returned to the model as a tool result. The model can adjust its reasoning rather than the entire loop crashing.
4. **Per-conversation isolation.** Each processor owns its own `ChainOfThought` instance. No shared mutable state between conversations.
5. **Testable boundaries.** `StopReasonHandler` is abstract, allowing test doubles that verify loop behavior without real AWS calls or tool execution.

### Negative

1. **Bedrock-specific.** The processor calls `bedrock_client.converse()` directly. OpenAI or Anthropic API users must write their own loop or adapt the pattern.
2. **Async-only.** Consumers must run in an asyncio event loop. Synchronous callers need `asyncio.run()` or thread delegation.
3. **Timeout configuration is class-level.** Timeouts are set in `__init__` and apply to all calls on that instance. Per-call timeout overrides are not supported.
4. **Max 20 iterations is hardcoded as a default.** The `max_iterations` parameter on `process_tool_loop` allows override, but the class attribute `_max_iterations` is not configurable via constructor.

### Tradeoffs

<!-- adr:tradeoffs -->
```yaml
tradeoffs:
  - gain: Complete Bedrock tool loop abstraction with timeout guards
    cost: Provider lock-in to AWS Bedrock Converse API surface
    acceptable: true
    rationale: >
      Bedrock is the primary integration target (ADR-0003). The Converse API
      is stable. Provider-generic abstraction would require a lowest-common-
      denominator interface that loses Bedrock-specific features like
      cross-region inference profiles.

  - gain: Error-as-status keeps the loop alive through tool failures
    cost: Consumers must inspect tool result status rather than catching exceptions
    acceptable: true
    rationale: >
      The model needs to see tool failures so it can adjust reasoning.
      Crashing the loop on a single tool timeout wastes all prior reasoning
      steps. The error-as-status pattern matches how Bedrock itself expects
      tool results to be structured.

  - gain: Per-conversation ChainOfThought instance ensures isolation
    cost: No cross-conversation reasoning without explicit import/export
    acceptable: true
    rationale: >
      Conversations are independent by definition. Sharing chain state
      between conversations would require synchronization logic that
      introduces bugs without clear use cases.

  - gain: Abstract StopReasonHandler allows alternative implementations
    cost: BedrockStopReasonHandler is the only concrete implementation
    acceptable: true
    rationale: >
      The abstract base costs nothing at runtime. It enables testing with
      mocks and leaves the door open for non-Bedrock handlers without
      requiring them now (YAGNI).
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

- [AWS Bedrock Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html)
- [ADR-0003: Bedrock Converse API as Primary Tool Spec Format](0003-bedrock-converse-api-as-primary-tool-spec-format.md)
- [ADR-0004: WeakValueDictionary Hybrid for Multi-Tenant Isolation](0004-weakvaluedictionary-hybrid-for-multi-tenant-isolation.md)
- [ADR-0005: Handler Factory with Cross-Cutting Concerns](0005-handler-factory-with-cross-cutting-concerns.md)
