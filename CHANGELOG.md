# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-05-14

### Added
- God module decomposition: split monolithic `core.py` into 7 focused modules
  - `handlers.py` - Handler wrapper functions
  - `bedrock.py` - AsyncChainOfThoughtProcessor, StopReasonHandler
  - `concurrency.py` - ThreadAwareChainOfThought, RateLimiter
  - `validators.py` - ParameterValidator (XSS, Unicode, injection prevention)
  - `security.py` - RequestValidator (Bedrock request sanitization)
  - `auxiliary.py` - HypothesisGenerator, AssumptionMapper, ConfidenceCalibrator
- PEP 561 `py.typed` marker for type checking support
- 16 Architecture Decision Records (ADR-0001 through ADR-0016)
- Comprehensive test suite: 321 tests with 80% coverage enforcement
- Security vulnerability tests (XSS, injection, JSON serialization, memory leaks)
- Thread safety and concurrency tests
- Async timeout and edge case tests
- Rate limiting with token bucket algorithm
- RequestValidator for Bedrock API request sanitization
- ServiceRegistry for dependency injection

### Changed
- ADR-0001: Removed model ID allowlist from security validation
- ADR-0007: Auxiliary tools documented as structural scaffolding with honest capability descriptions

## [0.2.0] - 2026-05-12

### Added
- `ParameterValidator` with comprehensive input validation
- `_safe_json_dumps` with security sanitization
- `ServiceRegistry` for thread-safe dependency injection
- `create_generic_handler` factory pattern for cross-cutting concerns

## [0.1.0] - 2026-05-10

### Added
- Initial release
- `ChainOfThought` core reasoning chain
- `ThoughtStep` dataclass with confidence tracking
- `ThreadAwareChainOfThought` for multi-tenant isolation
- `AsyncChainOfThoughtProcessor` for Bedrock tool loops
- 8 tool specifications in AWS Bedrock Converse API format
- Example Bedrock integration script
- Zero external dependencies
