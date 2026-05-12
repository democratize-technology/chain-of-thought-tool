# ADR-0001: Remove model ID allowlist from security validation

## Status

Accepted

## Date

2026-05-12

## Context

The `RequestValidator` in `security.py` maintained a hardcoded allowlist of Bedrock model ID patterns that only matched Anthropic Claude 3 and Claude 3.5 Sonnet models. This allowlist rejected:

- Cross-region inference profiles (`us.`, `eu.`, `apac.` prefixed)
- Any Anthropic model newer than Claude 3.5 Sonnet (Opus 4, Sonnet 4.5, Haiku 4.5, Opus 4.7)
- All non-Anthropic Bedrock providers (Meta Llama, DeepSeek, Amazon, Cohere, Mistral, AI21)

The error message ("Security validation failed") implied a security boundary was crossed when the caller simply chose a model not in the static list.

## Decision

Remove the model ID allowlist entirely. `_validate_model_id` now accepts any non-empty string and trusts the caller. AWS Bedrock rejects invalid model IDs at the API layer with accurate error messages.

## Rationale

1. **Wrong-layer policing.** A chain-of-thought reasoning library cannot know what models its caller is authorized to use. Model authorization lives in AWS IAM, billing setup, or the caller's own policy code.

2. **Static staleness.** Every new Bedrock model release silently broke the library until the allowlist was updated. This is a treadmill that serves no security purpose.

3. **Misleading errors.** "Security validation failed" for a valid model ID misrepresents what happened. No security boundary was crossed.

4. **Defeats cross-model diversity.** The library's value is enabling structured reasoning across LLMs. Gating model IDs prevents the cross-model diversity that makes structured reasoning valuable.

## Consequences

- Any non-empty string model ID is now accepted through the validator
- AWS Bedrock remains the authoritative rejection point for invalid model IDs
- Downstream consumers (devil-advocate-mcp, etc.) no longer need custom SecurityConfig overrides to use modern models
- The `allowed_model_patterns` field is removed from `SecurityConfig`
