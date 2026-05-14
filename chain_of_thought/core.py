"""
Chain of Thought Tool - Core Implementation
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import threading


import logging
import html
import math
import re
from .validators import ParameterValidator

# Sanitization and Security Limits
MAX_RECURSION_DEPTH = 50
MAX_LIST_SIZE = 100
MAX_STRING_LENGTH = 1000
MAX_JSON_SIZE = 100000  # 100KB limit
MAX_IMPORT_STEPS = 10_000  # DoS prevention: max steps allowed in import_chain

# Confidence and text constants are now in auxiliary.py
# Re-exported below after ServiceRegistry definition


# Configuration for tool handlers that the generic factory can create
TOOL_HANDLERS_CONFIG = {
    'chain_of_thought_step': {
        'service_name': 'chain_of_thought',
        'service_method': 'add_step'
    },
    'get_chain_summary': {
        'service_name': 'chain_of_thought',
        'service_method': 'generate_summary'
    },
    'clear_chain': {
        'service_name': 'chain_of_thought',
        'service_method': 'clear_chain'
    },
    'generate_hypotheses': {
        'service_name': 'hypothesis_generator',
        'service_method': 'generate_hypotheses'
    },
    'map_assumptions': {
        'service_name': 'assumption_mapper',
        'service_method': 'map_assumptions'
    },
    'calibrate_confidence': {
        'service_name': 'confidence_calibrator',
        'service_method': 'calibrate_confidence'
    }
}


def create_generic_handler(
    tool_name: str,
    registry: Optional['ServiceRegistry'] = None,
    rate_limiter: Optional['RateLimiter'] = None,
    client_id: str = "default"
) -> Callable:
    """
    Create a generic handler function for any configured tool.

    This function replaces the individual create_*_handler functions with
    a single configurable implementation that reduces code duplication.

    Args:
        tool_name: Name of the tool to create a handler for
        registry: Service registry to use. If None, uses default global registry.
        rate_limiter: Rate limiter to use. If None, uses global rate limiter.
        client_id: Client identifier for rate limiting.

    Returns:
        Handler function that handles rate limiting and service calls

    Raises:
        ValueError: If tool_name is not configured
    """
    if tool_name not in TOOL_HANDLERS_CONFIG:
        raise ValueError(f"Unknown tool '{tool_name}'. Available tools: {list(TOOL_HANDLERS_CONFIG.keys())}")

    config = TOOL_HANDLERS_CONFIG[tool_name]
    service_name = config['service_name']
    service_method = config['service_method']

    # Use provided rate limiter or global one (deferred import to avoid circular dependency)
    if rate_limiter is not None:
        limiter = rate_limiter
    else:
        from .concurrency import get_global_rate_limiter as _get_rate_limiter
        limiter = _get_rate_limiter()

    def handler(**kwargs) -> str:
        """Generic handler function with rate limiting and service injection."""

        # Check rate limit first
        if not limiter.check_rate_limit(client_id):
            retry_after = limiter.get_retry_after(client_id)
            return _safe_json_dumps({
                "status": "error",
                "message": f"Rate limit exceeded. Retry after {retry_after or 60} seconds.",
                "error_type": "rate_limit_exceeded",
                "retry_after": retry_after
            }, indent=2)

        try:
            service_registry = registry or get_service_registry()
            service = service_registry.get_service(service_name)

            # Call the service method with the provided kwargs
            method = getattr(service, service_method)
            result = method(**kwargs)

            return _safe_json_dumps(result, indent=2)

        except Exception as e:
            return _safe_json_dumps({"status": "error", "message": str(e)}, indent=2)

    return handler


class ServiceCreationError(Exception):
    """Raised when service creation fails in ServiceRegistry."""
    pass


class ServiceRegistry:
    """
    Thread-safe dependency injection container for managing service instances.

    Provides a clean way to manage service lifecycles while maintaining
    global singleton usage.
    """

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._lock = threading.RLock()

    def _register_default_factories(self):
        """Register default factories for all core services."""
        from .auxiliary import HypothesisGenerator, AssumptionMapper, ConfidenceCalibrator
        self._factories.update({
            'chain_of_thought': lambda: ChainOfThought(),
            'hypothesis_generator': lambda: HypothesisGenerator(),
            'assumption_mapper': lambda: AssumptionMapper(),
            'confidence_calibrator': lambda: ConfidenceCalibrator(),
        })

    def register_service(self, name: str, service: Any) -> None:
        """
        Register a service instance.

        Args:
            name: Service name
            service: Service instance to register
        """
        with self._lock:
            self._services[name] = service

    def register_factory(self, name: str, factory: Callable[[], Any]) -> None:
        """
        Register a factory function for lazy service creation.

        Args:
            name: Service name
            factory: Factory function that creates the service
        """
        with self._lock:
            self._factories[name] = factory
            # Remove any existing instance to force recreation
            self._services.pop(name, None)

    def get_service(self, name: str) -> Any:
        """
        Get a service instance, creating it lazily if needed.

        Args:
            name: Service name

        Returns:
            Service instance

        Raises:
            KeyError: If service is not registered
        """
        with self._lock:
            # Return existing instance if available
            if name in self._services:
                return self._services[name]

            # Create new instance using factory
            if name in self._factories:
                try:
                    service = self._factories[name]()

                    # Validate that factory returned a valid service
                    if service is None:
                        raise ServiceCreationError(
                            f"Failed to create service '{name}': factory returned None"
                        )

                    self._services[name] = service
                    return service

                except Exception as e:
                    # Log the error for debugging
                    logging.error(f"Failed to create service '{name}': {type(e).__name__}: {str(e)}")
                    raise ServiceCreationError(
                        f"Failed to create service '{name}': {str(e)}"
                    ) from e

            raise KeyError(f"Service '{name}' not registered")

    def has_service(self, name: str) -> bool:
        """Check if a service is registered."""
        with self._lock:
            return name in self._factories

    def clear_service(self, name: str) -> None:
        """Clear a service instance (will be recreated on next access)."""
        with self._lock:
            self._services.pop(name, None)

    def clear_all_services(self) -> None:
        """Clear all service instances."""
        with self._lock:
            self._services.clear()

    def initialize_default_services(self):
        """Initialize default service factories after all classes are defined."""
        self._register_default_factories()


# Global service registry
_default_registry = ServiceRegistry()


def get_service_registry() -> ServiceRegistry:
    return _default_registry


# Re-export auxiliary classes for backward compatibility.
# Placed after ServiceRegistry to avoid circular imports at module load time.
from .auxiliary import (  # noqa: E402
    Hypothesis,
    HypothesisGenerator,
    Assumption,
    AssumptionMapper,
    ConfidenceAssessment,
    ConfidenceCalibrator,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    MAX_PREDICTION_WORDS,
)


@dataclass
class ThoughtStep:
    """Represents a single step in the chain of thought."""
    thought: str
    step_number: int
    total_steps: int
    reasoning_stage: str = "Analysis"
    confidence: float = 0.8
    next_step_needed: bool = True
    dependencies: Optional[List[int]] = None
    contradicts: Optional[List[int]] = None
    evidence: Optional[List[str]] = None
    assumptions: Optional[List[str]] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.dependencies is None:
            self.dependencies = []
        if self.contradicts is None:
            self.contradicts = []
        if self.evidence is None:
            self.evidence = []
        if self.assumptions is None:
            self.assumptions = []


class ChainOfThought:

    def __init__(self):
        self.steps: List[ThoughtStep] = []
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "total_confidence": 0.0
        }
        self.validator = ParameterValidator()
        self._lock = threading.RLock()  # For thread safety
      
    def _validate_and_extract_params(
        self,
        thought: str,
        step_number: int,
        total_steps: int,
        next_step_needed: bool,
        reasoning_stage: str = "Analysis",
        confidence: float = 0.8,
        dependencies: Optional[List[int]] = None,
        contradicts: Optional[List[int]] = None,
        evidence: Optional[List[str]] = None,
        assumptions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Validate and extract input parameters for chain step."""
        return self.validator.validate_input(
            thought=thought,
            step_number=step_number,
            total_steps=total_steps,
            next_step_needed=next_step_needed,
            reasoning_stage=reasoning_stage,
            confidence=confidence,
            dependencies=dependencies,
            contradicts=contradicts,
            evidence=evidence,
            assumptions=assumptions
        )

    def _create_thought_step(self, validated_params: Dict[str, Any]) -> ThoughtStep:
        """Create a ThoughtStep instance from validated parameters."""
        return ThoughtStep(
            thought=validated_params["thought"],
            step_number=validated_params["step_number"],
            total_steps=validated_params["total_steps"],
            reasoning_stage=validated_params["reasoning_stage"],
            confidence=validated_params["confidence"],
            next_step_needed=validated_params["next_step_needed"],
            dependencies=validated_params["dependencies"],
            contradicts=validated_params["contradicts"],
            evidence=validated_params["evidence"],
            assumptions=validated_params["assumptions"]
        )

    def _handle_step_revision(self, step_number: int, validated_params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle revision of an existing step."""
        for i, step in enumerate(self.steps):
            if step.step_number == step_number:
                # This is a revision
                self.steps[i] = self._create_thought_step(validated_params)
                self._update_metadata()
                return self._generate_feedback(self.steps[i], is_revision=True)
        # If we reach here, the step number wasn't found - this shouldn't happen in normal operation
        # But can occur in race conditions during concurrent access
        return None

    def _handle_new_step(self, validated_params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle creation of a new step."""
        step = self._create_thought_step(validated_params)
        self.steps.append(step)
        self._update_metadata()
        return self._generate_feedback(step, is_revision=False)

    def add_step(
        self,
        thought: str,
        step_number: int,
        total_steps: int,
        next_step_needed: bool,
        reasoning_stage: str = "Analysis",
        confidence: float = 0.8,
        dependencies: Optional[List[int]] = None,
        contradicts: Optional[List[int]] = None,
        evidence: Optional[List[str]] = None,
        assumptions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Add a new step to the chain of thought.

        Returns analysis and feedback for the step.
        """
        with self._lock:
            validated_params = self._validate_and_extract_params(
                thought, step_number, total_steps, next_step_needed,
                reasoning_stage, confidence, dependencies, contradicts,
                evidence, assumptions
            )

            # Check if this is a revision of an existing step
            revision_result = self._handle_step_revision(
                validated_params["step_number"], validated_params
            )
            if revision_result:
                return revision_result

            # Handle new step
            return self._handle_new_step(validated_params)
    
    def _generate_feedback(self, step: ThoughtStep, is_revision: bool) -> Dict[str, Any]:
        """Generate feedback and guidance for the thought step."""
        
        feedback_parts = []
        
        stage_guidance = {
            "Problem Definition": "Foundation established. Ensure the problem is clearly scoped.",
            "Research": "Gathering information. Consider multiple sources and perspectives.",
            "Analysis": "Breaking down components. Look for patterns and relationships.",
            "Synthesis": "Integrating insights. Focus on connections and implications.",
            "Conclusion": "Finalizing reasoning. Ensure conclusions address the initial problem."
        }
        
        if step.reasoning_stage in stage_guidance:
            feedback_parts.append(stage_guidance[step.reasoning_stage])
        
        if step.confidence < 0.5:
            feedback_parts.append("Low confidence detected. Consider gathering more evidence.")
        elif step.confidence > 0.9:
            feedback_parts.append("High confidence. Ensure assumptions are well-founded.")
        
        if step.dependencies:
            feedback_parts.append(f"Building on steps: {', '.join(map(str, step.dependencies))}")
        
        if step.contradicts:
            feedback_parts.append(f"Contradicts steps: {', '.join(map(str, step.contradicts))}. Consider reconciliation.")
        
        progress = step.step_number / step.total_steps
        if progress >= 0.8 and step.next_step_needed:
            feedback_parts.append("Approaching conclusion. Consider synthesis of insights.")
        
        return {
            "status": "success",
            "step_processed": step.step_number,
            "progress": f"{step.step_number}/{step.total_steps}",
            "confidence": step.confidence,
            "feedback": " ".join(feedback_parts),
            "next_step_needed": step.next_step_needed,
            "total_steps_recorded": len(self.steps),
            "is_revision": is_revision
        }
    
    def _update_metadata(self):
        """Update chain metadata based on current steps."""
        if self.steps:
            total_confidence = sum(s.confidence for s in self.steps) / len(self.steps)
            self.metadata["total_confidence"] = round(total_confidence, 3)
            self.metadata["last_updated"] = datetime.now().isoformat()
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive summary of the chain of thought."""
        with self._lock:
            if not self.steps:
                return {
                    "status": "empty",
                    "message": "No thought steps have been recorded yet."
                }

            # Organize by stage
            stages = {}
            for step in self.steps:
                if step.reasoning_stage not in stages:
                    stages[step.reasoning_stage] = []
                stages[step.reasoning_stage].append(step)

            all_evidence = set()
            all_assumptions = set()
            contradiction_pairs = []

            for step in self.steps:
                all_evidence.update(step.evidence or [])
                all_assumptions.update(step.assumptions or [])
                if step.contradicts:
                    for contradicted in step.contradicts:
                        contradiction_pairs.append((step.step_number, contradicted))

            confidence_by_stage = {}
            for stage, steps_in_stage in stages.items():
                avg_confidence = sum(s.confidence for s in steps_in_stage) / len(steps_in_stage)
                confidence_by_stage[stage] = round(avg_confidence, 3)

            # Build content_synthesis: full thought text grouped by stage
            content_synthesis: Dict[str, List[str]] = {}
            for stage, steps_in_stage in stages.items():
                content_synthesis[stage] = [s.thought for s in steps_in_stage]

            # Build completion_status against the 5 canonical reasoning stages
            required_stages = [
                "Problem Definition", "Research", "Analysis", "Synthesis", "Conclusion"
            ]
            stages_present = set(stages.keys())
            stages_missing = [s for s in required_stages if s not in stages_present]
            stages_found = [s for s in required_stages if s in stages_present]
            completion_status = {
                "has_all_stages": len(stages_missing) == 0,
                "percent_complete": round(len(stages_found) / len(required_stages) * 100.0, 1),
                "stages_required": required_stages,
                "stages_missing": stages_missing
            }

            return {
                "status": "success",
                "total_steps": len(self.steps),
                "stages_covered": list(stages.keys()),
                "overall_confidence": self.metadata["total_confidence"],
                "confidence_by_stage": confidence_by_stage,
                "content_synthesis": content_synthesis,
                "completion_status": completion_status,
                "chain": [
                    {
                        "step": s.step_number,
                        "stage": s.reasoning_stage,
                        "thought_preview": s.thought[:100] + "..." if len(s.thought) > 100 else s.thought,
                        "confidence": s.confidence,
                        "has_evidence": bool(s.evidence),
                        "has_assumptions": bool(s.assumptions)
                    }
                    for s in sorted(self.steps, key=lambda x: x.step_number)
                ],
                "insights": {
                    "total_evidence": list(all_evidence),
                    "total_assumptions": list(all_assumptions),
                    "contradiction_pairs": contradiction_pairs,
                    "high_confidence_steps": [s.step_number for s in self.steps if s.confidence >= 0.8],
                    "low_confidence_steps": [s.step_number for s in self.steps if s.confidence < 0.5]
                },
                "metadata": self.metadata
            }
    
    def clear_chain(self) -> Dict[str, Any]:
        """Clear all steps and reset the chain of thought."""
        with self._lock:
            self.steps.clear()
            self.metadata = {
                "created_at": datetime.now().isoformat(),
                "total_confidence": 0.0
            }

            return {
                "status": "success",
                "message": "Chain of thought cleared. Ready for new reasoning sequence."
            }

    @staticmethod
    def _validate_file_path(file_path: str) -> None:
        """Validate file_path is a non-empty string."""
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path must be a non-empty string")

    def export_chain(self, file_path: str) -> Dict[str, Any]:
        """
        Serialize all steps to JSON and write to file.

        Args:
            file_path: Path to the output file.

        Returns:
            Status dict indicating success or error.
        """
        try:
            self._validate_file_path(file_path)
            data = {
                "steps": [asdict(step) for step in self.steps],
                "metadata": self.metadata
            }
            serialized = _safe_json_dumps(data, indent=2)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(serialized)
            return {
                "status": "success",
                "message": f"Chain exported to {file_path}",
                "steps_exported": len(self.steps)
            }
        except (OSError, IOError, ValueError) as e:
            return {"status": "error", "message": str(e)}

    def import_chain(self, file_path: str) -> Dict[str, Any]:
        """
        Load JSON from file and restore ThoughtStep objects, replacing the current chain.

        Step numbers are preserved as-is, including non-sequential numbering.
        Does NOT use add_step() to avoid triggering revision logic.

        Args:
            file_path: Path to the JSON file previously produced by export_chain.

        Returns:
            Status dict indicating success or error.
        """
        try:
            self._validate_file_path(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                raw = f.read()
            data = json.loads(raw)
        except FileNotFoundError:
            return {"status": "error", "message": f"File not found: {file_path}"}
        except (OSError, IOError) as e:
            return {"status": "error", "message": str(e)}
        except json.JSONDecodeError as e:
            return {"status": "error", "message": f"Invalid JSON: {e}"}
        except ValueError as e:
            return {"status": "error", "message": str(e)}

        if not isinstance(data, dict):
            return {"status": "error", "message": "Invalid JSON structure: expected a JSON object at root"}

        steps_data = data.get("steps", [])

        if not isinstance(steps_data, list):
            return {"status": "error", "message": "Invalid JSON structure: 'steps' must be a list"}

        if len(steps_data) > MAX_IMPORT_STEPS:
            return {
                "status": "error",
                "message": f"Import rejected: {len(steps_data)} steps exceeds maximum of {MAX_IMPORT_STEPS}"
            }

        required_keys = {"thought", "step_number", "total_steps", "next_step_needed"}
        restored: List[ThoughtStep] = []
        for idx, d in enumerate(steps_data):
            if not isinstance(d, dict):
                return {
                    "status": "error",
                    "message": f"Invalid step at index {idx}: each step must be a JSON object"
                }
            missing = required_keys - d.keys()
            if missing:
                return {
                    "status": "error",
                    "message": f"Invalid step at index {idx}: missing required fields {sorted(missing)}"
                }
            if not isinstance(d["thought"], str):
                return {
                    "status": "error",
                    "message": f"Invalid step at index {idx}: 'thought' must be a string"
                }
            if not isinstance(d["step_number"], int) or isinstance(d["step_number"], bool):
                return {
                    "status": "error",
                    "message": f"Invalid step at index {idx}: 'step_number' must be an integer"
                }
            if not isinstance(d["total_steps"], int) or isinstance(d["total_steps"], bool):
                return {
                    "status": "error",
                    "message": f"Invalid step at index {idx}: 'total_steps' must be an integer"
                }
            if not isinstance(d["next_step_needed"], bool):
                return {
                    "status": "error",
                    "message": f"Invalid step at index {idx}: 'next_step_needed' must be a boolean"
                }
            if "confidence" in d:
                conf = d["confidence"]
                if isinstance(conf, bool) or not isinstance(conf, (int, float)):
                    return {
                        "status": "error",
                        "message": f"Invalid step at index {idx}: 'confidence' must be a number"
                    }
                if isinstance(conf, float) and (math.isnan(conf) or math.isinf(conf)):
                    return {
                        "status": "error",
                        "message": f"Invalid step at index {idx}: 'confidence' must be a finite number"
                    }
                if conf < 0.0 or conf > 1.0:
                    return {
                        "status": "error",
                        "message": f"Invalid step at index {idx}: 'confidence' must be between 0.0 and 1.0"
                    }

            reasoning_stage_val = d.get("reasoning_stage", "Analysis")
            if not isinstance(reasoning_stage_val, str):
                return {
                    "status": "error",
                    "message": f"Invalid step at index {idx}: 'reasoning_stage' must be a string"
                }

            for int_list_field in ("dependencies", "contradicts"):
                field_val = d.get(int_list_field)
                if field_val is not None:
                    if not isinstance(field_val, list):
                        return {
                            "status": "error",
                            "message": f"Invalid step at index {idx}: '{int_list_field}' must be a list"
                        }
                    for elem in field_val:
                        if isinstance(elem, bool) or not isinstance(elem, int):
                            return {
                                "status": "error",
                                "message": (
                                    f"Invalid step at index {idx}: "
                                    f"'{int_list_field}' elements must be integers"
                                )
                            }

            for str_list_field in ("evidence", "assumptions"):
                field_val = d.get(str_list_field)
                if field_val is not None:
                    if not isinstance(field_val, list):
                        return {
                            "status": "error",
                            "message": f"Invalid step at index {idx}: '{str_list_field}' must be a list"
                        }
                    for elem in field_val:
                        if not isinstance(elem, str):
                            return {
                                "status": "error",
                                "message": (
                                    f"Invalid step at index {idx}: "
                                    f"'{str_list_field}' elements must be strings"
                                )
                            }

            timestamp_val = d.get("timestamp")
            if timestamp_val is not None and not isinstance(timestamp_val, str):
                return {
                    "status": "error",
                    "message": f"Invalid step at index {idx}: 'timestamp' must be a string or null"
                }

            thought_val = html.escape(d["thought"].strip())

            reasoning_stage_val_stripped = reasoning_stage_val.strip()
            if len(reasoning_stage_val_stripped) > 100:
                return {
                    "status": "error",
                    "message": f"Invalid step at index {idx}: 'reasoning_stage' cannot exceed 100 characters"
                }
            if not re.match(r'^[a-zA-Z0-9 _-]+$', reasoning_stage_val_stripped):
                return {
                    "status": "error",
                    "message": (
                        f"Invalid step at index {idx}: 'reasoning_stage' can only contain "
                        "letters, numbers, spaces, underscores, and hyphens"
                    )
                }

            evidence_list = d.get("evidence") or []
            evidence_sanitized = [html.escape(item.strip()) for item in evidence_list]

            assumptions_list = d.get("assumptions") or []
            assumptions_sanitized = [html.escape(item.strip()) for item in assumptions_list]

            step = ThoughtStep(
                thought=thought_val,
                step_number=d["step_number"],
                total_steps=d["total_steps"],
                reasoning_stage=reasoning_stage_val_stripped,
                confidence=d.get("confidence", 0.8),
                next_step_needed=d["next_step_needed"],
                dependencies=d.get("dependencies") or [],
                contradicts=d.get("contradicts") or [],
                evidence=evidence_sanitized,
                assumptions=assumptions_sanitized,
                timestamp=timestamp_val
            )
            restored.append(step)

        self.steps = restored
        if not restored:
            self.metadata["total_confidence"] = 0.0
            self.metadata.pop("last_updated", None)
        else:
            self._update_metadata()

        return {
            "status": "success",
            "message": f"Chain imported from {file_path}",
            "steps_imported": len(self.steps)
        }



# Security helper function for safe JSON serialization
def _safe_json_dumps(data: Any, indent: int = 2) -> str:
    """
    Safely serialize data to JSON with strict security controls.

    Implements defense-in-depth approach with multiple security layers:
    1. Whitelist-only type checking
    2. Sensitive key filtering
    3. Dangerous content detection
    4. Generic error handling (no information disclosure)

    Args:
        data: Data to serialize
        indent: JSON indentation level

    Returns:
        Safe JSON string with no sensitive data exposed
    """
    try:
        # Define whitelist of safe types (defense-in-depth)
        SAFE_TYPES = (dict, list, str, int, float, bool, type(None))

        # Define sensitive keys to filter (case-insensitive)
        SENSITIVE_KEYS = {
            'password', 'passwd', 'pwd', 'secret', 'token', 'key', 'apikey', 'api_key',
            'auth', 'authorization', 'auth_token', 'session', 'session_id',
            'credit_card', 'card', 'ssn', 'social_security', 'pin',
            'credential', 'private', 'confidential', 'internal'
        }

        # Define dangerous content patterns
        DANGEROUS_PATTERNS = {
            '__import__', 'eval(', 'exec(', 'open(', 'file(', 'input(',
            'subprocess', 'os.system', 'shell_exec', 'DROP TABLE', 'SELECT *',
            '<script', 'javascript:', 'data:', 'vbscript:', 'onload=', 'onerror='
        }

        def sanitize(obj, depth=0):
            """
            Recursively sanitize object for safe serialization.
            Uses whitelist approach with depth limiting to prevent recursion attacks.
            """
            # Prevent deep recursion attacks
            if depth > MAX_RECURSION_DEPTH:
                return {"status": "error", "message": "Data too deep"}

            if isinstance(obj, SAFE_TYPES):
                if isinstance(obj, dict):
                    sanitized_dict = {}
                    for key, value in obj.items():
                        # Filter sensitive keys (case-insensitive)
                        key_lower = str(key).lower()
                        is_sensitive = any(sensitive in key_lower for sensitive in SENSITIVE_KEYS)

                        if is_sensitive:
                            # Replace sensitive values with placeholder
                            sanitized_dict[key] = "[REDACTED]"
                        else:
                            # Recursively sanitize values
                            sanitized_dict[key] = sanitize(value, depth + 1)

                    return sanitized_dict

                elif isinstance(obj, list):
                    # Sanitize list elements recursively
                    try:
                        return [sanitize(item, depth + 1) for item in obj[:MAX_LIST_SIZE]]  # Limit list size
                    except Exception:
                        return [{"status": "error", "message": "List processing failed"}]

                elif isinstance(obj, str):
                    # Check for dangerous content in strings
                    content_lower = obj.lower()
                    for pattern in DANGEROUS_PATTERNS:
                        if pattern in content_lower:
                            return "[FILTERED_CONTENT]"
                    return obj[:MAX_STRING_LENGTH]  # Limit string length

                elif isinstance(obj, (int, float)):
                    # Check for dangerous numeric values
                    if isinstance(obj, float):
                        if obj != obj:  # NaN
                            return 0.0
                        if obj in (float('inf'), float('-inf')):  # Infinity
                            return 0.0
                    return obj

                elif isinstance(obj, bool) or obj is None:
                    return obj

            else:
                # Convert unknown objects to safe string representation
                # NEVER expose internal structure or methods
                obj_type = type(obj).__name__
                return f"[Object: {obj_type}]"

        # Apply sanitization
        sanitized_data = sanitize(data)

        # Final security check on result size
        json_string = json.dumps(
            sanitized_data,
            indent=indent,
            ensure_ascii=True,
            separators=(',', ': '),
            sort_keys=True
        )

        # Prevent DoS through huge JSON output
        if len(json_string) > MAX_JSON_SIZE:  # Prevent DoS through huge JSON output
            return json.dumps({
                "status": "error",
                "message": "Data processing failed"
            })

        return json_string

    except Exception:
        # NEVER expose internal error details - security principle
        # No information disclosure about internal errors, types, or stack traces
        return json.dumps({
            "status": "error",
            "message": "Data processing failed"
        })


# Initialize default service factories now that all classes are defined
_default_registry.initialize_default_services()

# Global instance for simple usage - now using the service registry
_chain_processor = _default_registry.get_service('chain_of_thought')

# Import security module components
from .security import RequestValidator, SecurityValidationError, default_validator


# Re-export concurrency classes from concurrency.py for backward compatibility.
from .concurrency import (  # noqa: E402
    RateLimiter,
    get_global_rate_limiter,
    set_global_rate_limiter,
    ThreadAwareChainOfThought,
    DEFAULT_MAX_REQUESTS_PER_MINUTE,
    DEFAULT_MAX_REQUESTS_PER_HOUR,
    DEFAULT_MAX_BURST_SIZE,
)

# Re-export handler functions from handlers.py for backward compatibility.
from .handlers import (  # noqa: E402
    chain_of_thought_step_handler,
    get_chain_summary_handler,
    clear_chain_handler,
    generate_hypotheses_handler,
    map_assumptions_handler,
    calibrate_confidence_handler,
    export_chain_handler,
    import_chain_handler,
    create_chain_of_thought_step_handler,
    create_get_chain_summary_handler,
    create_clear_chain_handler,
    create_generate_hypotheses_handler,
    create_map_assumptions_handler,
    create_calibrate_confidence_handler,
)


# Re-export Bedrock integration classes from bedrock.py for backward compatibility.
from .bedrock import (  # noqa: E402
    StopReasonHandler,
    BedrockStopReasonHandler,
    AsyncChainOfThoughtProcessor,
)
