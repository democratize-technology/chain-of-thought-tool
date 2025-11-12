"""
Chain of Thought Tool - Core Implementation
"""
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import asyncio
import threading
import weakref
import time
import logging
from abc import ABC, abstractmethod
from .validators import ParameterValidator


class ServiceCreationError(Exception):
    """Raised when service creation fails in ServiceRegistry."""
    pass


class ServiceRegistry:
    """
    Thread-safe dependency injection container for managing service instances.

    Provides a clean way to manage service lifecycles while maintaining
    backward compatibility with global singleton usage.
    """

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._lock = threading.RLock()

    def _register_default_factories(self):
        """Register default factories for all core services."""
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


# Global service registry for backward compatibility
_default_registry = ServiceRegistry()


def get_service_registry() -> ServiceRegistry:
    """Get the default service registry."""
    return _default_registry


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
    """
    Chain of Thought processor that tracks reasoning steps and provides analysis.
    """

    def __init__(self):
        self.steps: List[ThoughtStep] = []
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "total_confidence": 0.0
        }
        self.validator = ParameterValidator()
      
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

        # Validate and sanitize all input parameters for security
        validated_params = self.validator.validate_input(
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

        # Extract validated parameters
        thought = validated_params["thought"]
        step_number = validated_params["step_number"]
        total_steps = validated_params["total_steps"]
        reasoning_stage = validated_params["reasoning_stage"]
        confidence = validated_params["confidence"]
        next_step_needed = validated_params["next_step_needed"]
        dependencies = validated_params["dependencies"]
        contradicts = validated_params["contradicts"]
        evidence = validated_params["evidence"]
        assumptions = validated_params["assumptions"]
        
        # Check if this is a revision of an existing step
        for i, step in enumerate(self.steps):
            if step.step_number == step_number:
                # This is a revision
                self.steps[i] = ThoughtStep(
                    thought=thought,
                    step_number=step_number,
                    total_steps=total_steps,
                    reasoning_stage=reasoning_stage,
                    confidence=confidence,
                    next_step_needed=next_step_needed,
                    dependencies=dependencies,
                    contradicts=contradicts,
                    evidence=evidence,
                    assumptions=assumptions
                )
                self._update_metadata()
                return self._generate_feedback(self.steps[i], is_revision=True)
        
        step = ThoughtStep(
            thought=thought,
            step_number=step_number,
            total_steps=total_steps,
            reasoning_stage=reasoning_stage,
            confidence=confidence,
            next_step_needed=next_step_needed,
            dependencies=dependencies,
            contradicts=contradicts,
            evidence=evidence,
            assumptions=assumptions
        )
        
        self.steps.append(step)
        self._update_metadata()
        
        return self._generate_feedback(step, is_revision=False)
    
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
        
        return {
            "status": "success",
            "total_steps": len(self.steps),
            "stages_covered": list(stages.keys()),
            "overall_confidence": self.metadata["total_confidence"],
            "confidence_by_stage": confidence_by_stage,
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
        self.steps.clear()
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "total_confidence": 0.0
        }
        
        return {
            "status": "success",
            "message": "Chain of thought cleared. Ready for new reasoning sequence."
        }


@dataclass
class Hypothesis:
    """Represents a single hypothesis for explaining an observation."""
    hypothesis_text: str
    hypothesis_type: str  # scientific, intuitive, contrarian, systematic
    confidence: float = 0.8
    testability_score: float = 0.7
    reasoning: str = ""
    evidence_requirements: Optional[List[str]] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.evidence_requirements is None:
            self.evidence_requirements = []


class HypothesisGenerator:
    """
    Hypothesis generator that creates diverse explanations for observations.
    """
    
    def __init__(self):
        self.hypotheses: List[Hypothesis] = []
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "generation_count": 0
        }
    
    def generate_hypotheses(
        self,
        observation: str,
        hypothesis_count: int = 4
    ) -> Dict[str, Any]:
        """
        Generate diverse hypotheses for the given observation.
        
        Returns analysis and ranked hypotheses.
        """
        
        # Clear previous hypotheses for new observation
        self.hypotheses.clear()
        
        # Generate different types of hypotheses
        hypothesis_types = ["scientific", "intuitive", "contrarian", "systematic"]
        
        # Ensure we don't generate more than requested
        types_to_generate = hypothesis_types[:hypothesis_count]
        
        for i, hypothesis_type in enumerate(types_to_generate):
            hypothesis = self._generate_hypothesis_by_type(observation, hypothesis_type, i + 1)
            self.hypotheses.append(hypothesis)
        
        # Rank by testability
        ranked_hypotheses = sorted(self.hypotheses, key=lambda h: h.testability_score, reverse=True)
        
        self.metadata["generation_count"] += 1
        self.metadata["last_generated"] = datetime.now().isoformat()
        
        return {
            "status": "success",
            "observation": observation,
            "hypotheses_generated": len(ranked_hypotheses),
            "hypotheses": [
                {
                    "rank": i + 1,
                    "text": h.hypothesis_text,
                    "type": h.hypothesis_type,
                    "confidence": h.confidence,
                    "testability": h.testability_score,
                    "reasoning": h.reasoning,
                    "evidence_needed": h.evidence_requirements
                }
                for i, h in enumerate(ranked_hypotheses)
            ],
            "insights": {
                "most_testable": ranked_hypotheses[0].hypothesis_type if ranked_hypotheses else None,
                "highest_confidence": max((h.confidence for h in ranked_hypotheses), default=0),
                "types_generated": [h.hypothesis_type for h in ranked_hypotheses]
            },
            "metadata": self.metadata
        }
    
    def _generate_hypothesis_by_type(self, observation: str, hypothesis_type: str, rank: int) -> Hypothesis:
        """Generate a hypothesis of a specific type."""
        
        if hypothesis_type == "scientific":
            return Hypothesis(
                hypothesis_text=f"Based on empirical evidence, {observation.lower()} could be explained by measurable factors that follow established patterns or laws.",
                hypothesis_type="scientific",
                confidence=0.8,
                testability_score=0.9,
                reasoning="Scientific approach focuses on testable, measurable explanations",
                evidence_requirements=["Quantitative data", "Control groups", "Reproducible experiments"]
            )
        elif hypothesis_type == "intuitive":
            return Hypothesis(
                hypothesis_text=f"Pattern recognition suggests that {observation.lower()} fits a familiar template based on previous similar situations.",
                hypothesis_type="intuitive",
                confidence=0.7,
                testability_score=0.6,
                reasoning="Intuitive approach leverages pattern matching and heuristics",
                evidence_requirements=["Historical precedents", "Pattern analysis", "Expert judgment"]
            )
        elif hypothesis_type == "contrarian":
            return Hypothesis(
                hypothesis_text=f"Contrary to obvious explanations, {observation.lower()} might be caused by the opposite of what initially appears likely.",
                hypothesis_type="contrarian",
                confidence=0.6,
                testability_score=0.8,
                reasoning="Contrarian approach challenges conventional assumptions",
                evidence_requirements=["Alternative data sources", "Assumption validation", "Devil's advocate analysis"]
            )
        elif hypothesis_type == "systematic":
            return Hypothesis(
                hypothesis_text=f"A systematic breakdown of {observation.lower()} reveals multiple interconnected factors that must be analyzed hierarchically.",
                hypothesis_type="systematic",
                confidence=0.75,
                testability_score=0.85,
                reasoning="Systematic approach breaks complex observations into manageable components",
                evidence_requirements=["Component analysis", "System mapping", "Dependency tracking"]
            )
        else:
            return Hypothesis(
                hypothesis_text=f"General explanation for {observation.lower()} based on available information.",
                hypothesis_type="general",
                confidence=0.5,
                testability_score=0.5,
                reasoning="Default hypothesis when type is unrecognized"
            )




@dataclass
class Assumption:
    """Represents a single assumption identified in a statement."""
    statement: str
    assumption_type: str  # explicit, implicit
    confidence: float = 0.8
    dependencies: Optional[List[str]] = None
    is_critical: bool = False
    reasoning: str = ""
    validation_methods: Optional[List[str]] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.dependencies is None:
            self.dependencies = []
        if self.validation_methods is None:
            self.validation_methods = []


class AssumptionMapper:
    """
    Assumption mapper that identifies and categorizes assumptions in statements.
    """
    
    def __init__(self):
        self.assumptions: List[Assumption] = []
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "mapping_count": 0
        }
    
    def extract_explicit_assumptions(self, statement: str) -> List[Assumption]:
        """Extract explicitly stated assumptions from the statement."""
        assumptions = []
        
        # Look for explicit assumption indicators
        assumption_indicators = [
            "assuming", "given that", "if we assume", "provided that",
            "taking for granted", "presupposing", "based on the premise"
        ]
        
        # Simulate finding explicit assumptions based on linguistic patterns
        if any(indicator in statement.lower() for indicator in assumption_indicators):
            assumptions.append(Assumption(
                statement=f"Explicit assumption found in: '{statement[:50]}...'",
                assumption_type="explicit",
                confidence=0.9,
                is_critical=True,
                reasoning="Statement contains explicit assumption indicators",
                validation_methods=["Textual analysis", "Logical parsing"]
            ))
        
        # Look for conditional statements that reveal assumptions
        if any(word in statement.lower() for word in ["if", "when", "unless", "provided"]):
            assumptions.append(Assumption(
                statement=f"Conditional assumption in statement about prerequisites",
                assumption_type="explicit",
                confidence=0.8,
                is_critical=False,
                reasoning="Conditional language reveals explicit preconditions",
                validation_methods=["Conditional logic analysis"]
            ))
        
        return assumptions
    
    def identify_implicit_assumptions(self, statement: str) -> List[Assumption]:
        """Identify unstated assumptions underlying the statement."""
        assumptions = []
        
        # Domain-specific implicit assumptions
        if "market" in statement.lower() or "business" in statement.lower():
            assumptions.append(Assumption(
                statement="Market behavior follows rational economic principles",
                assumption_type="implicit",
                confidence=0.6,
                is_critical=True,
                reasoning="Business statements often assume market rationality",
                validation_methods=["Market research", "Economic data analysis"]
            ))
        
        # Causal implicit assumptions
        if "because" in statement.lower() or "leads to" in statement.lower():
            assumptions.append(Assumption(
                statement="Causal relationships are direct and measurable",
                assumption_type="implicit",
                confidence=0.7,
                is_critical=True,
                reasoning="Causal language assumes direct cause-effect relationships",
                validation_methods=["Causal analysis", "Controlled experiments"]
            ))
        
        # Temporal implicit assumptions
        if any(word in statement.lower() for word in ["will", "future", "predict", "forecast"]):
            assumptions.append(Assumption(
                statement="Future conditions will remain similar to current conditions",
                assumption_type="implicit",
                confidence=0.5,
                is_critical=True,
                reasoning="Future-oriented statements assume continuity",
                validation_methods=["Trend analysis", "Scenario planning"]
            ))
        
        # Scale/scope implicit assumptions
        if any(word in statement.lower() for word in ["all", "every", "always", "never"]):
            assumptions.append(Assumption(
                statement="Universal quantifiers apply without exceptions",
                assumption_type="implicit",
                confidence=0.4,
                is_critical=True,
                reasoning="Absolute statements assume no edge cases",
                validation_methods=["Edge case analysis", "Exception testing"]
            ))
        
        return assumptions
    
    def identify_critical_assumptions(self, assumptions: List[Assumption]) -> List[Assumption]:
        """Identify which assumptions are load-bearing (critical to the argument)."""
        critical_assumptions = []
        
        for assumption in assumptions:
            # Mark as critical if it has high confidence and affects core logic
            if assumption.confidence >= 0.7:
                assumption.is_critical = True
                critical_assumptions.append(assumption)
            
            # Mark causal assumptions as critical
            if "causal" in assumption.reasoning.lower():
                assumption.is_critical = True
                critical_assumptions.append(assumption)
                
            # Mark universal assumptions as critical due to fragility
            if "universal" in assumption.reasoning.lower() or "absolute" in assumption.reasoning.lower():
                assumption.is_critical = True
                critical_assumptions.append(assumption)
        
        return critical_assumptions
    
    def map_assumptions(
        self,
        statement: str,
        depth: str = "surface"
    ) -> Dict[str, Any]:
        """
        Map all assumptions in the given statement.
        
        Args:
            statement: The statement to analyze
            depth: Analysis depth - "surface" for basic, "deep" for comprehensive
            
        Returns analysis with categorized assumptions and criticality assessment.
        """
        
        # Clear previous assumptions for new statement
        self.assumptions.clear()
        
        # Extract different types of assumptions
        explicit_assumptions = self.extract_explicit_assumptions(statement)
        implicit_assumptions = self.identify_implicit_assumptions(statement)
        
        # Apply depth-specific analysis
        if depth == "deep":
            # In deep mode, generate additional implicit assumptions
            additional_implicit = []
            
            # Look for data quality assumptions
            if "data" in statement.lower() or "research" in statement.lower():
                additional_implicit.append(Assumption(
                    statement="Data sources are accurate and representative",
                    assumption_type="implicit",
                    confidence=0.6,
                    is_critical=True,
                    reasoning="Data-dependent statements assume source quality",
                    validation_methods=["Data validation", "Source verification"]
                ))
            
            # Look for stakeholder assumptions
            if "people" in statement.lower() or "users" in statement.lower():
                additional_implicit.append(Assumption(
                    statement="Human behavior is predictable and consistent",
                    assumption_type="implicit",
                    confidence=0.5,
                    is_critical=True,
                    reasoning="People-focused statements assume behavioral predictability",
                    validation_methods=["User research", "Behavioral analysis"]
                ))
                
            implicit_assumptions.extend(additional_implicit)
        
        # Combine all assumptions
        all_assumptions = explicit_assumptions + implicit_assumptions
        self.assumptions = all_assumptions
        
        # Identify critical assumptions
        critical_assumptions = self.identify_critical_assumptions(all_assumptions)
        
        # Build dependency relationships
        dependency_graph = self._build_dependency_graph(all_assumptions)
        
        self.metadata["mapping_count"] += 1
        self.metadata["last_mapped"] = datetime.now().isoformat()
        
        return {
            "status": "success",
            "statement": statement,
            "depth": depth,
            "assumptions_found": len(all_assumptions),
            "explicit": [
                {
                    "statement": a.statement,
                    "confidence": a.confidence,
                    "is_critical": a.is_critical,
                    "reasoning": a.reasoning,
                    "validation_methods": a.validation_methods
                }
                for a in explicit_assumptions
            ],
            "implicit": [
                {
                    "statement": a.statement,
                    "confidence": a.confidence,
                    "is_critical": a.is_critical,
                    "reasoning": a.reasoning,
                    "validation_methods": a.validation_methods
                }
                for a in implicit_assumptions
            ],
            "critical": [
                {
                    "statement": a.statement,
                    "type": a.assumption_type,
                    "confidence": a.confidence,
                    "reasoning": a.reasoning
                }
                for a in critical_assumptions
            ],
            "insights": {
                "total_critical": len(critical_assumptions),
                "highest_risk": min((a.confidence for a in critical_assumptions), default=1.0),
                "dependency_complexity": len(dependency_graph),
                "assumption_types": list(set(a.assumption_type for a in all_assumptions))
            },
            "graph": dependency_graph,
            "metadata": self.metadata
        }
    
    def _build_dependency_graph(self, assumptions: List[Assumption]) -> Dict[str, List[str]]:
        """Build a simple dependency graph between assumptions."""
        graph = {}
        
        for i, assumption in enumerate(assumptions):
            assumption_id = f"assumption_{i}"
            graph[assumption_id] = []
            
            # Simple heuristic: critical assumptions depend on less critical ones
            for j, other_assumption in enumerate(assumptions):
                if i != j and assumption.is_critical and not other_assumption.is_critical:
                    graph[assumption_id].append(f"assumption_{j}")
        
        return graph




@dataclass
class ConfidenceAssessment:
    """Represents a confidence calibration assessment."""
    original_confidence: float
    calibrated_confidence: float
    confidence_band: tuple  # (lower_bound, upper_bound)
    overconfidence_indicators: Optional[List[str]] = None
    calibration_reasoning: str = ""
    uncertainty_factors: Optional[List[str]] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.overconfidence_indicators is None:
            self.overconfidence_indicators = []
        if self.uncertainty_factors is None:
            self.uncertainty_factors = []
        
        # Ensure confidence values are in valid range
        self.original_confidence = max(0.0, min(1.0, self.original_confidence))
        self.calibrated_confidence = max(0.0, min(1.0, self.calibrated_confidence))


class ConfidenceCalibrator:
    """
    Confidence calibrator that adjusts overconfident predictions and provides uncertainty bounds.
    """
    
    def __init__(self):
        self.assessments: List[ConfidenceAssessment] = []
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "calibration_count": 0
        }
    
    def detect_overconfidence_patterns(self, prediction: str, confidence: float) -> Dict[str, Any]:
        """Detect patterns that suggest overconfidence."""
        indicators = []
        overconfidence_score = 0.0
        
        # Very high confidence (>0.9) is often overconfident
        if confidence > 0.9:
            indicators.append("Very high initial confidence (>90%)")
            overconfidence_score += 0.3
        
        # Absolute language suggests overconfidence
        absolute_words = ["always", "never", "definitely", "certainly", "absolutely", "guaranteed", "impossible"]
        if any(word in prediction.lower() for word in absolute_words):
            indicators.append("Contains absolute language suggesting overconfidence")
            overconfidence_score += 0.2
        
        # Future predictions are inherently uncertain
        future_words = ["will", "going to", "by 2030", "by 2025", "next year", "soon"]
        if any(word in prediction.lower() for word in future_words):
            indicators.append("Future prediction with inherent uncertainty")
            overconfidence_score += 0.15
        
        # Complex predictions (multiple factors) often overconfident
        complexity_indicators = ["and", "because", "due to", "multiple", "various", "complex"]
        complexity_count = sum(1 for word in complexity_indicators if word in prediction.lower())
        if complexity_count >= 2:
            indicators.append("Complex prediction with multiple factors")
            overconfidence_score += 0.1
        
        # Technology predictions are notoriously overconfident
        tech_words = ["ai", "artificial intelligence", "agi", "technology", "innovation", "breakthrough"]
        if any(word in prediction.lower() for word in tech_words):
            indicators.append("Technology prediction (historically overconfident domain)")
            overconfidence_score += 0.1
        
        # Statistical/quantitative claims without evidence
        if any(char.isdigit() for char in prediction) and confidence > 0.8:
            indicators.append("Quantitative claim with high confidence but no cited evidence")
            overconfidence_score += 0.15
        
        return {
            "indicators": indicators,
            "overconfidence_score": min(1.0, overconfidence_score),
            "risk_level": "high" if overconfidence_score > 0.4 else "medium" if overconfidence_score > 0.2 else "low"
        }
    
    def calculate_uncertainty_bands(self, confidence: float) -> tuple:
        """Calculate realistic uncertainty bands around the confidence estimate."""
        
        # Base uncertainty depends on confidence level
        if confidence > 0.95:
            # Very high confidence - add significant uncertainty
            uncertainty = 0.15
        elif confidence > 0.8:
            # High confidence - moderate uncertainty
            uncertainty = 0.1
        elif confidence > 0.6:
            # Medium confidence - some uncertainty
            uncertainty = 0.08
        else:
            # Low confidence - less additional uncertainty needed
            uncertainty = 0.05
        
        # Calculate bounds
        lower_bound = max(0.0, confidence - uncertainty)
        upper_bound = min(1.0, confidence + uncertainty)
        
        return (round(lower_bound, 3), round(upper_bound, 3))
    
    def apply_calibration_adjustment(self, original_confidence: float, overconfidence_score: float) -> float:
        """Apply calibration adjustment based on overconfidence indicators."""
        
        # Calculate adjustment factor based on overconfidence score
        # Higher overconfidence score = larger downward adjustment
        adjustment_factor = overconfidence_score * 0.3  # Max 30% reduction
        
        # Apply adjustment
        adjusted_confidence = original_confidence * (1 - adjustment_factor)
        
        # Ensure we don't go below a reasonable minimum
        adjusted_confidence = max(0.1, adjusted_confidence)
        
        return round(adjusted_confidence, 3)
    
    def calibrate_confidence(
        self,
        prediction: str,
        initial_confidence: float,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Calibrate confidence for the given prediction.
        
        Args:
            prediction: The prediction or claim to calibrate
            initial_confidence: Initial confidence level (0.0-1.0)
            context: Optional additional context for calibration
            
        Returns calibrated confidence with uncertainty bands and reasoning.
        """
        
        # Validate inputs
        initial_confidence = max(0.0, min(1.0, initial_confidence))
        
        # Detect overconfidence patterns
        overconfidence_analysis = self.detect_overconfidence_patterns(prediction, initial_confidence)
        
        # Apply calibration adjustment
        calibrated_confidence = self.apply_calibration_adjustment(
            initial_confidence, 
            overconfidence_analysis["overconfidence_score"]
        )
        
        # Calculate uncertainty bands
        uncertainty_band = self.calculate_uncertainty_bands(calibrated_confidence)
        
        # Identify uncertainty factors
        uncertainty_factors = []
        
        # Add context-specific uncertainty factors
        if "future" in prediction.lower() or any(word in prediction.lower() for word in ["will", "going to", "by 20"]):
            uncertainty_factors.append("Temporal uncertainty - future events")
        
        if "technology" in prediction.lower() or "ai" in prediction.lower():
            uncertainty_factors.append("Technology uncertainty - rapid change domain")
        
        if len(prediction.split()) > 20:
            uncertainty_factors.append("Complexity uncertainty - multiple interconnected factors")
        
        if context and "limited data" in context.lower():
            uncertainty_factors.append("Data uncertainty - limited information available")
        
        # Generate calibration reasoning
        adjustment_magnitude = abs(calibrated_confidence - initial_confidence)
        
        if adjustment_magnitude > 0.15:
            reasoning = f"Significant confidence reduction ({adjustment_magnitude:.2f}) due to strong overconfidence indicators."
        elif adjustment_magnitude > 0.05:
            reasoning = f"Moderate confidence adjustment ({adjustment_magnitude:.2f}) due to uncertainty factors."
        else:
            reasoning = f"Minor confidence adjustment ({adjustment_magnitude:.2f}) - original estimate reasonably calibrated."
        
        if overconfidence_analysis["risk_level"] == "high":
            reasoning += " High overconfidence risk detected."
        
        # Create assessment
        assessment = ConfidenceAssessment(
            original_confidence=initial_confidence,
            calibrated_confidence=calibrated_confidence,
            confidence_band=uncertainty_band,
            overconfidence_indicators=overconfidence_analysis["indicators"],
            calibration_reasoning=reasoning,
            uncertainty_factors=uncertainty_factors
        )
        
        self.assessments.append(assessment)
        self.metadata["calibration_count"] += 1
        self.metadata["last_calibrated"] = datetime.now().isoformat()
        
        return {
            "status": "success",
            "prediction": prediction,
            "original_confidence": initial_confidence,
            "calibrated_confidence": calibrated_confidence,
            "confidence_band": {
                "lower_bound": uncertainty_band[0],
                "upper_bound": uncertainty_band[1],
                "range": round(uncertainty_band[1] - uncertainty_band[0], 3)
            },
            "adjustment": {
                "magnitude": round(adjustment_magnitude, 3),
                "direction": "down" if calibrated_confidence < initial_confidence else "up",
                "reasoning": reasoning
            },
            "overconfidence_analysis": {
                "risk_level": overconfidence_analysis["risk_level"],
                "indicators": overconfidence_analysis["indicators"],
                "score": overconfidence_analysis["overconfidence_score"]
            },
            "uncertainty_factors": uncertainty_factors,
            "insights": {
                "confidence_appropriate": adjustment_magnitude < 0.1,
                "high_uncertainty": uncertainty_band[1] - uncertainty_band[0] > 0.2,
                "needs_more_evidence": len(overconfidence_analysis["indicators"]) > 2
            },
            "metadata": self.metadata
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
            if depth > 50:
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
                        return [sanitize(item, depth + 1) for item in obj[:100]]  # Limit list size
                    except Exception:
                        return [{"status": "error", "message": "List processing failed"}]

                elif isinstance(obj, str):
                    # Check for dangerous content in strings
                    content_lower = obj.lower()
                    for pattern in DANGEROUS_PATTERNS:
                        if pattern in content_lower:
                            return "[FILTERED_CONTENT]"
                    return obj[:1000]  # Limit string length

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
        if len(json_string) > 100000:  # 100KB limit
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


class RateLimiter:
    """
    Thread-safe rate limiting to prevent DoS attacks on handler functions.

    Implements token bucket algorithm with multiple time windows:
    - Burst limit: Immediate consecutive requests
    - Per-minute limit: Requests within 1-minute window
    - Per-hour limit: Requests within 1-hour window

    Each client is tracked separately to ensure isolation.
    """

    def __init__(self, max_requests_per_minute: int = 60, max_requests_per_hour: int = 1000, max_burst_size: int = 10):
        """
        Initialize rate limiter with configurable limits.

        Args:
            max_requests_per_minute: Maximum requests per minute per client
            max_requests_per_hour: Maximum requests per hour per client
            max_burst_size: Maximum consecutive immediate requests per client
        """
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour
        self.max_burst_size = max_burst_size

        # Track request counts and timestamps per client
        self._request_counts: Dict[str, int] = {}  # Current burst counts
        self._request_timestamps: Dict[str, List[float]] = {}  # Timestamps for sliding windows
        self._lock = threading.RLock()  # Thread-safe access

    def _cleanup_old_timestamps(self, client_id: str, current_time: float) -> None:
        """Remove timestamps older than 1 hour from tracking."""
        if client_id not in self._request_timestamps:
            return

        # Remove timestamps older than 1 hour
        one_hour_ago = current_time - 3600.0
        timestamps = self._request_timestamps[client_id]
        self._request_timestamps[client_id] = [
            ts for ts in timestamps if ts > one_hour_ago
        ]

        # Clean up empty timestamp lists
        if not self._request_timestamps[client_id]:
            del self._request_timestamps[client_id]

    def _get_minute_count(self, client_id: str, current_time: float) -> int:
        """Count requests in the last minute for a client."""
        if client_id not in self._request_timestamps:
            return 0

        one_minute_ago = current_time - 60.0
        return sum(1 for ts in self._request_timestamps[client_id] if ts > one_minute_ago)

    def _get_hour_count(self, client_id: str, current_time: float) -> int:
        """Count requests in the last hour for a client."""
        if client_id not in self._request_timestamps:
            return 0

        one_hour_ago = current_time - 3600.0
        return sum(1 for ts in self._request_timestamps[client_id] if ts > one_hour_ago)

    def check_rate_limit(self, client_id: str = "default") -> bool:
        """
        Check if a request from the client should be allowed.

        Args:
            client_id: Unique identifier for the client (IP address, session ID, etc.)

        Returns:
            True if request should be allowed, False if rate limited
        """
        current_time = time.time()

        with self._lock:
            # Clean up old timestamps
            self._cleanup_old_timestamps(client_id, current_time)

            # Check burst limit (immediate consecutive requests)
            current_burst = self._request_counts.get(client_id, 0)
            if current_burst >= self.max_burst_size:
                return False

            # Check per-minute limit
            minute_count = self._get_minute_count(client_id, current_time)
            if minute_count >= self.max_requests_per_minute:
                return False

            # Check per-hour limit
            hour_count = self._get_hour_count(client_id, current_time)
            if hour_count >= self.max_requests_per_hour:
                return False

            # Request is allowed - update tracking
            self._request_counts[client_id] = current_burst + 1

            # Add timestamp for sliding window tracking
            if client_id not in self._request_timestamps:
                self._request_timestamps[client_id] = []
            self._request_timestamps[client_id].append(current_time)

            return True

    def get_retry_after(self, client_id: str = "default") -> Optional[int]:
        """
        Get suggested retry-after seconds for a rate-limited client.

        Args:
            client_id: Unique identifier for the client

        Returns:
            Seconds to wait before retry, or None if not rate limited
        """
        current_time = time.time()

        with self._lock:
            # Check burst limit
            current_burst = self._request_counts.get(client_id, 0)
            if current_burst >= self.max_burst_size:
                return 1  # Very short delay for burst limit

            # Check minute limit
            minute_count = self._get_minute_count(client_id, current_time)
            if minute_count >= self.max_requests_per_minute:
                if client_id in self._request_timestamps and self._request_timestamps[client_id]:
                    oldest_timestamp = min(self._request_timestamps[client_id])
                    retry_after = int(60 - (current_time - oldest_timestamp)) + 1
                    return max(retry_after, 1)

            # Check hour limit
            hour_count = self._get_hour_count(client_id, current_time)
            if hour_count >= self.max_requests_per_hour:
                if client_id in self._request_timestamps and self._request_timestamps[client_id]:
                    oldest_timestamp = min(self._request_timestamps[client_id])
                    retry_after = int(3600 - (current_time - oldest_timestamp)) + 1
                    return max(retry_after, 60)  # At least 1 minute

            return None  # Not rate limited

    def reset_client(self, client_id: str = "default") -> None:
        """Reset rate limiting tracking for a specific client."""
        with self._lock:
            if client_id in self._request_counts:
                del self._request_counts[client_id]
            if client_id in self._request_timestamps:
                del self._request_timestamps[client_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get current rate limiting statistics."""
        with self._lock:
            return {
                "active_clients": len(self._request_counts),
                "total_tracked_timestamps": sum(len(timestamps) for timestamps in self._request_timestamps.values()),
                "max_requests_per_minute": self.max_requests_per_minute,
                "max_requests_per_hour": self.max_requests_per_hour,
                "max_burst_size": self.max_burst_size
            }


# Global rate limiter instance
_global_rate_limiter: Optional[RateLimiter] = None
_rate_limiter_lock = threading.Lock()


def get_global_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter instance."""
    global _global_rate_limiter

    if _global_rate_limiter is None:
        with _rate_limiter_lock:
            if _global_rate_limiter is None:  # Double-check
                _global_rate_limiter = RateLimiter()

    return _global_rate_limiter


def set_global_rate_limiter(limiter: RateLimiter) -> None:
    """Set a custom global rate limiter instance."""
    global _global_rate_limiter

    with _rate_limiter_lock:
        _global_rate_limiter = limiter


# Initialize default service factories now that all classes are defined
_default_registry.initialize_default_services()

# Global instance for simple usage - now using the service registry
_chain_processor = _default_registry.get_service('chain_of_thought')
_hypothesis_generator = _default_registry.get_service('hypothesis_generator')
_assumption_mapper = _default_registry.get_service('assumption_mapper')
_confidence_calibrator = _default_registry.get_service('confidence_calibrator')

# Import security module components
from .security import RequestValidator, SecurityValidationError, default_validator


def create_chain_of_thought_step_handler(registry: Optional[ServiceRegistry] = None, rate_limiter: Optional[RateLimiter] = None, client_id: str = "default"):
    """
    Create a chain_of_thought_step handler with dependency injection and rate limiting.

    Args:
        registry: Service registry to use. If None, uses default global registry.
        rate_limiter: Rate limiter to use. If None, uses global rate limiter.
        client_id: Client identifier for rate limiting.

    Returns:
        Handler function
    """
    # Use provided rate limiter or global one
    limiter = rate_limiter or get_global_rate_limiter()

    def handler(**kwargs) -> str:
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
            chain_processor = service_registry.get_service('chain_of_thought')
            result = chain_processor.add_step(**kwargs)
            return _safe_json_dumps(result, indent=2)
        except Exception as e:
            return _safe_json_dumps({"status": "error", "message": str(e)}, indent=2)
    return handler


def create_get_chain_summary_handler(registry: Optional[ServiceRegistry] = None, rate_limiter: Optional[RateLimiter] = None, client_id: str = "default"):
    """
    Create a get_chain_summary handler with dependency injection and rate limiting.

    Args:
        registry: Service registry to use. If None, uses default global registry.
        rate_limiter: Rate limiter to use. If None, uses global rate limiter.
        client_id: Client identifier for rate limiting.

    Returns:
        Handler function
    """
    limiter = rate_limiter or get_global_rate_limiter()

    def handler() -> str:
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
            chain_processor = service_registry.get_service('chain_of_thought')
            result = chain_processor.generate_summary()
            return _safe_json_dumps(result, indent=2)
        except Exception as e:
            return _safe_json_dumps({"status": "error", "message": str(e)}, indent=2)
    return handler


def create_clear_chain_handler(registry: Optional[ServiceRegistry] = None, rate_limiter: Optional[RateLimiter] = None, client_id: str = "default"):
    """
    Create a clear_chain handler with dependency injection and rate limiting.

    Args:
        registry: Service registry to use. If None, uses default global registry.
        rate_limiter: Rate limiter to use. If None, uses global rate limiter.
        client_id: Client identifier for rate limiting.

    Returns:
        Handler function
    """
    limiter = rate_limiter or get_global_rate_limiter()

    def handler() -> str:
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
            chain_processor = service_registry.get_service('chain_of_thought')
            result = chain_processor.clear_chain()
            return _safe_json_dumps(result, indent=2)
        except Exception as e:
            return _safe_json_dumps({"status": "error", "message": str(e)}, indent=2)
    return handler


def create_generate_hypotheses_handler(registry: Optional[ServiceRegistry] = None, rate_limiter: Optional[RateLimiter] = None, client_id: str = "default"):
    """
    Create a generate_hypotheses handler with dependency injection and rate limiting.

    Args:
        registry: Service registry to use. If None, uses default global registry.
        rate_limiter: Rate limiter to use. If None, uses global rate limiter.
        client_id: Client identifier for rate limiting.

    Returns:
        Handler function
    """
    limiter = rate_limiter or get_global_rate_limiter()

    def handler(**kwargs) -> str:
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
            hypothesis_generator = service_registry.get_service('hypothesis_generator')
            result = hypothesis_generator.generate_hypotheses(**kwargs)
            return _safe_json_dumps(result, indent=2)
        except Exception as e:
            return _safe_json_dumps({"status": "error", "message": str(e)}, indent=2)
    return handler


def create_map_assumptions_handler(registry: Optional[ServiceRegistry] = None, rate_limiter: Optional[RateLimiter] = None, client_id: str = "default"):
    """
    Create a map_assumptions handler with dependency injection and rate limiting.

    Args:
        registry: Service registry to use. If None, uses default global registry.
        rate_limiter: Rate limiter to use. If None, uses global rate limiter.
        client_id: Client identifier for rate limiting.

    Returns:
        Handler function
    """
    limiter = rate_limiter or get_global_rate_limiter()

    def handler(**kwargs) -> str:
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
            assumption_mapper = service_registry.get_service('assumption_mapper')
            result = assumption_mapper.map_assumptions(**kwargs)
            return _safe_json_dumps(result, indent=2)
        except Exception as e:
            return _safe_json_dumps({"status": "error", "message": str(e)}, indent=2)
    return handler


def create_calibrate_confidence_handler(registry: Optional[ServiceRegistry] = None, rate_limiter: Optional[RateLimiter] = None, client_id: str = "default"):
    """
    Create a calibrate_confidence handler with dependency injection and rate limiting.

    Args:
        registry: Service registry to use. If None, uses default global registry.
        rate_limiter: Rate limiter to use. If None, uses global rate limiter.
        client_id: Client identifier for rate limiting.

    Returns:
        Handler function
    """
    limiter = rate_limiter or get_global_rate_limiter()

    def handler(**kwargs) -> str:
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
            confidence_calibrator = service_registry.get_service('confidence_calibrator')
            result = confidence_calibrator.calibrate_confidence(**kwargs)
            return _safe_json_dumps(result, indent=2)
        except Exception as e:
            return _safe_json_dumps({"status": "error", "message": str(e)}, indent=2)
    return handler


# Backward compatibility: create default handlers using the global registry
def chain_of_thought_step_handler(**kwargs) -> str:
    """Handler function for the chain_of_thought_step tool (backward compatibility)."""
    return create_chain_of_thought_step_handler()(**kwargs)


def get_chain_summary_handler() -> str:
    """Handler function for the get_chain_summary tool (backward compatibility)."""
    return create_get_chain_summary_handler()()


def clear_chain_handler() -> str:
    """Handler function for the clear_chain tool (backward compatibility)."""
    return create_clear_chain_handler()()


def generate_hypotheses_handler(**kwargs) -> str:
    """Handler function for the generate_hypotheses tool (backward compatibility)."""
    return create_generate_hypotheses_handler()(**kwargs)


def map_assumptions_handler(**kwargs) -> str:
    """Handler function for the map_assumptions tool (backward compatibility)."""
    return create_map_assumptions_handler()(**kwargs)


def calibrate_confidence_handler(**kwargs) -> str:
    """Handler function for the calibrate_confidence tool (backward compatibility)."""
    return create_calibrate_confidence_handler()(**kwargs)


class StopReasonHandler(ABC):
    """Abstract base for handling stopReason integration with CoT."""
    
    @abstractmethod
    async def should_continue_reasoning(self, chain: ChainOfThought) -> bool:
        """Return True if reasoning should continue, False if end_turn."""
        pass
    
    @abstractmethod
    async def execute_tool_call(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call and return the result."""
        pass


class BedrockStopReasonHandler(StopReasonHandler):
    """Bedrock-specific stop reason handler that integrates with CoT flow."""
    
    def __init__(self, handlers: Optional[Dict[str, Callable]] = None, chain: Optional[ChainOfThought] = None):
        self.chain = chain  # If provided, use this chain instead of global
        if self.chain is not None:
            # Create instance-specific handlers
            self.handlers = handlers or {
                "chain_of_thought_step": self._create_chain_step_handler(),
                "get_chain_summary": self._create_summary_handler(),
                "clear_chain": self._create_clear_handler()
            }
        else:
            # Use global handlers
            self.handlers = handlers or {
                "chain_of_thought_step": chain_of_thought_step_handler,
                "get_chain_summary": get_chain_summary_handler,
                "clear_chain": clear_chain_handler
            }
    
    def _create_handler_factory(self, method_name: str, takes_kwargs: bool = False):
        """
        Create a generic handler factory for any method on this instance's chain.

        Args:
            method_name: Name of the method to call on self.chain
            takes_kwargs: Whether the method accepts keyword arguments

        Returns:
            A handler function bound to this instance's chain
        """
        def handler(**kwargs):
            try:
                method = getattr(self.chain, method_name)
                if takes_kwargs:
                    result = method(**kwargs)
                else:
                    result = method()
                return _safe_json_dumps(result, indent=2)
            except Exception as e:
                return _safe_json_dumps({"status": "error", "message": str(e)}, indent=2)
        return handler

    def _create_chain_step_handler(self):
        """Create a chain step handler bound to this instance's chain."""
        return self._create_handler_factory("add_step", takes_kwargs=True)

    def _create_summary_handler(self):
        """Create a summary handler bound to this instance's chain."""
        return self._create_handler_factory("generate_summary", takes_kwargs=False)

    def _create_clear_handler(self):
        """Create a clear handler bound to this instance's chain."""
        return self._create_handler_factory("clear_chain", takes_kwargs=False)
    
    async def should_continue_reasoning(self, chain: ChainOfThought) -> bool:
        """Check if CoT indicates more steps needed."""
        if not chain.steps:
            return True  # No steps yet, continue
        
        last_step = chain.steps[-1]
        return last_step.next_step_needed
    
    async def execute_tool_call(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CoT tool call asynchronously."""
        if tool_name not in self.handlers:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        handler = self.handlers[tool_name]
        
        # Run handler in executor if it's synchronous
        if asyncio.iscoroutinefunction(handler):
            result = await handler(**tool_args)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: handler(**tool_args))
        
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                result = {"status": "error", "message": "Invalid JSON response"}
        
        return result


class AsyncChainOfThoughtProcessor:
    """Async wrapper for CoT that integrates with Bedrock tool loops."""
    
    def __init__(self, conversation_id: str, stop_handler: Optional[StopReasonHandler] = None,
                 request_validator: Optional[RequestValidator] = None,
                 aws_call_timeout: float = 30.0, tool_call_timeout: float = 10.0):
        """
        Initialize AsyncChainOfThoughtProcessor with configurable timeouts.

        Args:
            conversation_id: Unique identifier for the conversation
            stop_handler: Handler for stopReason logic
            request_validator: Security request validator
            aws_call_timeout: Timeout in seconds for AWS API calls
            tool_call_timeout: Timeout in seconds for tool handler calls
        """
        self.conversation_id = conversation_id
        self.chain = ChainOfThought()
        # Pass the chain instance to the handler so it uses this specific chain
        self.stop_handler = stop_handler or BedrockStopReasonHandler(chain=self.chain)
        self.request_validator = request_validator or default_validator
        self._tool_use_count = 0
        self._max_iterations = 20

        # Timeout configuration
        self.aws_call_timeout = aws_call_timeout
        self.tool_call_timeout = tool_call_timeout

    async def _safe_aws_call(self, bedrock_client, **kwargs) -> Dict[str, Any]:
        """
        Execute AWS Bedrock call with proper timeout handling.

        Args:
            bedrock_client: AWS Bedrock client
            **kwargs: Parameters for converse call

        Returns:
            AWS response

        Raises:
            TimeoutError: If AWS call exceeds timeout
        """
        try:
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: bedrock_client.converse(**kwargs)
                ),
                timeout=self.aws_call_timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"AWS Bedrock call timed out after {self.aws_call_timeout} seconds")

    async def _safe_tool_call(self, handler_func: Callable, **kwargs) -> str:
        """
        Execute tool handler call with proper timeout handling.

        Args:
            handler_func: Tool handler function
            **kwargs: Parameters for handler

        Returns:
            Handler response JSON string

        Raises:
            TimeoutError: If tool call exceeds timeout
        """
        try:
            # Run tool handler in thread pool with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(handler_func, **kwargs),
                timeout=self.tool_call_timeout
            )
            return result
        except asyncio.TimeoutError:
            raise TimeoutError(f"Tool call timed out after {self.tool_call_timeout} seconds")

    async def process_tool_loop(self,
                              bedrock_client,
                              initial_request: Dict[str, Any],
                              max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """Process Bedrock tool loop with CoT integration."""

        # Validate and sanitize the initial request to prevent injection attacks
        try:
            sanitized_request = self.request_validator.validate_and_sanitize_request(initial_request)
        except SecurityValidationError as e:
            raise SecurityValidationError(f"Security validation failed: {str(e)}")

        max_iter = max_iterations or self._max_iterations
        messages = sanitized_request.get("messages", []).copy()

        for iteration in range(max_iter):
            # Use safe AWS call with timeout protection
            response = await self._safe_aws_call(
                bedrock_client,
                **{**sanitized_request, "messages": messages}
            )
            
            stop_reason = response.get("stopReason")
            
            if stop_reason == "end_turn":
                # Check if CoT actually wants to continue
                should_continue = await self.stop_handler.should_continue_reasoning(self.chain)
                if not should_continue:
                    return response
                # If CoT wants to continue but Bedrock says end_turn, we're done
                return response
            
            elif stop_reason == "tool_use":
                message_content = response.get("output", {}).get("message", {}).get("content", [])
                tool_results = []
                
                for content_item in message_content:
                    if "toolUse" in content_item:
                        tool_use = content_item["toolUse"]
                        tool_name = tool_use["name"]
                        tool_input = tool_use["input"]
                        tool_use_id = tool_use["toolUseId"]
                        
                        try:
                            # Use safe tool call with timeout protection
                            result = await asyncio.wait_for(
                                self.stop_handler.execute_tool_call(tool_name, tool_input),
                                timeout=self.tool_call_timeout
                            )
                            tool_results.append({
                                "toolResult": {
                                    "toolUseId": tool_use_id,
                                    "content": [{"text": _safe_json_dumps(result)}]
                                }
                            })
                        except asyncio.TimeoutError:
                            tool_results.append({
                                "toolResult": {
                                    "toolUseId": tool_use_id,
                                    "content": [{"text": _safe_json_dumps({"error": f"Tool call timed out after {self.tool_call_timeout} seconds"})}],
                                    "status": "error"
                                }
                            })
                        except Exception as e:
                            tool_results.append({
                                "toolResult": {
                                    "toolUseId": tool_use_id,
                                    "content": [{"text": _safe_json_dumps({"error": str(e)})}],
                                    "status": "error"
                                }
                            })
                
                messages.append(response["output"]["message"])
                if tool_results:
                    messages.append({
                        "role": "user",
                        "content": tool_results
                    })
                
                self._tool_use_count += len(tool_results)
            
            else:
                # Unexpected stop reason
                return response
        
        return {
            "stopReason": "max_tokens",
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Maximum reasoning iterations reached."}]
                }
            }
        }

    async def process_tool_loop_with_timeout(self,
                                           bedrock_client,
                                           initial_request: Dict[str, Any],
                                           max_iterations: Optional[int] = None,
                                           overall_timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Process Bedrock tool loop with overall timeout protection.

        Args:
            bedrock_client: AWS Bedrock client
            initial_request: Initial Bedrock request
            max_iterations: Maximum number of tool loop iterations
            overall_timeout: Overall timeout for the entire process

        Returns:
            Bedrock response or timeout error response
        """
        timeout = overall_timeout or (self.aws_call_timeout * 2)  # Default to 2x AWS timeout

        try:
            return await asyncio.wait_for(
                self.process_tool_loop(bedrock_client, initial_request, max_iterations),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return {
                "stopReason": "timeout",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": f"Request timeout after {timeout} seconds"}]
                    }
                }
            }
    
    async def get_reasoning_summary(self) -> Dict[str, Any]:
        """Get summary of the reasoning process."""
        return self.chain.generate_summary()
    
    def clear_reasoning(self) -> Dict[str, Any]:
        """Clear the reasoning chain."""
        self._tool_use_count = 0
        return self.chain.clear_chain()


class ThreadAwareChainOfThought:
    """Thread-safe version for production use with dependency injection support."""

    # Hybrid approach: WeakValueDictionary for automatic cleanup + strong refs for active conversations
    _instances: 'weakref.WeakValueDictionary[str, ChainOfThought]' = weakref.WeakValueDictionary()
    _strong_refs: Dict[str, ChainOfThought] = {}  # Keep strong refs to prevent premature GC
    _lock = threading.RLock()

    @classmethod
    def for_conversation(cls, conversation_id: str, registry: Optional[ServiceRegistry] = None) -> ChainOfThought:
        """Get or create a ChainOfThought instance for a conversation."""
        with cls._lock:
            # Try to get existing instance from strong references first
            if conversation_id in cls._strong_refs:
                return cls._strong_refs[conversation_id]

            # Try to get from weak references (may be None if GC'd)
            try:
                weak_instance = cls._instances[conversation_id]
                if weak_instance is not None:
                    # Found in weak refs, promote to strong refs
                    cls._strong_refs[conversation_id] = weak_instance
                    return weak_instance
            except KeyError:
                pass  # Instance doesn't exist, create new one

            # Create new instance
            service_registry = registry or get_service_registry()
            new_instance = ChainOfThought()

            # Store in both weak and strong references
            cls._instances[conversation_id] = new_instance
            cls._strong_refs[conversation_id] = new_instance
            return new_instance

    @classmethod
    def clear_conversation(cls, conversation_id: str) -> bool:
        """Explicitly clear a conversation from the cache.

        Args:
            conversation_id: The conversation ID to clear

        Returns:
            True if conversation was removed, False if not found
        """
        with cls._lock:
            removed_from_weak = cls._instances.pop(conversation_id, None) is not None
            removed_from_strong = cls._strong_refs.pop(conversation_id, None) is not None
            return removed_from_weak or removed_from_strong

    @classmethod
    def clear_all_conversations(cls) -> int:
        """Clear all conversations and return count cleared.

        Returns:
            Number of conversations that were cleared
        """
        with cls._lock:
            count = max(len(cls._instances), len(cls._strong_refs))
            cls._instances.clear()
            cls._strong_refs.clear()
            return count

    @classmethod
    def get_cached_conversation_count(cls) -> int:
        """Get the current number of cached conversations.

        Returns:
            Number of conversations currently cached
        """
        with cls._lock:
            return max(len(cls._instances), len(cls._strong_refs))

    @classmethod
    def release_conversation(cls, conversation_id: str) -> bool:
        """Release strong reference for a conversation, allowing weak reference cleanup.

        This is the key method for memory management - call this when conversation
        is no longer actively needed but should remain available for weak reference GC.

        Args:
            conversation_id: The conversation ID to release

        Returns:
            True if conversation was released, False if not found
        """
        with cls._lock:
            return cls._strong_refs.pop(conversation_id, None) is not None

    def __init__(self, conversation_id: str, registry: Optional[ServiceRegistry] = None):
        self.conversation_id = conversation_id
        self.registry = registry or get_service_registry()
        self.chain = self.for_conversation(conversation_id, self.registry)

    def get_tool_specs(self):
        """Get tool specs for this instance."""
        from . import TOOL_SPECS
        return TOOL_SPECS

    def get_handlers(self):
        """Get handlers bound to this instance using dependency injection."""
        # Create a service registry with this instance's ChainOfThought
        instance_registry = ServiceRegistry()

        # Copy all factories from the main registry
        for service_name in ['hypothesis_generator', 'assumption_mapper', 'confidence_calibrator']:
            if self.registry.has_service(service_name):
                instance_registry.register_factory(service_name, lambda name=service_name: self.registry.get_service(name))

        # Register this instance's ChainOfThought
        instance_registry.register_service('chain_of_thought', self.chain)

        # Create handlers using the instance registry
        return {
            "chain_of_thought_step": create_chain_of_thought_step_handler(instance_registry),
            "get_chain_summary": create_get_chain_summary_handler(instance_registry),
            "clear_chain": create_clear_chain_handler(instance_registry),
            "generate_hypotheses": create_generate_hypotheses_handler(instance_registry),
            "map_assumptions": create_map_assumptions_handler(instance_registry),
            "calibrate_confidence": create_calibrate_confidence_handler(instance_registry)
        }
