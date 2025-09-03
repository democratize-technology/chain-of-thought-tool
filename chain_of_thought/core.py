"""
Chain of Thought Tool - Core Implementation
"""
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import asyncio
from abc import ABC, abstractmethod


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


# Global instance for simple usage
_hypothesis_generator = HypothesisGenerator()


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


# Global instance for simple usage
_assumption_mapper = AssumptionMapper()


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


# Global instance for simple usage
_confidence_calibrator = ConfidenceCalibrator()


# Global instance for simple usage
_chain_processor = ChainOfThought()


def chain_of_thought_step_handler(**kwargs) -> str:
    """Handler function for the chain_of_thought_step tool."""
    try:
        result = _chain_processor.add_step(**kwargs)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, indent=2)


def get_chain_summary_handler() -> str:
    """Handler function for the get_chain_summary tool."""
    try:
        result = _chain_processor.generate_summary()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, indent=2)


def clear_chain_handler() -> str:
    """Handler function for the clear_chain tool."""
    try:
        result = _chain_processor.clear_chain()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, indent=2)


def generate_hypotheses_handler(**kwargs) -> str:
    """Handler function for the generate_hypotheses tool."""
    try:
        result = _hypothesis_generator.generate_hypotheses(**kwargs)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, indent=2)


def map_assumptions_handler(**kwargs) -> str:
    """Handler function for the map_assumptions tool."""
    try:
        result = _assumption_mapper.map_assumptions(**kwargs)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, indent=2)


def calibrate_confidence_handler(**kwargs) -> str:
    """Handler function for the calibrate_confidence tool."""
    try:
        result = _confidence_calibrator.calibrate_confidence(**kwargs)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, indent=2)


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
    
    def __init__(self, handlers: Optional[Dict[str, Callable]] = None):
        self.handlers = handlers or {
            "chain_of_thought_step": chain_of_thought_step_handler,
            "get_chain_summary": get_chain_summary_handler,
            "clear_chain": clear_chain_handler
        }
    
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
    
    def __init__(self, conversation_id: str, stop_handler: Optional[StopReasonHandler] = None):
        self.conversation_id = conversation_id
        self.chain = ChainOfThought()
        self.stop_handler = stop_handler or BedrockStopReasonHandler()
        self._tool_use_count = 0
        self._max_iterations = 20
    
    async def process_tool_loop(self, 
                              bedrock_client,
                              initial_request: Dict[str, Any],
                              max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """Process Bedrock tool loop with CoT integration."""
        
        max_iter = max_iterations or self._max_iterations
        messages = initial_request.get("messages", []).copy()
        
        for iteration in range(max_iter):
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: bedrock_client.converse(**{**initial_request, "messages": messages})
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
                            result = await self.stop_handler.execute_tool_call(tool_name, tool_input)
                            tool_results.append({
                                "toolResult": {
                                    "toolUseId": tool_use_id,
                                    "content": [{"text": json.dumps(result)}]
                                }
                            })
                        except Exception as e:
                            tool_results.append({
                                "toolResult": {
                                    "toolUseId": tool_use_id,
                                    "content": [{"text": json.dumps({"error": str(e)})}],
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
    
    async def get_reasoning_summary(self) -> Dict[str, Any]:
        """Get summary of the reasoning process."""
        return self.chain.generate_summary()
    
    def clear_reasoning(self) -> Dict[str, Any]:
        """Clear the reasoning chain."""
        self._tool_use_count = 0
        return self.chain.clear_chain()


class ThreadAwareChainOfThought:
    """Thread-safe version for production use."""
    
    _instances: Dict[str, ChainOfThought] = {}
    
    @classmethod
    def for_conversation(cls, conversation_id: str) -> ChainOfThought:
        """Get or create a ChainOfThought instance for a conversation."""
        if conversation_id not in cls._instances:
            cls._instances[conversation_id] = ChainOfThought()
        return cls._instances[conversation_id]
    
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.chain = self.for_conversation(conversation_id)
    
    def get_tool_specs(self):
        """Get tool specs for this instance."""
        from . import TOOL_SPECS
        return TOOL_SPECS
    
    def get_handlers(self):
        """Get handlers bound to this instance."""
        return {
            "chain_of_thought_step": lambda **kwargs: json.dumps(
                self.chain.add_step(**kwargs), indent=2
            ),
            "get_chain_summary": lambda: json.dumps(
                self.chain.generate_summary(), indent=2
            ),
            "clear_chain": lambda: json.dumps(
                self.chain.clear_chain(), indent=2
            )
        }
