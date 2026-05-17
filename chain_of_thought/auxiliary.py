"""
Auxiliary reasoning scaffolding tools for the Chain of Thought library.

These tools provide structured output shapes using template and heuristic
implementations. The LLM consuming the output performs the actual analysis.

MODULE-SIZE-JUSTIFICATION: This module extracts three tightly-coupled
dataclass+class pairs (Hypothesis/HypothesisGenerator, Assumption/AssumptionMapper,
ConfidenceAssessment/ConfidenceCalibrator) from core.py as part of an ADR-driven
decomposition. Each pair shares the same domain vocabulary and is too small to
warrant its own module. Splitting further into hypothesis.py, assumption.py, and
confidence.py would create three files under 250 lines each with no independent
reusability -- pure fragmentation with no cohesion benefit. The previous location
(core.py) was 2289 lines; this extraction is the meaningful decomposition boundary.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import math


HIGH_CONFIDENCE_THRESHOLD = 0.15
MEDIUM_CONFIDENCE_THRESHOLD = 0.05

MAX_PREDICTION_WORDS = 20


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

        self.hypotheses.clear()

        hypothesis_types = ["scientific", "intuitive", "contrarian", "systematic"]

        types_to_generate = hypothesis_types[:hypothesis_count]

        for i, hypothesis_type in enumerate(types_to_generate):
            hypothesis = self._generate_hypothesis_by_type(observation, hypothesis_type, i + 1)
            self.hypotheses.append(hypothesis)

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

    def __init__(self):
        self.assumptions: List[Assumption] = []
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "mapping_count": 0
        }

    def extract_explicit_assumptions(self, statement: str) -> List[Assumption]:
        """Extract explicitly stated assumptions from the statement."""
        assumptions = []

        assumption_indicators = [
            "assuming", "given that", "if we assume", "provided that",
            "taking for granted", "presupposing", "based on the premise"
        ]

        if any(indicator in statement.lower() for indicator in assumption_indicators):
            assumptions.append(Assumption(
                statement=f"Explicit assumption found in: '{statement[:50]}...'",
                assumption_type="explicit",
                confidence=0.9,
                is_critical=True,
                reasoning="Statement contains explicit assumption indicators",
                validation_methods=["Textual analysis", "Logical parsing"]
            ))

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

        if "market" in statement.lower() or "business" in statement.lower():
            assumptions.append(Assumption(
                statement="Market behavior follows rational economic principles",
                assumption_type="implicit",
                confidence=0.6,
                is_critical=True,
                reasoning="Business statements often assume market rationality",
                validation_methods=["Market research", "Economic data analysis"]
            ))

        if "because" in statement.lower() or "leads to" in statement.lower():
            assumptions.append(Assumption(
                statement="Causal relationships are direct and measurable",
                assumption_type="implicit",
                confidence=0.7,
                is_critical=True,
                reasoning="Causal language assumes direct cause-effect relationships",
                validation_methods=["Causal analysis", "Controlled experiments"]
            ))

        if any(word in statement.lower() for word in ["will", "future", "predict", "forecast"]):
            assumptions.append(Assumption(
                statement="Future conditions will remain similar to current conditions",
                assumption_type="implicit",
                confidence=0.5,
                is_critical=True,
                reasoning="Future-oriented statements assume continuity",
                validation_methods=["Trend analysis", "Scenario planning"]
            ))

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
            if assumption.confidence >= 0.7:
                assumption.is_critical = True
                critical_assumptions.append(assumption)

            if "causal" in assumption.reasoning.lower():
                assumption.is_critical = True
                critical_assumptions.append(assumption)

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

        self.assumptions.clear()

        explicit_assumptions = self.extract_explicit_assumptions(statement)
        implicit_assumptions = self.identify_implicit_assumptions(statement)

        if depth == "deep":
            additional_implicit = []

            if "data" in statement.lower() or "research" in statement.lower():
                additional_implicit.append(Assumption(
                    statement="Data sources are accurate and representative",
                    assumption_type="implicit",
                    confidence=0.6,
                    is_critical=True,
                    reasoning="Data-dependent statements assume source quality",
                    validation_methods=["Data validation", "Source verification"]
                ))

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

        all_assumptions = explicit_assumptions + implicit_assumptions
        self.assumptions = all_assumptions

        critical_assumptions = self.identify_critical_assumptions(all_assumptions)

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

        self.original_confidence = max(0.0, min(1.0, self.original_confidence))
        self.calibrated_confidence = max(0.0, min(1.0, self.calibrated_confidence))


class ConfidenceCalibrator:

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

        if confidence > 0.9:
            indicators.append("Very high initial confidence (>90%)")
            overconfidence_score += 0.3

        absolute_words = ["always", "never", "definitely", "certainly", "absolutely", "guaranteed", "impossible"]
        if any(word in prediction.lower() for word in absolute_words):
            indicators.append("Contains absolute language suggesting overconfidence")
            overconfidence_score += 0.2

        future_words = ["will", "going to", "by 2030", "by 2025", "next year", "soon"]
        if any(word in prediction.lower() for word in future_words):
            indicators.append("Future prediction with inherent uncertainty")
            overconfidence_score += 0.15

        complexity_indicators = ["and", "because", "due to", "multiple", "various", "complex"]
        complexity_count = sum(1 for word in complexity_indicators if word in prediction.lower())
        if complexity_count >= 2:
            indicators.append("Complex prediction with multiple factors")
            overconfidence_score += 0.1

        tech_words = ["ai", "artificial intelligence", "agi", "technology", "innovation", "breakthrough"]
        if any(word in prediction.lower() for word in tech_words):
            indicators.append("Technology prediction (historically overconfident domain)")
            overconfidence_score += 0.1

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

        if confidence > 0.95:
            uncertainty = 0.15
        elif confidence > 0.8:
            uncertainty = 0.1
        elif confidence > 0.6:
            uncertainty = 0.08
        else:
            uncertainty = 0.05


        lower_bound = max(0.0, confidence - uncertainty)
        upper_bound = min(1.0, confidence + uncertainty)

        return (round(lower_bound, 3), round(upper_bound, 3))

    def apply_calibration_adjustment(self, original_confidence: float, overconfidence_score: float) -> float:
        """Apply calibration adjustment based on overconfidence indicators."""

        adjustment_factor = overconfidence_score * 0.3

        adjusted_confidence = original_confidence * (1 - adjustment_factor)

        adjusted_confidence = max(0.1, adjusted_confidence)

        return round(adjusted_confidence, 3)

    def _identify_uncertainty_factors(self, prediction: str, context: str) -> List[str]:
        """Identify uncertainty factors based on prediction and context."""
        uncertainty_factors = []

        if "future" in prediction.lower() or any(word in prediction.lower() for word in ["will", "going to", "by 20"]):
            uncertainty_factors.append("Temporal uncertainty - future events")

        if "technology" in prediction.lower() or "ai" in prediction.lower():
            uncertainty_factors.append("Technology uncertainty - rapid change domain")

        if len(prediction.split()) > MAX_PREDICTION_WORDS:
            uncertainty_factors.append("Complexity uncertainty - multiple interconnected factors")

        if context and "limited data" in context.lower():
            uncertainty_factors.append("Data uncertainty - limited information available")

        return uncertainty_factors

    def _generate_calibration_reasoning(
        self,
        adjustment_magnitude: float,
        risk_level: str
    ) -> str:
        """Generate reasoning text for confidence calibration."""
        if adjustment_magnitude > HIGH_CONFIDENCE_THRESHOLD:
            reasoning = f"Significant confidence reduction ({adjustment_magnitude:.2f}) due to strong overconfidence indicators."
        elif adjustment_magnitude > MEDIUM_CONFIDENCE_THRESHOLD:
            reasoning = f"Moderate confidence adjustment ({adjustment_magnitude:.2f}) due to uncertainty factors."
        else:
            reasoning = f"Minor confidence adjustment ({adjustment_magnitude:.2f}) - original estimate reasonably calibrated."

        if risk_level == "high":
            reasoning += " High overconfidence risk detected."

        return reasoning

    def _create_confidence_assessment(
        self,
        initial_confidence: float,
        calibrated_confidence: float,
        uncertainty_band: tuple,
        overconfidence_analysis: Dict[str, Any],
        reasoning: str,
        uncertainty_factors: List[str]
    ) -> ConfidenceAssessment:
        """Create a ConfidenceAssessment instance."""
        return ConfidenceAssessment(
            original_confidence=initial_confidence,
            calibrated_confidence=calibrated_confidence,
            confidence_band=uncertainty_band,
            overconfidence_indicators=overconfidence_analysis["indicators"],
            calibration_reasoning=reasoning,
            uncertainty_factors=uncertainty_factors
        )

    def _build_calibration_response(
        self,
        prediction: str,
        initial_confidence: float,
        calibrated_confidence: float,
        uncertainty_band: tuple,
        overconfidence_analysis: Dict[str, Any],
        uncertainty_factors: List[str],
        reasoning: str
    ) -> Dict[str, Any]:
        """Build the calibration response dictionary."""
        adjustment_magnitude = abs(calibrated_confidence - initial_confidence)

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
        if not isinstance(initial_confidence, (int, float)) or isinstance(initial_confidence, bool):
            raise ValueError("initial_confidence must be a number")
        if isinstance(initial_confidence, float) and (math.isnan(initial_confidence) or math.isinf(initial_confidence)):
            raise ValueError("initial_confidence must be a finite number")
        if initial_confidence < 0.0 or initial_confidence > 1.0:
            raise ValueError("initial_confidence must be between 0.0 and 1.0")

        overconfidence_analysis = self.detect_overconfidence_patterns(prediction, initial_confidence)

        calibrated_confidence = self.apply_calibration_adjustment(
            initial_confidence,
            overconfidence_analysis["overconfidence_score"]
        )

        uncertainty_band = self.calculate_uncertainty_bands(calibrated_confidence)

        uncertainty_factors = self._identify_uncertainty_factors(prediction, context)

        adjustment_magnitude = abs(calibrated_confidence - initial_confidence)
        reasoning = self._generate_calibration_reasoning(
            adjustment_magnitude, overconfidence_analysis["risk_level"]
        )

        assessment = self._create_confidence_assessment(
            initial_confidence, calibrated_confidence, uncertainty_band,
            overconfidence_analysis, reasoning, uncertainty_factors
        )

        self.assessments.append(assessment)
        self.metadata["calibration_count"] += 1
        self.metadata["last_calibrated"] = datetime.now().isoformat()

        return self._build_calibration_response(
            prediction, initial_confidence, calibrated_confidence,
            uncertainty_band, overconfidence_analysis, uncertainty_factors, reasoning
        )
