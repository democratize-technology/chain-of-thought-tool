"""
Chain of Thought Tool - Core Implementation
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json


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
        
        # Validate step number
        if step_number != len(self.steps) + 1 and step_number not in [s.step_number for s in self.steps]:
            # Allow for revision of existing steps
            for i, step in enumerate(self.steps):
                if step.step_number == step_number:
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
                    return self._generate_feedback(self.steps[i], is_revision=True)
        
        # Create new step
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
        
        # Stage-specific guidance
        stage_guidance = {
            "Problem Definition": "Foundation established. Ensure the problem is clearly scoped.",
            "Research": "Gathering information. Consider multiple sources and perspectives.",
            "Analysis": "Breaking down components. Look for patterns and relationships.",
            "Synthesis": "Integrating insights. Focus on connections and implications.",
            "Conclusion": "Finalizing reasoning. Ensure conclusions address the initial problem."
        }
        
        if step.reasoning_stage in stage_guidance:
            feedback_parts.append(stage_guidance[step.reasoning_stage])
        
        # Confidence assessment
        if step.confidence < 0.5:
            feedback_parts.append("Low confidence detected. Consider gathering more evidence.")
        elif step.confidence > 0.9:
            feedback_parts.append("High confidence. Ensure assumptions are well-founded.")
        
        # Dependency analysis
        if step.dependencies:
            feedback_parts.append(f"Building on steps: {', '.join(map(str, step.dependencies))}")
        
        # Contradiction handling
        if step.contradicts:
            feedback_parts.append(f"Contradicts steps: {', '.join(map(str, step.contradicts))}. Consider reconciliation.")
        
        # Progress tracking
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
        
        # Collect all evidence and assumptions
        all_evidence = set()
        all_assumptions = set()
        contradiction_pairs = []
        
        for step in self.steps:
            all_evidence.update(step.evidence or [])
            all_assumptions.update(step.assumptions or [])
            if step.contradicts:
                for contradicted in step.contradicts:
                    contradiction_pairs.append((step.step_number, contradicted))
        
        # Calculate confidence metrics
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
