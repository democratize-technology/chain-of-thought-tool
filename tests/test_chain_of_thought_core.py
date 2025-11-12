"""
Unit tests for core ChainOfThought class.

Tests cover:
- Step addition and validation
- Summary generation
- Chain clearing
- Metadata management
- Feedback generation
- Edge cases and error conditions
"""
import pytest
from datetime import datetime
from chain_of_thought.core import ChainOfThought, ThoughtStep


class TestThoughtStep:
    """Test ThoughtStep dataclass functionality."""
    
    def test_thought_step_creation(self):
        """Test creating a ThoughtStep with minimal parameters."""
        step = ThoughtStep(
            thought="Test thought",
            step_number=1,
            total_steps=3,
            next_step_needed=True
        )
        
        assert step.thought == "Test thought"
        assert step.step_number == 1
        assert step.total_steps == 3
        assert step.next_step_needed is True
        assert step.reasoning_stage == "Analysis"  # default
        assert step.confidence == 0.8  # default
        assert step.timestamp is not None
        assert isinstance(step.dependencies, list)
        assert isinstance(step.contradicts, list)
        assert isinstance(step.evidence, list)
        assert isinstance(step.assumptions, list)
    
    def test_thought_step_full_parameters(self):
        """Test creating a ThoughtStep with all parameters."""
        step = ThoughtStep(
            thought="Complex analysis step",
            step_number=2,
            total_steps=5,
            reasoning_stage="Synthesis",
            confidence=0.9,
            next_step_needed=True,
            dependencies=[1],
            contradicts=[3],
            evidence=["Market data", "User research"],
            assumptions=["Stable conditions"]
        )
        
        assert step.reasoning_stage == "Synthesis"
        assert step.confidence == 0.9
        assert step.dependencies == [1]
        assert step.contradicts == [3]
        assert step.evidence == ["Market data", "User research"]
        assert step.assumptions == ["Stable conditions"]
    
    def test_thought_step_timestamp_auto_generated(self):
        """Test that timestamp is automatically generated."""
        before = datetime.now().isoformat()
        step = ThoughtStep("test", 1, 1, True)
        after = datetime.now().isoformat()
        
        assert before <= step.timestamp <= after


class TestChainOfThought:
    """Test ChainOfThought class functionality."""
    
    def setup_method(self):
        """Set up fresh ChainOfThought instance for each test."""
        self.cot = ChainOfThought()
    
    def test_initialization(self):
        """Test ChainOfThought initialization."""
        assert len(self.cot.steps) == 0
        assert "created_at" in self.cot.metadata
        assert self.cot.metadata["total_confidence"] == 0.0
    
    def test_add_step_basic(self):
        """Test adding a basic step."""
        result = self.cot.add_step(
            thought="First step",
            step_number=1,
            total_steps=3,
            next_step_needed=True
        )
        
        assert result["status"] == "success"
        assert result["step_processed"] == 1
        assert result["progress"] == "1/3"
        assert result["next_step_needed"] is True
        assert result["is_revision"] is False
        assert len(self.cot.steps) == 1
        assert self.cot.steps[0].thought == "First step"
    
    def test_add_step_with_metadata(self):
        """Test adding step with full metadata."""
        result = self.cot.add_step(
            thought="Analysis step",
            step_number=1,
            total_steps=2,
            next_step_needed=True,
            reasoning_stage="Analysis",
            confidence=0.7,
            dependencies=[],
            contradicts=[],
            evidence=["Data point 1"],
            assumptions=["Assumption 1"]
        )
        
        assert result["confidence"] == 0.7
        step = self.cot.steps[0]
        assert step.reasoning_stage == "Analysis"
        assert step.evidence == ["Data point 1"]
        assert step.assumptions == ["Assumption 1"]
    
    def test_add_multiple_steps(self):
        """Test adding multiple steps in sequence."""
        # Add first step
        self.cot.add_step("Step 1", 1, 3, True)
        # Add second step  
        self.cot.add_step("Step 2", 2, 3, True)
        # Add final step
        self.cot.add_step("Step 3", 3, 3, False)
        
        assert len(self.cot.steps) == 3
        assert self.cot.steps[0].thought == "Step 1"
        assert self.cot.steps[1].thought == "Step 2"
        assert self.cot.steps[2].thought == "Step 3"
        assert self.cot.steps[2].next_step_needed is False
    
    def test_revise_existing_step(self):
        """Test revising an existing step."""
        # Add initial step
        self.cot.add_step("Original thought", 1, 2, True)
        
        # Revise the step
        result = self.cot.add_step("Revised thought", 1, 2, True)
        
        assert result["is_revision"] is True
        assert len(self.cot.steps) == 1  # Still only one step
        assert self.cot.steps[0].thought == "Revised thought"
    
    def test_generate_summary_empty_chain(self):
        """Test summary generation for empty chain."""
        summary = self.cot.generate_summary()
        
        assert summary["status"] == "empty"
        assert "message" in summary
    
    def test_generate_summary_with_steps(self):
        """Test summary generation with steps."""
        # Add steps across different stages
        self.cot.add_step(
            "Problem definition",
            1, 4, True,
            reasoning_stage="Problem Definition",
            confidence=0.9,
            evidence=["Requirement doc"]
        )
        self.cot.add_step(
            "Research findings", 
            2, 4, True,
            reasoning_stage="Research",
            confidence=0.7
        )
        self.cot.add_step(
            "Analysis result",
            3, 4, True, 
            reasoning_stage="Analysis",
            confidence=0.8,
            assumptions=["Market stability"]
        )
        self.cot.add_step(
            "Final conclusion",
            4, 4, False,
            reasoning_stage="Conclusion",
            confidence=0.85
        )
        
        summary = self.cot.generate_summary()
        
        assert summary["status"] == "success"
        assert summary["total_steps"] == 4
        assert len(summary["stages_covered"]) == 4
        assert "Problem Definition" in summary["stages_covered"]
        assert "Research" in summary["stages_covered"]
        assert "Analysis" in summary["stages_covered"]
        assert "Conclusion" in summary["stages_covered"]
        
        # Check confidence calculations
        assert summary["overall_confidence"] == 0.813  # Average of 0.9, 0.7, 0.8, 0.85 rounded to 3 decimal places
        assert "Problem Definition" in summary["confidence_by_stage"]
        
        # Check insights
        insights = summary["insights"]
        assert "Requirement doc" in insights["total_evidence"]
        assert "Market stability" in insights["total_assumptions"]
        assert 1 in insights["high_confidence_steps"]  # confidence >= 0.8 (0.9)
        assert 3 in insights["high_confidence_steps"]  # confidence >= 0.8 (0.8)
        assert 4 in insights["high_confidence_steps"]  # confidence >= 0.8 (0.85)
        assert len(insights["low_confidence_steps"]) == 0  # No steps with confidence < 0.5
    
    def test_generate_summary_with_contradictions(self):
        """Test summary generation with step contradictions."""
        self.cot.add_step("First hypothesis", 1, 3, True, confidence=0.6)
        self.cot.add_step("Conflicting evidence", 2, 3, True, contradicts=[1], confidence=0.7)
        self.cot.add_step("Resolution", 3, 3, False, dependencies=[1, 2], confidence=0.8)
        
        summary = self.cot.generate_summary()
        
        assert (2, 1) in summary["insights"]["contradiction_pairs"]
    
    def test_clear_chain(self):
        """Test clearing the chain."""
        # Add some steps
        self.cot.add_step("Step 1", 1, 2, True)
        self.cot.add_step("Step 2", 2, 2, False)
        
        assert len(self.cot.steps) == 2
        
        # Clear the chain
        result = self.cot.clear_chain()
        
        assert result["status"] == "success"
        assert len(self.cot.steps) == 0
        assert self.cot.metadata["total_confidence"] == 0.0
        assert "created_at" in self.cot.metadata
    
    def test_metadata_updates(self):
        """Test metadata updates as steps are added."""
        # Initially empty
        assert self.cot.metadata["total_confidence"] == 0.0
        
        # Add step with confidence 0.6
        self.cot.add_step("Step 1", 1, 2, True, confidence=0.6)
        assert self.cot.metadata["total_confidence"] == 0.6
        
        # Add step with confidence 0.8
        self.cot.add_step("Step 2", 2, 2, False, confidence=0.8)
        assert self.cot.metadata["total_confidence"] == 0.7  # (0.6 + 0.8) / 2
        assert "last_updated" in self.cot.metadata
    
    def test_feedback_generation_stages(self):
        """Test feedback generation for different reasoning stages."""
        # Test Problem Definition stage
        result = self.cot.add_step(
            "Define the problem",
            1, 3, True,
            reasoning_stage="Problem Definition"
        )
        assert "Foundation established" in result["feedback"]
        
        # Test Research stage
        self.cot.clear_chain()
        result = self.cot.add_step(
            "Gather information", 
            1, 3, True,
            reasoning_stage="Research"
        )
        assert "Gathering information" in result["feedback"]
    
    def test_feedback_generation_confidence_levels(self):
        """Test feedback generation for different confidence levels."""
        # Low confidence
        result = self.cot.add_step(
            "Uncertain step",
            1, 2, True,
            confidence=0.3
        )
        assert "Low confidence detected" in result["feedback"]
        
        # High confidence
        self.cot.clear_chain()
        result = self.cot.add_step(
            "Very confident step",
            1, 2, True, 
            confidence=0.95
        )
        assert "High confidence" in result["feedback"]
    
    def test_feedback_generation_dependencies_contradictions(self):
        """Test feedback for steps with dependencies and contradictions."""
        # Add base steps
        self.cot.add_step("Base step", 1, 4, True)
        self.cot.add_step("Another base", 2, 4, True)
        
        # Add step with dependencies
        result = self.cot.add_step(
            "Dependent step",
            3, 4, True,
            dependencies=[1, 2]
        )
        assert "Building on steps: 1, 2" in result["feedback"]
        
        # Add step with contradictions
        result = self.cot.add_step(
            "Contradicting step",
            4, 4, False,
            contradicts=[1]
        )
        assert "Contradicts steps: 1" in result["feedback"]
    
    def test_feedback_generation_progress_tracking(self):
        """Test feedback generation for progress tracking."""
        # Add steps approaching completion
        self.cot.add_step("Step 1", 1, 5, True)
        self.cot.add_step("Step 2", 2, 5, True)
        self.cot.add_step("Step 3", 3, 5, True)
        
        # Step 4 of 5 (80% progress) should suggest synthesis
        result = self.cot.add_step("Step 4", 4, 5, True)
        assert "Approaching conclusion" in result["feedback"]


class TestChainOfThoughtEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up fresh ChainOfThought instance for each test."""
        self.cot = ChainOfThought()
    
    def test_add_step_invalid_confidence(self):
        """Test adding step with invalid confidence values."""
        # Confidence should be clamped or handled appropriately
        # The current implementation doesn't validate, but we test the behavior
        result = self.cot.add_step(
            "Test step",
            1, 1, False,
            confidence=1.5  # > 1.0
        )
        
        assert result["status"] == "success"
        assert result["confidence"] == 1.5  # Current implementation allows this
    
    def test_empty_thought_content(self):
        """Test adding step with empty thought content."""
        result = self.cot.add_step("", 1, 1, False)
        
        assert result["status"] == "success"
        assert self.cot.steps[0].thought == ""
    
    def test_negative_step_numbers(self):
        """Test behavior with negative step numbers.""" 
        result = self.cot.add_step("Test", -1, 1, False)
        
        assert result["status"] == "success"
        assert self.cot.steps[0].step_number == -1
    
    def test_large_numbers(self):
        """Test handling of very large step numbers."""
        result = self.cot.add_step("Test", 999999, 1000000, False)
        
        assert result["status"] == "success"
        assert result["progress"] == "999999/1000000"
    
    def test_unicode_content(self):
        """Test handling of unicode content."""
        unicode_thought = "思考步骤 with émojis 🤔 and símb●ls"
        result = self.cot.add_step(unicode_thought, 1, 1, False)
        
        assert result["status"] == "success"
        assert self.cot.steps[0].thought == unicode_thought
    
    def test_very_long_content(self):
        """Test handling of very long thought content."""
        long_thought = "x" * 10000  # 10k characters
        result = self.cot.add_step(long_thought, 1, 1, False)
        
        assert result["status"] == "success"
        summary = self.cot.generate_summary()
        
        # Summary should truncate long thoughts
        chain_entry = summary["chain"][0]
        assert len(chain_entry["thought_preview"]) <= 103  # 100 + "..."
    
    def test_none_values_in_optional_fields(self):
        """Test handling of None values in optional fields."""
        result = self.cot.add_step(
            "Test step",
            1, 1, False,
            dependencies=None,
            contradicts=None,
            evidence=None,
            assumptions=None
        )
        
        assert result["status"] == "success"
        step = self.cot.steps[0]
        assert isinstance(step.dependencies, list)
        assert isinstance(step.contradicts, list)
        assert isinstance(step.evidence, list)
        assert isinstance(step.assumptions, list)