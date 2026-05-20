"""
Tests for export_chain / import_chain (Issue #1) and enhanced generate_summary (Issue #2).
"""
import json
import os
import tempfile

import pytest

from chain_of_thought.core import ChainOfThought, ThoughtStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_basic_chain(cot: ChainOfThought) -> None:
    """Populate a chain with three steps using add_step()."""
    cot.add_step("First thought", 1, 3, True, reasoning_stage="Problem Definition", confidence=0.9)
    cot.add_step("Second thought", 2, 3, True, reasoning_stage="Analysis", confidence=0.7)
    cot.add_step("Third thought", 3, 3, False, reasoning_stage="Conclusion", confidence=0.85)


# ---------------------------------------------------------------------------
# Issue #1: export_chain / import_chain
# ---------------------------------------------------------------------------

class TestExportChain:
    """Tests for ChainOfThought.export_chain()."""

    def setup_method(self):
        self.cot = ChainOfThought()

    def test_export_chain_saves_all_steps_including_non_sequential(self):
        """Regression: non-sequential step_numbers (1, 2, 6) are all exported."""
        # Directly set steps with a gap in numbering (bypass add_step revision logic)
        self.cot.steps = [
            ThoughtStep(thought="Step one", step_number=1, total_steps=6, next_step_needed=True),
            ThoughtStep(thought="Step two", step_number=2, total_steps=6, next_step_needed=True),
            ThoughtStep(thought="Step six", step_number=6, total_steps=6, next_step_needed=False),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            result = self.cot.export_chain(tmp_path)
            assert result["status"] == "success"
            assert result["steps_exported"] == 3

            with open(tmp_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            step_numbers = [s["step_number"] for s in data["steps"]]
            assert step_numbers == [1, 2, 6], f"Expected [1, 2, 6], got {step_numbers}"
        finally:
            os.unlink(tmp_path)

    def test_export_chain_invalid_path(self):
        """Graceful error on invalid export path."""
        self.cot.add_step("A thought", 1, 1, False)
        result = self.cot.export_chain("/nonexistent_directory_xyz/output.json")
        assert result["status"] == "error"
        assert "message" in result

    def test_export_chain_empty_chain(self):
        """Exporting an empty chain succeeds with zero steps."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            result = self.cot.export_chain(tmp_path)
            assert result["status"] == "success"
            assert result["steps_exported"] == 0

            with open(tmp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["steps"] == []
        finally:
            os.unlink(tmp_path)


class TestImportChain:
    """Tests for ChainOfThought.import_chain()."""

    def setup_method(self):
        self.cot = ChainOfThought()

    def test_import_chain_restores_all_steps(self):
        """All steps survive the round-trip (export then import)."""
        _make_basic_chain(self.cot)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            self.cot.export_chain(tmp_path)

            fresh = ChainOfThought()
            result = fresh.import_chain(tmp_path)

            assert result["status"] == "success"
            assert result["steps_imported"] == 3
            assert len(fresh.steps) == 3
        finally:
            os.unlink(tmp_path)

    def test_import_chain_replaces_existing_chain(self):
        """Import clears previous state and replaces it."""
        # Put something in the chain first
        self.cot.add_step("Old thought", 1, 1, False)
        assert len(self.cot.steps) == 1

        # Build the chain to import
        source = ChainOfThought()
        _make_basic_chain(source)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            source.export_chain(tmp_path)
            result = self.cot.import_chain(tmp_path)

            assert result["status"] == "success"
            # Old step gone, new steps present
            assert len(self.cot.steps) == 3
            thoughts = [s.thought for s in self.cot.steps]
            assert "Old thought" not in thoughts
        finally:
            os.unlink(tmp_path)

    def test_export_import_roundtrip_preserves_fields(self):
        """All ThoughtStep fields survive export/import without mutation."""
        original = ThoughtStep(
            thought="Detailed analysis",
            step_number=2,
            total_steps=5,
            reasoning_stage="Analysis",
            confidence=0.75,
            next_step_needed=True,
            dependencies=[1],
            contradicts=[3],
            evidence=["Source A", "Source B"],
            assumptions=["Assumption X"],
            timestamp="2026-01-01T00:00:00"
        )
        self.cot.steps = [original]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            self.cot.export_chain(tmp_path)

            fresh = ChainOfThought()
            fresh.import_chain(tmp_path)

            assert len(fresh.steps) == 1
            restored = fresh.steps[0]

            assert restored.thought == original.thought
            assert restored.step_number == original.step_number
            assert restored.total_steps == original.total_steps
            assert restored.reasoning_stage == original.reasoning_stage
            assert restored.confidence == original.confidence
            assert restored.next_step_needed == original.next_step_needed
            assert restored.dependencies == original.dependencies
            assert restored.contradicts == original.contradicts
            assert restored.evidence == original.evidence
            assert restored.assumptions == original.assumptions
            assert restored.timestamp == original.timestamp
        finally:
            os.unlink(tmp_path)

    def test_import_chain_file_not_found(self):
        """Graceful error on missing file."""
        result = self.cot.import_chain("/no/such/file_xyz_abc.json")
        assert result["status"] == "error"
        assert "message" in result

    def test_import_chain_non_sequential_step_numbers_preserved(self):
        """Non-sequential step numbers (1, 2, 6) are preserved after import."""
        self.cot.steps = [
            ThoughtStep(thought="A", step_number=1, total_steps=6, next_step_needed=True),
            ThoughtStep(thought="B", step_number=2, total_steps=6, next_step_needed=True),
            ThoughtStep(thought="C", step_number=6, total_steps=6, next_step_needed=False),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            self.cot.export_chain(tmp_path)

            fresh = ChainOfThought()
            fresh.import_chain(tmp_path)

            step_numbers = [s.step_number for s in fresh.steps]
            assert step_numbers == [1, 2, 6]
        finally:
            os.unlink(tmp_path)

    def test_import_chain_updates_metadata(self):
        """Metadata is updated after import."""
        source = ChainOfThought()
        source.add_step("Step", 1, 1, False, confidence=0.6)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            source.export_chain(tmp_path)
            self.cot.import_chain(tmp_path)
            # After import, metadata should reflect the imported step's confidence
            assert self.cot.metadata["total_confidence"] == pytest.approx(0.6)
        finally:
            os.unlink(tmp_path)

    def test_import_chain_rejects_empty_file_path(self):
        """import_chain returns an error dict when given an empty file path."""
        result = self.cot.import_chain("")
        assert result["status"] == "error"

    def test_import_chain_rejects_non_dict_json(self, tmp_path):
        """import_chain returns an error when the JSON root is not an object."""
        p = tmp_path / "bad.json"
        p.write_text("[1, 2, 3]")
        result = self.cot.import_chain(str(p))
        assert result["status"] == "error"

    def test_import_chain_rejects_missing_required_field(self, tmp_path):
        """import_chain returns an error when a step is missing required fields."""
        p = tmp_path / "bad.json"
        p.write_text('{"steps": [{"step_number": 1}]}')
        result = self.cot.import_chain(str(p))
        assert result["status"] == "error"

    def test_import_chain_rejects_wrong_type_for_step_number(self, tmp_path):
        """import_chain returns an error when step_number is not an integer."""
        p = tmp_path / "bad.json"
        p.write_text(
            '{"steps": [{"thought": "x", "step_number": "one", '
            '"total_steps": 1, "next_step_needed": false}]}'
        )
        result = self.cot.import_chain(str(p))
        assert result["status"] == "error"

    def test_import_chain_rejects_too_many_steps(self, tmp_path):
        """import_chain rejects payloads that exceed MAX_IMPORT_STEPS."""
        from chain_of_thought.core import MAX_IMPORT_STEPS
        p = tmp_path / "big.json"
        steps = [
            {
                "thought": "x",
                "step_number": i + 1,
                "total_steps": MAX_IMPORT_STEPS + 1,
                "next_step_needed": True,
            }
            for i in range(MAX_IMPORT_STEPS + 1)
        ]
        p.write_text(json.dumps({"steps": steps}))
        result = self.cot.import_chain(str(p))
        assert result["status"] == "error"
        assert str(MAX_IMPORT_STEPS) in result["message"]

    def test_export_chain_rejects_empty_file_path(self):
        """export_chain returns an error dict when given an empty file path."""
        self.cot.add_step("thought", 1, 1, False)
        result = self.cot.export_chain("")
        assert result["status"] == "error"

    def test_import_chain_preserves_data_without_double_escaping(self, tmp_path):
        """import_chain preserves data as-is without double HTML escaping."""
        p = tmp_path / "roundtrip.json"
        p.write_text(json.dumps({"steps": [{
            "thought": "<script>alert(1)</script>",
            "step_number": 1,
            "total_steps": 1,
            "next_step_needed": False,
            "evidence": ["<img src=x onerror=alert(1)>"],
            "assumptions": ["<b>bold</b>"]
        }]}))
        result = self.cot.import_chain(str(p))
        assert result["status"] == "success"
        step = self.cot.steps[0]
        assert step.thought == "<script>alert(1)</script>"
        assert step.evidence is not None
        assert step.evidence[0] == "<img src=x onerror=alert(1)>"
        assert step.assumptions is not None
        assert step.assumptions[0] == "<b>bold</b>"

    def test_import_chain_rejects_invalid_reasoning_stage_regex(self, tmp_path):
        """import_chain rejects reasoning_stage with injection characters."""
        import json
        p = tmp_path / "bad_stage.json"
        p.write_text(json.dumps({"steps": [{
            "thought": "test",
            "step_number": 1,
            "total_steps": 1,
            "next_step_needed": False,
            "reasoning_stage": "<script>bad</script>"
        }]}))
        result = self.cot.import_chain(str(p))
        assert result["status"] == "error"
        assert "reasoning_stage" in result["message"]

    def test_import_chain_rejects_nan_and_inf_confidence(self, tmp_path):
        """import_chain rejects NaN and Infinity in confidence field."""
        # Python's json.dumps cannot produce NaN, but json.loads accepts it
        # because Python's json module is non-strict by default.
        p = tmp_path / "nan_conf.json"
        p.write_text('{"steps": [{"thought": "t", "step_number": 1, '
                     '"total_steps": 1, "next_step_needed": false, "confidence": NaN}]}')
        result = self.cot.import_chain(str(p))
        assert result["status"] == "error"
        assert "finite" in result["message"]

        p2 = tmp_path / "inf_conf.json"
        p2.write_text('{"steps": [{"thought": "t", "step_number": 1, '
                      '"total_steps": 1, "next_step_needed": false, "confidence": Infinity}]}')
        result2 = self.cot.import_chain(str(p2))
        assert result2["status"] == "error"
        assert "finite" in result2["message"]

    def test_import_chain_resets_metadata_when_importing_empty_chain(self, tmp_path):
        """Importing an empty chain resets total_confidence to 0."""
        # Start with a non-empty chain
        self.cot.add_step("step", 1, 1, False, confidence=0.9)
        assert self.cot.metadata["total_confidence"] == pytest.approx(0.9)

        # Export an empty chain and import it
        empty_cot = ChainOfThought()
        p = tmp_path / "empty.json"
        empty_cot.export_chain(str(p))
        result = self.cot.import_chain(str(p))
        assert result["status"] == "success"
        assert result["steps_imported"] == 0
        assert self.cot.metadata["total_confidence"] == pytest.approx(0.0)
        assert "last_updated" not in self.cot.metadata

    def test_generate_summary_no_zerodivision_when_total_steps_zero(self):
        """Validator rejects step_number=0 (must be 1-1000), preventing ZeroDivisionError."""
        with pytest.raises(ValueError, match="step_number"):
            self.cot.add_step("thought", 0, 0, False)

    def test_import_chain_preserves_original_created_at(self, tmp_path):
        """created_at from the exported chain survives import round-trip."""
        source = ChainOfThought()
        original_created_at = source.metadata["created_at"]
        source.add_step("Step", 1, 1, False, confidence=0.7)

        p = tmp_path / "meta.json"
        source.export_chain(str(p))

        fresh = ChainOfThought()
        fresh.import_chain(str(p))

        assert fresh.metadata["created_at"] == original_created_at
        assert fresh.metadata["total_confidence"] == pytest.approx(0.7)

    def test_import_chain_preserves_thought_whitespace(self, tmp_path):
        """Thought text with leading/trailing whitespace is preserved."""
        p = tmp_path / "ws.json"
        p.write_text(json.dumps({"steps": [{
            "thought": "  indented thought  ",
            "step_number": 1,
            "total_steps": 1,
            "next_step_needed": False,
        }]}))
        result = self.cot.import_chain(str(p))
        assert result["status"] == "success"
        assert self.cot.steps[0].thought == "  indented thought  "

    def test_import_chain_preserves_evidence_assumptions_whitespace(self, tmp_path):
        """Evidence and assumptions with leading/trailing whitespace are preserved."""
        p = tmp_path / "ws_ea.json"
        p.write_text(json.dumps({"steps": [{
            "thought": "test",
            "step_number": 1,
            "total_steps": 1,
            "next_step_needed": False,
            "evidence": ["  evidence with spaces  "],
            "assumptions": ["\tassumption with tab\t"],
        }]}))
        result = self.cot.import_chain(str(p))
        assert result["status"] == "success"
        assert self.cot.steps[0].evidence == ["  evidence with spaces  "]
        assert self.cot.steps[0].assumptions == ["\tassumption with tab\t"]


# ---------------------------------------------------------------------------
# Issue #2: Enhanced generate_summary
# ---------------------------------------------------------------------------

class TestGenerateSummaryContentSynthesis:
    """Tests for the content_synthesis field added to generate_summary()."""

    def setup_method(self):
        self.cot = ChainOfThought()

    def test_generate_summary_includes_content_synthesis(self):
        """content_synthesis field is present and contains actual thought text."""
        self.cot.add_step(
            "Define the scope of the problem",
            1, 3, True,
            reasoning_stage="Problem Definition"
        )
        self.cot.add_step(
            "Analyse available data",
            2, 3, True,
            reasoning_stage="Analysis"
        )
        self.cot.add_step(
            "Final conclusion reached",
            3, 3, False,
            reasoning_stage="Conclusion"
        )

        summary = self.cot.generate_summary()

        assert "content_synthesis" in summary
        cs = summary["content_synthesis"]

        assert "Problem Definition" in cs
        assert "Analysis" in cs
        assert "Conclusion" in cs

        # Full text, not truncated
        assert cs["Problem Definition"] == ["Define the scope of the problem"]
        assert cs["Analysis"] == ["Analyse available data"]
        assert cs["Conclusion"] == ["Final conclusion reached"]

    def test_generate_summary_content_synthesis_multiple_steps_per_stage(self):
        """Multiple steps in the same stage all appear in content_synthesis."""
        self.cot.add_step("Analysis A", 1, 2, True, reasoning_stage="Analysis")
        self.cot.add_step("Analysis B", 2, 2, False, reasoning_stage="Analysis")

        summary = self.cot.generate_summary()
        cs = summary["content_synthesis"]

        assert "Analysis" in cs
        assert "Analysis A" in cs["Analysis"]
        assert "Analysis B" in cs["Analysis"]

    def test_generate_summary_content_synthesis_full_text_not_truncated(self):
        """Content synthesis stores the full thought text, not a 100-char preview."""
        long_thought = "x" * 500
        self.cot.add_step(long_thought, 1, 1, False, reasoning_stage="Analysis")

        summary = self.cot.generate_summary()
        cs = summary["content_synthesis"]

        assert cs["Analysis"][0] == long_thought


class TestGenerateSummaryCompletionStatus:
    """Tests for the completion_status field added to generate_summary()."""

    def setup_method(self):
        self.cot = ChainOfThought()

    def test_generate_summary_has_all_stages_false_when_missing_stages(self):
        """has_all_stages is False when fewer than 5 required stages are present."""
        self.cot.add_step("Problem", 1, 2, True, reasoning_stage="Problem Definition")
        self.cot.add_step("Analysis", 2, 2, False, reasoning_stage="Analysis")

        summary = self.cot.generate_summary()
        cs = summary["completion_status"]

        assert cs["has_all_stages"] is False

    def test_generate_summary_has_all_stages_true_when_complete(self):
        """has_all_stages is True when all 5 required stages are present."""
        stages = ["Problem Definition", "Research", "Analysis", "Synthesis", "Conclusion"]
        for i, stage in enumerate(stages, start=1):
            self.cot.add_step(f"Step {i}", i, 5, i < 5, reasoning_stage=stage)

        summary = self.cot.generate_summary()
        cs = summary["completion_status"]

        assert cs["has_all_stages"] is True
        assert cs["percent_complete"] == 100.0
        assert cs["stages_missing"] == []

    def test_generate_summary_percent_complete(self):
        """percent_complete correctly reflects 4/5 stages as 80.0."""
        self.cot = ChainOfThought()
        self.cot.add_step("PD", 1, 4, True, reasoning_stage="Problem Definition")
        self.cot.add_step("R", 2, 4, True, reasoning_stage="Research")
        self.cot.add_step("A", 3, 4, True, reasoning_stage="Analysis")
        self.cot.add_step("C", 4, 4, False, reasoning_stage="Conclusion")

        summary = self.cot.generate_summary()
        cs = summary["completion_status"]

        assert cs["percent_complete"] == 80.0
        assert "Synthesis" in cs["stages_missing"]
        assert cs["has_all_stages"] is False

    def test_generate_summary_stages_required_documented(self):
        """stages_required lists exactly the 5 canonical stage names."""
        self.cot.add_step("A step", 1, 1, False)
        summary = self.cot.generate_summary()
        cs = summary["completion_status"]

        expected = ["Problem Definition", "Research", "Analysis", "Synthesis", "Conclusion"]
        assert cs["stages_required"] == expected

    def test_generate_summary_stages_missing_lists_absent_stages(self):
        """stages_missing correctly lists only the absent required stages."""
        self.cot.add_step("PD", 1, 1, False, reasoning_stage="Problem Definition")

        summary = self.cot.generate_summary()
        cs = summary["completion_status"]

        assert set(cs["stages_missing"]) == {"Research", "Analysis", "Synthesis", "Conclusion"}


class TestGenerateSummaryBackwardCompatibility:
    """Verify existing summary fields still present after the enhancement."""

    def setup_method(self):
        self.cot = ChainOfThought()
        _make_basic_chain(self.cot)

    def test_generate_summary_backward_compatible(self):
        """Existing fields (total_steps, stages_covered, chain, insights) still present."""
        summary = self.cot.generate_summary()

        assert summary["status"] == "success"
        assert "total_steps" in summary
        assert "stages_covered" in summary
        assert "overall_confidence" in summary
        assert "confidence_by_stage" in summary
        assert "chain" in summary
        assert "insights" in summary
        assert "metadata" in summary

    def test_generate_summary_chain_field_still_has_thought_preview(self):
        """chain entries still have thought_preview (truncated) alongside full text in synthesis."""
        long_thought = "y" * 200
        self.cot.add_step(long_thought, 4, 4, False, reasoning_stage="Synthesis")

        summary = self.cot.generate_summary()

        # Find the chain entry for step 4
        entry = next(e for e in summary["chain"] if e["step"] == 4)
        assert entry["thought_preview"].endswith("...")
        assert len(entry["thought_preview"]) <= 103

        # But content_synthesis has the full text
        assert summary["content_synthesis"]["Synthesis"][0] == long_thought

    def test_generate_summary_empty_chain_unchanged(self):
        """Empty chain still returns status=empty, not the new fields."""
        fresh = ChainOfThought()
        summary = fresh.generate_summary()
        assert summary["status"] == "empty"
        assert "content_synthesis" not in summary
        assert "completion_status" not in summary
