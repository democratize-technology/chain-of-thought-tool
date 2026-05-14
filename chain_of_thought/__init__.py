"""
Chain of Thought Tool

A lightweight Python package that provides structured Chain of Thought reasoning capabilities for LLMs.

Usage:
    from chain_of_thought import TOOL_SPECS, HANDLERS
    
    # Add to your LLM tools
    tools = [
        *TOOL_SPECS,
        # ... other tools
    ]
    
    # Handle tool calls
    if tool_name in HANDLERS:
        result = HANDLERS[tool_name](**tool_args)
"""

from .core import (
    chain_of_thought_step_handler,
    get_chain_summary_handler,
    clear_chain_handler,
    generate_hypotheses_handler,
    map_assumptions_handler,
    calibrate_confidence_handler,
    export_chain_handler,
    import_chain_handler,
    # Generic handler factory
    create_generic_handler,
    TOOL_HANDLERS_CONFIG,
    # Service registry
    ServiceRegistry,
    get_service_registry,
    # Core classes
    ChainOfThought,
    ThoughtStep as ThoughtStep,
    ThreadAwareChainOfThought,
    StopReasonHandler,
    BedrockStopReasonHandler,
    AsyncChainOfThoughtProcessor
)
from .validators import ParameterValidator

# Tool specifications compatible with Converse API format
TOOL_SPECS = [
    {
        "toolSpec": {
            "name": "chain_of_thought_step",
            "description": "Add a step to structured chain-of-thought reasoning. Enables systematic problem-solving with confidence tracking, evidence, and assumptions.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "thought": {
                            "type": "string",
                            "description": "The reasoning content for this step"
                        },
                        "step_number": {
                            "type": "integer",
                            "description": "Current step number (starting from 1)",
                            "minimum": 1
                        },
                        "total_steps": {
                            "type": "integer",
                            "description": "Estimated total steps needed",
                            "minimum": 1
                        },
                        "next_step_needed": {
                            "type": "boolean",
                            "description": "Whether another step is needed"
                        },
                        "reasoning_stage": {
                            "type": "string",
                            "enum": ["Problem Definition", "Research", "Analysis", "Synthesis", "Conclusion"],
                            "default": "Analysis"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence level (0.0-1.0)",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.8
                        },
                        "dependencies": {
                            "type": "array",
                            "description": "Step numbers this depends on",
                            "items": {"type": "integer"}
                        },
                        "contradicts": {
                            "type": "array",
                            "description": "Step numbers this contradicts",
                            "items": {"type": "integer"}
                        },
                        "evidence": {
                            "type": "array",
                            "description": "Supporting evidence for this step",
                            "items": {"type": "string"}
                        },
                        "assumptions": {
                            "type": "array",
                            "description": "Assumptions made in this step",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["thought", "step_number", "total_steps", "next_step_needed"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "get_chain_summary",
            "description": "Get a comprehensive summary of the chain of thought reasoning process. Returns: total_steps, stages_covered, overall_confidence, confidence_by_stage, chain (with thought_preview per step), insights (evidence, assumptions, contradictions), content_synthesis (full thought text grouped by stage), completion_status (has_all_stages, percent_complete, stages_required, stages_missing), and metadata.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "clear_chain",
            "description": "Clear the chain of thought and start fresh",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "generate_hypotheses",
            "description": "Generate a structured hypothesis framework for an observation. Returns scientific, intuitive, contrarian, and systematic hypothesis templates that scaffold reasoning — the LLM consuming these provides the actual analysis.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "observation": {
                            "type": "string",
                            "description": "The observation or phenomenon to generate hypotheses for"
                        },
                        "hypothesis_count": {
                            "type": "integer",
                            "description": "Number of hypotheses to generate (1-4)",
                            "minimum": 1,
                            "maximum": 4,
                            "default": 4
                        }
                    },
                    "required": ["observation"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "map_assumptions",
            "description": "Map assumptions in statements using keyword-based heuristic patterns. Identifies linguistic indicators of assumptions (e.g., 'clearly', 'obviously', 'must') and returns a structured framework for the LLM to reason about.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": "The statement or claim to analyze for assumptions"
                        },
                        "depth": {
                            "type": "string",
                            "enum": ["surface", "deep"],
                            "default": "surface",
                            "description": "Analysis depth - 'surface' for basic patterns, 'deep' for comprehensive analysis"
                        }
                    },
                    "required": ["statement"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "calibrate_confidence",
            "description": "Calibrate confidence using heuristic pattern matching. Detects absolute language and domain-specific patterns to produce adjusted confidence with uncertainty bands. Provides a calibration framework — the LLM interprets and reasons about the output.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "prediction": {
                            "type": "string",
                            "description": "The prediction or claim to calibrate confidence for"
                        },
                        "initial_confidence": {
                            "type": "number",
                            "description": "Initial confidence level (0.0-1.0)",
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional additional context for calibration analysis",
                            "default": ""
                        }
                    },
                    "required": ["prediction", "initial_confidence"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "export_chain",
            "description": "Export the current chain of thought to a JSON file for later import.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute or relative path to the output JSON file",
                            "minLength": 1
                        }
                    },
                    "required": ["file_path"],
                    "additionalProperties": False
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "import_chain",
            "description": "Import a chain of thought from a JSON file, replacing the current chain.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute or relative path to the JSON file to import",
                            "minLength": 1
                        }
                    },
                    "required": ["file_path"],
                    "additionalProperties": False
                }
            }
        }
    }
]

# Handler mapping for easy tool execution
HANDLERS = {
    "chain_of_thought_step": chain_of_thought_step_handler,
    "get_chain_summary": get_chain_summary_handler,
    "clear_chain": clear_chain_handler,
    "generate_hypotheses": generate_hypotheses_handler,
    "map_assumptions": map_assumptions_handler,
    "calibrate_confidence": calibrate_confidence_handler,
    "export_chain": export_chain_handler,
    "import_chain": import_chain_handler
}

# Convenience exports
__all__ = [
    "TOOL_SPECS",
    "HANDLERS",
    # Core classes
    "ChainOfThought",
    "ThoughtStep",
    "ThreadAwareChainOfThought",
    "StopReasonHandler",
    "BedrockStopReasonHandler",
    "AsyncChainOfThoughtProcessor",
    # Dependency injection
    "ServiceRegistry",
    "get_service_registry",
    # Generic handler factory
    "create_generic_handler",
    "TOOL_HANDLERS_CONFIG",
    # Handler functions
    "chain_of_thought_step_handler",
    "get_chain_summary_handler",
    "clear_chain_handler",
    "generate_hypotheses_handler",
    "map_assumptions_handler",
    "calibrate_confidence_handler",
    "export_chain_handler",
    "import_chain_handler",
    # Utilities
    "ParameterValidator",
]

# Version info
__version__ = "0.3.0"
