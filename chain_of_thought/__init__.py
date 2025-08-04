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
    ChainOfThought,
    ThreadAwareChainOfThought
)

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
            "description": "Get a comprehensive summary of the chain of thought reasoning process",
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
    }
]

# Handler mapping for easy tool execution
HANDLERS = {
    "chain_of_thought_step": chain_of_thought_step_handler,
    "get_chain_summary": get_chain_summary_handler,
    "clear_chain": clear_chain_handler
}

# Convenience exports
__all__ = [
    "TOOL_SPECS",
    "HANDLERS", 
    "ChainOfThought",
    "ThreadAwareChainOfThought",
    "chain_of_thought_step_handler",
    "get_chain_summary_handler",
    "clear_chain_handler"
]

# Version info
__version__ = "0.1.0"
