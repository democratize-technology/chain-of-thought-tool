"""
Handler functions for Chain of Thought tools.

Convenience wrappers and factory-created handlers for tool execution.
All imports from core.py are deferred to avoid circular dependency.
"""


def chain_of_thought_step_handler(**kwargs) -> str:
    from .core import create_generic_handler
    return create_generic_handler('chain_of_thought_step')(**kwargs)


def get_chain_summary_handler(**kwargs) -> str:
    from .core import create_generic_handler
    return create_generic_handler('get_chain_summary')(**kwargs)


def clear_chain_handler(**kwargs) -> str:
    from .core import create_generic_handler
    return create_generic_handler('clear_chain')(**kwargs)


def generate_hypotheses_handler(**kwargs) -> str:
    from .core import create_generic_handler
    return create_generic_handler('generate_hypotheses')(**kwargs)


def map_assumptions_handler(**kwargs) -> str:
    from .core import create_generic_handler
    return create_generic_handler('map_assumptions')(**kwargs)


def calibrate_confidence_handler(**kwargs) -> str:
    from .core import create_generic_handler
    return create_generic_handler('calibrate_confidence')(**kwargs)


def export_chain_handler(**kwargs) -> str:
    from .core import _chain_processor, _safe_json_dumps
    try:
        result = _chain_processor.export_chain(**kwargs)
        return _safe_json_dumps(result, indent=2)
    except Exception as e:
        return _safe_json_dumps({"status": "error", "message": str(e)}, indent=2)


def import_chain_handler(**kwargs) -> str:
    from .core import _chain_processor, _safe_json_dumps
    try:
        result = _chain_processor.import_chain(**kwargs)
        return _safe_json_dumps(result, indent=2)
    except Exception as e:
        return _safe_json_dumps({"status": "error", "message": str(e)}, indent=2)


def create_chain_of_thought_step_handler(registry=None, rate_limiter=None, client_id="default"):
    from .core import create_generic_handler
    return create_generic_handler('chain_of_thought_step', registry, rate_limiter, client_id)


def create_get_chain_summary_handler(registry=None, rate_limiter=None, client_id="default"):
    from .core import create_generic_handler
    return create_generic_handler('get_chain_summary', registry, rate_limiter, client_id)


def create_clear_chain_handler(registry=None, rate_limiter=None, client_id="default"):
    from .core import create_generic_handler
    return create_generic_handler('clear_chain', registry, rate_limiter, client_id)


def create_generate_hypotheses_handler(registry=None, rate_limiter=None, client_id="default"):
    from .core import create_generic_handler
    return create_generic_handler('generate_hypotheses', registry, rate_limiter, client_id)


def create_map_assumptions_handler(registry=None, rate_limiter=None, client_id="default"):
    from .core import create_generic_handler
    return create_generic_handler('map_assumptions', registry, rate_limiter, client_id)


def create_calibrate_confidence_handler(registry=None, rate_limiter=None, client_id="default"):
    from .core import create_generic_handler
    return create_generic_handler('calibrate_confidence', registry, rate_limiter, client_id)
