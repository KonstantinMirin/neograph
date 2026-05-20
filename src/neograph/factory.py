"""Node-function construction — turns Node definitions into LangGraph callables."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from neograph._dispatch import (  # noqa: F401 — re-exported for tests/backward compat
    ModeDispatch,
    NodeInput,
    NodeOutput,
    ScriptedDispatch,
    ThinkDispatch,
    ToolDispatch,
    _dispatch_for_mode,
    _render_input,
)
from neograph._execute import (  # noqa: F401 — re-exported for tests
    _execute_node,
    _extract_context,
    _type_name,
)
from neograph._input_shape import (  # noqa: F401 — internal helpers re-exported for tests
    InputShape,
    _classify_input_shape,
    _extract_each_item,
    _extract_fan_in_dict,
    _extract_input,
    _extract_loop_reentry,
    _extract_single_type,
)
from neograph._oracle import (  # noqa: F401 — re-exported so compiler.py imports stay stable
    _build_oracle_merge_result,
    _inject_oracle_config,
    _unwrap_oracle_results,
    make_each_redirect_fn,
    make_eachoracle_redirect_fn,
    make_oracle_merge_fn,
    make_oracle_redirect_fn,
)
from neograph._registry import registry
from neograph._runtime_registry import (  # noqa: F401 — re-exported for public API
    lookup_condition,
    lookup_scripted,
    register_condition,
    register_scripted,
    register_tool_factory,
)
from neograph._state_bus import adapt_state  # noqa: F401 — re-exported for tests
from neograph._state_write import (  # noqa: F401 — re-exported for tests
    _apply_skip_when,
    _build_state_update,
)
from neograph._subconstruct import make_subgraph_fn  # noqa: F401 — re-exported

# Backward-compat re-exports for tests that imported these helpers from
# factory.py before the §4 split. Each is `noqa`'d individually because ruff
# strips items inside a parenthesized import group even with a line-level noqa.
from neograph.di import _isinstance_safe as _is_instance_safe  # noqa: F401
from neograph.di import (
    _unwrap_each_dict,  # noqa: F401
    _unwrap_loop_value,  # noqa: F401
)
from neograph.errors import ConfigurationError
from neograph.naming import field_name_for
from neograph.node import Node

log = structlog.get_logger()


def make_node_fn(node: Node) -> Callable:
    """Create a LangGraph node function from a Node definition.

    This is the core of NeoGraph — the generic factory that eliminates
    the 70% boilerplate from every hand-coded node.

    Raw nodes get a minimal observability wrapper. All other modes
    (scripted, think, agent, act) go through _execute_node with a
    mode-specific ModeDispatch.
    """
    # Raw node — wrap with observability so node_start/node_complete fire
    if node.raw_fn is not None:
        return _make_raw_wrapper(node)

    # Validate scripted registration early
    if node.mode == "scripted" and node.scripted_fn not in registry.scripted:
        raise ConfigurationError.build(
            f"Scripted function '{node.scripted_fn}' not registered",
            hint="Use register_scripted() to register it before compilation",
            node=node.name,
        )

    dispatch = _dispatch_for_mode(node)
    field_name = field_name_for(node.name)

    def node_wrapper(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        return _execute_node(node, state, config, dispatch)

    node_wrapper.__name__ = field_name
    return node_wrapper


def _make_raw_wrapper(node: Node) -> Callable:
    """Wrap a raw_fn dispatch with observability (node_start/node_complete).

    Only used for explicit ``mode='raw'`` escape-hatch nodes. Raw nodes
    bypass the unified _execute_node path — no DI/input/output wrapping,
    only logging.
    """
    assert node.raw_fn is not None, f"node '{node.name}' has mode='raw' but no raw_fn"
    raw_fn = node.raw_fn
    field_name = field_name_for(node.name)

    def raw_node_wrapper(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        node_log = log.bind(node=node.name, mode="raw")
        node_log.info("node_start", input_type=_type_name(node.inputs), output_type=_type_name(node.outputs))
        t0 = time.monotonic()

        result = raw_fn(state, config)

        elapsed = time.monotonic() - t0
        node_log.info("node_complete", duration_s=round(elapsed, 3))
        return result

    raw_node_wrapper.__name__ = field_name
    return raw_node_wrapper
