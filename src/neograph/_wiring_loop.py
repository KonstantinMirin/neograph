"""Loop routing helpers — the pure decision/unwrap functions for ``Loop``.

Extracted from ``_wiring.py`` (neograph-3ffdg.2) as a pure file split — the
functions below are unchanged, only their home moved.

Scope note: ``_add_loop_back_edge`` and ``_add_subgraph_loop`` deliberately
STAYED in ``_wiring.py``. Both call ``_resolve_condition``, which is genuinely
cross-cutting (the agent-cycle tool gate and the Operator check use it too, and
``factory.py`` reaches it via a function-local import carrying its own allowlist
entry), so it stays put. Moving the two back-edge builders here would have meant
either importing back into ``_wiring.py`` — a cycle — or injecting the resolver
as a parameter, which would have grown ``compiler.py``'s two call sites past its
own size ceiling. A smaller extraction beats both. What lives here is the pure
part: routing decisions and value unwrapping, no condition resolution.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog

from neograph._normalize import primary_output_field
from neograph._state_bus import StateBus, adapt_state
from neograph.di import _unwrap_loop_value
from neograph.errors import ExecutionError
from neograph.modifiers import Loop
from neograph.node import Node

log = structlog.get_logger()

LangGraphRouterFn = Callable[[Any], str]
LangGraphLoopUnwrapFn = Callable[[StateBus, str], Any]


def _make_loop_router(
    item_name: str,
    field_name: str,
    count_field: str,
    loop: Loop,
    condition: Callable[[Any], bool],
    exit_name: str,
    reenter_target: str,
    unwrap_fn: LangGraphLoopUnwrapFn,
) -> LangGraphRouterFn:
    """Build a loop_router closure shared by Node and Construct loop wiring.

    Parameters
    ----------
    unwrap_fn:
        ``(state, field_name) -> value`` -- extracts the latest output from
        state.  Node path handles dict-form outputs and list unwrapping;
        Construct path reads the field and delegates to _unwrap_loop_value.
    """

    def loop_router(state: Any) -> str:
        bus = adapt_state(state)
        # Counter bootstrap (absent/None -> 0) lives in StateBus.get_counter.
        count = bus.get_counter(count_field)
        if count >= loop.max_iterations:
            if loop.on_exhaust == "error":
                raise ExecutionError.build(
                    "loop exceeded max_iterations",
                    expected=f"convergence within {loop.max_iterations} iterations",
                    found=f"{loop.max_iterations} iterations exhausted",
                    node=item_name,
                )
            return exit_name
        val = unwrap_fn(bus, field_name)
        try:
            should_continue = condition(val)
        except (AttributeError, TypeError) as exc:
            raise ExecutionError.build(
                f"loop condition raised {type(exc).__name__}",
                found=f"value {val!r}",
                hint=str(exc),
                node=item_name,
            ) from exc
        if should_continue:
            return reenter_target
        return exit_name

    return loop_router


def _node_loop_unwrap(node: Node, field_name: str) -> LangGraphLoopUnwrapFn:
    """Unwrap callback for Node loop routers.

    Handles dict-form outputs (primary key) and list unwrapping from the
    append-reducer that Loop uses.
    """

    def unwrap(state: StateBus, _field_name: str) -> Any:
        # Dict-form outputs: primary value lands on {field}_{first_key}.
        state_field = primary_output_field(_field_name, node.outputs)
        # StateBus.get optional: loop-bootstrap — first router pass may have
        # not-yet-populated list; user condition expected to handle None.
        # Empty list -> None (no output yet, e.g. skip_when with no skip_value)
        # so user conditions like `lambda d: d is None or ...` work; the
        # construct-loop path delegates to the same helper.
        own_val = state.get(state_field)
        return _unwrap_loop_value(own_val, object)

    return unwrap


def _construct_loop_unwrap(state: StateBus, field_name: str) -> Any:
    """Unwrap callback for Construct loop routers.

    Receives a pre-adapted StateBus from ``loop_router``.
    """
    # StateBus.get optional: loop-bootstrap — sub-construct output absent on
    # first pass; condition handles None.
    val = state.get(field_name)
    return _unwrap_loop_value(val, object)
