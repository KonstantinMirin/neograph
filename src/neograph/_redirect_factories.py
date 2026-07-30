"""Redirect factories — the fan-out wrappers for Oracle, Each, and their fusion.

Extracted from ``_oracle.py`` (neograph-3ffdg.16) as a pure file split — the
three factories below are unchanged, only their home moved. ``_oracle.py``
re-exports them, so existing imports keep resolving.

The ticket left the size of this cut to the implementer. All three redirect
factories moved rather than only ``make_each_redirect_fn``: they are the same
concern (build the wrapper that turns one node into N fanned-out sends), none of
them touches the merge half of the file, and taking all three is what brings
``_oracle.py`` under the 500-line cap.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.errors import GraphBubbleUp

from neograph._state_bus import adapt_state
from neograph._state_keys import StateKeys
from neograph.modifiers import Each, EachFailure
from neograph.naming import split_output_field
from neograph.node import HasName


def make_oracle_redirect_fn(
    raw_fn: Runnable,
    field_name: str,
    collector_field: str,
    item: HasName,
) -> Runnable:
    """Wrap a node function to redirect output from field_name to collector_field.

    Used by Oracle generators: the node writes to the collector (list reducer)
    instead of the consumer-facing field.

    Handles both single-type outputs (result has field_name key) and dict-form
    outputs (result has {field_name}_{key} keys). For dict-form, collects the
    full result dict into the collector so the merge fn can process per-key.

    ``item`` carries the IR node/construct; ``item.name`` is the user-facing
    label. This factory does not currently raise StateMissingError, so it does
    not read ``item.name`` — the parameter is kept for signature parity with
    the other two redirect factories (a future get_required here asks
    ``item.name`` directly, never a threaded string nor ``raw_fn.__name__``).
    """

    def _project(result: dict) -> dict:
        val = result.get(field_name)
        if val is not None:
            return {collector_field: val}
        # Dict-form outputs: per-key fields like {field_name}_{key}
        if any(split_output_field(k, field_name) is not None for k in result):
            return {collector_field: result}
        return result

    def oracle_redirect_fn(state: Any, config: RunnableConfig) -> dict:
        return _project(raw_fn.invoke(state, config))

    async def aoracle_redirect_fn(state: Any, config: RunnableConfig) -> dict:
        return _project(await raw_fn.ainvoke(state, config))

    # Dual-path: the gen node the redirect wraps has its own sync/async twins;
    # under ainvoke we MUST await raw_fn.ainvoke() or an Oracle-wrapped LLM node
    # would be threadpooled and block the loop (review H2). Shared _project()
    # keeps sync/async post-processing from drifting.
    return RunnableLambda(oracle_redirect_fn, afunc=aoracle_redirect_fn)


def make_eachoracle_redirect_fn(
    raw_fn: Runnable,
    field_name: str,
    collector_field: str,
    each_key: str,
    item: HasName,
) -> Runnable:
    """Wrap a node function for Each x Oracle fusion.

    Like make_oracle_redirect_fn, but tags each result with the each_key
    extracted from neo_each_item. The collector accumulates (key, result) tuples.

    ``item`` is the captured IR node/construct; ``item.name`` is the
    user-declared label surfaced in StateMissingError messages. The closure
    asks the IR object directly (Information Expert) rather than receiving a
    pre-extracted string.
    """

    def _project(state: Any, result: dict) -> dict:
        # REQUIRED: flat Each×Oracle router always populates EACH_ITEM in the
        # Send payload. Absence = wiring bug.
        each_item = adapt_state(state).get_required(
            StateKeys.EACH_ITEM,
            node_label=item.name,
        )
        key = getattr(each_item, each_key, str(each_item))
        # Single-type outputs: result has {field_name: val}
        val = result.get(field_name)
        if val is not None:
            return {collector_field: [(key, val)]}
        # Dict-form outputs: result has per-key fields.
        # Collect the full per-key dict as the tagged value.
        per_key = {ok: v for k, v in result.items() if (ok := split_output_field(k, field_name)) is not None}
        if per_key:
            return {collector_field: [(key, per_key)]}
        return result

    def eachoracle_redirect_fn(state: Any, config: RunnableConfig) -> dict:
        return _project(state, raw_fn.invoke(state, config))

    async def aeachoracle_redirect_fn(state: Any, config: RunnableConfig) -> dict:
        return _project(state, await raw_fn.ainvoke(state, config))

    return RunnableLambda(eachoracle_redirect_fn, afunc=aeachoracle_redirect_fn)


def make_each_redirect_fn(
    raw_fn: Runnable,
    field_name: str,
    each: Each,
    item: HasName,
) -> Runnable:
    """Wrap a node function to key the result by the Each item's key field.

    Reads neo_each_item from state, uses each.key to extract the dispatch key.
    ``item`` is the captured IR node/construct; ``item.name`` is the
    user-declared label surfaced in StateMissingError messages — asked of the
    IR object directly, never threaded as a string.
    """

    def _key_val(state: Any) -> Any:
        # REQUIRED: Each router always populates EACH_ITEM in the Send payload.
        each_item = adapt_state(state).get_required(
            StateKeys.EACH_ITEM,
            node_label=item.name,
        )
        return getattr(each_item, each.key, str(each_item))

    def _project(state: Any, result: dict) -> dict:
        val = result.get(field_name)
        if val is not None:
            return {field_name: {_key_val(state): val}}
        return result

    def _project_failure(state: Any, exc: Exception) -> dict:
        # on_error='collect': key a typed EachFailure into the barrier so the
        # barrier always completes with one entry per planned key.
        key_val = _key_val(state)
        failure = EachFailure(
            key=str(key_val),
            error_type=type(exc).__name__,
            message=str(exc),
        )
        return {field_name: {key_val: failure}}

    # raw_fn is a RunnableLambda (make_node_fn or a wrapped subgraph_fn);
    # .invoke(state, None) is safe (langchain synthesizes a config).
    def each_redirect_fn(state: Any, config: RunnableConfig = None) -> dict:  # type: ignore[assignment]
        if each.on_error == "collect":
            try:
                return _project(state, raw_fn.invoke(state, config))
            except GraphBubbleUp:
                # HITL interrupt / Command routing / cancellation must propagate,
                # never be collected into an EachFailure.
                raise
            except Exception as exc:  # noqa: BLE001 — collect any per-item fault
                return _project_failure(state, exc)
        return _project(state, raw_fn.invoke(state, config))

    async def aeach_redirect_fn(state: Any, config: RunnableConfig = None) -> dict:  # type: ignore[assignment]
        if each.on_error == "collect":
            try:
                return _project(state, await raw_fn.ainvoke(state, config))
            except GraphBubbleUp:
                # HITL interrupt / Command routing / cancellation must propagate,
                # never be collected into an EachFailure.
                raise
            except Exception as exc:  # noqa: BLE001 — collect any per-item fault
                return _project_failure(state, exc)
        return _project(state, await raw_fn.ainvoke(state, config))

    return RunnableLambda(each_redirect_fn, afunc=aeach_redirect_fn)
