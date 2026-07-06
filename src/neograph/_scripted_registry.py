"""Scripted-shim registration for @node-decorated functions.

Extracted from _construct_builder.py per neograph-3zai. Houses
_register_node_scripted, the helper that builds the (input_data, config) ->
fn(*args) adapter closure and stashes it on the Node via the _scripted_shim
PrivateAttr. The compiler walks the construct and builds a fresh per-compile
scripted dict from these shims (compiler._collect_scripted_shims).

Imports only leaf modules (_sidecar, node) -- never decorators.py.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from neograph._sidecar import _get_param_res, _get_sidecar
from neograph.di import DIKind

if TYPE_CHECKING:
    from neograph.node import Node


def _register_node_scripted(
    n: Node,
    fan_out: set[str] | None = None,
    port_param_map: dict[str, str] | None = None,
    loop_renames: dict[str, str] | None = None,
) -> str | None:
    """Build the scripted shim, attach it to the Node, and return the lookup name.

    Post-ticket-bbov: the shim no longer registers into a process-global
    `Registry` singleton. Instead it is stored on the Node via
    `_scripted_shim` (PrivateAttr) and the lookup name is just `n.name`,
    so `compile()` can walk the construct and build a fresh per-compile
    scripted dict.

    Returns the lookup name to set on `node.scripted_fn` (via model_copy),
    or None if the node has no sidecar.
    """
    sidecar = _get_sidecar(n)
    if sidecar is None:
        return None
    fn, param_names = sidecar
    param_res = _get_param_res(n)
    _port_map = port_param_map or {}
    _loop_map = loop_renames or {}

    def _input_arg(pname: str, input_data: Any) -> Any:
        # Port param or loop rename: key was rewritten (e.g. "claim" →
        # "neo_subgraph_input", or "draft" → "seed" for loop self-ref). Fan-out
        # and upstream params are already in input_data under the param name
        # (fan-out via node.fan_out_param → neo_each_item, upstream via
        # factory._extract_input).
        lookup_key = _port_map.get(pname, _loop_map.get(pname, pname))
        return input_data.get(lookup_key) if isinstance(input_data, dict) else input_data

    def scripted_shim(input_data: Any, config: Any) -> Any:
        """Adapter: (input_data, config) → fn(*positional_args)."""
        args = []
        for pname in param_names:
            binding = param_res.get(pname)
            if binding is not None:
                args.append(binding.resolve(config))
            else:
                args.append(_input_arg(pname, input_data))
        return fn(*args)

    # A FROM_RESOURCE binding's value is AWAITED (the fetch), so the shim must be
    # a coroutine — resolution cannot happen on the sync path. ScriptedDispatch
    # .aexecute awaits an awaitable shim result; .execute fails loud on one
    # per neograph-khff, so run() surfaces the "use arun()" error and arun()
    # resolves. No new dispatch class — the exact one-line-twin property async
    # tool factories have in w74k.3.1.
    has_resource = any(
        b.kind == DIKind.FROM_RESOURCE for b in param_res.values()
    )

    async def ascripted_shim(input_data: Any, config: Any) -> Any:
        """Async adapter: awaits FROM_RESOURCE bindings before calling fn."""
        args = []
        for pname in param_names:
            binding = param_res.get(pname)
            if binding is not None:
                if binding.kind == DIKind.FROM_RESOURCE:
                    args.append(await binding.aresolve(config))
                else:
                    args.append(binding.resolve(config))
            else:
                args.append(_input_arg(pname, input_data))
        result = fn(*args)
        if inspect.isawaitable(result):
            result = await result
        return result

    # __name__ stays informational; the shim is registered under
    # n.scripted_fn (compiler._collect_scripted_shims), never via __name__.
    # See neograph-y20i.
    # Store the shim on the Node via PrivateAttr — compile() reads it and
    # inserts the entry into the per-compile scripted dict.
    n._scripted_shim = ascripted_shim if has_resource else scripted_shim
    return n.name
