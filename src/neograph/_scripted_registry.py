"""Scripted-shim registration for @node-decorated functions.

Extracted from _construct_builder.py per neograph-3zai. Houses
_register_node_scripted, the helper that builds the (input_data, config) ->
fn(*args) adapter closure and stashes it on the Node via the _scripted_shim
PrivateAttr. The compiler walks the construct and builds a fresh per-compile
scripted dict from these shims (compiler._collect_scripted_shims).

Imports only leaf modules (_sidecar, node) -- never decorators.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neograph._sidecar import _get_param_res, _get_sidecar

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

    def scripted_shim(input_data: Any, config: Any) -> Any:
        """Adapter: (input_data, config) → fn(*positional_args)."""
        args = []
        for pname in param_names:
            binding = param_res.get(pname)
            if binding is not None:
                args.append(binding.resolve(config))
            else:
                # Port param or loop rename: key was rewritten
                # (e.g. "claim" → "neo_subgraph_input", or
                # "draft" → "seed" for loop self-ref). Look up rewritten key.
                lookup_key = _port_map.get(pname, _loop_map.get(pname, pname))
                # Fan-out or upstream param — both are already in
                # input_data under the param name (fan-out via
                # node.fan_out_param → neo_each_item, upstream via
                # factory._extract_input).
                args.append(
                    input_data.get(lookup_key)
                    if isinstance(input_data, dict)
                    else input_data
                )
        return fn(*args)

    # __name__ stays informational; the shim is registered under
    # n.scripted_fn (compiler._collect_scripted_shims), never via __name__.
    # See neograph-y20i.
    # Store the shim on the Node via PrivateAttr — compile() reads it and
    # inserts the entry into the per-compile scripted dict.
    n._scripted_shim = scripted_shim
    return n.name
