"""Construct-building functions extracted from decorators.py.

Houses construct_from_module, construct_from_functions, and their
internal helpers (_build_construct_from_decorated, _register_node_scripted,
_resolve_dict_output_param, _resolve_loop_self_param).

These functions depend on sidecar / DI helpers that remain in decorators.py
and are imported back from there.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel

from neograph._construct_validation import ConstructError
from neograph.construct import Construct
from neograph._sidecar import (
    _get_node_source,
    _get_param_res,
    _get_sidecar,
    _set_param_res,
    infer_oracle_gen_type,
)
from neograph.di import DIBinding, DIKind
from neograph.naming import field_name_for
from neograph.node import Node


def construct_from_module(
    mod: Any,
    name: str | None = None,
    *,
    llm_config: dict[str, Any] | None = None,
    input: type[BaseModel] | None = None,
    output: type[BaseModel] | None = None,
) -> Construct:
    """Walk a module's Node instances, sort topologically, return a Construct.

    Walks `vars(mod)`, collecting both @node-decorated functions (Node instances
    with sidecars) and plain `Node(...)` instances. Builds adjacency from each
    decorated node's parameter-name tuple and each plain node's dict-form inputs
    keys. Unknown parameter names raise `ConstructError`; cycles raise
    `ConstructError`.

    The returned Construct is a regular Construct — compile/run operate on it
    unchanged. The existing `_validate_node_chain` walker runs via
    `Construct.__init__`, so type-compatibility is enforced as usual.

    Args:
        mod: The module to walk.
        name: Construct name. Default: module's short name.
        llm_config: Default LLM config inherited by every node. Per-node
            llm_config merges over this (node wins on conflicts).
        input: Input type for sub-construct boundary.
        output: Output type for sub-construct boundary.
    """
    nodes: list[Node] = []
    plain_nodes: list[Node] = []
    source_label = f"module '{mod.__name__}'"
    for attr in vars(mod).values():
        if isinstance(attr, Node):
            if _get_sidecar(attr) is not None:
                nodes.append(attr)
            else:
                plain_nodes.append(attr)

    construct_name = name or mod.__name__.split(".")[-1]
    return _build_construct_from_decorated(
        nodes, construct_name, source_label, llm_config,
        construct_input=input, construct_output=output,
        plain_nodes=plain_nodes,
    )


def construct_from_functions(
    name: str,
    functions: list[Any],
    *,
    llm_config: dict[str, Any] | None = None,
    input: type[BaseModel] | None = None,
    output: type[BaseModel] | None = None,
) -> Construct:
    """Build a Construct from an explicit list of @node-decorated functions.

    Use this when multiple pipelines share a file — `construct_from_module()`
    walks the whole module and cannot partition @nodes into separate
    Constructs. Pass the subset explicitly:

        pipelineA = construct_from_functions("A", [fn1, fn2, fn3])
        pipelineB = construct_from_functions("B", [fn4, fn5])

    When building a sub-construct, pass ``input=`` / ``output=`` to define the
    state boundary:

        sub = construct_from_functions("verify", [explore, score],
                                       input=VerifyClaim, output=ClaimResult)

    Same topological sort, validation, and error messages as
    `construct_from_module()`. The returned Construct is a regular Construct.

    Args:
        name: Construct name.
        functions: List of @node-decorated functions (in any order —
            topological sort handles ordering). Each element must be a Node
            instance returned by @node; plain callables raise ConstructError.
        llm_config: Default LLM config inherited by every node. Per-node
            llm_config merges over this (node wins on conflicts).
        input: Input type for sub-construct boundary. When set, the Construct
            receives an isolated state with this type at ``neo_subgraph_input``.
        output: Output type for sub-construct boundary.
    """
    source_label = f"construct '{name}'"
    nodes: list[Node] = []
    sub_constructs: list[Construct] = []
    for item in functions:
        if isinstance(item, Construct):
            if item.output is None:
                raise ConstructError.build(
                    f"Construct '{item.name}' has no output type",
                    construct=name,
                    hint="sub-constructs must declare output= so downstream @nodes can resolve dependencies",
                )
            sub_constructs.append(item)
        elif isinstance(item, Node) and _get_sidecar(item) is not None:
            nodes.append(item)
        else:
            raise ConstructError.build(
                "argument is not decorated with @node or a Construct",
                construct=name,
                found=type(item).__name__,
                hint="every list element must be a function decorated with @node or a Construct with declared output",
            )

    return _build_construct_from_decorated(
        nodes, name, source_label, llm_config,
        construct_input=input, construct_output=output,
        sub_constructs=sub_constructs,
    )


def _resolve_dict_output_param(
    pname: str,
    decorated: dict[str, Node],
) -> str | None:
    """If pname is {upstream}_{output_key} for a dict-output upstream, return the upstream name.

    Tries longest-prefix matching against decorated node names with dict outputs.
    Returns None if no match.
    """
    for upstream_name, upstream_node in decorated.items():
        prefix = f"{upstream_name}_"
        if pname.startswith(prefix) and isinstance(upstream_node.outputs, dict):
            output_key = pname[len(prefix):]
            if output_key in upstream_node.outputs:
                return upstream_name
    return None


def _resolve_loop_self_param(
    node: Node,
    pname: str,
    decorated: dict[str, Node],
    sub_by_field: dict[str, Any],
) -> str | None:
    """For a Loop node, resolve a param by type when name doesn't match upstream.

    Returns the upstream field_name if exactly one upstream produces a compatible
    type. Returns None if no match. Raises ConstructError if ambiguous (multiple
    matches).
    """
    from neograph._construct_validation import _types_compatible, effective_producer_type

    if not isinstance(node.inputs, dict):
        return None
    param_type = node.inputs.get(pname)
    if param_type is None:
        return None

    field_name = field_name_for(node.name)
    candidates: list[str] = []
    all_upstreams = {**decorated, **sub_by_field}
    for up_field, upstream in all_upstreams.items():
        if up_field == field_name:
            continue  # skip self
        up_type = effective_producer_type(upstream)
        if up_type is not None and _types_compatible(up_type, param_type):
            candidates.append(up_field)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ConstructError.build(
            f"loop self-reference param '{pname}' matches multiple upstreams by type",
            node=node.name,
            found=str(sorted(candidates)),
            hint="name the param after the specific upstream to disambiguate",
            location=_get_node_source(node),
        )
    return None


def _build_decorated_dict(
    nodes: list[Node],
    plain_nodes: list[Node],
    sub_by_field: dict[str, Construct],
    construct_name: str,
) -> tuple[dict[str, Node], set[str]]:
    """Build field_name -> Node dict with collision detection.

    Returns (decorated, plain_fields) where plain_fields is the set of
    field names belonging to plain Node instances (no sidecar).
    """
    decorated: dict[str, Node] = {}
    plain_fields: set[str] = set()
    for n in nodes:
        field_name = field_name_for(n.name)
        if field_name in decorated or field_name in sub_by_field:
            existing_name = decorated[field_name].name if field_name in decorated else sub_by_field[field_name].name
            raise ConstructError.build(
                f"name collision: two items resolve to field name '{field_name}'",
                construct=construct_name,
                found=f"'{existing_name}' and '{n.name}'",
                hint="pass explicit name= to @node on one of them",
            )
        decorated[field_name] = n

    for n in plain_nodes:
        field_name = field_name_for(n.name)
        if field_name in decorated or field_name in sub_by_field:
            existing_name = decorated[field_name].name if field_name in decorated else sub_by_field[field_name].name
            raise ConstructError.build(
                f"name collision: two items resolve to field name '{field_name}'",
                construct=construct_name,
                found=f"'{existing_name}' and '{n.name}'",
                hint="rename one of the nodes to avoid the collision",
            )
        decorated[field_name] = n
        plain_fields.add(field_name)
    return decorated, plain_fields


def _identify_port_params(
    decorated: dict[str, Node],
    construct_input: type[BaseModel] | None,
    construct_name: str,
) -> dict[str, set[str]]:
    """Identify port params: params whose type matches construct_input.

    Returns field_name -> {param_names} mapping. Port params read from
    neo_subgraph_input, not from a peer @node.
    """
    port_params: dict[str, set[str]] = {}
    if construct_input is None:
        return port_params
    for field_name, n in decorated.items():
        if not isinstance(n.inputs, dict):
            continue
        ports: set[str] = set()
        for pname, ptype in n.inputs.items():
            if pname in decorated:
                continue  # peer @node takes priority
            try:
                if isinstance(ptype, type) and issubclass(ptype, construct_input):
                    ports.add(pname)
            except TypeError:  # pragma: no cover — isinstance(ptype, type) guards this
                pass  # generic types fail issubclass — skip
        if len(ports) > 1:
            raise ConstructError.build(
                f"{len(ports)} parameters match construct input type {construct_input.__name__}",
                node=n.name,
                construct=construct_name,
                found=str(sorted(ports)),
                hint="only one port param is allowed per node -- rename one or use FromInput annotation",
            )
        if ports:
            port_params[field_name] = ports
    return port_params


def _detect_fan_out_params(
    decorated: dict[str, Node],
    plain_fields: set[str],
    port_params: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Detect fan-out params for Each-modified nodes.

    Fan-out params don't match any @node name and are Each item receivers
    — they must be skipped in adjacency wiring.
    """
    fan_out_params: dict[str, set[str]] = {}
    for field_name, n in decorated.items():
        if field_name in plain_fields:
            continue
        if n.modifier_set.each is not None:
            sidecar = _get_sidecar(n)
            if sidecar is None:
                raise ConstructError.build(
                    "lost sidecar metadata (function + param names)",
                    node=n.name,
                    hint="a modifier was likely applied via | without re-registering the sidecar on the new Node copy",
                )
            _, pnames = sidecar
            di_params = set(_get_param_res(n))
            _ports = port_params.get(field_name, set())
            fan_out_params[field_name] = {p for p in pnames if p not in decorated and p not in di_params and p not in _ports}
    return fan_out_params


def _classify_constants(
    decorated: dict[str, Node],
    plain_fields: set[str],
    sub_by_field: dict[str, Construct],
    fan_out_params: dict[str, set[str]],
    port_params: dict[str, set[str]],
) -> None:
    """Classify default-value constants as DI CONSTANT bindings.

    Params with defaults that don't match any decorated @node and aren't
    already classified as from_input/from_config get tagged as constants.
    Mutates param_res in-place via _set_param_res.
    """
    for field_name, n in decorated.items():
        if field_name in plain_fields:
            continue
        sidecar = _get_sidecar(n)
        if sidecar is None:
            raise ConstructError.build(
                "lost sidecar metadata (function + param names)",
                node=n.name,
                hint="a modifier was likely applied via | without re-registering the sidecar on the new Node copy",
            )
        fn, pnames = sidecar
        param_res = _get_param_res(n)
        sig = inspect.signature(fn)
        updated = False
        for pname in pnames:
            if pname in param_res:
                continue
            if pname in fan_out_params.get(field_name, set()):
                continue
            if pname in port_params.get(field_name, set()):
                continue
            if pname not in decorated and pname not in sub_by_field:
                p = sig.parameters.get(pname)
                if p is not None and p.default is not inspect.Parameter.empty:
                    param_res[pname] = DIBinding(
                        name=pname, kind=DIKind.CONSTANT,
                        inner_type=type(p.default), required=False,
                        default_value=p.default,
                    )
                    updated = True
        if updated:
            _set_param_res(n, param_res)


def _check_di_collisions(
    decorated: dict[str, Node],
    plain_fields: set[str],
    sub_by_field: dict[str, Construct],
) -> None:
    """Raise if any DI param has the same name as a known producer node.

    This prevents silent dependency drops where a FromInput/FromConfig
    annotation shadows an upstream edge.
    """
    for _field_name, n in decorated.items():
        if _field_name in plain_fields:
            continue
        param_res = _get_param_res(n)
        for pname, binding in param_res.items():
            if binding.kind in (DIKind.FROM_INPUT, DIKind.FROM_INPUT_MODEL, DIKind.FROM_CONFIG, DIKind.FROM_CONFIG_MODEL):
                if pname in decorated or pname in sub_by_field:
                    di_label = "FromInput" if "input" in binding.kind.value else "FromConfig"
                    raise ConstructError.build(
                        f"parameter '{pname}' is annotated as {di_label} but '{pname}' is also a known upstream node/sub-construct",
                        node=n.name,
                        hint="this would silently drop the dependency edge -- rename either the parameter or the upstream node",
                    )


def _build_adjacency(
    decorated: dict[str, Node],
    plain_fields: set[str],
    sub_by_field: dict[str, Construct],
    fan_out_params: dict[str, set[str]],
    port_params: dict[str, set[str]],
    source_label: str,
) -> tuple[dict[str, list[str]], dict[str, dict[str, str]], dict[str, None]]:
    """Build adjacency graph and detect loop self-reference renames.

    Returns (adjacency, loop_param_renames, all_known).
    """
    loop_param_renames: dict[str, dict[str, str]] = {}
    all_known = {**dict.fromkeys(decorated), **dict.fromkeys(sub_by_field)}
    adjacency: dict[str, list[str]] = {k: [] for k in all_known}
    for field_name, n in decorated.items():
        if field_name in plain_fields:
            # Plain Node — derive adjacency from dict-form inputs keys.
            if isinstance(n.inputs, dict):
                for dep_name in n.inputs:
                    dep_field = field_name_for(dep_name)
                    if dep_field in decorated or dep_field in sub_by_field:
                        adjacency[field_name].append(dep_field)
            continue
        sidecar = _get_sidecar(n)
        if sidecar is None:  # pragma: no cover — earlier phases catch this first
            raise ConstructError.build(
                "lost sidecar metadata (function + param names)",
                node=n.name,
                hint="a modifier was likely applied via | without re-registering the sidecar on the new Node copy",
            )
        _, param_names = sidecar
        param_res = _get_param_res(n)
        skip = fan_out_params.get(field_name, set())
        _ports = port_params.get(field_name, set())
        seen_deps: set[str] = set()
        for pname in param_names:
            if pname in skip:
                continue
            if pname in param_res:
                continue
            if pname in _ports:
                continue  # port param — reads from neo_subgraph_input, not a peer
            if pname in sub_by_field:
                if pname not in seen_deps:
                    adjacency[field_name].append(pname)
                    seen_deps.add(pname)
                continue
            if pname not in decorated:
                resolved_upstream = _resolve_dict_output_param(pname, decorated)
                if resolved_upstream is not None:
                    if resolved_upstream not in seen_deps:
                        adjacency[field_name].append(resolved_upstream)
                        seen_deps.add(resolved_upstream)
                    continue
                if n.modifier_set.loop is not None:
                    loop_upstream = _resolve_loop_self_param(n, pname, decorated, sub_by_field)
                    if loop_upstream is not None:
                        if loop_upstream not in seen_deps:
                            adjacency[field_name].append(loop_upstream)
                            seen_deps.add(loop_upstream)
                        loop_param_renames.setdefault(field_name, {})[pname] = loop_upstream
                        continue
                all_names = sorted(set(decorated.keys()) | set(sub_by_field.keys()))
                raise ConstructError.build(
                    f"parameter '{pname}' does not match any @node or sub-construct in {source_label}",
                    node=n.name,
                    hint="all parameters must name an upstream @node/Construct, use FromInput/FromConfig annotation, or have a default value\n  available items: " + str(all_names),
                    location=_get_node_source(n),
                )
            if pname == field_name:
                raise ConstructError.build(
                    f"parameter '{pname}' refers to itself",
                    node=n.name,
                    hint="self-dependency is not allowed",
                    location=_get_node_source(n),
                )
            if pname not in seen_deps:
                adjacency[field_name].append(pname)
                seen_deps.add(pname)
    return adjacency, loop_param_renames, all_known


def _topo_sort(
    adjacency: dict[str, list[str]],
    all_known: dict[str, None],
    decorated: dict[str, Node],
    sub_by_field: dict[str, Construct],
    construct_name: str,
) -> list[Any]:
    """Topological sort via DFS with cycle detection.

    Returns ordered list of Node and Construct items in dependency order.
    """
    ordered: list[Any] = []
    marks: dict[str, str] = {}

    def visit(field: str) -> None:
        state = marks.get(field)
        if state == "black":
            return
        if state == "gray":
            item = decorated.get(field) or sub_by_field.get(field)
            raise ConstructError.build(
                f"cycle detected involving '{field}'",
                construct=construct_name,
                hint="cyclical dependencies are not allowed",
                location=_get_node_source(item) if isinstance(item, Node) else None,
            )
        marks[field] = "gray"
        for dep in adjacency[field]:
            visit(dep)
        marks[field] = "black"
        if field in decorated:
            ordered.append(decorated[field])
        elif field in sub_by_field:
            ordered.append(sub_by_field[field])

    for field in all_known:
        visit(field)
    return ordered


def _cleanup_inputs_and_register(
    decorated: dict[str, Node],
    plain_fields: set[str],
    sub_by_field: dict[str, Construct],
    fan_out_params: dict[str, set[str]],
    port_params: dict[str, set[str]],
    loop_param_renames: dict[str, dict[str, str]],
    ordered: list[Any],
) -> list[Any]:
    """Clean up inputs, register scripted shims, infer oracle types.

    Strips DI params from inputs, rewrites port/loop params, sets
    fan_out_param, registers scripted shims, and infers deferred
    oracle_gen_type. All via model_copy -- never mutates originals.

    Returns the final ordered list with model_copy replacements.
    """
    # Single-pass: accumulate all updates per node, then one model_copy each.
    # Previously 3 separate loops with up to 3 model_copy calls per node.
    for field in list(decorated):
        n = decorated[field]
        if field in plain_fields:
            continue
        updates: dict[str, Any] = {}

        # Phase 1: Strip DI params, rewrite port/loop keys, set fan_out_param.
        if isinstance(n.inputs, dict):
            skip = fan_out_params.get(field, set())
            _ports = port_params.get(field, set())
            renames = loop_param_renames.get(field, {})
            filtered: dict[str, Any] = {}
            for k, v in n.inputs.items():
                if k in _ports:
                    filtered["neo_subgraph_input"] = v
                elif k in renames:
                    filtered[renames[k]] = v
                elif (
                    (k in decorated and k != field)
                    or k in sub_by_field
                    or k in skip
                    or _resolve_dict_output_param(k, decorated) is not None
                ):
                    filtered[k] = v
            if filtered != n.inputs:
                updates["inputs"] = filtered
            if skip:
                updates["fan_out_param"] = next(iter(skip))

        # Phase 2: Register scripted shim.
        if n.mode == "scripted" and n.raw_fn is None:
            synthetic_name = _register_node_scripted(
                n, fan_out_params.get(field, set()),
                port_param_map=dict.fromkeys(port_params.get(field, set()), "neo_subgraph_input"),
                loop_renames=loop_param_renames.get(field),
            )
            if synthetic_name is not None:
                updates["scripted_fn"] = synthetic_name

        # Phase 3: Deferred oracle_gen_type inference.
        if isinstance(n, Node) and n.modifier_set.oracle is not None and n.oracle_gen_type is None:
            oracle_mod = n.modifier_set.oracle
            if oracle_mod is not None and oracle_mod.merge_fn is not None:
                gen_type = infer_oracle_gen_type(oracle_mod.merge_fn)
                if gen_type is not None and gen_type is not n.outputs:
                    updates["oracle_gen_type"] = gen_type

        # Single model_copy with all accumulated updates.
        if updates:
            decorated[field] = n.model_copy(update=updates)

    # Rebuild ordered list to pick up model_copy replacements.
    return [
        decorated.get(field_name_for(item.name), item) if isinstance(item, Node) else item
        for item in ordered
    ]


def _build_construct_from_decorated(
    nodes: list[Node],
    construct_name: str,
    source_label: str,
    llm_config: dict[str, Any] | None,
    construct_input: type[BaseModel] | None = None,
    construct_output: type[BaseModel] | None = None,
    sub_constructs: list[Construct] | None = None,
    plain_nodes: list[Node] | None = None,
) -> Construct:
    """Core pipeline builder shared by construct_from_module and
    construct_from_functions. Delegates to named step helpers for each
    phase of the build pipeline.
    """
    _sub_constructs = sub_constructs or []
    _plain_nodes = plain_nodes or []
    if not nodes and not _sub_constructs and not _plain_nodes:
        raise ConstructError.build(
            "Construct has no nodes",
            construct=construct_name,
            hint="add at least one @node function or sub-Construct",
        )

    sub_by_field: dict[str, Construct] = {}
    for sc in _sub_constructs:
        sub_by_field[field_name_for(sc.name)] = sc

    decorated, plain_fields = _build_decorated_dict(
        nodes, _plain_nodes, sub_by_field, construct_name,
    )
    port_params = _identify_port_params(decorated, construct_input, construct_name)
    fan_out_params = _detect_fan_out_params(decorated, plain_fields, port_params)
    _classify_constants(decorated, plain_fields, sub_by_field, fan_out_params, port_params)
    _check_di_collisions(decorated, plain_fields, sub_by_field)
    adjacency, loop_param_renames, all_known = _build_adjacency(
        decorated, plain_fields, sub_by_field, fan_out_params, port_params, source_label,
    )
    ordered = _topo_sort(adjacency, all_known, decorated, sub_by_field, construct_name)
    ordered = _cleanup_inputs_and_register(
        decorated, plain_fields, sub_by_field,
        fan_out_params, port_params, loop_param_renames, ordered,
    )

    return Construct(
        name=construct_name,
        nodes=ordered,
        llm_config=llm_config or {},
        input=construct_input,
        output=construct_output,
    )


def _register_node_scripted(
    n: Node,
    fan_out: set[str] | None = None,
    port_param_map: dict[str, str] | None = None,
    loop_renames: dict[str, str] | None = None,
) -> str | None:
    """Register a scripted shim and return the synthetic name.

    Returns the synthetic name string so the caller can set it on the Node
    via model_copy (immutable IR). Returns None if the node has no sidecar.

    When ``port_param_map`` is set, port params (whose keys were rewritten
    in ``n.inputs`` to ``neo_subgraph_input``) are looked up under the
    rewritten key in ``input_data``.

    When ``loop_renames`` is set, loop self-reference params (whose keys
    were rewritten in ``n.inputs`` from original name to resolved upstream)
    are looked up under the rewritten key in ``input_data``.
    """
    from neograph.factory import register_scripted

    sidecar = _get_sidecar(n)
    if sidecar is None:
        return None
    fn, param_names = sidecar
    param_res = _get_param_res(n)
    _fan_out = fan_out or set()
    _port_map = port_param_map or {}
    _loop_map = loop_renames or {}

    # Synthesize a unique name for the registered shim. Use id(n) so
    # parallel pipelines with the same node names don't collide.
    synthetic_name = f"_node_{n.name}_{id(n):x}"

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

    scripted_shim.__name__ = field_name_for(n.name)
    register_scripted(synthetic_name, scripted_shim)
    return synthetic_name
