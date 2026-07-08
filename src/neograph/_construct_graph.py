"""DAG construction for @node construct assembly.

Extracted from _construct_builder.py per neograph-3zai. Houses the helpers that
turn the `decorated` field-name dict into a dependency graph and an ordered
node list:

- _build_decorated_dict       -- field_name -> Node dict with collision detection
- _resolve_dict_output_param  -- resolve {upstream}_{output_key} params
- _resolve_loop_self_param    -- resolve a Loop self-reference param by type
- _build_adjacency            -- adjacency graph + loop self-reference renames
- _topo_sort                  -- DFS topological sort with cycle detection

Imports only leaf modules -- never decorators.py. The function-local import of
_types_compatible / effective_producer_type inside _resolve_loop_self_param is
the sole user of that cycle-break (allowlisted in
test_guards_sidecar_imports.FUNCTION_LOCAL_IMPORT_ALLOWLIST).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neograph._construct_validation import ConstructError
from neograph._normalize import normalize_inputs, normalize_outputs
from neograph._sidecar import _get_node_source, _get_param_res, _get_sidecar
from neograph.naming import field_name_for, split_output_field
from neograph.node import Node

if TYPE_CHECKING:
    from neograph.construct import Construct


def _resolve_dict_output_param(
    pname: str,
    decorated: dict[str, Node],
) -> str | None:
    """If pname is {upstream}_{output_key} for a dict-output upstream, return the upstream name.

    Tries longest-prefix matching against decorated node names with dict outputs.
    Returns None if no match.
    """
    for upstream_name, upstream_node in decorated.items():
        output_key = split_output_field(pname, upstream_name)
        if output_key is None:
            continue
        up_no = normalize_outputs(upstream_node.outputs)
        if not up_no.is_dict_form:
            continue
        if output_key in up_no.all_keys:
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

    ni = normalize_inputs(node.inputs)
    if not ni.is_dict_form:
        return None
    param_type = ni.by_name.get(pname)
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
            n_inputs_norm = normalize_inputs(n.inputs)
            if n_inputs_norm.is_dict_form:
                for dep_name in n_inputs_norm.by_name:
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
                    hint="all parameters must name an upstream @node/Construct, use FromInput/FromConfig annotation, or have a default value\n  available items: "
                    + str(all_names),
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
