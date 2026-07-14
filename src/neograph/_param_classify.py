"""Parameter classification for @node construct assembly.

Extracted from _construct_builder.py per neograph-3zai. Houses the helpers that
classify a node's parameters over the `decorated` field-name dict:

- _identify_port_params   -- params whose type matches construct_input (sub-construct port)
- _detect_fan_out_params  -- Each item-receiver params (skipped in adjacency wiring)
- _classify_constants     -- default-value params tagged as DI CONSTANT bindings
- _check_di_collisions    -- FromInput/FromConfig params shadowing an upstream node

These operate on the assembly-time `decorated` dict; the IR-level inferences
that must be identical across all API surfaces (fan_out_param, oracle_gen_type)
are owned by neograph._ir_normalize, not here.

Imports only leaf modules -- never decorators.py.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from neograph._construct_graph import _resolve_dict_output_param
from neograph._construct_validation import ConstructError
from neograph._normalize import normalize_inputs
from neograph._sidecar import _get_param_res, _get_sidecar, _set_param_res
from neograph.di import DIBinding, DIKind

if TYPE_CHECKING:
    from pydantic import BaseModel

    from neograph.construct import Construct
    from neograph.node import Node


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
        ni = normalize_inputs(n.inputs)
        if not ni.is_dict_form:
            continue
        ports: set[str] = set()
        for pname, ptype in ni.by_name.items():
            if pname in decorated:
                continue  # peer @node takes priority
            if _resolve_dict_output_param(pname, decorated) is not None:
                # A {upstream}_{output_key} reference to a dict-output producer takes the same
                # priority as a peer @node — otherwise a dict output whose type subclasses the
                # construct input is misclassified as a port and fan-in validation rejects it.
                continue
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
            fan_out_params[field_name] = {
                p for p in pnames if p not in decorated and p not in di_params and p not in _ports
            }
    return fan_out_params


def _detect_handoff_params(
    decorated: dict[str, Node],
    plain_fields: set[str],
    port_params: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Detect the reserved ``"handoff"`` param for Portal mesh members.

    A Portal member reads its handoff payload from the entry-keyed mesh
    channel via the reserved ``"handoff"`` inputs key (design §3.3), NOT from a
    peer @node. So — exactly like the Each fan-out receiver
    (:func:`_detect_fan_out_params`) — the ``handoff`` signature param names no
    upstream node and must be SKIPPED in adjacency wiring (and kept in the
    node's inputs so the normalizer detects the reserved key and
    ``factory._extract_input`` reads the channel). This is @node topology-wiring
    classification, not an IR-field write: ``handoff_param`` / ``handoff_channel``
    remain owned solely by ``_ir_normalize`` for all three surfaces (G3).
    """
    handoff_params: dict[str, set[str]] = {}
    for field_name, n in decorated.items():
        if field_name in plain_fields:
            continue
        if n.modifier_set.portal is None:
            continue
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
        if "handoff" in pnames and "handoff" not in decorated and "handoff" not in di_params and "handoff" not in _ports:
            handoff_params[field_name] = {"handoff"}
    return handoff_params


def _detect_channel_skip_params(
    decorated: dict[str, Node],
    plain_fields: set[str],
    port_params: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Union of framework-channel params to skip in @node adjacency wiring:
    Each fan-out receivers AND Portal reserved-``"handoff"`` receivers.

    Both name no upstream node, are skipped in adjacency, kept in the node's
    inputs, and passed to the scripted shim identically. This is @node topology
    classification only — the IR fields ``fan_out_param`` / ``handoff_param`` /
    ``handoff_channel`` stay owned solely by ``_ir_normalize`` for all three
    surfaces (guard G3).
    """
    skip = _detect_fan_out_params(decorated, plain_fields, port_params)
    for field_name, hp in _detect_handoff_params(decorated, plain_fields, port_params).items():
        skip.setdefault(field_name, set()).update(hp)
    return skip


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
                        name=pname,
                        kind=DIKind.CONSTANT,
                        inner_type=type(p.default),
                        required=False,
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
            if binding.kind in (
                DIKind.FROM_INPUT,
                DIKind.FROM_INPUT_MODEL,
                DIKind.FROM_CONFIG,
                DIKind.FROM_CONFIG_MODEL,
            ):
                if pname in decorated or pname in sub_by_field:
                    di_label = "FromInput" if "input" in binding.kind.value else "FromConfig"
                    raise ConstructError.build(
                        f"parameter '{pname}' is annotated as {di_label} but '{pname}' is also a known upstream node/sub-construct",
                        node=n.name,
                        hint="this would silently drop the dependency edge -- rename either the parameter or the upstream node",
                    )
