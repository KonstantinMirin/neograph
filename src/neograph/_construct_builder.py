"""Construct-building orchestration extracted from decorators.py.

Houses the public entry points (construct_from_module, construct_from_functions),
the pipeline orchestrator (_build_construct_from_decorated), and the @node-specific
input-cleanup pass (_cleanup_inputs_and_register).

The per-phase helpers live in cohesive sibling modules. See neograph-3zai:
  - graph construction      -> neograph._construct_graph
  - parameter classification -> neograph._param_classify
  - scripted-shim wiring     -> neograph._scripted_registry
  - member selection        -> neograph._member_select

These functions depend on sidecar / DI helpers that remain in decorators.py
and are imported back from there (one-way; _construct_builder never imports
decorators.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pydantic import BaseModel

from neograph._construct_graph import (
    _build_adjacency,
    _build_decorated_dict,
    _resolve_dict_output_param,
    _topo_sort,
)
from neograph._construct_validation import ConstructError
from neograph._llm_config import LlmConfig
from neograph._member_select import _bucket_members
from neograph._normalize import normalize_inputs
from neograph._param_classify import (
    _check_di_collisions,
    _classify_constants,
    _detect_channel_skip_params,
    _identify_port_params,
)
from neograph._scripted_registry import _register_node_scripted
from neograph._state_keys import StateKeys
from neograph.construct import Construct
from neograph.naming import field_name_for
from neograph.node import Node


def construct_from_module(
    mod: Any,
    name: str | None = None,
    *,
    llm_config: dict[str, Any] | LlmConfig | None = None,
    input: type[BaseModel] | None = None,
    output: type[BaseModel] | None = None,
) -> Construct:
    """Walk a module's pipeline members, sort topologically, return a Construct.

    Walks `vars(mod)`, collecting every pipeline member: @node-decorated
    functions (Node instances with sidecars), plain `Node(...)` instances, and
    sub-`Construct`s (wired via their `output=` boundary). A module-level
    Construct WITHOUT `output=` is treated as a stored top-level pipeline
    artifact and skipped with a `ConstructArtifactSkipped` warning — never
    silently. Non-member attributes (helpers, constants, imports) are skipped.
    Membership is decided by the module-level binding — an imported member is
    collected the same as one defined in the module. Builds adjacency from
    each decorated node's
    parameter-name tuple and each plain node's dict-form inputs keys. Unknown
    parameter names raise `ConstructError`; cycles raise `ConstructError`.

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
    construct_name = name or mod.__name__.split(".")[-1]
    return _build_construct_from_decorated(
        list(vars(mod).values()),
        construct_name,
        f"module '{mod.__name__}'",
        llm_config,
        construct_input=input,
        construct_output=output,
        source="module",
    )


def construct_from_functions(
    name: str,
    functions: list[Any],
    *,
    llm_config: dict[str, Any] | LlmConfig | None = None,
    input: type[BaseModel] | None = None,
    output: type[BaseModel] | None = None,
) -> Construct:
    """Build a Construct from an explicit list of pipeline members.

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
        functions: List of pipeline members (in any order — topological sort
            handles ordering). Each element must be an @node-decorated
            function, a plain ``Node`` instance, or a Construct with declared
            output; anything else raises ConstructError.
        llm_config: Default LLM config inherited by every node. Per-node
            llm_config merges over this (node wins on conflicts).
        input: Input type for sub-construct boundary. When set, the Construct
            receives an isolated state with this type at ``neo_subgraph_input``.
        output: Output type for sub-construct boundary.
    """
    return _build_construct_from_decorated(
        functions,
        name,
        f"construct '{name}'",
        llm_config,
        construct_input=input,
        construct_output=output,
    )


def _cleanup_inputs_and_register(
    decorated: dict[str, Node],
    plain_fields: set[str],
    sub_by_field: dict[str, Construct],
    fan_out_params: dict[str, set[str]],
    port_params: dict[str, set[str]],
    loop_param_renames: dict[str, dict[str, str]],
    ordered: list[Any],
) -> list[Any]:
    """Clean up inputs and register scripted shims (@node-specific work).

    Strips DI params from inputs, rewrites port/loop params, sets the
    signature-derived fan_out_param, and registers scripted shims. All via
    model_copy -- never mutates originals.

    IR-level inferences that must be identical across all API surfaces
    (oracle_gen_type, and fan_out_param for non-@node surfaces) are owned by
    neograph._ir_normalize, run from Construct.__init__ -- NOT here.

    Returns the final ordered list with model_copy replacements.
    """
    # Single-pass: accumulate all updates per node, then one model_copy each.
    # Previously 3 separate loops with up to 3 model_copy calls per node.
    for field in list(decorated):
        n = decorated[field]
        if field in plain_fields:
            continue
        updates: dict[str, Any] = {}

        # Phase 1: Strip DI params, rewrite port/loop keys.
        # fan_out_param is NOT written here — it is owned exclusively by
        # neograph._ir_normalize (run from Construct.__init__), which
        # re-derives it from the same fan_out_candidates rule the validator
        # uses. `skip` is still computed: it keeps the fan-out receiver key in
        # the filtered inputs and is passed to _register_node_scripted below.
        # The normalizer is the sole writer of fan_out_param. See neograph-k7bg.
        ni = normalize_inputs(n.inputs)
        if ni.is_dict_form:
            skip = fan_out_params.get(field, set())
            _ports = port_params.get(field, set())
            renames = loop_param_renames.get(field, {})
            filtered: dict[str, Any] = {}
            for k, v in ni.by_name.items():
                if k in _ports:
                    filtered[StateKeys.SUBGRAPH_INPUT] = v
                elif k in renames:
                    filtered[renames[k]] = v
                elif (
                    (k in decorated and k != field)
                    or k in sub_by_field
                    or k in skip
                    or _resolve_dict_output_param(k, decorated) is not None
                ):
                    filtered[k] = v
            if filtered != ni.by_name:
                updates["inputs"] = filtered

        # Phase 2: Register scripted shim.
        if n.mode == "scripted" and n.raw_fn is None:
            synthetic_name = _register_node_scripted(
                n,
                fan_out_params.get(field, set()),
                port_param_map=dict.fromkeys(port_params.get(field, set()), StateKeys.SUBGRAPH_INPUT),
                loop_renames=loop_param_renames.get(field),
            )
            if synthetic_name is not None:
                updates["scripted_fn"] = synthetic_name

        # Neither oracle_gen_type NOR fan_out_param is written here: both are
        # IR-level inferences owned exclusively by neograph._ir_normalize (run
        # from Construct.__init__). Writing them here would re-create the
        # two-site drift class neograph-20xq/k7bg addressed. This assembly path
        # does only @node-specific work (input cleanup + scripted shims).

        # Single model_copy with all accumulated updates.
        if updates:
            decorated[field] = n.model_copy(update=updates)

    # Rebuild ordered list to pick up model_copy replacements. Sub-construct
    # names are keyed in sub_by_field, never in decorated, so .get() passes
    # them through unchanged — no isinstance dispatch needed.
    return [decorated.get(field_name_for(item.name), item) for item in ordered]


def _build_construct_from_decorated(
    members: list[Any],
    construct_name: str,
    source_label: str,
    llm_config: dict[str, Any] | LlmConfig | None,
    construct_input: type[BaseModel] | None = None,
    construct_output: type[BaseModel] | None = None,
    source: Literal["module", "list"] = "list",
) -> Construct:
    """Core pipeline builder shared by construct_from_module and
    construct_from_functions. Takes ONE heterogeneous member list and buckets
    it exactly once via `_bucket_members` (per-call-site bucketing is where
    the neograph-xv9ay drift lived); `source` declares the input kind and ALL
    skip/warn/raise policy lives inside `_bucket_members`. Delegates to named
    step helpers for each phase of the build pipeline.
    """
    nodes, _plain_nodes, _sub_constructs = _bucket_members(members, construct_name, source)
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
        nodes,
        _plain_nodes,
        sub_by_field,
        construct_name,
    )
    port_params = _identify_port_params(decorated, construct_input, construct_name)
    fan_out_params = _detect_channel_skip_params(decorated, plain_fields, port_params)
    _classify_constants(decorated, plain_fields, sub_by_field, fan_out_params, port_params)
    _check_di_collisions(decorated, plain_fields, sub_by_field)
    adjacency, loop_param_renames, all_known = _build_adjacency(
        decorated,
        plain_fields,
        sub_by_field,
        fan_out_params,
        port_params,
        source_label,
    )
    ordered = _topo_sort(adjacency, all_known, decorated, sub_by_field, construct_name)
    ordered = _cleanup_inputs_and_register(
        decorated,
        plain_fields,
        sub_by_field,
        fan_out_params,
        port_params,
        loop_param_renames,
        ordered,
    )

    return Construct(
        name=construct_name,
        nodes=ordered,
        llm_config=(llm_config if isinstance(llm_config, LlmConfig) else LlmConfig(**(llm_config or {}))),
        input=construct_input,
        output=construct_output,
    )
