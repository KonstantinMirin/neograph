"""State bus compiler — auto-generates Pydantic state from Node I/O union.

Each Construct gets its own state model with exactly the fields its Nodes need.
No monolithic state that grows with every derivation type.
"""

from __future__ import annotations

import warnings
from typing import Annotated, Any, get_args, get_origin

import structlog
from pydantic import BaseModel, create_model

from neograph._ir_branch import _BranchNode
from neograph.construct import Construct
from neograph.errors import CompileError
from neograph.naming import field_name_for, output_field_name
from neograph.spec_types import lookup_type

log = structlog.get_logger()
from typing import assert_never

from neograph._normalize import _declared_output, normalize_outputs
from neograph._state_keys import StateKeys
from neograph.modifiers import EachFailure, ModifierCombo, classify_modifiers
from neograph.node import Node


def _last_write_wins(existing: Any, new: Any) -> Any:
    """Reducer: last write wins (default for sequential nodes)."""
    return new


def _append_loop_result(existing: Any, new: Any) -> list:
    """Reducer: append each loop iteration's result to a list."""
    if existing is None:
        existing = []
    return [*existing, new]


def _concat_reducer(existing: Any, new: Any) -> list:
    """Reducer: concatenate list-valued writes onto an accumulator.

    The single list-append reducer shared by every additive channel:
      - oracle fan-out results (``list[sub.output]``)
      - Each×Oracle tagged (key, result) tuples
      - agent-cycle ToolInteraction records (``tool_log``, per-turn concat)
      - agent-cycle ResourceRef records (``resource_manifest``, per-turn concat)

    A per-turn write is a ``list`` (extend); a single value is appended. These
    four channels were byte-identical functions (neograph-yrph item 4). LangGraph
    keys channels by FIELD NAME, not reducer identity, so one shared operator is
    safe — the same pattern ``_last_write_wins``/``_merge_dicts`` already use
    across many distinct channels. A structural guard bans re-planting a
    byte-identical concat twin.
    """
    if existing is None:
        existing = []
    if isinstance(new, list):
        return existing + new
    return [*existing, new]


def _merge_dicts(existing: Any, new: dict) -> dict:
    """Reducer: merge dicts additively (for fan-out results).

    On duplicate keys, keeps the existing (first) value. Logs a single
    summary instead of per-key warnings (neograph-o0tv: noisy on resume).
    """
    if existing is None:
        existing = {}
    if not isinstance(existing, dict):
        existing = {}
    if not isinstance(new, dict):
        return existing
    merged = {**existing}
    dupes = []
    for key, val in new.items():
        if key in merged:
            dupes.append(key)
            continue
        merged[key] = val
    if dupes:
        log.debug(
            "each_duplicate_keys", count=len(dupes), keys=dupes[:5], action="kept_existing", truncated=len(dupes) > 5
        )
    return merged


def compile_state_model(
    construct: Construct,
    *,
    context_types: dict[str, type] | None = None,
) -> type[BaseModel]:
    """Generate a Pydantic state model from the union of Node I/O fields.

    Each Node's output becomes a state field. Fan-out nodes get dict reducers.
    The resulting model is used as the LangGraph StateGraph schema.

    Args:
        context_types: When compiling a subconstruct, the parent passes concrete
            types for context fields (instead of Any). Keys are field_name_for'd
            context names, values are the parent's output types.
    """
    fields: dict[str, Any] = {}

    nodes_only = [n for n in construct.nodes if isinstance(n, Node)]
    sub_constructs = [n for n in construct.nodes if isinstance(n, Construct)]
    branch_nodes = [n for n in construct.nodes if isinstance(n, _BranchNode)]

    # Detect field-name collisions from hyphen/underscore normalization.
    # Two nodes "my-node" and "my_node" both map to state field "my_node",
    # which would silently share loop counters, reducers, etc.
    seen_fields: dict[str, str] = {}  # field_name → original node name
    for item in nodes_only + sub_constructs:
        field_name = field_name_for(item.name)
        if field_name in seen_fields:
            raise CompileError.build(
                "node name collision",
                expected="unique state field names",
                found=f"'{item.name}' and '{seen_fields[field_name]}' both map to state field '{field_name}'",
                hint="rename one of them so the normalized field names differ",
            )
        seen_fields[field_name] = item.name

    for node in nodes_only:
        _add_output_field(node, fields)
        _add_agent_channels(node, fields)

    # Branch arm nodes: add state fields for nodes inside branch arms.
    # Arms can contain both Nodes and Constructs (e.g., self.loop() in
    # ForwardConstruct produces a Construct in the branch arm).
    for branch in branch_nodes:
        meta = branch._neo_branch_meta
        for arm_item in meta.true_arm_nodes + meta.false_arm_nodes:
            if isinstance(arm_item, Construct):
                # Construct in branch arm — same handling as sub-constructs
                if arm_item.output is None:
                    continue
                field_name = field_name_for(arm_item.name)
                arm_combo, _ = classify_modifiers(arm_item)
                if arm_combo in (ModifierCombo.LOOP, ModifierCombo.LOOP_OPERATOR):
                    fields[field_name] = (
                        Annotated[list[arm_item.output], _append_loop_result],  # type: ignore[name-defined]
                        [],
                    )
                    fields[StateKeys.loop_count(field_name)] = (int, 0)
                else:
                    fields[field_name] = (arm_item.output | None, None)
            else:
                _add_output_field(arm_item, fields)

    # Sub-constructs: handle modifiers same as nodes
    for sub in sub_constructs:
        if sub.output is None:
            raise CompileError.build(
                "sub-construct has no output type",
                hint="declare output=SomeModel on the sub-construct",
                construct=sub.name,
            )
        field_name = field_name_for(sub.name)

        combo, mods = classify_modifiers(sub)
        match combo:
            case ModifierCombo.ORACLE | ModifierCombo.ORACLE_OPERATOR:
                # Oracle on Construct: collector + consumer field
                collector_field = StateKeys.oracle_collector(field_name)
                fields[collector_field] = (
                    Annotated[list[sub.output], _concat_reducer],  # type: ignore[name-defined]
                    [],
                )
                fields[field_name] = (sub.output | None, None)  # type: ignore[name-defined]
            case ModifierCombo.EACH | ModifierCombo.EACH_OPERATOR:
                # Each on Construct: dict field. Under on_error='collect' the
                # barrier may hold a typed EachFailure per thrown item, so the
                # value type widens to accept it (default 'raise' unchanged).
                each_mod = mods["each"]
                value_type: Any = (
                    sub.output | EachFailure  # type: ignore[name-defined]
                    if each_mod.on_error == "collect"
                    else sub.output
                )
                field_type = dict[str, value_type] | None  # type: ignore[name-defined,valid-type]
                fields[field_name] = (
                    Annotated[field_type, _merge_dicts],
                    None,
                )
            case ModifierCombo.LOOP | ModifierCombo.LOOP_OPERATOR:
                # Loop on Construct: append-list + iteration counter
                fields[field_name] = (
                    Annotated[list[sub.output], _append_loop_result],  # type: ignore[name-defined]
                    [],
                )
                fields[StateKeys.loop_count(field_name)] = (int, 0)
            case ModifierCombo.BARE | ModifierCombo.OPERATOR:
                fields[field_name] = (sub.output | None, None)
            case ModifierCombo.EACH_ORACLE | ModifierCombo.EACH_ORACLE_OPERATOR:
                # Each×Oracle on Constructs not supported
                # Compiler raises earlier; this is a defensive fallback.
                fields[field_name] = (sub.output | None, None)
            case ModifierCombo.KEYMAKER:
                # Keymaker on a Construct is rejected at assembly (D-MESH-LEVEL:
                # mesh members are sibling Nodes, not sub-constructs), so this
                # arm is defensively-unreachable — mirror the EACH_ORACLE
                # fallback rather than crash the state build.
                fields[field_name] = (sub.output | None, None)
            case _ as unreachable:
                assert_never(unreachable)

    # Oracle support: generator ID + optional model override passed via state
    all_items = nodes_only + sub_constructs
    has_any_oracle = False
    has_any_each = False
    for item in all_items:
        item_combo, _ = classify_modifiers(item)
        if item_combo in (
            ModifierCombo.ORACLE,
            ModifierCombo.ORACLE_OPERATOR,
            ModifierCombo.EACH_ORACLE,
            ModifierCombo.EACH_ORACLE_OPERATOR,
        ):
            has_any_oracle = True
        if item_combo in (
            ModifierCombo.EACH,
            ModifierCombo.EACH_OPERATOR,
            ModifierCombo.EACH_ORACLE,
            ModifierCombo.EACH_ORACLE_OPERATOR,
        ):
            has_any_each = True
    if has_any_oracle:
        fields[StateKeys.ORACLE_GEN_ID] = (str | None, None)
        fields[StateKeys.ORACLE_MODEL] = (str | None, None)
    if has_any_each:
        fields[StateKeys.EACH_ITEM] = (Any, None)

    # Loop support: iteration counter per looped node
    for n in nodes_only:
        n_combo, n_mods = classify_modifiers(n)
        if n_combo in (ModifierCombo.LOOP, ModifierCombo.LOOP_OPERATOR):
            field_name = field_name_for(n.name)
            fields[StateKeys.loop_count(field_name)] = (int, 0)
            loop = n_mods["loop"]

    # Keymaker support: per-mesh hop counter + shared payload channel, keyed off
    # the mesh ENTRY (the first Keymaker member in node order — design §3.1
    # "the first member is the entry"; assembly validation guarantees one
    # contiguous mesh at this level). Both are neo_-prefixed → excluded from the
    # schema fingerprint (member OUTPUT fields carry the fingerprint). The
    # channel/counter are runtime-inert until T2 lowering reads them.
    def _is_dispatch(n: Node) -> bool:
        km = n.modifier_set.keymaker
        return km is not None and km.is_dispatch

    # PEER-mode members only: a dispatch node (route="decide") is NOT a mesh member
    # — it has no hop counter / mesh channel; it gets a {field}_dispatch field below.
    keymaker_members = [
        n
        for n in nodes_only
        if classify_modifiers(n)[0] == ModifierCombo.KEYMAKER and not _is_dispatch(n)
    ]
    if keymaker_members:
        entry = keymaker_members[0]
        entry_field = field_name_for(entry.name)
        # Single-type by assembly validation (dict-form members rejected); typed
        # Any so the `| None` field spec matches the sibling arms' pattern.
        payload: Any = _declared_output(entry)
        fields[StateKeys.handoff_hops(entry_field)] = (int, 0)
        fields[StateKeys.handoff_payload(entry_field)] = (payload | None, None)

    # Keymaker DISPATCH support (design §4.2): a route="decide" node writes the
    # dispatched flow's typed result to a regular (fingerprinted, NON-neo_-prefixed)
    # field `{field_name}_dispatch` — an output-contract change correctly
    # invalidates checkpoints. The node's OWN output (the emitted spec/input model)
    # is written to its plain output field by the KEYMAKER arm in
    # `_add_single_output_field`; this is the SEPARATE dispatch-result field.
    for n in nodes_only:
        if _is_dispatch(n):
            km = n.modifier_set.keymaker
            assert km is not None  # _is_dispatch guarantees it
            out_spec = km.output
            assert out_spec is not None  # dispatch-mode invariant (T1 validation)
            dispatch_field = output_field_name(field_name_for(n.name), "dispatch")
            resolved = lookup_type(out_spec) if isinstance(out_spec, str) else out_spec
            fields[dispatch_field] = (resolved | None, None)

    # Subgraph input port — when this Construct declares an input type
    if construct.input is not None:
        fields[StateKeys.SUBGRAPH_INPUT] = (construct.input | None, None)

    # Context fields — forwarded from parent state for nodes that declare context=
    # When context_types is provided (subconstruct compilation), use the concrete
    # type from the parent instead of Any. This ensures the msgpack allowlist
    # includes the context field types for checkpoint serialization.
    _ctx_types = context_types or {}
    for n in nodes_only:
        if n.context:
            for ctx_name in n.context:
                ctx_field = field_name_for(ctx_name)
                if ctx_field not in fields:
                    ctx_type = _ctx_types.get(ctx_field, Any)
                    fields[ctx_field] = (ctx_type if ctx_type is not Any else Any, None)
    # Also check branch arm nodes (skip Constructs — they handle context internally)
    for branch in branch_nodes:
        meta = branch._neo_branch_meta
        for arm_node in meta.true_arm_nodes + meta.false_arm_nodes:
            if isinstance(arm_node, Construct):
                continue
            if arm_node.context:
                for ctx_name in arm_node.context:
                    ctx_field = field_name_for(ctx_name)
                    if ctx_field not in fields:
                        ctx_type = _ctx_types.get(ctx_field, Any)
                        fields[ctx_field] = (ctx_type if ctx_type is not Any else Any, None)

    # Framework fields — always present
    # node_id and project_root have defaults so consumers can omit them
    # in run(input=...); they're still accessible via config["configurable"]
    # for node functions that need pipeline metadata.
    fields[StateKeys.NODE_ID] = (str, "")
    fields[StateKeys.PROJECT_ROOT] = (str, "")
    fields[StateKeys.HUMAN_FEEDBACK] = (dict[str, Any] | None, None)
    fields[StateKeys.SCHEMA_FINGERPRINT] = (str, "")
    fields[StateKeys.NODE_FINGERPRINTS] = (dict[str, str], {})

    return create_model(f"{construct.name}State", **fields)


def build_output_schema_model(state_model: type[BaseModel]) -> type[BaseModel]:
    """Build the StateGraph ``output_schema``: every state field NOT ``neo_``-prefixed.

    Declared at compile time (``StateGraph(state_model, output_schema=...)``) so the
    ENGINE itself filters framework plumbing out of ``invoke``/``ainvoke`` results —
    replacing the hand-rolled ``_strip_internals`` wrap the runner and sub-construct
    exits used to carry (neograph-pjqe: declare, don't wrap). The filter is the
    ``neo_`` prefix, mirroring ``_strip_internals`` EXACTLY: the three non-``neo_``
    framework-injected fields (``node_id``/``project_root``/``human_feedback``) still
    surface, so the user-visible contract is unchanged; only enforcement moves from a
    runtime wrapper we own to a compile-time declaration the engine honours.

    Field annotations (including reducer ``Annotated`` metadata) are preserved via
    ``rebuild_annotation`` so the output channels match the state channels exactly.
    See docs/design/langgraph-output-schema-research-2026-07-03.md (R1/R3).
    """
    fields: dict[str, Any] = {
        name: (finfo.rebuild_annotation(), finfo)
        for name, finfo in state_model.model_fields.items()
        if not name.startswith(StateKeys.FRAMEWORK_PREFIX)
    }
    # A user node named e.g. ``validate`` produces a state field that shadows a
    # BaseModel attribute; Pydantic already warned once when ``state_model`` was
    # built. This synthesized Output model mirrors the same fields, so it would
    # re-emit the identical warning — a duplicate the user can do nothing about.
    # Suppress ONLY the framework copy here; the user-facing original still fires
    # on their own state model. See neograph-tj53.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Field name .* shadows an attribute in parent .*",
            category=UserWarning,
        )
        return create_model(f"{state_model.__name__}Output", **fields)


def _type_signature(typ: Any) -> str:
    """Structural signature of a type, used by both fingerprint computations.

    Qualname alone is too coarse: two structurally-different models that share a
    ``__qualname__`` (or the same class after a field-level edit) collide into a
    false negative, so the schema/node fingerprints never change and the
    checkpoint auto-rewind never triggers (neograph-v63o / review 080726 PAT-03).

    This folds one level of field detail into the signature — the same
    ``(field_name, str(annotation))`` detail ``compute_schema_fingerprint``
    records — so a same-name field add/remove/retype changes the signature:

    - Pydantic model  -> ``module.Qualname`` + sorted ``(field, str(annotation))``
      pairs. Nested models contribute their ``str(annotation)`` (not their own
      structure) to stay cycle-safe, matching the schema fingerprint's depth.
    - Generic (``list[X]``, ``dict[K,V]``, Each's ``dict[str, X]``) -> unwrapped
      so a field change on the wrapped model ``X`` is still visible.
    - Anything else -> ``str(typ)`` (already carries module + qualname).
    """
    args = get_args(typ)
    if args:
        origin = get_origin(typ)
        origin_name = getattr(origin, "__qualname__", str(origin))
        return f"{origin_name}[{','.join(_type_signature(a) for a in args)}]"
    if isinstance(typ, type) and issubclass(typ, BaseModel):
        fields = sorted((fname, str(finfo.annotation)) for fname, finfo in typ.model_fields.items())
        return f"{typ.__module__}.{typ.__qualname__}{fields!r}"
    return str(typ)


def compute_node_fingerprints(construct: Any) -> dict[str, str]:
    """Compute per-node output type fingerprints for checkpoint invalidation.

    Returns {field_name: sha256_prefix} for each node in the construct.
    Used to identify which specific nodes changed between runs.
    """
    import hashlib

    def _fp(name: str, typ: Any) -> str:
        # The fingerprint contract: sha256('{name}:{type_signature}')[:12]. The
        # :12 width and '{name}:{sig}' layout are load-bearing — schema and node
        # fingerprints move in lockstep, neograph-v63o, so the two branches
        # (dict-form per-key + singular) MUST share one definition, neograph-2yi7q.
        return hashlib.sha256(f"{name}:{_type_signature(typ)}".encode()).hexdigest()[:12]

    from neograph.naming import field_name_for

    result: dict[str, str] = {}

    def _fingerprint_item(item: Any) -> None:
        """Fingerprint one Node (per output key) or Construct (its output).

        Shared between top-level items and branch-arm items so an arm node's
        output type is invalidated on change exactly like a top-level node's.
        Kept as its own walk rather than routed through ``iter_nodes`` to
        preserve the top-level-only granularity: a sub-construct is
        fingerprinted by its declared output, not by its internal nodes.
        """
        # _declared_output abstracts the Node.outputs (plural) / Construct.output
        # (singular) split — Node dict-form is fingerprinted per key, a Construct's
        # single declared output as one field. No hand-rolled hasattr discrimination.
        declared = _declared_output(item)
        if declared is None:
            return
        fname = field_name_for(item.name)
        no = normalize_outputs(declared)
        if no.is_dict_form:
            # Dict-form outputs: fingerprint each key
            for key, typ in no.all_keys.items():
                full_name = output_field_name(fname, key)
                result[full_name] = _fp(full_name, typ)
        else:
            typ = no.primary
            result[fname] = _fp(fname, typ)

    for item in construct.nodes:
        if isinstance(item, _BranchNode):
            meta = item._neo_branch_meta
            for arm_item in meta.true_arm_nodes + meta.false_arm_nodes:
                _fingerprint_item(arm_item)
        else:
            _fingerprint_item(item)
    return result


def compute_schema_fingerprint(state_model: type[BaseModel]) -> str:
    """Compute a stable fingerprint from the state model's non-framework fields.

    The fingerprint changes when node output types change (field added/removed,
    type changed, class renamed). Framework fields (neo_*, node_id, project_root,
    human_feedback) are excluded — they change with modifier config, not schema.
    """
    import hashlib

    _FRAMEWORK_PREFIXES = (
        StateKeys.FRAMEWORK_PREFIX,
        StateKeys.NODE_ID,
        StateKeys.PROJECT_ROOT,
        StateKeys.HUMAN_FEEDBACK,
    )
    items = []
    for fname, finfo in state_model.model_fields.items():
        if any(fname.startswith(p) or fname == p for p in _FRAMEWORK_PREFIXES):
            continue
        # _type_signature (not bare str(annotation)) so a same-qualname field
        # change opens the gate -- otherwise the enriched node fingerprint below
        # is never reached; see neograph-v63o. Keeps both fingerprints in lockstep.
        items.append((fname, _type_signature(finfo.annotation)))
    items.sort()
    raw = repr(items).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _add_agent_channels(node: Node, fields: dict[str, Any]) -> None:
    """Add the agent-cycle state channels for an agent/act node.

    These carry the ReAct loop's per-turn state — message history, tool_log,
    resource manifest, and budget/iteration counters — so the inline agent-cycle
    expander (``_wiring._add_agent_cycle``) can make every turn a checkpointed
    superstep. All are ``neo_``-prefixed, so ``_strip_internals`` removes them
    from returned state and ``compute_schema_fingerprint`` excludes them.

    Only agent/act nodes get channels; think/scripted/raw nodes never enter a
    ReAct loop.
    """
    from langgraph.graph.message import add_messages

    if node.mode not in ("agent", "act"):
        return

    field_name = field_name_for(node.name)
    fields[StateKeys.agent_messages(field_name)] = (Annotated[list, add_messages], [])
    fields[StateKeys.agent_tool_log(field_name)] = (Annotated[list, _concat_reducer], [])
    fields[StateKeys.resource_manifest(field_name)] = (Annotated[list, _concat_reducer], [])
    fields[StateKeys.agent_budget(field_name)] = (dict | None, None)


def _add_output_field(node: Node, fields: dict[str, Any]) -> None:
    """Add a node's output type(s) as field(s) on the state model.

    When outputs is a dict (multi-output), creates one field per key:
    ``{node_name}_{output_key}``. Each/Oracle modifiers apply per key.
    When outputs is a single type (backward compat), creates ``{node_name}``.
    """
    if node.outputs is None:
        raise CompileError.build(
            "node has no output type",
            hint="every node must declare outputs=SomeModel",
            node=node.name,
        )

    field_name = field_name_for(node.name)
    no = normalize_outputs(node.outputs)

    # Dict-form outputs: one state field per key (neograph-1bp.2).
    if no.is_dict_form:
        combo, mods = classify_modifiers(node)
        match combo:
            case ModifierCombo.EACH_ORACLE | ModifierCombo.EACH_ORACLE_OPERATOR:
                # Each×Oracle fusion + dict-form: tagged collector + dict output
                # per key. Same as single-type fusion but per-key.
                collector_field = StateKeys.eachoracle_collector(field_name)
                fields[collector_field] = (
                    Annotated[list, _concat_reducer],
                    [],
                )
                for output_key, output_type in no.all_keys.items():
                    key_field = output_field_name(field_name, output_key)
                    field_type = dict[str, output_type] | None  # type: ignore[valid-type]
                    fields[key_field] = (
                        Annotated[field_type, _merge_dicts],
                        None,
                    )
            case ModifierCombo.ORACLE | ModifierCombo.ORACLE_OPERATOR:
                # Oracle + dict-form: single collector for the whole result dict,
                # per-key consumer fields without per-key collectors.
                collector_field = StateKeys.oracle_collector(field_name)
                fields[collector_field] = (
                    Annotated[list[dict], _concat_reducer],
                    [],
                )
                for output_key, output_type in no.all_keys.items():
                    key_field = output_field_name(field_name, output_key)
                    fields[key_field] = (output_type | None, None)
            case (
                ModifierCombo.BARE
                | ModifierCombo.OPERATOR
                | ModifierCombo.EACH
                | ModifierCombo.EACH_OPERATOR
                | ModifierCombo.LOOP
                | ModifierCombo.LOOP_OPERATOR
                | ModifierCombo.KEYMAKER
            ):
                # KEYMAKER dict-form is rejected at assembly (D-DICT-OUTPUTS);
                # the arm is defensively-unreachable and defers to the per-key
                # single-output builder (which treats KEYMAKER as bare).
                for output_key, output_type in no.all_keys.items():
                    key_field = output_field_name(field_name, output_key)
                    _add_single_output_field(node, key_field, output_type, fields)
            case _ as unreachable:
                assert_never(unreachable)
        return

    # Single-type outputs (backward compat): one field named after the node.
    _add_single_output_field(node, field_name, no.primary, fields)


def _add_single_output_field(
    node: Node,
    field_name: str,
    output_type: Any,
    fields: dict[str, Any],
) -> None:
    """Add one output field to the state model, applying modifier wrapping."""
    combo, mods = classify_modifiers(node)
    match combo:
        case ModifierCombo.EACH_ORACLE | ModifierCombo.EACH_ORACLE_OPERATOR:
            # Each×Oracle fusion: tagged collector + dict output
            collector_field = StateKeys.eachoracle_collector(field_name)
            fields[collector_field] = (
                Annotated[list, _concat_reducer],
                [],
            )
            # Final output: same shape as Each alone (dict[str, merged_type])
            field_type = dict[str, output_type] | None  # type: ignore[valid-type]
            fields[field_name] = (
                Annotated[field_type, _merge_dicts],
                None,
            )
        case ModifierCombo.EACH | ModifierCombo.EACH_OPERATOR:
            field_type = dict[str, output_type] | None  # type: ignore[valid-type, misc, assignment]
            fields[field_name] = (
                Annotated[field_type, _merge_dicts],
                None,
            )
        case ModifierCombo.ORACLE | ModifierCombo.ORACLE_OPERATOR:
            collector_field = StateKeys.oracle_collector(field_name)
            # When oracle_gen_type is set, the collector holds per-variant types
            # (list[gen_type]), not the post-merge type. The consumer-facing field
            # keeps node.outputs (the post-merge type).
            collector_type = node.oracle_gen_type if node.oracle_gen_type is not None else output_type
            fields[collector_field] = (
                Annotated[list[collector_type], _concat_reducer],  # type: ignore[valid-type]
                [],
            )
            fields[field_name] = (output_type | None, None)
        case ModifierCombo.LOOP | ModifierCombo.LOOP_OPERATOR:
            # Loop: append-list reducer. Each iteration pushes to the list.
            # _extract_input unwraps [-1] for the node on re-entry.
            # Downstream nodes after loop exit see the final value (unwrapped).
            fields[field_name] = (
                Annotated[list[output_type], _append_loop_result],
                [],
            )
        case ModifierCombo.BARE | ModifierCombo.OPERATOR | ModifierCombo.KEYMAKER:
            # A Keymaker mesh member writes its OWN output field as a plain
            # value (like a bare node); the mesh channel + hop counter are
            # separate neo_-prefixed fields added per mesh entry below.
            fields[field_name] = (output_type | None, None)
        case _ as unreachable:
            assert_never(unreachable)
