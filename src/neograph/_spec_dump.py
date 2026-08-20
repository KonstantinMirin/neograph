"""Construct -> spec dict. The inverse of :mod:`neograph.loader`.

``Construct.model_dump_json()`` raises: the IR holds live Pydantic *classes* in
``Node.inputs`` / ``Node.outputs`` / ``Construct.input`` / ``Construct.output``,
and a class is not JSON. ``load_spec`` existed with no counterpart, so a pipeline
could be built from data but never turned back into it (GH issue #9).

Two rules shape this module.

**One lowering semantics.** Every function here mirrors its counterpart in
``loader.py`` -- ``_dump_construct``/``_build_construct``,
``_dump_node``/``_build_node``, ``_dump_modifiers``/``_apply_modifiers`` -- so the
two directions cannot drift into disagreeing about what a spec means. Structural
dispatch walks RAW ``construct.nodes``, deliberately NOT
``_ir_branch.iter_with_arms``: that iterator DROPS the ``_BranchNode`` sentinel
and yields both arms in sequence, which would emit a linear pipeline in which
both arms look unconditional.

**Losses are marked in band.** A Construct holds values that are not data and
never can be -- ``Loop(when=lambda ...)``, ``skip_when``, ``raw_fn``, the ``@node``
function itself. Each is emitted as a sentinel object AT ITS OWN SITE::

    "loop": {"max_iterations": 3,
             "when": {"neograph/unrepresentable": "callable_loop_when",
                      "ref": "myproj.scoring:score_ok",
                      "source": "src/scoring.py:41"}}

so no consumer can mistake an unrepresentable value for an absent one -- a differ
comparing two pipelines whose only difference is the ``when`` lambda must not
report "identical". The ``neograph/losses`` array is an *index* over the same
sentinels (path-keyed, sorted), not the sole record of them.

**Scope.** This is part (a) of GH #9: dump + manifest. Where the spec format has
no slot for an IR shape at all -- dict-form ``Node.outputs``, a boundary-less
``Construct``, a ``_BranchNode`` -- this module emits a sentinel rather than
widening the schema; closing those gaps is part (b). Consequently the output is
NOT guaranteed to reload: ``load_spec(dump_spec(c))`` is not a fixed point and is
not claimed to be one.
"""

from __future__ import annotations

import inspect
import pathlib
import types
from typing import Any, NamedTuple, Union, get_args, get_origin

from pydantic import BaseModel

from neograph._ir_branch import _BranchNode
from neograph._normalize import normalize_inputs, normalize_outputs
from neograph.construct import Construct
from neograph.errors import ConfigurationError
from neograph.modifiers import classify_modifiers
from neograph.node import Node
from neograph.spec_types import _type_registry

__all__ = ["DUMP_LOSS_META", "dump_spec"]

#: The key that marks a value the spec format cannot carry. A dict shape is used
#: rather than a magic string because every spec slot that could hold a callable
#: is typed ``str``, so a dict here can never collide with a legitimate value.
UNREPRESENTABLE_KEY = "neograph/unrepresentable"

#: Top-level index over every sentinel in the document.
LOSSES_KEY = "neograph/losses"


class DumpLossMeta(NamedTuple):
    """What one loss id means. ``tier`` follows the GH #9 taxonomy."""

    tier: str
    meaning: str


#: Single source of truth for every id this module can emit. A both-directions
#: guard test pins that no unregistered id is emitted and no entry goes stale.
DUMP_LOSS_META: dict[str, DumpLossMeta] = {
    "callable_loop_when": DumpLossMeta(
        "NO_REPR", "Loop.when is a Python predicate; the spec slot holds a condition name"
    ),
    "callable_gate_tools_when": DumpLossMeta(
        "NO_REPR", "Node.gate_tools_when is a Python predicate, not a registered name"
    ),
    "callable_skip_when": DumpLossMeta("NO_REPR", "Node.skip_when is a Python predicate"),
    "callable_skip_value": DumpLossMeta("NO_REPR", "Node.skip_value is a Python callable"),
    "raw_fn": DumpLossMeta("NO_REPR", "Node.raw_fn is a Python function"),
    "renderer": DumpLossMeta("NO_REPR", "renderer is a live Renderer instance"),
    "oracle_merge_hook": DumpLossMeta(
        "NO_REPR", "Oracle merge_pre_process/merge_post_process/merge_fallback are Python callables"
    ),
    "absent_outputs": DumpLossMeta("NO_SLOT", "NodeSpec.outputs is required; Node.outputs is None"),
    "unregistered_type": DumpLossMeta(
        "NO_SLOT", "the annotation is not a BaseModel subclass, so it has no schema or registry name"
    ),
    "absent_construct_boundary": DumpLossMeta(
        "NO_SLOT", "ConstructSpec.input/output are required; this Construct declares none"
    ),
    "branch_node": DumpLossMeta(
        "NO_SLOT", "a _BranchNode has no spec representation; branch topology is spec-format v2"
    ),
}


def _repo_relative(path: str | None) -> str | None:
    """Repo-relative form of *path*, so two dumps from different checkouts match."""
    if not path:
        return None
    resolved = pathlib.Path(path).resolve()
    root = pathlib.Path(__file__).resolve().parents[2]
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        # Outside the repo (site-packages, a notebook): the basename still
        # identifies it without leaking a machine-specific prefix.
        return resolved.name


def _reference(value: Any) -> tuple[str | None, str | None]:
    """``(module:qualname, repo-relative file:lineno)`` for a live Python object.

    This is the only identity an unrepresentable value can carry into data, and
    it is why a sentinel beats an omission: ``ref`` is stable enough to diff.
    """
    target = value
    if not (inspect.isfunction(target) or inspect.isclass(target) or inspect.ismethod(target)):
        target = type(value)
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    ref = f"{module}:{qualname}" if module and qualname else None

    try:
        file = _repo_relative(inspect.getsourcefile(target))
        _, lineno = inspect.getsourcelines(target)
        source = f"{file}:{lineno}" if file else None
    except (OSError, TypeError):
        source = None
    return ref, source


class _Dump:
    """One dump in progress: the flat pools plus the loss index being built."""

    def __init__(self) -> None:
        self.types: dict[str, dict[str, Any]] = {}
        self.nodes: list[dict[str, Any]] = []
        self.constructs: list[dict[str, Any]] = []
        self.losses: list[dict[str, Any]] = []

    # -- losses ------------------------------------------------------------

    def lose(self, loss_id: str, path: str, value: Any = None) -> dict[str, Any]:
        """Record a loss and return the in-band sentinel to embed at *path*."""
        if loss_id not in DUMP_LOSS_META:
            raise ConfigurationError.build(
                "dump_spec emitted an unregistered loss id",
                found=loss_id,
                hint="add it to _spec_dump.DUMP_LOSS_META with its tier and meaning.",
            )
        meta = DUMP_LOSS_META[loss_id]
        sentinel: dict[str, Any] = {UNREPRESENTABLE_KEY: loss_id}
        ref, source = _reference(value) if value is not None else (None, None)
        if ref:
            sentinel["ref"] = ref
        if source:
            sentinel["source"] = source

        entry = {"id": loss_id, "tier": meta.tier, "path": path, "meaning": meta.meaning}
        if ref:
            entry["ref"] = ref
        if source:
            entry["source"] = source
        self.losses.append(entry)
        return sentinel

    # -- types -------------------------------------------------------------

    def type_ref(self, annotation: Any, path: str, *, loss_id: str = "unregistered_type") -> Any:
        """A spec type reference for *annotation*: a name, or a sentinel.

        Containers over a model resolve rather than being refused --
        ``list[X]`` -> ``"[X]"``, ``dict[str, X]`` -> ``"{str: X}"``,
        ``X | None`` -> ``"X?"`` -- because those are the shapes real pipelines
        declare (a tool log is a ``list[Entry]``), and refusing them meant every
        agent node lost its output contract (GH #9). Each member model is
        recursed into, so its schema lands in ``types:`` too.

        The schema is ALWAYS emitted alongside the name, so the document is
        self-contained -- a bare registry name is only meaningful to a process
        that already imported the project, and the stated consumer is an
        external viewer.
        """
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            name = _registered_name(annotation) or annotation.__name__
            if name not in self.types:
                self.types[name] = annotation.model_json_schema()
            return name

        rendered = self._container_ref(annotation, path)
        if rendered is not None:
            return rendered

        return self.lose(loss_id, path, annotation)

    def _container_ref(self, annotation: Any, path: str) -> str | None:
        """Render a container/union over models, or None when unrecognised.

        Deliberately narrow: only the shapes whose members can themselves be
        resolved. Anything else falls through to a sentinel rather than being
        rendered as a repr a consumer could mistake for a real type name.
        """
        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin is None or not args:
            return None

        def member(arg: Any) -> str | None:
            if arg is type(None):
                return None
            resolved = self.type_ref(arg, path)
            # A member that is itself unrepresentable makes the whole container
            # unrepresentable -- do not paper over it with a partial name.
            return resolved if isinstance(resolved, str) else None

        if origin is Union or origin is types.UnionType:
            non_none = [a for a in args if a is not type(None)]
            rendered = [member(a) for a in non_none]
            if any(r is None for r in rendered):
                return None
            body = " | ".join(r for r in rendered if r)
            return f"{body}?" if len(non_none) != len(args) else body

        if origin in (list, set, frozenset, tuple):
            inner = member(args[0])
            return None if inner is None else f"[{inner}]"

        if origin is dict and len(args) == 2:
            key = args[0].__name__ if isinstance(args[0], type) else None
            value = member(args[1])
            if key is None or value is None:
                return None
            return f"{{{key}: {value}}}"

        return None


def _registered_name(cls: type) -> str | None:
    """The spec-registry name for *cls*, or None.

    DERIVED by scanning the registry, never cached in a parallel map: the test
    suite clears ``_type_registry`` directly, which a parallel map would survive,
    so a cache would emit names for types no longer registered. First
    registration wins, which is the documented insertion-order tie-break.
    """
    for name, registered in _type_registry.items():
        if registered is cls:
            return name
    return None


def _dump_modifiers(item: Any, dump: _Dump, path: str) -> dict[str, Any]:
    """Mirror of ``loader._apply_modifiers``."""
    out: dict[str, Any] = {}
    _, mods = classify_modifiers(item)

    oracle = mods.get("oracle")
    if oracle is not None:
        spec: dict[str, Any] = {}
        for field in ("n", "models", "merge_fn", "merge_prompt"):
            value = getattr(oracle, field, None)
            if value is not None:
                spec[field] = value
        if getattr(oracle, "merge_model", None):
            spec["merge_model"] = oracle.merge_model
        for hook in ("merge_pre_process", "merge_post_process", "merge_fallback"):
            value = getattr(oracle, hook, None)
            if value is not None:
                spec[hook] = dump.lose("oracle_merge_hook", f"{path}.oracle.{hook}", value)
        out["oracle"] = spec

    each = mods.get("each")
    if each is not None:
        out["each"] = {"over": each.over, "key": each.key}

    loop = mods.get("loop")
    if loop is not None:
        when: Any = loop.when
        if not isinstance(when, str):
            when = dump.lose("callable_loop_when", f"{path}.loop.when", loop.when)
        out["loop"] = {
            "when": when,
            "max_iterations": loop.max_iterations,
            "on_exhaust": loop.on_exhaust,
        }

    operator = mods.get("operator")
    if operator is not None:
        out["operator"] = {"when": operator.when}

    return out


def _dump_tool(tool: Any) -> dict[str, Any]:
    """Mirror of ``loader._resolve_tool``."""
    return {
        "name": tool.name,
        "budget": getattr(tool, "budget", 0),
        "config": dict(getattr(tool, "config", {}) or {}),
    }


def _dump_node(node: Node, dump: _Dump, path: str) -> dict[str, Any]:
    """Mirror of ``loader._build_node``."""
    out: dict[str, Any] = {"name": node.name, "mode": node.mode}

    # The outputs/inputs shape discrimination is monopolized by _normalize --
    # a hand-rolled isinstance(..., dict) here would be a second discriminator.
    outputs = normalize_outputs(node.outputs)
    if outputs.is_none:
        out["outputs"] = dump.lose("absent_outputs", f"{path}.outputs")
    elif outputs.is_dict_form:
        out["outputs"] = {
            key: dump.type_ref(value, f"{path}.outputs.{key}")
            for key, value in outputs.all_keys.items()
        }
    else:
        out["outputs"] = dump.type_ref(outputs.primary, f"{path}.outputs")

    inputs = normalize_inputs(node.inputs)
    if inputs.is_dict_form:
        out["inputs"] = {
            key: dump.type_ref(value, f"{path}.inputs.{key}")
            for key, value in inputs.by_name.items()
        }
    elif not inputs.is_none:
        out["inputs"] = dump.type_ref(inputs.single_type, f"{path}.inputs")

    for field in ("prompt", "model", "scripted_fn", "context"):
        value = getattr(node, field, None)
        if value:
            out[field] = value

    if node.tools:
        out["tools"] = [_dump_tool(t) for t in node.tools]

    for field, loss_id in (
        ("raw_fn", "raw_fn"),
        ("skip_when", "callable_skip_when"),
        ("skip_value", "callable_skip_value"),
        ("renderer", "renderer"),
    ):
        value = getattr(node, field, None)
        if value is not None:
            out[field] = dump.lose(loss_id, f"{path}.{field}", value)

    gate = getattr(node, "gate_tools_when", None)
    if gate is not None and not isinstance(gate, str):
        out["gate_tools_when"] = dump.lose("callable_gate_tools_when", f"{path}.gate_tools_when", gate)
    elif isinstance(gate, str):
        out["gate_tools_when"] = gate

    out.update(_dump_modifiers(node, dump, path))
    return out


def _dump_sub_construct(sub: Construct, dump: _Dump, path: str) -> dict[str, Any]:
    """Mirror of ``loader._build_sub_construct``.

    Members go into the flat node pool and are referenced by name, which is the
    spec format's own shape (``ConstructSpec.nodes`` holds strings).
    """
    out: dict[str, Any] = {"name": sub.name}

    for field in ("input", "output"):
        declared = getattr(sub, field, None)
        if declared is None:
            out[field] = dump.lose("absent_construct_boundary", f"{path}.{field}")
        else:
            out[field] = dump.type_ref(declared, f"{path}.{field}")

    refs: list[Any] = []
    for index, member in enumerate(sub.nodes):
        member_path = f"{path}.nodes[{index}]"
        if isinstance(member, Node):
            dump.nodes.append(_dump_node(member, dump, member_path))
            refs.append(member.name)
        elif isinstance(member, Construct):
            dump.constructs.append(_dump_sub_construct(member, dump, member_path))
            refs.append(member.name)
        else:
            refs.append(dump.lose("branch_node", member_path, member))
    out["nodes"] = refs

    out.update(_dump_modifiers(sub, dump, path))
    return out


def dump_spec(construct: Construct, *, strict: bool = False) -> dict[str, Any]:
    """Render *construct* as a JSON-serializable spec dict.

    The inverse of :func:`neograph.load_spec`, for tooling that needs a pipeline
    as data: a graph viewer, a diff between two pipeline versions, a CI check
    comparing two arms' tool bindings (GH issue #9).

    Parameters
    ----------
    construct:
        The pipeline to render.
    strict:
        When True, raise instead of returning a document that contains any
        unrepresentable value. The default is False because this output is meant
        to be READ, not executed on a foreign runtime -- and because every loss
        is marked in band at its own site plus indexed under
        ``"neograph/losses"``, so a lossy dump is self-describing rather than
        silently incomplete.

    Returns
    -------
    dict
        A spec-shaped dict that survives ``json.dumps``. Unrepresentable values
        appear as ``{"neograph/unrepresentable": <id>, "ref": ..., "source": ...}``
        sentinels. Reloading is not guaranteed -- see the module docstring.

    Raises
    ------
    ConfigurationError
        When ``strict=True`` and the document contains at least one sentinel.
    """
    dump = _Dump()

    refs: list[Any] = []
    # RAW construct.nodes, deliberately not iter_with_arms -- that iterator drops
    # the _BranchNode, and a dropped branch is exactly the kind of silent
    # omission this module exists to prevent.
    for index, item in enumerate(construct.nodes):
        path = f"pipeline.nodes[{index}]"
        if isinstance(item, Node):
            dump.nodes.append(_dump_node(item, dump, f"nodes[{index}]"))
            refs.append(item.name)
        elif isinstance(item, Construct):
            dump.constructs.append(_dump_sub_construct(item, dump, f"constructs[{index}]"))
            refs.append(item.name)
        elif isinstance(item, _BranchNode):
            refs.append(dump.lose("branch_node", path, item))
        else:  # pragma: no cover - the IR admits no fourth member kind
            refs.append(dump.lose("branch_node", path, item))

    payload: dict[str, Any] = {
        "version": "1",
        "name": construct.name,
        "description": getattr(construct, "description", "") or "",
        "types": {name: dump.types[name] for name in sorted(dump.types)},
        "nodes": dump.nodes,
        "constructs": dump.constructs,
        "pipeline": {"nodes": refs},
        LOSSES_KEY: sorted(dump.losses, key=lambda entry: (entry["path"], entry["id"])),
    }

    if strict and dump.losses:
        first = payload[LOSSES_KEY][0]
        raise ConfigurationError.build(
            "dump_spec(strict=True) cannot represent this construct",
            found=f"{first['id']} at {first['path']}",
            expected="a construct whose every field has a spec representation",
            hint="drop strict= to receive the dump with in-band sentinels and a "
            f'"{LOSSES_KEY}" index.',
        )

    return payload
