"""@node decorator and construct_from_module — Dagster-style pipeline definition.

Ergonomic front-end on top of Node + Construct: the function signature IS the
dependency graph.

    @node(mode="scripted", output=Claims)
    def decompose(topic: RawText) -> Claims: ...

    @node(mode="scripted", output=Classified)
    def classify(decompose: Claims) -> Classified: ...
    # parameter name 'decompose' matches upstream node 'decompose' → auto-wires

    pipeline = construct_from_module(sys.modules[__name__])

Design notes

* Mirrors `src/neograph/tool.py:89-133` — same two-form call shape (`@node`
  vs `@node(...)`), same function-local factory import to dodge the
  `factory → node → decorators` circular path, same "return a spec instance
  rather than the wrapped function" contract.

* The decorator stashes the original function and its parameter-name tuple in
  a module-level dict keyed by `id(Node)`, with a `weakref.finalize` callback
  that evicts the entry when the Node is garbage-collected. `Node` itself is
  a pydantic BaseModel and is not mutated to add sidecar fields — the beads
  brief explicitly forbids editing `node.py`.

* Scripted @node functions are dispatched via the existing `raw_fn` field on
  `Node` (which the `factory._make_raw_wrapper` branch already supports for
  `raw_node`). Using the raw-fn path — rather than `register_scripted` — lets
  us pass the full state to a closure that reads N upstream values by name,
  without editing `factory.py` (which is out of scope per the brief). Non-
  scripted modes (produce / gather / execute) never see the raw_fn path and
  keep their existing LLM dispatch; their parameter annotations only drive
  topology + type inference.

* `construct_from_module` walks `vars(mod)` once, keeps only Node instances
  that appear in the sidecar (so plain `Node(...)` at module scope is
  ignored), builds adjacency from each node's parameter-name tuple, DFS
  topological-sorts with a visiting set for cycle detection, and hands the
  sorted list to `Construct(name=..., nodes=...)`. No new validation path:
  the existing `_validate_node_chain` runs via `Construct.__init__`.

* Name convention: function `foo_bar` → node name `'foo-bar'`; a downstream
  parameter `foo_bar: T` looks up the node via `name.replace("-", "_")`.
  Matches the state-field convention everywhere else in the codebase.

* v1 scope: every parameter must name an upstream `@node` in the module.
  Scalars and run-input kwargs are out of scope (they raise `ConstructError`).
  `*args` / `**kwargs` are rejected at decoration time.
"""

from __future__ import annotations

import inspect
import weakref
from typing import Any, Callable, Literal

from neograph._construct_validation import ConstructError
from neograph.construct import Construct
from neograph.node import Node
from neograph.tool import Tool


# Sidecar: id(Node) -> (original_fn, param_names_tuple).
# Keyed by id() so `Node` is not mutated. A `weakref.finalize` callback evicts
# entries when the Node is garbage-collected, so the dict cannot leak.
_node_sidecar: dict[int, tuple[Callable, tuple[str, ...]]] = {}


def _register_sidecar(n: Node, fn: Callable, param_names: tuple[str, ...]) -> None:
    node_id = id(n)
    _node_sidecar[node_id] = (fn, param_names)
    weakref.finalize(n, _node_sidecar.pop, node_id, None)


def _get_sidecar(n: Node) -> tuple[Callable, tuple[str, ...]] | None:
    return _node_sidecar.get(id(n))


def node(
    fn: Callable | None = None,
    *,
    mode: Literal["produce", "gather", "execute", "scripted"] = "produce",
    input: Any = None,
    output: Any = None,
    model: str | None = None,
    prompt: str | None = None,
    llm_config: dict[str, Any] | None = None,
    tools: list[Tool] | None = None,
    name: str | None = None,
) -> Any:
    """Decorator that turns a function into a Node spec with signature-inferred
    dependencies. Supports both `@node` and `@node(...)` call forms.

    Inference rules — explicit kwargs always win over annotations:
        * `name`   ← kwarg, else `fn.__name__.replace("_", "-")`
        * `output` ← kwarg, else function return annotation
        * `input`  ← kwarg, else annotation of the first annotated parameter

    For `mode='scripted'`, the function is executed via `Node.raw_fn` set at
    `construct_from_module` time — this keeps `factory.py` untouched and
    supports fan-in (>1 parameter) nodes uniformly.
    """

    def decorator(f: Callable) -> Node:
        sig = inspect.signature(f)

        # Reject *args / **kwargs early — they have no sensible mapping to
        # upstream nodes.
        for p in sig.parameters.values():
            if p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                msg = (
                    f"@node decorator on '{f.__name__}': parameter '{p.name}' "
                    f"is *args/**kwargs, which has no upstream-node mapping. "
                    f"Use explicit named parameters."
                )
                raise ConstructError(msg)

        param_names = tuple(p.name for p in sig.parameters.values())
        node_name = (name or f.__name__).replace("_", "-")

        # Output inference: explicit kwarg wins; fall back to return annotation.
        inferred_output = output
        if inferred_output is None:
            ret = sig.return_annotation
            if ret is not inspect.Signature.empty:
                inferred_output = ret

        # Input inference: explicit kwarg wins. Otherwise use the annotation
        # of the first annotated parameter. The per-node `.input` field only
        # tracks a single type (matches the existing Node contract); fan-in
        # still drives topology via param names, only the first annotated
        # param feeds `.input`.
        inferred_input = input
        if inferred_input is None:
            for p in sig.parameters.values():
                if p.annotation is not inspect.Parameter.empty:
                    inferred_input = p.annotation
                    break

        n = Node(
            name=node_name,
            mode=mode,
            input=inferred_input,
            output=inferred_output,
            model=model,
            prompt=prompt,
            llm_config=llm_config or {},
            tools=tools or [],
        )

        _register_sidecar(n, f, param_names)
        return n

    # Support both @node and @node(...) forms (see tool.py:130-132).
    if fn is not None:
        return decorator(fn)
    return decorator


def construct_from_module(mod: Any, name: str | None = None) -> Construct:
    """Walk a module's @node-built Nodes, sort topologically, return a Construct.

    Walks `vars(mod)`, keeping only Node instances that appear in the @node
    sidecar (plain `Node(...)` instances at module scope are ignored). Builds
    adjacency from each node's parameter-name tuple: a parameter named `foo_bar`
    is treated as a dependency on the node whose state-field form is `foo_bar`
    (i.e. the node named `'foo-bar'`). Unknown parameter names raise
    `ConstructError`; cycles raise `ConstructError`.

    The returned Construct is a regular Construct — compile/run operate on it
    unchanged. The existing `_validate_node_chain` walker runs via
    `Construct.__init__`, so type-compatibility is enforced as usual.
    """
    decorated: dict[str, Node] = {}
    for attr in vars(mod).values():
        if isinstance(attr, Node) and _get_sidecar(attr) is not None:
            field_name = attr.name.replace("-", "_")
            decorated[field_name] = attr

    if not decorated:
        construct_name = name or mod.__name__.split(".")[-1]
        return Construct(name=construct_name, nodes=[])

    # Build adjacency: for each node, which other nodes does it depend on?
    adjacency: dict[str, list[str]] = {k: [] for k in decorated}
    for field_name, n in decorated.items():
        sidecar = _get_sidecar(n)
        assert sidecar is not None  # filtered above
        _, param_names = sidecar
        seen_deps: set[str] = set()
        for pname in param_names:
            if pname not in decorated:
                msg = (
                    f"@node '{n.name}' parameter '{pname}' does not match any "
                    f"@node in module '{mod.__name__}'. All parameters must "
                    f"name an upstream @node (v1 does not support scalar or "
                    f"run-input parameters).\n"
                    f"  available @nodes: {sorted(decorated.keys())}"
                )
                raise ConstructError(msg)
            if pname == field_name:
                msg = (
                    f"@node '{n.name}' has a parameter '{pname}' that refers "
                    f"to itself — self-dependency is not allowed."
                )
                raise ConstructError(msg)
            if pname not in seen_deps:
                adjacency[field_name].append(pname)
                seen_deps.add(pname)

    # Topological sort via DFS with a visiting set for cycle detection.
    # Dict insertion order is preserved, giving deterministic output for
    # the same module.
    ordered: list[Node] = []
    marks: dict[str, str] = {}  # missing=white, "gray"=visiting, "black"=done

    def visit(field: str) -> None:
        state = marks.get(field)
        if state == "black":
            return
        if state == "gray":
            msg = (
                f"@node cycle detected in module involving '{field}'. "
                f"Cyclical parameter-name dependencies are not allowed."
            )
            raise ConstructError(msg)
        marks[field] = "gray"
        for dep in adjacency[field]:
            visit(dep)
        marks[field] = "black"
        ordered.append(decorated[field])

    for field in decorated:
        visit(field)

    # Install the execution dispatch for scripted nodes. We use Node.raw_fn
    # (which factory._make_raw_wrapper already handles) rather than
    # register_scripted, because raw_fn receives the full LangGraph state
    # and can read N upstream values by name — the only v1 way to support
    # fan-in without editing factory.py.
    for n in ordered:
        if n.mode == "scripted":
            _attach_scripted_raw_fn(n)

    construct_name = name or mod.__name__.split(".")[-1]
    return Construct(name=construct_name, nodes=ordered)


def _attach_scripted_raw_fn(n: Node) -> None:
    """Wrap the original user function in a raw-node adapter that reads
    upstream values from LangGraph state by parameter name.

    The adapter reads each `param_name` from state (dict or BaseModel form),
    calls the user function positionally, and returns a state update dict
    keyed by this node's state-field name. If the user function produces
    `None`, nothing is written — matching `factory._make_scripted_wrapper`
    semantics.
    """
    sidecar = _get_sidecar(n)
    if sidecar is None:
        return
    fn, param_names = sidecar
    field_name = n.name.replace("-", "_")

    def raw_adapter(state: Any, config: Any) -> dict:
        def _get(key: str) -> Any:
            if isinstance(state, dict):
                return state.get(key)
            return getattr(state, key, None)

        args = [_get(pname) for pname in param_names]
        result = fn(*args)

        if result is None:
            return {}
        return {field_name: result}

    raw_adapter.__name__ = field_name

    # Node is a pydantic v2 BaseModel without `frozen=True`, so direct
    # attribute assignment is supported. `raw_fn` is a declared field on
    # Node (used by the existing @raw_node decorator) so assigning it does
    # not mutate the schema — only the field value.
    n.raw_fn = raw_adapter
