"""Symbolic-execution core for ForwardConstruct tracing.

Extracted from ``forward.py`` (neograph-3ffdg.12) as a pure file split — the
helpers and classes below are unchanged, only their home moved. ``forward.py``
re-exports them, so existing imports keep resolving.

What lives here: the proxy/tracer engine that records what ``forward()`` did.
``_Proxy`` and ``_ConditionProxy`` stand in for real values and record attribute
chains and comparisons; ``_Tracer`` accumulates ``_NodeCall`` records and branch
decisions; ``_BranchPoint`` / ``_BranchTrace`` capture one branch decision and
one whole-body re-trace. The three free helpers parse proxy names and attribute
chains and are shared by this engine, the trace orchestrator, and the DX builder
classes that stay in ``forward.py``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import Any

from neograph._ir_branch import _ConditionSpec
from neograph._normalize import _declared_output, normalize_outputs
from neograph.conditions import OPERATORS
from neograph.construct import Construct
from neograph.errors import ConstructError
from neograph.naming import field_name_for
from neograph.node import Node

# Ceiling on distinct branch points in one forward() body. Lives here with the
# tracer that enforces it.
_MAX_BRANCHES = 8


def _primary_type(item: Node | Construct) -> Any:
    """The primary declared-output type of a node/construct — the downstream
    state-bus type the tracer uses for edge and port typing.

    Returns the post-merge output type (consumer-facing). This INTENTIONALLY
    diverges from ``_dispatch._resolve_primary_output``, neograph-2yi7q:
      - here / edges / ports: post-merge type B (what flows on the state bus —
        ``state.py`` writes ``output_type`` = ``node.outputs`` for the
        consumer-facing field)
      - ``_resolve_primary_output``: the per-variant ``gen_type`` A passed to the
        LLM as the invocation schema (``_dispatch`` -> ``invoke_structured``)
    For a type-transforming Oracle (``merge_fn`` A->B) these MUST differ, and
    ``_ir_normalize.oracle_gen_type_for`` fires the override exactly when the
    declared output is the post-merge B. Do NOT route these call sites through
    ``_resolve_primary_output`` — that would infer gen_type A for downstream
    edges while the bus carries B, breaking edge type-inference.
    """
    return normalize_outputs(_declared_output(item)).primary


def _attr_chain_after_prefix(source_node: Node | Construct | None, full_name: str) -> list[str]:
    """Strip the ``out_of_<node>`` proxy-name prefix and return the attr chain.

    The symbolic tracer names node-output proxies ``out_of_<node_name>`` and
    appends a dotted attribute path (e.g. ``out_of_check.items.severity``). This
    returns the path components after the prefix (``["items", "severity"]``), or
    ``[]`` when there is no source node or the prefix does not match. Shared by
    ``_ConditionProxy._build_runtime_condition`` and ``_Tracer.record_iteration``.
    """
    if source_node is None:
        return []
    prefix = f"out_of_{source_node.name}"
    if not full_name.startswith(prefix):
        return []
    remainder = full_name[len(prefix) :]
    return [p for p in remainder.split(".") if p]


def _over_path_for_proxy(source_node: Node | Construct | None, full_name: str) -> str:
    """Build a dotted ``Each.over`` state path from a traced proxy.

    ``out_of_seed.items`` with source ``seed`` becomes ``seed.items`` — the
    producer's state field name followed by the attribute chain. Shared by
    ``_Tracer.record_iteration`` (bare-``for`` sugar) and ``_EachCall``.
    """
    if source_node is None:
        return full_name
    field_name = field_name_for(source_node.name)
    attr_parts = _attr_chain_after_prefix(source_node, full_name)
    return ".".join([field_name, *attr_parts]) if attr_parts else field_name


class _Proxy:
    """A stand-in for a real value during forward() tracing.

    Carries the Node that produced it (or None for the initial input).
    Supports attribute access (returns child proxies), comparison operators
    (returns _ConditionProxy), and __bool__ (delegates to tracer for branch
    recording).
    """

    __slots__ = ("_neo_source", "_neo_name", "_neo_tracer")

    def __init__(
        self,
        source_node: Node | Construct | None,
        name: str,
        tracer: _Tracer | None = None,
    ) -> None:
        self._neo_source = source_node
        self._neo_name = name
        self._neo_tracer = tracer

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_neo_"):
            raise AttributeError(name)
        # Return a child proxy for attribute access (e.g., classified.confidence)
        return _Proxy(self._neo_source, f"{self._neo_name}.{name}", self._neo_tracer)

    def __lt__(self, other: Any) -> _ConditionProxy:
        return _ConditionProxy(self, "<", other)

    def __le__(self, other: Any) -> _ConditionProxy:
        return _ConditionProxy(self, "<=", other)

    def __gt__(self, other: Any) -> _ConditionProxy:
        return _ConditionProxy(self, ">", other)

    def __ge__(self, other: Any) -> _ConditionProxy:
        return _ConditionProxy(self, ">=", other)

    def __eq__(self, other: Any) -> _ConditionProxy:  # type: ignore[override]
        return _ConditionProxy(self, "==", other)

    def __ne__(self, other: Any) -> _ConditionProxy:  # type: ignore[override]
        return _ConditionProxy(self, "!=", other)

    def __hash__(self) -> int:
        # Required because we defined __eq__
        return id(self)

    def __bool__(self) -> bool:
        tracer = self._neo_tracer
        if tracer is None:
            raise TypeError("Cannot use proxy in boolean context outside tracing")
        return tracer.record_branch(self)

    def __iter__(self):
        tracer = self._neo_tracer
        if tracer is None:
            raise TypeError("Cannot iterate proxy outside tracing")
        return tracer.record_iteration(self)

    def __repr__(self) -> str:
        src = self._neo_source.name if self._neo_source else "<input>"
        return f"_Proxy(from={src}, name={self._neo_name})"


class _ConditionProxy:
    """Records a comparison for branch lowering.

    Created by _Proxy comparison operators (e.g., proxy.score < 0.7).
    When used in a boolean context (if statement), delegates to the
    tracer to record the branch point.
    """

    __slots__ = ("_left", "_op", "_right", "_neo_tracer")

    def __init__(self, left: _Proxy, op: str, right: Any) -> None:
        self._left = left
        self._op = op
        self._right = right
        self._neo_tracer = getattr(left, "_neo_tracer", None)

    def __bool__(self) -> bool:
        tracer = self._neo_tracer
        if tracer is None:
            raise TypeError("Cannot use condition in boolean context outside tracing")
        # v1 limit (neograph-e9zse.7): branch conditions must compare a proxy
        # attribute against a CONSTANT. A proxy right-hand side previously
        # traced silently and misbehaved at runtime (the threshold would be
        # the _Proxy object itself) — fail loud with the declarative escape.
        if isinstance(self._right, (_Proxy, _ConditionProxy)):
            raise ConstructError.build(
                "forward() branch conditions must compare against constants",
                expected="a constant right-hand side (e.g. proxy.score < 0.7)",
                found="a traced proxy on the right-hand side",
                hint="richer conditions are a v1 limitation of forward() tracing — "
                "compute the comparison inside a node, or use the declarative "
                "Construct form with a registered condition",
            )
        return tracer.record_branch(self)

    def _build_runtime_condition(self) -> Any:
        """Build a callable that evaluates this condition against live state.

        Parses the attribute path on the left-hand proxy to determine which
        state field + attribute chain to read at runtime.

        Example: _Proxy(from=check, name="out_of_check.score") > 0.5
        becomes: lambda state: getattr(state, "br_check").score > 0.5

        Returns (source_node, attr_chain, op_fn, threshold) tuple that
        the compiler uses to build the router function.
        """
        left = self._left
        op_fn = OPERATORS[self._op]
        threshold = self._right

        # Parse the proxy name to extract state field + attribute chain
        # e.g., "out_of_br-check.score" → source "br-check", attrs ["score"]
        source_node = left._neo_source
        full_name = left._neo_name

        # Extract attribute chain after the "out_of_<node>" proxy prefix.
        attr_chain = _attr_chain_after_prefix(source_node, full_name)

        return _ConditionSpec(
            source_node=source_node,
            attr_chain=attr_chain,
            op_fn=op_fn,
            op_str=self._op,
            threshold=threshold,
        )


@dataclasses.dataclass
class _BranchPoint:
    """A recorded branch point during tracing."""

    branch_id: int
    condition: _ConditionProxy | _Proxy
    decision: bool


@dataclasses.dataclass
class _BranchTrace:
    """Result of tracing both arms of a branch."""

    branch: _BranchPoint
    true_nodes: list[Node | Construct]
    false_nodes: list[Node | Construct]


class _Tracer:
    """Collects node invocations during a single forward() trace run.

    Deduplicates by identity (id(node)): repeated calls to the same Node
    instance collapse into one entry. Supports branch recording for if/else
    tracing via the re-trace strategy.
    """

    def __init__(self, branch_decisions: dict[int, bool] | None = None) -> None:
        self._ordered: list[Node | Construct] = []
        self._seen: set[int] = set()
        self._branches: list[_BranchPoint] = []
        self._branch_decisions = branch_decisions or {}
        self._next_branch_id = 0
        # Loop-mode tracking: maps node id → Each over-path for nodes in loop body
        self._loop_stack: list[str] = []  # stack of over-paths (for nested detection)
        self._loop_body_nodes: dict[int, str] = {}  # id(node) → over-path
        # Deterministic sub-construct naming: per-(kind, slug) occurrence
        # counter — a loop and an each over the same body slug must not
        # share a counter.
        self._occurrences: dict[tuple[str, str], int] = {}

    def record(self, node: Node) -> None:
        key = id(node)
        if key in self._seen:
            return
        self._seen.add(key)
        self._ordered.append(node)
        # If we're inside a loop body, tag this node for Each wrapping
        if self._loop_stack:
            self._loop_body_nodes[key] = self._loop_stack[-1]

    def record_construct(self, construct: Construct) -> None:
        """Record a sub-construct (e.g., from self.loop()) in the node list."""
        self._ordered.append(construct)

    def next_occurrence(self, kind: str, body_slug: str) -> int:
        """Return the next occurrence index for a sub-construct of ``kind``
        ('loop' or 'each') with the given body slug.

        Deterministic within a single trace pass: the first loop with body
        'review-revise' gets 0, the second gets 1. Re-trace passes produce
        the same sequence because forward() is called the same way. Keyed
        per (kind, slug) so a loop and an each over the same body do not
        collide.
        """
        count = self._occurrences.get((kind, body_slug), 0)
        self._occurrences[(kind, body_slug)] = count + 1
        return count

    def record_iteration(self, proxy: _Proxy) -> Iterator[_Proxy]:
        """Record a for-loop iteration over a proxy attribute.

        Enters loop mode: nodes recorded while in this mode get tagged
        with an Each modifier. Yields a single proxy item (enough for
        tracing — the loop body runs once to discover node calls).
        """
        # Build the Each over-path from the proxy's attribute chain
        # e.g., _Proxy(source=make, name="out_of_make.groups") → "make.groups"
        source_node = proxy._neo_source
        full_name = proxy._neo_name
        over_path = _over_path_for_proxy(source_node, full_name)

        self._loop_stack.append(over_path)
        # Yield a single proxy item — enough for tracing the loop body once
        item_proxy = _Proxy(
            source_node=source_node,
            name=f"{full_name}.__item__",
            tracer=self,
        )

        def _iter():
            try:
                yield item_proxy
            finally:
                self._loop_stack.pop()

        return _iter()

    def record_branch(self, condition: _ConditionProxy | _Proxy) -> bool:
        """Record a branch point and return the decision for this trace pass.

        Default decision is True (take the true arm). Pre-configured
        decisions in self._branch_decisions override this.

        Raises ConstructError if more than _MAX_BRANCHES branches are encountered.
        """
        branch_id = self._next_branch_id
        if branch_id >= _MAX_BRANCHES:
            raise ConstructError.build(
                "too many branches in forward()",
                expected=f"at most {_MAX_BRANCHES} branches",
                found=f"{branch_id + 1} branches",
                hint="branch discovery re-traces forward() per branch (2^N cost) — "
                "simplify your forward(), extract sub-pipelines, or use the "
                "declarative Construct form for richer branching",
            )
        self._next_branch_id += 1

        if branch_id in self._branch_decisions:
            decision = self._branch_decisions[branch_id]
        else:
            decision = True

        self._branches.append(
            _BranchPoint(
                branch_id=branch_id,
                condition=condition,
                decision=decision,
            )
        )
        return decision

    @property
    def nodes(self) -> list[Node | Construct]:
        return list(self._ordered)

    @property
    def branches(self) -> list[_BranchPoint]:
        return list(self._branches)


class _NodeCall:
    """Callable wrapper that records a node invocation into the active Tracer."""

    __slots__ = ("_node", "_tracer")

    def __init__(self, node: Node, tracer: _Tracer) -> None:
        self._node = node
        self._tracer = tracer

    def __call__(self, *args: Any, **kwargs: Any) -> _Proxy:
        self._tracer.record(self._node)
        return _Proxy(
            source_node=self._node,
            name=f"out_of_{self._node.name}",
            tracer=self._tracer,
        )
