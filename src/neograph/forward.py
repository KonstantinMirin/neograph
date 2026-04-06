"""ForwardConstruct — class-based pipeline definition with forward() tracing.

    class MyPipeline(ForwardConstruct):
        extract = Node.scripted("extract", fn="extract_fn", outputs=RawText)
        classify = Node(mode="produce", outputs=Claims, prompt="rw/classify", model="fast")

        def forward(self, topic):
            raw = self.extract(topic)
            return self.classify(raw)

    graph = compile(MyPipeline())

ForwardConstruct discovers Node class attributes via MRO walk and populates
self.nodes by tracing forward() with symbolic proxies (torch.fx-style). The
resulting node list is identical to what a declarative Construct(nodes=[...])
produces, so compile() works unchanged.

Strategy: Symbolic Proxy — see `.claude/spikes/neograph-pub/design.md`.

Branching support (neograph-w5z):
    `if` branches in forward() are handled via the **re-trace** strategy
    (torch.fx pattern). The tracer intercepts Proxy.__bool__ to record
    branch points, then re-runs forward() with alternate branch decisions
    to discover both arms. The result is a node list annotated with
    _BranchMeta that the compiler lowers to add_conditional_edges.

    Design decision: re-trace (not AST inspection). Justification:
    - Simple, proven (torch.fx does the same thing)
    - Exponential cost (2^N traces) is acceptable for N <= 8 branches
    - Avoids AST walking complexity
    - Each trace is a normal Python execution — no special IR needed

    Limitations (v1):
    - Only comparisons against constants are supported (proxy.attr < 0.7)
    - Arbitrary expressions in conditions are deferred
    - Max 8 branches per forward() (raises ValueError beyond that)

try/except support (neograph-xi0, v1):
    try/except in forward() does NOT compile to a fallback graph. During
    tracing, proxy operations (node calls) never raise — they are symbolic
    recordings — so the except block is unreachable dead code. Only real
    Python errors (e.g., ``1/0``) before or between node calls can route
    tracing into the except block.

    Consequence: if both try and except arms call nodes, only the try-body
    nodes appear in the compiled graph. For retry/fallback patterns, use
    the declarative API or a future mechanism (see design.md, P3).
"""

from __future__ import annotations

import dataclasses
import operator as op_module
from typing import Any

from neograph.construct import Construct
from neograph.errors import ConstructError
from neograph.modifiers import Each
from neograph.node import Node

__all__ = ["ForwardConstruct"]

# Map comparison operator strings to callables for runtime evaluation
_OP_MAP = {
    "<": op_module.lt,
    "<=": op_module.le,
    ">": op_module.gt,
    ">=": op_module.ge,
    "==": op_module.eq,
    "!=": op_module.ne,
}

_MAX_BRANCHES = 8


class ForwardConstruct(Construct):
    """A Construct whose node list is discovered from class attributes.

    Subclass this, declare Node attributes at class level, and override
    forward() to define the execution order. The tracer populates self.nodes
    from the forward() call order so compile() works unchanged.

    Usage::

        class MyPipeline(ForwardConstruct):
            a = Node.scripted("a", fn="a_fn", outputs=RawText)
            b = Node.scripted("b", fn="b_fn", outputs=Claims)

            def forward(self, topic):
                raw = self.a(topic)
                return self.b(raw)

        pipeline = MyPipeline()
        graph = compile(pipeline)

    Note on try/except (v1):
        try/except blocks in forward() are valid Python and do not break
        tracing, but the except block is dead code during tracing because
        proxy operations never raise. Only the try-body nodes are recorded.
        try/except does not compile to a fallback graph in v1.
    """

    # Tell Pydantic to ignore Node instances as class attributes — they are
    # pipeline declarations, not model fields. Without this, Pydantic raises
    # "non-annotated attribute" errors for Node class attrs on subclasses.
    model_config = {"arbitrary_types_allowed": True, "ignored_types": (Node,)}

    def __init__(self, name_: str | None = None, /, **kwargs: Any) -> None:
        # Discover Node class attributes
        discovered = self._discover_node_attrs()

        if not discovered and type(self) is ForwardConstruct:
            msg = (
                "ForwardConstruct cannot be instantiated directly. "
                "Subclass it and declare Node attributes."
            )
            raise TypeError(msg)

        # Check that forward() is overridden
        if type(self).forward is ForwardConstruct.forward:
            msg = (
                f"{type(self).__name__} must override forward(). "
                "Define a forward() method that calls self.<node>(...) "
                "to specify execution order."
            )
            raise TypeError(msg)

        # Trace forward() to get nodes in call order
        traced_nodes = _trace_forward(self, discovered)

        # Default name from class name if not provided
        if name_ is None and "name" not in kwargs:
            kwargs["name"] = type(self).__name__

        # Pass traced nodes to Construct.__init__ (triggers _validate_node_chain)
        kwargs["nodes"] = traced_nodes
        super().__init__(name_, **kwargs)

    @classmethod
    def _discover_node_attrs(cls) -> dict[str, Node]:
        """Walk cls.__mro__ in reverse; return attr_name -> Node for every
        class attribute that isinstance(v, Node). Subclass attrs shadow base.

        Reverse MRO walk means root-to-leaf, so subclass values overwrite
        parent values — matching nn.Module parameter discovery semantics.
        """
        discovered: dict[str, Node] = {}
        for klass in reversed(cls.__mro__):
            for attr_name, attr_val in klass.__dict__.items():
                if isinstance(attr_val, Node):
                    discovered[attr_name] = attr_val
        return discovered

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Override this method to define the pipeline execution order.

        Call self.<node>(...) for each node in the desired order. The tracer
        records these calls and populates self.nodes.
        """
        raise NotImplementedError(
            "ForwardConstruct subclasses must override forward()"
        )


# ─────────────────────────── Tracing machinery ───────────────────────────


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
        source_node: Node | None,
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

    def __eq__(self, other: Any) -> _ConditionProxy:
        return _ConditionProxy(self, "==", other)

    def __ne__(self, other: Any) -> _ConditionProxy:
        return _ConditionProxy(self, "!=", other)

    def __hash__(self) -> int:
        # Required because we defined __eq__
        return id(self)

    def __bool__(self) -> bool:
        tracer = self._neo_tracer
        if tracer is None:
            raise TypeError(
                "Cannot use proxy in boolean context outside tracing"
            )
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
            raise TypeError(
                "Cannot use condition in boolean context outside tracing"
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
        op_fn = _OP_MAP[self._op]
        threshold = self._right

        # Parse the proxy name to extract state field + attribute chain
        # e.g., "out_of_br-check.score" → source "br-check", attrs ["score"]
        source_node = left._neo_source
        full_name = left._neo_name

        # Extract attribute chain after the proxy name prefix
        # The prefix is "out_of_<node_name>" for node outputs
        if source_node is not None:
            prefix = f"out_of_{source_node.name}"
            if full_name.startswith(prefix):
                remainder = full_name[len(prefix):]
                # remainder is like ".score" or ".items.first.severity"
                attr_chain = [p for p in remainder.split(".") if p]
            else:
                attr_chain = []
        else:
            attr_chain = []

        return _ConditionSpec(
            source_node=source_node,
            attr_chain=attr_chain,
            op_fn=op_fn,
            op_str=self._op,
            threshold=threshold,
        )


@dataclasses.dataclass(frozen=True)
class _ConditionSpec:
    """Parsed condition specification for compiler lowering."""
    source_node: Node | None
    attr_chain: list[str]
    op_fn: Any  # operator callable
    op_str: str  # e.g., "<", ">"
    threshold: Any  # right-hand side constant


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
    true_nodes: list[Node]
    false_nodes: list[Node]


@dataclasses.dataclass
class _BranchMeta:
    """Branch metadata attached to the node list for compiler consumption.

    This is stored on a sentinel _BranchNode that the compiler recognizes
    and lowers to add_conditional_edges.
    """
    condition_spec: _ConditionSpec
    # Nodes that only appear in the true arm
    true_arm_nodes: list[Node]
    # Nodes that only appear in the false arm
    false_arm_nodes: list[Node]


class _BranchNode:
    """Sentinel that carries _BranchMeta in the node list.

    The compiler checks for this type and wires conditional edges instead
    of adding a regular node. It carries a synthetic name for graph wiring.
    """

    def __init__(self, branch_meta: _BranchMeta, branch_id: int) -> None:
        self._neo_branch_meta = branch_meta
        self.name = f"__branch_{branch_id}"
        # Satisfy Construct validation: pretend to be a Node-like
        self.modifiers = []
        self.output = None

    def has_modifier(self, mod_type: type) -> bool:
        return False

    def get_modifier(self, mod_type: type) -> None:
        return None


class _Tracer:
    """Collects node invocations during a single forward() trace run.

    Deduplicates by identity (id(node)): repeated calls to the same Node
    instance collapse into one entry. Supports branch recording for if/else
    tracing via the re-trace strategy.
    """

    def __init__(self, branch_decisions: dict[int, bool] | None = None) -> None:
        self._ordered: list[Node] = []
        self._seen: set[int] = set()
        self._branches: list[_BranchPoint] = []
        self._branch_decisions = branch_decisions or {}
        self._next_branch_id = 0
        # Loop-mode tracking: maps node id → Each over-path for nodes in loop body
        self._loop_stack: list[str] = []  # stack of over-paths (for nested detection)
        self._loop_body_nodes: dict[int, str] = {}  # id(node) → over-path

    def record(self, node: Node) -> None:
        key = id(node)
        if key in self._seen:
            return
        self._seen.add(key)
        self._ordered.append(node)
        # If we're inside a loop body, tag this node for Each wrapping
        if self._loop_stack:
            self._loop_body_nodes[key] = self._loop_stack[-1]

    def record_iteration(self, proxy: _Proxy) -> iter:
        """Record a for-loop iteration over a proxy attribute.

        Enters loop mode: nodes recorded while in this mode get tagged
        with an Each modifier. Yields a single proxy item (enough for
        tracing — the loop body runs once to discover node calls).
        """
        # Build the Each over-path from the proxy's attribute chain
        # e.g., _Proxy(source=make, name="out_of_make.groups") → "make.groups"
        source_node = proxy._neo_source
        full_name = proxy._neo_name

        if source_node is not None:
            prefix = f"out_of_{source_node.name}"
            field_name = source_node.name.replace("-", "_")
            if full_name.startswith(prefix):
                remainder = full_name[len(prefix):]
                attr_parts = [p for p in remainder.split(".") if p]
                over_path = ".".join([field_name] + attr_parts)
            else:
                over_path = field_name
        else:
            over_path = full_name

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
            msg = (
                f"Too many branches in forward(): {branch_id + 1} exceeds "
                f"the limit of {_MAX_BRANCHES}. Simplify your forward() or "
                "extract sub-pipelines."
            )
            raise ConstructError(msg)
        self._next_branch_id += 1

        if branch_id in self._branch_decisions:
            decision = self._branch_decisions[branch_id]
        else:
            decision = True

        self._branches.append(_BranchPoint(
            branch_id=branch_id,
            condition=condition,
            decision=decision,
        ))
        return decision

    @property
    def nodes(self) -> list[Node]:
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


class _ForwardSelf:
    """Replacement self used during tracing.

    Attribute lookups for known node names return _NodeCall recording
    callables. Any other attribute falls through to the real instance,
    so forward() can still read user-defined helpers and constants.
    """

    def __init__(
        self,
        node_attrs: dict[str, Node],
        tracer: _Tracer,
        real_self: ForwardConstruct,
    ) -> None:
        object.__setattr__(self, "_neo_nodes", node_attrs)
        object.__setattr__(self, "_neo_tracer", tracer)
        object.__setattr__(self, "_neo_real", real_self)

    def __getattr__(self, name: str) -> Any:
        node_attrs: dict[str, Node] = object.__getattribute__(self, "_neo_nodes")
        if name in node_attrs:
            tracer: _Tracer = object.__getattribute__(self, "_neo_tracer")
            return _NodeCall(node_attrs[name], tracer)
        real: ForwardConstruct = object.__getattribute__(self, "_neo_real")
        return getattr(real, name)

    def __setattr__(self, name: str, value: Any) -> None:
        real: ForwardConstruct = object.__getattribute__(self, "_neo_real")
        setattr(real, name, value)


def _run_trace(
    instance: ForwardConstruct,
    node_attrs: dict[str, Node],
    branch_decisions: dict[int, bool] | None = None,
) -> tuple[_Tracer, list[Node]]:
    """Run a single trace pass of forward() and return (tracer, nodes)."""
    tracer = _Tracer(branch_decisions=branch_decisions)
    shim = _ForwardSelf(node_attrs, tracer, real_self=instance)
    seed = _Proxy(source_node=None, name="forward_input", tracer=tracer)
    type(instance).forward(shim, seed)
    nodes = _apply_loop_modifiers(tracer)
    return tracer, nodes


def _apply_loop_modifiers(tracer: _Tracer) -> list[Node]:
    """Replace loop-body nodes with Each-modified copies.

    Nodes recorded during a for-loop iteration over a proxy get an
    Each modifier attached. Non-loop nodes pass through unchanged.
    """
    if not tracer._loop_body_nodes:
        return tracer.nodes

    result = []
    for node in tracer._ordered:
        key = id(node)
        if key in tracer._loop_body_nodes:
            over_path = tracer._loop_body_nodes[key]
            node = node | Each(over=over_path, key="label")
        result.append(node)
    return result


def _trace_forward(
    instance: ForwardConstruct,
    node_attrs: dict[str, Node],
) -> list[Node]:
    """Trace forward() to discover node call order.

    For straight-line pipelines (no if/else), returns nodes in call order.

    For branching pipelines, uses the re-trace strategy:
    1. First trace: all branches take True arm → discover true-arm nodes
    2. For each branch: re-trace with that branch flipped to False
    3. Diff traces to identify true-only and false-only nodes
    4. Build _BranchNode sentinels that the compiler lowers to conditional edges
    """
    # Pass 1: all branches True (default)
    true_tracer, true_nodes = _run_trace(instance, node_attrs)
    branches = true_tracer.branches

    if not branches:
        # Straight-line pipeline — no branches, return as-is
        return true_nodes

    # Re-trace for each branch with that branch flipped to False
    branch_traces: list[_BranchTrace] = []
    for branch in branches:
        false_tracer, false_nodes = _run_trace(
            instance, node_attrs,
            branch_decisions={branch.branch_id: False},
        )
        branch_traces.append(_BranchTrace(
            branch=branch,
            true_nodes=true_nodes,
            false_nodes=false_nodes,
        ))

    return _merge_branch_traces(true_nodes, branch_traces, branches)


def _merge_branch_traces(
    true_nodes: list[Node],
    branch_traces: list[_BranchTrace],
    branches: list[_BranchPoint],
) -> list:
    """Merge true and false trace results into a node list with branch metadata.

    For each branch, identifies:
    - Shared prefix: nodes that appear in both traces (before the divergence)
    - True-only nodes: nodes unique to the true arm
    - False-only nodes: nodes unique to the false arm
    - Shared suffix: nodes that appear in both traces (after the convergence)

    Returns a flat list containing Node instances for shared nodes and
    _BranchNode sentinels for branches. The compiler recognizes _BranchNode
    and emits conditional edges.
    """
    if len(branch_traces) == 1:
        return _merge_single_branch(
            branch_traces[0], branches[0],
        )

    # Multiple sequential branches: merge one at a time
    # Each branch splits the linear flow; we process them in order
    return _merge_sequential_branches(branch_traces, branches)


def _merge_single_branch(
    trace: _BranchTrace,
    branch: _BranchPoint,
) -> list:
    """Merge a single branch into a node list with a _BranchNode sentinel."""
    true_names = [n.name for n in trace.true_nodes]
    false_names = [n.name for n in trace.false_nodes]

    true_set = set(true_names)
    false_set = set(false_names)

    # Find shared prefix (nodes before the branch point)
    shared_prefix = []
    for node in trace.true_nodes:
        if node.name in false_set:
            shared_prefix.append(node)
        else:
            break

    prefix_names = {n.name for n in shared_prefix}

    # True-only and false-only nodes
    true_only = [n for n in trace.true_nodes if n.name not in false_set]
    false_only = [n for n in trace.false_nodes if n.name not in true_set]

    # Build condition spec from the branch's condition
    condition = branch.condition
    if isinstance(condition, _ConditionProxy):
        cond_spec = condition._build_runtime_condition()
    else:
        # Plain proxy used as bool — less common, create a truthy-check spec
        cond_spec = _ConditionSpec(
            source_node=condition._neo_source,
            attr_chain=[],
            op_fn=op_module.truth,
            op_str="truthy",
            threshold=None,
        )

    branch_meta = _BranchMeta(
        condition_spec=cond_spec,
        true_arm_nodes=true_only,
        false_arm_nodes=false_only,
    )

    # Shared suffix: nodes in both traces that aren't in prefix or branch arms
    branch_arm_names = {n.name for n in true_only} | {n.name for n in false_only}
    shared_suffix = [
        n for n in trace.true_nodes
        if n.name not in prefix_names and n.name not in branch_arm_names
    ]

    # Build the result: prefix + branch sentinel + suffix
    result: list = list(shared_prefix)
    result.append(_BranchNode(branch_meta, branch.branch_id))
    result.extend(shared_suffix)
    return result


def _merge_sequential_branches(
    branch_traces: list[_BranchTrace],
    branches: list[_BranchPoint],
) -> list:
    """Merge multiple sequential branches.

    For sequential branches (not nested), each branch adds a _BranchNode
    sentinel at the appropriate position in the node list.
    """
    # Use the first (all-true) trace as the base ordering
    base_nodes = branch_traces[0].true_nodes

    # For each branch, compute its true-only and false-only nodes
    result: list = []
    processed_names: set[str] = set()

    for i, (trace, branch) in enumerate(zip(branch_traces, branches)):
        true_names = {n.name for n in trace.true_nodes}
        false_names = {n.name for n in trace.false_nodes}

        # True-only nodes for this branch
        true_only = [n for n in trace.true_nodes if n.name not in false_names]
        false_only = [n for n in trace.false_nodes if n.name not in true_names]

        # Build condition spec
        condition = branch.condition
        if isinstance(condition, _ConditionProxy):
            cond_spec = condition._build_runtime_condition()
        else:
            cond_spec = _ConditionSpec(
                source_node=condition._neo_source,
                attr_chain=[],
                op_fn=op_module.truth,
                op_str="truthy",
                threshold=None,
            )

        branch_meta = _BranchMeta(
            condition_spec=cond_spec,
            true_arm_nodes=true_only,
            false_arm_nodes=false_only,
        )

        # Add shared nodes before this branch's divergence point
        for node in base_nodes:
            if node.name in processed_names:
                continue
            if node.name in {n.name for n in true_only}:
                # Hit the divergence — insert branch sentinel
                result.append(_BranchNode(branch_meta, branch.branch_id))
                processed_names.update(n.name for n in true_only)
                processed_names.update(n.name for n in false_only)
                break
            result.append(node)
            processed_names.add(node.name)
        else:
            # Branch divergence not found in remaining base nodes — append sentinel
            result.append(_BranchNode(branch_meta, branch.branch_id))
            processed_names.update(n.name for n in true_only)
            processed_names.update(n.name for n in false_only)

    # Add any remaining shared nodes from the base
    for node in base_nodes:
        if node.name not in processed_names:
            result.append(node)
            processed_names.add(node.name)

    return result
