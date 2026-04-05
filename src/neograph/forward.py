"""ForwardConstruct — class-based pipeline definition with forward() tracing.

    class MyPipeline(ForwardConstruct):
        extract = Node.scripted("extract", fn="extract_fn", output=RawText)
        classify = Node(mode="produce", output=Claims, prompt="rw/classify", model="fast")

        def forward(self, topic):
            raw = self.extract(topic)
            return self.classify(raw)

    graph = compile(MyPipeline())

ForwardConstruct discovers Node class attributes via MRO walk and populates
self.nodes by tracing forward() with symbolic proxies (torch.fx-style). The
resulting node list is identical to what a declarative Construct(nodes=[...])
produces, so compile() works unchanged.

Strategy: Symbolic Proxy — see `.claude/spikes/neograph-pub/design.md`.

Straight-line only for v1. Branching (__bool__), looping (__iter__), and
argument-to-producer wiring are future subtasks.
"""

from __future__ import annotations

from typing import Any

from neograph.construct import Construct
from neograph.node import Node

__all__ = ["ForwardConstruct"]


class ForwardConstruct(Construct):
    """A Construct whose node list is discovered from class attributes.

    Subclass this, declare Node attributes at class level, and override
    forward() to define the execution order. The tracer populates self.nodes
    from the forward() call order so compile() works unchanged.

    Usage::

        class MyPipeline(ForwardConstruct):
            a = Node.scripted("a", fn="a_fn", output=RawText)
            b = Node.scripted("b", fn="b_fn", output=Claims)

            def forward(self, topic):
                raw = self.a(topic)
                return self.b(raw)

        pipeline = MyPipeline()
        graph = compile(pipeline)
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

    Carries the Node that produced it (or None for the initial input). For
    the straight-line MVP, proxies only need to flow through calls — deeper
    interception (__bool__, __getattr__, __iter__) lands with branching and
    looping subtasks.
    """

    __slots__ = ("_neo_source", "_neo_name")

    def __init__(self, source_node: Node | None, name: str) -> None:
        self._neo_source = source_node
        self._neo_name = name

    def __repr__(self) -> str:
        src = self._neo_source.name if self._neo_source else "<input>"
        return f"_Proxy(from={src}, name={self._neo_name})"


class _Tracer:
    """Collects node invocations during a single forward() trace run.

    Deduplicates by identity (id(node)): repeated calls to the same Node
    instance collapse into one entry. Loop unrolling is a future subtask —
    for the straight-line MVP this is the correct behavior.
    """

    def __init__(self) -> None:
        self._ordered: list[Node] = []
        self._seen: set[int] = set()

    def record(self, node: Node) -> None:
        key = id(node)
        if key in self._seen:
            return
        self._seen.add(key)
        self._ordered.append(node)

    @property
    def nodes(self) -> list[Node]:
        return list(self._ordered)


class _NodeCall:
    """Callable wrapper that records a node invocation into the active Tracer."""

    __slots__ = ("_node", "_tracer")

    def __init__(self, node: Node, tracer: _Tracer) -> None:
        self._node = node
        self._tracer = tracer

    def __call__(self, *args: Any, **kwargs: Any) -> _Proxy:
        self._tracer.record(self._node)
        return _Proxy(source_node=self._node, name=f"out_of_{self._node.name}")


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


def _trace_forward(
    instance: ForwardConstruct,
    node_attrs: dict[str, Node],
) -> list[Node]:
    """Trace forward() to discover node call order.

    Builds a _ForwardSelf shim, invokes forward() with a seed _Proxy,
    and returns the nodes in the order they were called.
    """
    tracer = _Tracer()
    shim = _ForwardSelf(node_attrs, tracer, real_self=instance)

    # Seed the call with a single proxy input
    seed = _Proxy(source_node=None, name="forward_input")

    # Unbound call: shim plays self
    type(instance).forward(shim, seed)

    return tracer.nodes
