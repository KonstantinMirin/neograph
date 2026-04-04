"""Prototype: Construct.forward() tracing via symbolic proxies.

Throwaway proof-of-concept. Proves that a class with `forward()` containing
straight-line Python calls against `self.<node>(...)` attributes can be
*traced* (without executing the nodes) to produce a nodes list equivalent to
the declarative `Construct(nodes=[decompose, classify])`.

NOT production code. Lives only under .claude/spikes/. No src/neograph/
files are touched by this spike.

Strategy: Symbolic Proxy (torch.fx-style).
    1. Walk `cls.__dict__` to discover class attributes that are Node instances.
    2. Bind `self.<node_name>` to a proxy callable that, when called, records
       a trace entry and returns a `Proxy` whose `._neo_node_name` identifies
       the producer.
    3. Invoke `forward(proxy_for_initial_input)` once at compile time.
    4. Read the trace; each entry is a Node invocation in the order it was
       called. De-duplicate by identity to produce the linear nodes list.

MVP scope: straight-line only. No branches, no loops, no tries.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Make the real neograph package importable without installing the spike.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pydantic import BaseModel  # noqa: E402

from neograph.node import Node  # noqa: E402


# ─────────────────────────── Tracing machinery ───────────────────────────


class Proxy:
    """A stand-in for a real value during forward() tracing.

    Carries the Node that produced it (or None for the initial input). Every
    real-world Python op against a Proxy — attribute access, comparison,
    subscript, iteration — is intercepted by the tracer. For the MVP we only
    need proxies to flow through calls; deeper interception (for branch
    support) is a follow-up subtask.
    """

    def __init__(self, source_node: Node | None, name: str):
        self._neo_source = source_node
        self._neo_name = name

    def __repr__(self) -> str:  # debug aid
        src = self._neo_source.name if self._neo_source else "<input>"
        return f"Proxy(from={src}, name={self._neo_name})"


class _NodeCall:
    """A callable wrapper that records invocations into the active Tracer."""

    def __init__(self, node: Node, tracer: "Tracer"):
        self._node = node
        self._tracer = tracer

    def __call__(self, *args: Any, **kwargs: Any) -> Proxy:
        # Record the call as an edge in the trace. Args are ignored for the
        # straight-line MVP — name-based edge order is sufficient to emit a
        # linear Construct. Argument-to-producer tracking is a follow-up
        # needed for fan-in DAGs (subtask: forward-fan-in).
        self._tracer.record(self._node)
        return Proxy(source_node=self._node, name=f"out_of_{self._node.name}")


class Tracer:
    """Collects node invocations during a single forward() trace run."""

    def __init__(self) -> None:
        self._ordered: list[Node] = []
        self._seen: set[int] = set()  # by id(node) to preserve instance identity

    def record(self, node: Node) -> None:
        key = id(node)
        if key in self._seen:
            # Loops/rebinding would legitimately call the same node twice.
            # Out of scope for MVP — a future "loop support" subtask needs
            # to decide whether repeated calls unroll or collapse.
            return
        self._seen.add(key)
        self._ordered.append(node)

    @property
    def nodes(self) -> list[Node]:
        return list(self._ordered)


# ─────────────────────── ForwardConstruct base class ──────────────────────


class ForwardConstruct:
    """Minimal stand-in for the real Construct — just enough to validate the
    tracing strategy. The real integration subtask will subclass
    `neograph.construct.Construct` and populate `nodes` from the trace so the
    existing compile() pipeline consumes the result unchanged.
    """

    name: str = "forward-pipeline"

    def __init__(self) -> None:
        self.nodes: list[Node] = self._trace()

    # Subclass override.
    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401
        raise NotImplementedError

    # ─── tracing ───────────────────────────────────────────────────────────

    @classmethod
    def _discover_node_attrs(cls) -> dict[str, Node]:
        """Walk the MRO and collect class attributes that are Node instances.

        Walking the MRO (not just `cls.__dict__`) lets users inherit nodes
        from a base ForwardConstruct — mirrors how nn.Module gathers params.
        Subclass attributes shadow parents, which `dict.setdefault` in reverse
        MRO order gives us for free.
        """
        discovered: dict[str, Node] = {}
        for klass in reversed(cls.__mro__):
            for attr_name, attr_val in klass.__dict__.items():
                if isinstance(attr_val, Node):
                    discovered[attr_name] = attr_val
        return discovered

    def _trace(self) -> list[Node]:
        node_attrs = self._discover_node_attrs()
        tracer = Tracer()

        # Build a stand-in `self` shim: every discovered node becomes a
        # recording callable. The shim is what the user's forward() sees
        # when it does `self.decompose(...)`. Using a shim (instead of
        # rebinding on `self`) keeps the real instance attrs untouched for
        # unit-test use cases where the user calls forward() for real.
        shim = _ForwardSelf(node_attrs, tracer, real_self=self)

        # Seed the call with a single proxy input. Multi-arg forward() is
        # a cosmetic extension — pass one proxy per positional arg.
        seed = Proxy(source_node=None, name="forward_input")
        self.__class__.forward(shim, seed)  # unbound call, `shim` plays self

        return tracer.nodes


class _ForwardSelf:
    """A replacement `self` used during tracing.

    Attribute lookups for known node names return recording callables. Any
    other attribute falls through to the real instance, so forward() can
    still read user-defined helpers/constants. Writes fall through too so
    the user's forward() can set bookkeeping attrs without polluting the
    trace.
    """

    def __init__(
        self,
        node_attrs: dict[str, Node],
        tracer: Tracer,
        real_self: ForwardConstruct,
    ) -> None:
        object.__setattr__(self, "_neo_nodes", node_attrs)
        object.__setattr__(self, "_neo_tracer", tracer)
        object.__setattr__(self, "_neo_real", real_self)

    def __getattr__(self, name: str) -> Any:
        node_attrs: dict[str, Node] = object.__getattribute__(self, "_neo_nodes")
        if name in node_attrs:
            tracer: Tracer = object.__getattribute__(self, "_neo_tracer")
            return _NodeCall(node_attrs[name], tracer)
        real: ForwardConstruct = object.__getattribute__(self, "_neo_real")
        return getattr(real, name)

    def __setattr__(self, name: str, value: Any) -> None:
        real: ForwardConstruct = object.__getattribute__(self, "_neo_real")
        setattr(real, name, value)


# ─────────────────────────────── Demo use ────────────────────────────────


class Claims(BaseModel):
    items: list[str] = []


class Classified(BaseModel):
    confidence: float = 0.0


class Pipeline(ForwardConstruct):
    decompose = Node.scripted("decompose", fn="decompose_fn", output=Claims)
    classify = Node.scripted(
        "classify", fn="classify_fn", input=Claims, output=Classified
    )

    def forward(self, topic):  # type: ignore[override]
        claims = self.decompose(topic)
        classified = self.classify(claims)
        return classified


if __name__ == "__main__":
    p = Pipeline()
    names = [n.name for n in p.nodes]
    print(f"traced nodes: {names}")
    assert names == ["decompose", "classify"], f"got {names!r}"
    assert p.nodes[0] is Pipeline.decompose
    assert p.nodes[1] is Pipeline.classify
    print("PASS: forward() traced to equivalent of Construct(nodes=[decompose, classify])")
