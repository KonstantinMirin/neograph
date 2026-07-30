"""ForwardConstruct — class-based pipeline definition with forward() tracing.

    class MyPipeline(ForwardConstruct):
        extract = Node.scripted("extract", fn="extract_fn", outputs=RawText)
        classify = Node(mode="think", outputs=Claims, prompt="rw/classify", model="fast")

        def forward(self, topic):
            raw = self.extract(topic)
            return self.classify(raw)

    graph = compile(MyPipeline())

ForwardConstruct discovers Node class attributes via MRO walk and populates
self.nodes by tracing forward() with symbolic proxies (torch.fx-style). The
resulting node list is identical to what a declarative Construct(nodes=[...])
produces, so compile() works unchanged.

Strategy: Symbolic Proxy — see `.claude/spikes/neograph-pub/design.md`.

Branching support:
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

    Limitations (v1 — CAPPED, neograph-e9zse.7; the declarative Construct
    form is the escape for anything richer):
    - Only comparisons against constants are supported (proxy.attr < 0.7).
      A proxy right-hand side raises ConstructError at trace time.
    - Arbitrary expressions in conditions are deferred
    - Max 8 branches per forward() (raises ConstructError beyond that)

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

# --- names forward.py imported and RE-EXPORTED before the split. The moved code
# --- was their only consumer here, so ruff sees them as unused -- but tests and
# --- downstream code import them from neograph.forward, so the surface must hold.
import dataclasses  # noqa: E402,F401
import operator as op_module  # noqa: E402,F401
from collections.abc import Iterator  # noqa: E402,F401
from typing import Any, Literal, cast

from pydantic import ConfigDict

from neograph._construct_validation import (
    _MISSING,
    _extract_list_element,
    _resolve_field_annotation,
    effective_producer_type,
    effective_producer_type_for,
)

# --- extracted clusters (neograph-3ffdg.12), re-exported so existing
# --- `from neograph.forward import ...` call sites keep resolving unchanged.
from neograph._forward_proxy import (  # noqa: E402,F401
    _MAX_BRANCHES,
    _attr_chain_after_prefix,
    _BranchPoint,
    _BranchTrace,
    _ConditionProxy,
    _NodeCall,
    _over_path_for_proxy,
    _primary_type,
    _Proxy,
    _Tracer,
)
from neograph._forward_trace import (  # noqa: E402,F401
    _apply_loop_modifiers,
    _build_condition_spec,
    _merge_branch_traces,
    _merge_sequential_branches,
    _merge_single_branch,
    _run_trace,
    _trace_forward,
)
from neograph._ir_branch import (  # noqa: E402,F401
    _BranchMeta,
    _BranchNode,
    _ConditionSpec,
)
from neograph._normalize import _declared_output, normalize_outputs  # noqa: E402,F401
from neograph._state_keys import StateKeys
from neograph.conditions import OPERATORS  # noqa: E402,F401
from neograph.construct import Construct
from neograph.errors import ConstructError
from neograph.modifiers import Each, Loop, Operator, Oracle, Portal
from neograph.naming import field_name_for
from neograph.node import Node, TypeSpecStatic

__all__ = ["ForwardConstruct"]


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

    # arbitrary_types_allowed: inherited need from Construct (``nodes`` holds
    # the ``_BranchNode`` sentinel, ``renderer`` is a Protocol). Also required
    # so the ``ignored_types=(Node,)`` config can keep Node class attributes —
    # pipeline declarations, not model fields — without Pydantic raising
    # "non-annotated attribute" errors on ForwardConstruct subclasses.
    model_config = ConfigDict(arbitrary_types_allowed=True, ignored_types=(Node,))

    def __init__(self, name_: str | None = None, /, **kwargs: Any) -> None:
        # Discover Node class attributes
        discovered = self._discover_node_attrs()

        if not discovered and type(self) is ForwardConstruct:
            raise ConstructError.build(
                "ForwardConstruct cannot be instantiated directly",
                expected="a subclass that declares Node attributes",
                found="bare ForwardConstruct()",
                hint="subclass ForwardConstruct and declare Node attributes on the class",
            )

        # Check that forward() is overridden
        if type(self).forward is ForwardConstruct.forward:
            raise ConstructError.build(
                f"{type(self).__name__} must override forward()",
                expected="a forward() method that calls self.<node>(...) to specify execution order",
                found="no forward() override",
                hint="define forward(self) on your subclass",
            )

        # Trace forward() to get nodes in call order
        # _ForwardSelf stays here (its builder factories construct the DX classes
        # that also stay), so the trace module receives it rather than importing it.
        traced_nodes = _trace_forward(self, discovered, shim_factory=_ForwardSelf)

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

        Deliberately EXEMPT from `_member_select._classify_member` (the
        xv9ay membership-predicate monopoly, evaluated under neograph-gtzkd):
        this walk is a name -> Node lookup table feeding the tracer, not a
        membership classifier — pipeline membership is fixed by forward()
        tracing, and sub-pipelines enter a trace via self.each() /
        self.loop() / self.ensemble(), never as class attrs. The exemption
        is safe only while a Construct class attribute fails LOUD at
        class-definition time (PydanticUserError via the model_config
        ``ignored_types=(Node,)`` gate above) — pinned by
        tests/test_forward.py::TestConstructClassAttrFailsLoud.
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
        raise NotImplementedError("ForwardConstruct subclasses must override forward()")


# ─────────────────────────── Tracing machinery ───────────────────────────


def _declared_primary_of_body_item(item: Any) -> TypeSpecStatic:
    """Declared primary output of a deferred body item, for fallback type
    inference — a node ref's declared primary, or (recursively) the last
    body item's primary for deferred self.each()/self.loop() builders."""
    if isinstance(item, _NodeCall):
        return _primary_type(item._node)
    if isinstance(item, (_EachCall, _LoopCall)):
        return _declared_primary_of_body_item(item._body[-1]) if item._body else None
    if isinstance(item, _ModifierWrapCall):
        target = item._target
        if isinstance(target, _NodeCall):
            return _primary_type(target._node)
        return _declared_primary_of_body_item(target[-1]) if target else None
    return None


class _LoopCall:
    """Returned by _ForwardSelf.loop(). When called with a proxy, builds a
    sub-construct with Loop modifier and records it in the tracer.

    Usage in forward()::

        d = self.loop(
            body=[self.review, self.revise],
            when=lambda r: r.score < 0.8,
            max_iterations=5,
        )(d)
    """

    def __init__(
        self,
        body: list[_NodeCall],
        when: Any,
        max_iterations: int,
        on_exhaust: str,
        tracer: _Tracer,
    ) -> None:
        self._body = body
        self._when = when
        self._max_iterations = max_iterations
        self._on_exhaust = on_exhaust
        self._tracer = tracer

    def __call__(self, input_proxy: _Proxy) -> _Proxy:
        self._validate_body_kinds()

        # Infer the loop's input port from the proxy source. source_node can
        # be None when the proxy comes from a previous self.loop() or a
        # branch — fall back to the last body member's declared output so
        # the construct compiles. Input and output can differ
        # (produce+validate pattern, neograph-vt4y).
        source_node = input_proxy._neo_source
        if source_node is not None:
            input_type = _primary_type(source_node)
        else:
            input_type = _declared_primary_of_body_item(self._body[-1]) if self._body else None

        if input_type is None:
            raise ConstructError.build(
                "self.loop() input type could not be inferred",
                hint="call self.loop()(proxy) where proxy is a node output",
            )

        sub = self._materialize(input_type)

        # Record the sub-construct in the tracer (it appears in the node list)
        self._tracer.record_construct(sub)

        # Return a proxy whose source is the sub-construct so downstream
        # loops/branches can infer input types from it.
        return _Proxy(
            source_node=sub,
            name=f"out_of_{sub.name}",
            tracer=self._tracer,
        )

    def _materialize_nested(
        self,
        preceding: list[Node | Construct],
        outer_input_type: TypeSpecStatic,
    ) -> Construct:
        """Build this loop as a NESTED body member of an enclosing loop.

        The nested loop's input port is the previous materialized member's
        declared output (or the outer loop's port when first). Never records
        into the tracer — the enclosing construct owns the placement.
        """
        if preceding:
            input_type = _primary_type(preceding[-1])
        else:
            input_type = outer_input_type
        if input_type is None:
            raise ConstructError.build(
                "nested self.loop() input type could not be inferred",
                hint="ensure the preceding loop-body member declares an output type",
            )
        return self._materialize(input_type)

    def _validate_body_kinds(self) -> None:
        """Fail loud on an empty body or a member that is neither a node
        reference nor a deferred self.each()/self.loop() builder — BEFORE
        any type inference, so the author sees the body error first."""
        if not self._body:
            raise ConstructError.build(
                "self.loop() body must contain at least one node",
                hint="pass at least one node reference: self.loop(body=[self.some_node], ...)",
            )
        for nc in self._body:
            if not isinstance(nc, (_NodeCall, _EachCall, _LoopCall, _ModifierWrapCall)):
                raise ConstructError.build(
                    "self.loop() body must contain node references (self.node_name)",
                    expected="node reference (self.node_name) or a deferred "
                    "self.each()/self.loop()/self.ensemble()/self.interrupt() builder",
                    found=type(nc).__name__,
                )

    def _materialize(self, input_type: TypeSpecStatic) -> Construct:
        """Build the ``Construct | Loop`` sub-construct WITHOUT recording it.

        Body members may be node references or deferred ``self.each()`` /
        ``self.loop()`` builders — ``Construct(nodes=...)`` already accepts
        ``list[Node | Construct]``, so the tracer must not be stricter than
        the IR it targets (neograph-e9zse.2).
        """
        self._validate_body_kinds()

        # Materialize members in order. Node members that PRECEDE the first
        # sub-construct member consume the loop port and get inputs filled
        # when None (copy-not-mutate, neograph-2o9n). A None-inputs node
        # AFTER a sub-construct member consumes that construct's state field,
        # not the port — a blanket fill would silently mis-wire, so fail loud.
        members: list[Node | Construct] = []
        seen_construct = False
        for nc in self._body:
            if isinstance(nc, _NodeCall):
                bn = nc._node
                if bn.inputs is None:
                    if seen_construct:
                        raise ConstructError.build(
                            f"self.loop() body node '{bn.name}' follows a sub-construct member and must declare inputs",
                            expected="an explicit inputs= declaration",
                            found="inputs=None",
                            hint="declare inputs= on the node (e.g. "
                            "inputs={'each_verify': dict[str, X]}) or name its "
                            "@node parameter after the sub-construct's state field",
                        )
                    bn = bn.model_copy(update={"inputs": input_type})
                members.append(bn)
            else:
                # Deferred self.each()/self.loop()/self.ensemble() builder —
                # kinds are guaranteed by _validate_body_kinds above. A
                # node-form ensemble materializes as a bare Node | Oracle
                # member (form-aware nesting), so gate the inputs-fill rule
                # on what actually materialized, not on the builder kind.
                materialized = nc._materialize_nested(members, input_type)
                members.append(materialized)
                seen_construct = seen_construct or isinstance(materialized, Construct)

        # Boundary ports are single types; a dict-form declared output collapses
        # to its primary key (secondary outputs are not the loop value).
        # _declared_output abstracts the Node.outputs / Construct.output split.
        output_type = _primary_type(members[-1])

        # A fanned-out terminal member writes dict[str, X] to the bus, which
        # can never satisfy the output-boundary contract (X). The assembly
        # validator would also reject it; failing here gives the actionable
        # hint. effective_producer_type_for is the single source of truth
        # for modifier-aware type effects — never inline modifier checks.
        effective_last = effective_producer_type_for(output_type, getattr(members[-1], "modifier_set", None))
        if effective_last is not output_type:
            raise ConstructError.build(
                "self.loop() body cannot end with a fanned-out member",
                expected="a terminal member producing the loop's output type",
                found=f"'{members[-1].name}' writes {effective_last} to the state bus",
                hint="add a collector node consuming the fan-out dict "
                "(e.g. inputs={'each_x': dict[str, X]}) after self.each(...)",
            )

        # Name is deterministic from member names so re-trace (branch
        # discovery) produces identical names across passes. A per-tracer
        # (kind, slug) occurrence counter disambiguates duplicates.
        body_slug = "-".join(m.name for m in members)
        occurrence = self._tracer.next_occurrence("loop", body_slug)
        name = f"loop-{body_slug}" if occurrence == 0 else f"loop-{body_slug}-{occurrence}"

        return Construct(
            name=name,
            input=input_type,
            output=output_type,
            nodes=members,
        ) | Loop(
            when=self._when,
            max_iterations=self._max_iterations,
            on_exhaust=self._on_exhaust,
        )


class _EachCall:
    """Returned by _ForwardSelf.each(). When called with a proxy attribute
    (or a raw dotted state path), builds a sub-construct with an Each
    modifier and records it in the tracer — IR-identical to the declarative
    ``Construct(input=item, output=..., nodes=body) | Each(over, key)`` twin.

    Usage in forward()::

        each_verify = self.each(body=[self.verify], key="item_id")
        results = each_verify(items.claims)      # proxy form
        results = each_verify("seed.claims")     # raw-string form
    """

    def __init__(
        self,
        body: list[_NodeCall],
        key: str,
        on_error: str,
        tracer: _Tracer,
        over: str | None = None,
    ) -> None:
        self._body = body
        self._key = key
        self._on_error = on_error
        self._tracer = tracer
        self._over = over

    def __call__(self, over: _Proxy | str | None = None) -> _Proxy:
        if self._over is not None and over is not None:
            raise ConstructError.build(
                "self.each() over is bound twice",
                expected="over= at construction OR a call argument, not both",
                found=f"construction over='{self._over}' and a call argument",
            )
        effective_over = over if over is not None else self._over
        if effective_over is None:
            raise ConstructError.build(
                "self.each() requires an over binding",
                hint="call each_x(proxy.attr) or construct with self.each(..., over='seed.items')",
            )

        body_nodes = self._validated_body_nodes()
        source_node, attr_parts, over_path = self._resolve_over(effective_over)
        item_type = self._infer_item_type(source_node, attr_parts, over_path)
        sub = self._build(body_nodes, item_type, over_path)

        self._tracer.record_construct(sub)

        return _Proxy(
            source_node=sub,
            name=f"out_of_{sub.name}",
            tracer=self._tracer,
        )

    def _materialize_nested(
        self,
        preceding: list[Node | Construct],
        outer_input_type: TypeSpecStatic,
    ) -> Construct:
        """Build this each as a NESTED body member of an enclosing loop.

        A nested deferred each is never __call__'d, so its over path must be
        the construction-time raw string. The root resolves against the two
        forms the assembly validator accepts inside a sub-construct: a
        preceding body member's state field name (root type =
        effective_producer_type, the modifier-aware single source of truth)
        or ``neo_subgraph_input`` (the enclosing loop's input port). Never
        records into the tracer — the enclosing construct owns the placement.
        """
        if self._over is None:
            raise ConstructError.build(
                "self.each() nested in a loop body requires over=",
                expected="a construction-time over path",
                hint="construct it as self.each(body=[...], key=..., "
                "over='peer.field' or 'neo_subgraph_input.<field>')",
            )

        body_nodes = self._validated_body_nodes()

        parts = [p for p in self._over.split(".") if p]
        if not parts:
            raise ConstructError.build(
                "self.each() over path must not be empty",
                hint="pass a dotted state path like 'seed.claims'",
            )
        root, attr_parts = parts[0], parts[1:]

        if root == StateKeys.SUBGRAPH_INPUT:
            start_type: TypeSpecStatic = outer_input_type
        else:
            start_type = None
            for peer in preceding:
                if field_name_for(peer.name) == root:
                    start_type = effective_producer_type(peer)
                    break
            if start_type is None:
                raise ConstructError.build(
                    f"self.each() over root '{root}' does not match any preceding loop-body member",
                    expected=f"a preceding body member's state field name or '{StateKeys.SUBGRAPH_INPUT}.<field>'",
                    found=f"'{self._over}'",
                    hint="place the producer before self.each(...) in the loop body",
                )

        item_type = self._walk_item_type(start_type, attr_parts, self._over)
        return self._build(body_nodes, item_type, self._over)

    def _validated_body_nodes(self) -> list[Node]:
        for nc in self._body:
            if not isinstance(nc, _NodeCall):
                raise ConstructError.build(
                    "self.each() body must contain node references (self.node_name)",
                    expected="node reference (self.node_name)",
                    found=type(nc).__name__,
                )

        body_nodes = [nc._node for nc in self._body]

        if not body_nodes:
            raise ConstructError.build(
                "self.each() body must contain at least one node",
                hint="pass at least one node reference: self.each(body=[self.some_node], ...)",
            )
        return body_nodes

    def _build(
        self,
        body_nodes: list[Node],
        item_type: TypeSpecStatic,
        over_path: str,
    ) -> Construct:
        """Build the ``Construct | Each`` sub-construct WITHOUT recording it."""
        # Copy-not-mutate: never write inputs onto class-level Nodes
        # (same rule as _LoopCall, neograph-2o9n).
        body_nodes_copy = []
        for bn in body_nodes:
            if bn.inputs is None:
                bn = bn.model_copy(update={"inputs": item_type})
            body_nodes_copy.append(bn)

        output_type = _primary_type(body_nodes[-1])

        body_slug = "-".join(n.name for n in body_nodes)
        occurrence = self._tracer.next_occurrence("each", body_slug)
        name = f"each-{body_slug}" if occurrence == 0 else f"each-{body_slug}-{occurrence}"

        return Construct(
            name=name,
            input=item_type,
            output=output_type,
            nodes=body_nodes_copy,
        ) | Each(
            over=over_path,
            key=self._key,
            on_error=self._on_error,  # type: ignore[arg-type]
        )

    def _resolve_over(self, over: _Proxy | str) -> tuple[Node | Construct, list[str], str]:
        """Resolve ``over`` to (producer, attr chain, dotted over path).

        Proxy form: producer and attr chain come from the traced proxy.
        Raw-string form: the root segment is reverse-resolved to the traced
        node whose state field name matches — failing loud at trace time
        when no traced producer matches.
        """
        if isinstance(over, _Proxy):
            source_node = over._neo_source
            if source_node is None:
                raise ConstructError.build(
                    "self.each() over must be a node output attribute",
                    found="the forward() input proxy",
                    hint="fan out over a traced node's collection field, e.g. each_x(items.claims)",
                )
            attr_parts = _attr_chain_after_prefix(source_node, over._neo_name)
            return source_node, attr_parts, _over_path_for_proxy(source_node, over._neo_name)

        if isinstance(over, str):
            parts = [p for p in over.split(".") if p]
            if not parts:
                raise ConstructError.build(
                    "self.each() over path must not be empty",
                    hint="pass a dotted state path like 'seed.claims'",
                )
            root, attr_parts = parts[0], parts[1:]
            for item in self._tracer._ordered:
                if field_name_for(item.name) == root:
                    return item, attr_parts, over
            raise ConstructError.build(
                f"self.each() over root '{root}' does not match any traced node",
                expected="the state field name of a node already called in forward()",
                found=f"'{over}'",
                hint="call the producer node before self.each(...), or pass its output proxy",
            )

        raise ConstructError.build(
            "self.each() over must be a proxy attribute or a dotted string path",
            found=type(over).__name__,
        )

    def _infer_item_type(
        self,
        source_node: Node | Construct,
        attr_parts: list[str],
        over_path: str,
    ) -> TypeSpecStatic:
        """Walk the producer's declared output along the attr chain and
        extract the fanned item type (the sub-construct's input port)."""
        typ = _primary_type(source_node)
        return self._walk_item_type(typ, attr_parts, over_path)

    def _walk_item_type(
        self,
        typ: TypeSpecStatic,
        attr_parts: list[str],
        over_path: str,
    ) -> TypeSpecStatic:
        """Walk a start type along the attr chain and extract the fanned
        item type — failing loud on unresolvable segments or non-list
        terminals."""
        for attr in attr_parts:
            resolved = _resolve_field_annotation(typ, attr) if typ is not None else cast("TypeSpecStatic", _MISSING)
            if resolved is _MISSING:
                raise ConstructError.build(
                    f"self.each() over path '{over_path}' does not resolve",
                    expected=f"a field '{attr}' on {getattr(typ, '__name__', typ)}",
                    hint="check the collection path against the producer's output model",
                )
            typ = resolved
        item_type = _extract_list_element(typ) if typ is not None else None
        if item_type is None:
            raise ConstructError.build(
                f"self.each() over path '{over_path}' is not a list field",
                expected="a list[...] collection field to fan out over",
                found=str(typ),
            )
        return item_type


class _ModifierWrapCall:
    """Shared machinery for tracer builders that wrap a node reference or a
    node list with one pipeable modifier (Oracle, Operator). Form-aware:

    - node form (target is a node reference) → a bare ``Node | <modifier>``
      member, exactly as the declarative ``node | Modifier(...)`` pipe;
    - body form (target is a list of node references) → a
      ``Construct(input=, output=, nodes=body) | <modifier>`` sub-construct
      with the deterministic ``{kind}-{slug}`` name.

    Subclasses set ``_kind`` (name prefix + occurrence-counter key) and
    ``_surface`` (the user-facing method name, for error messages).
    """

    _kind: str
    _surface: str

    def __init__(
        self,
        target: _NodeCall | list[_NodeCall],
        modifier: Any,
        tracer: _Tracer,
    ) -> None:
        self._target = target
        self._modifier = modifier
        self._tracer = tracer

    def __call__(self, input_proxy: _Proxy) -> _Proxy:
        if isinstance(self._target, _NodeCall):
            member = self._target._node | self._modifier
            self._tracer.record(member)
            return _Proxy(
                source_node=member,
                name=f"out_of_{member.name}",
                tracer=self._tracer,
            )

        body_nodes = self._validated_body_nodes()

        source_node = input_proxy._neo_source
        if source_node is not None:
            input_type = _primary_type(source_node)
        else:
            input_type = _primary_type(body_nodes[-1])
        if input_type is None:
            raise ConstructError.build(
                f"{self._surface} input type could not be inferred",
                hint=f"call {self._surface[:-2]}(...)(proxy) where proxy is a node output",
            )

        sub = self._materialize(body_nodes, input_type)
        self._tracer.record_construct(sub)
        return _Proxy(
            source_node=sub,
            name=f"out_of_{sub.name}",
            tracer=self._tracer,
        )

    def _materialize_nested(
        self,
        preceding: list[Node | Construct],
        outer_input_type: TypeSpecStatic,
    ) -> Node | Construct:
        """Build this wrap as a NESTED body member of an enclosing loop —
        form-aware: a node-form target stays a bare ``Node | <modifier>``
        member (a Construct wrap would be non-twin IR); a body-form target
        becomes the ``Construct | <modifier>`` sub-construct. Never records
        into the tracer — the enclosing construct owns the placement.
        """
        if isinstance(self._target, _NodeCall):
            return self._target._node | self._modifier

        body_nodes = self._validated_body_nodes()
        if preceding:
            input_type = _primary_type(preceding[-1])
        else:
            input_type = outer_input_type
        if input_type is None:
            raise ConstructError.build(
                f"nested {self._surface} input type could not be inferred",
                hint="ensure the preceding loop-body member declares an output type",
            )
        return self._materialize(body_nodes, input_type)

    def _validated_body_nodes(self) -> list[Node]:
        if not isinstance(self._target, list):
            raise ConstructError.build(
                f"{self._surface} target must be a node reference or a list of them",
                expected="self.node_name or [self.a, self.b]",
                found=type(self._target).__name__,
            )
        for nc in self._target:
            if not isinstance(nc, _NodeCall):
                raise ConstructError.build(
                    f"{self._surface} body must contain node references (self.node_name)",
                    expected="node reference (self.node_name)",
                    found=type(nc).__name__,
                )
        if not self._target:
            raise ConstructError.build(
                f"{self._surface} body must contain at least one node",
                hint=f"pass at least one node reference: {self._surface[:-2]}([self.some_node], ...)",
            )
        return [nc._node for nc in self._target]

    def _materialize(self, body_nodes: list[Node], input_type: TypeSpecStatic) -> Construct:
        """Build the body-form ``Construct | <modifier>`` WITHOUT recording it."""
        # Copy-not-mutate: never write inputs onto class-level Nodes
        # (same rule as _LoopCall/_EachCall, neograph-2o9n).
        body_nodes_copy = []
        for bn in body_nodes:
            if bn.inputs is None:
                bn = bn.model_copy(update={"inputs": input_type})
            body_nodes_copy.append(bn)

        output_type = _primary_type(body_nodes[-1])

        body_slug = "-".join(n.name for n in body_nodes)
        occurrence = self._tracer.next_occurrence(self._kind, body_slug)
        name = f"{self._kind}-{body_slug}" if occurrence == 0 else f"{self._kind}-{body_slug}-{occurrence}"

        return (
            Construct(
                name=name,
                input=input_type,
                output=output_type,
                nodes=body_nodes_copy,
            )
            | self._modifier
        )


class _EnsembleCall(_ModifierWrapCall):
    """Returned by _ForwardSelf.ensemble() — Oracle wrap.

    Usage in forward()::

        best = self.ensemble(self.gen, n=3, merge_fn="combine")(draft)
        best = self.ensemble([self.draft, self.polish], n=3, merge_fn="combine")(t)
    """

    _kind = "ensemble"
    _surface = "self.ensemble()"


class _InterruptCall(_ModifierWrapCall):
    """Returned by _ForwardSelf.interrupt() — Operator (HITL) wrap.

    Usage in forward()::

        approved = self.interrupt(self.validate, when="any_test_failed")(result)
        approved = self.interrupt([self.check, self.approve], when="needs_review")(t)
    """

    _kind = "interrupt"
    _surface = "self.interrupt()"


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

    def loop(
        self,
        body: list,
        when: Any,
        max_iterations: int = 10,
        on_exhaust: str = "error",
    ) -> _LoopCall:
        """Define a loop body with explicit nodes and exit condition.

        Returns a callable that, when called with a proxy, builds a
        sub-construct with Loop modifier.

        Usage::

            d = self.loop(
                body=[self.review, self.revise],
                when=lambda r: r.score < 0.8,
                max_iterations=5,
            )(d)
        """
        tracer: _Tracer = object.__getattribute__(self, "_neo_tracer")
        return _LoopCall(body, when, max_iterations, on_exhaust, tracer)

    def each(
        self,
        body: list,
        key: str,
        on_error: str = "raise",
        over: str | None = None,
    ) -> _EachCall:
        """Define a fan-out body applied once per item of a collection.

        Returns a callable that, when called with a proxy attribute (or a
        raw dotted state path), builds a sub-construct with an Each modifier
        — the same IR as the declarative
        ``Construct(input=, output=, nodes=body) | Each(over, key, on_error)``.

        Usage::

            each_verify = self.each(body=[self.verify], key="item_id")
            results = each_verify(items.claims)

        ``over=`` (raw dotted string) is the construction-time binding for a
        deferred each placed inside a ``self.loop(body=[...])`` — it is never
        __call__'d there, so the over path must be supplied up front::

            self.loop(
                body=[
                    self.get_claims,
                    self.each(body=[self.verify], key="cid", over="get_claims.claims"),
                    self.collect,
                ],
                when=..., max_iterations=3,
            )(batch)

        The bare ``for x in proxy`` form remains as sugar for the trivial
        single-node, ``key="label"`` case; ``self.each()`` is the general
        form (custom key, multi-node body, ``on_error='collect'``).
        """
        tracer: _Tracer = object.__getattribute__(self, "_neo_tracer")
        return _EachCall(body, key, on_error, tracer, over=over)

    def ensemble(
        self,
        target: Any,
        *,
        n: int | None = None,
        models: list[str] | None = None,
        merge_prompt: str | None = None,
        merge_model: str = "reason",
        merge_fn: str | None = None,
        merge_pre_process: Any = None,
        merge_post_process: Any = None,
        merge_fallback: Any = None,
    ) -> _EnsembleCall:
        """Define an Oracle ensemble (N parallel generators + judge-merge).

        Kwargs mirror the ``Oracle`` modifier fields 1:1 (``n=None`` defers
        to Oracle's default of 3, or ``len(models)`` when ``models=`` is
        given). Returns a callable that, when called with a proxy, emits the
        declarative twin IR — form-aware:

        - ``self.ensemble(self.gen, ...)`` → bare ``Node | Oracle(...)``
        - ``self.ensemble([self.a, self.b], ...)`` →
          ``Construct(input=, output=, nodes=[...]) | Oracle(...)``

        Usage::

            best = self.ensemble(self.gen, n=3, merge_fn="combine")(draft)

        Oracle's own validation (merge strategy required, hooks only with
        merge_prompt) fires here at construction time — fail loud early.
        """
        oracle_kwargs: dict[str, Any] = {
            "merge_prompt": merge_prompt,
            "merge_model": merge_model,
            "merge_fn": merge_fn,
            "merge_pre_process": merge_pre_process,
            "merge_post_process": merge_post_process,
            "merge_fallback": merge_fallback,
        }
        # Pass n/models only when supplied — Oracle infers n from
        # len(models) ONLY when 'n' is absent from model_fields_set.
        if n is not None:
            oracle_kwargs["n"] = n
        if models is not None:
            oracle_kwargs["models"] = models

        tracer: _Tracer = object.__getattribute__(self, "_neo_tracer")
        return _EnsembleCall(target, Oracle(**oracle_kwargs), tracer)

    def interrupt(self, target: Any, *, when: str) -> _InterruptCall:
        """Define a human-in-the-loop gate (Operator modifier).

        ``when`` is a registered condition name, exactly as the declarative
        ``node | Operator(when=...)`` pipe. Returns a callable that, when
        called with a proxy, emits the declarative twin IR — form-aware:

        - ``self.interrupt(self.validate, when=...)`` →
          bare ``Node | Operator(...)``
        - ``self.interrupt([self.check, self.approve], when=...)`` →
          ``Construct(input=, output=, nodes=[...]) | Operator(...)``

        Usage::

            approved = self.interrupt(self.validate, when="any_test_failed")(result)
        """
        tracer: _Tracer = object.__getattribute__(self, "_neo_tracer")
        return _InterruptCall(target, Operator(when=when), tracer)

    def handoff(
        self,
        members: list[Any],
        to: dict[str, list[str]],
        max_hops: int = 10,
        on_exhaust: Literal["error", "exit"] = "error",
        entry: Any = None,
    ) -> None:
        """Define a Portal mode-(a) peer-routing mesh.

        D-FORWARD-EXEMPT: unlike ``self.interrupt``/``self.ensemble``, a mesh
        has NO static dataflow to thread through a proxy -- every member is
        simultaneously producer and consumer of the mesh channel (design
        §3.3). So ``self.handoff`` is a batch-RECORD, not a wrap-and-return-
        callable: it records each member DIRECTLY into the tracer's node
        list, matching ``examples/28_portal_swarm.py``'s declarative
        ``Construct(nodes=[member | Portal(to=[...]), ...])`` shape
        byte-for-byte (no wrapping sub-Construct). ``max_hops``/``on_exhaust``
        are ENTRY-only knobs (mirrors ``Portal`` itself) -- applied ONLY to
        the entry member (``entry``, or ``members[0]`` when omitted); every
        other member gets a bare ``Portal(to=...)`` with neither kwarg set,
        so ``_validation_portal._check_portal_mesh``'s entry-only-knobs rule
        is satisfied by construction, not by caller discipline.

        ``to`` maps each member's DX name -> its declared peer list (mirrors
        ``Portal.to`` 1:1) -- never resurrect ``peers=``.

        Scope: v1 supports a terminal/whole-graph mesh only (self.handoff has
        no proxy input/output); mid-pipeline mesh embedding is not supported.

        Usage::

            self.handoff(
                members=[self.triage, self.billing],
                to={"triage": ["billing"], "billing": []},
                max_hops=6,
            )
        """
        tracer: _Tracer = object.__getattribute__(self, "_neo_tracer")

        def _unwrap(item: Any) -> Node:
            return item._node if isinstance(item, _NodeCall) else item

        nodes = [_unwrap(m) for m in members]
        entry_node = _unwrap(entry) if entry is not None else nodes[0]

        for node in nodes:
            peers = to.get(node.name, [])
            if node is entry_node:
                wrapped = node | Portal(to=peers, max_hops=max_hops, on_exhaust=on_exhaust)
            else:
                wrapped = node | Portal(to=peers)
            tracer.record(wrapped)
