"""Arm-scoped producer-visibility bookkeeping for ``_validate_node_chain``.

Extracted from ``_construct_validation.py`` (neograph-ftnxl.2) to keep the
orchestrator under its 400-line cap — a pure mechanical split, no behavior
change at the extraction boundary (see ``docs/file-split-procedure.md``).

The concern this owns: a Branch's two arms are mutually exclusive at
runtime (``_wiring_branch.py`` wires exactly one via ``add_conditional_edges``),
so a producer registered by one arm's node must NOT be visible to the other
arm, nor to anything after the branch's join, by NAME — the soundness gap
neograph-ftnxl.2 fixes (previously arms were flattened into one shared
``ProducerMap``, so a false-arm consumer of a true-arm-only producer
validated successfully; see the ticket and neograph-vn5f site 1).

Two arm nodes can never share a literal ``.name`` (LangGraph's ``add_node``
requires graph-wide unique names), so a same-name "all arms produce it"
promotion for the fan-in/context checks is unreachable in any construct that
would also compile — this class deliberately does NOT attempt it. The one
TYPE-based (not name-based) exception is the ``construct.output`` boundary
check: two DIFFERENTLY-named arm producers can each independently satisfy a
declared output type, which IS reachable and compilable, so
``output_reachable_on_every_arm`` tracks that separately.
"""

from __future__ import annotations

from collections import OrderedDict

from neograph._ir_protocols import ConstructLike
from neograph._state_keys import StateKeys
from neograph._validation_types import NodeItem, Producer, ProducerMap, _fmt_type, _source_location, _types_compatible
from neograph.errors import ConstructError, NeographError
from neograph.node import TypeSpecStatic


class ArmScopedProducers:
    """Tracks producer registration + visibility across one
    ``_validate_node_chain`` walk over ``iter_with_arm_ids``.

    Usage (one instance per ``_validate_node_chain`` call):
      1. ``seed_subgraph_input(construct)`` once, before the walk.
      2. Per item, in walk order: ``visible_for(arm_key)`` (read), then zero
         or more ``register(field, producer, arm_key)`` calls (write) for
         that SAME item.
      3. ``finalize()`` once after the walk ends.
      4. Read ``.producers`` / ``.output_reachable_on_every_arm`` for the
         output-boundary check; ``.all_producers`` is diagnostics-only (never
         a pass/fail input) for cross-arm error messages.
    """

    def __init__(self, declared_output: TypeSpecStatic | None) -> None:
        self.producers: ProducerMap = OrderedDict()
        self.all_producers: ProducerMap = OrderedDict()
        self.output_reachable_on_every_arm = False
        self._declared_output = declared_output
        self._current_branch: dict[bool, ProducerMap] = {True: OrderedDict(), False: OrderedDict()}
        self._current_branch_id: int | None = None

    def seed_subgraph_input(self, construct_input: TypeSpecStatic | None, construct_name: str) -> None:
        """Register the Construct's own input port as the first producer, if
        declared — used by inner nodes reading from ``neo_subgraph_input``."""
        if construct_input is None:
            return
        port_producer = Producer(
            field_name=StateKeys.SUBGRAPH_INPUT,
            effective_type=construct_input,
            label=f"construct '{construct_name}' input port",
        )
        self.producers[StateKeys.SUBGRAPH_INPUT] = port_producer
        self.all_producers[StateKeys.SUBGRAPH_INPUT] = port_producer

    def visible_for(self, arm_key: tuple[int, bool] | None) -> ProducerMap:
        """Call once per walk item, in order, BEFORE registering that item's
        own producer(s). Flushes the previous branch when this item starts a
        new one (or returns to top level) and returns the producer map THIS
        item may read from: top-level producers alone for a non-arm item, or
        top-level UNION this arm's producers-so-far for an arm item.
        """
        branch_id = arm_key[0] if arm_key is not None else None
        if branch_id != self._current_branch_id:
            if self._current_branch_id is not None:
                self._flush_branch()
            self._current_branch_id = branch_id
        if arm_key is None:
            return self.producers
        visible = OrderedDict(self.producers)
        visible.update(self._current_branch[arm_key[1]])
        return visible

    def register(self, field_name: str, producer: Producer, arm_key: tuple[int, bool] | None) -> None:
        """Register ``producer`` under ``field_name``, scoped by ``arm_key``
        (the SAME key passed to the preceding ``visible_for`` call for this
        item) — plus unconditionally into the diagnostics-only
        ``all_producers``."""
        self.all_producers[field_name] = producer
        if arm_key is None:
            self.producers[field_name] = producer
        else:
            self._current_branch[arm_key[1]][field_name] = producer

    def finalize(self) -> None:
        """Call once after the walk ends — flushes a branch left open when
        the last top-level item was a ``_BranchNode``."""
        if self._current_branch_id is not None:
            self._flush_branch()

    def _flush_branch(self) -> None:
        """Update ``output_reachable_on_every_arm`` when the just-finished
        branch's every arm has at least one producer compatible with the
        declared output, then reset per-arm state for the next branch.
        Arm-registered producers are NEVER promoted into ``producers`` by
        name — see the module docstring for why that would be dead code.
        """
        if self._declared_output is not None and not self.output_reachable_on_every_arm:
            true_ok = any(
                p.effective_type is not None and _types_compatible(p.effective_type, self._declared_output)
                for p in self._current_branch[True].values()
            )
            false_ok = any(
                p.effective_type is not None and _types_compatible(p.effective_type, self._declared_output)
                for p in self._current_branch[False].values()
            )
            if true_ok and false_ok:
                self.output_reachable_on_every_arm = True
        self._current_branch[True] = OrderedDict()
        self._current_branch[False] = OrderedDict()


def _build_cross_arm_error(
    construct: ConstructLike,
    item: NodeItem,
    upstream_name: str,
    expected_type: TypeSpecStatic,
    producers: ProducerMap,
    all_producers: ProducerMap,
    *,
    each_over: str | None = None,
) -> NeographError:
    """The dedicated cross-arm variant of the 'unknown upstream' error
    (neograph-ftnxl.2): ``upstream_name`` exists SOMEWHERE in this construct
    level (``all_producers``) but is not in the caller's VISIBLE set
    (``producers``) — i.e. it is produced only on a branch arm this
    consumer cannot reach on every path. Deliberately distinct wording from
    the plain 'no upstream node named X exists' error: that phrasing is
    factually wrong here (the node DOES exist) and could otherwise collide
    with a should_fail check_fixture's regex for the genuinely-unknown case.
    """
    what = (
        f"Each(over='{each_over}') root '{upstream_name}' is produced only on a branch arm this node cannot reach"
        if each_over is not None
        else f"declares inputs['{upstream_name}']={_fmt_type(expected_type)} but "
        f"'{upstream_name}' is produced only on a branch arm this node cannot reach"
    )
    return ConstructError.build(
        what,
        expected="a producer reachable on every path to this node",
        found=f"'{upstream_name}' exists but is not reachable here; available upstreams: {sorted(producers)}",
        hint="move the producer above the branch, or have every arm produce a compatible value under the same name",
        node=item.name,
        construct=construct.name,
        location=_source_location(),
    )
