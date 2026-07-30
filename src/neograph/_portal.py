"""``Portal`` — dynamic peer-to-peer handoff between mesh members.

Extracted from ``modifiers.py`` (neograph-3ffdg.5) as a pure file split — the
class and helper below are unchanged, only their home moved. ``modifiers.py``
re-exports them, so every existing ``from neograph.modifiers import Portal``
keeps working and no call site changed.

Portal is the mechanism behind the north-star claim: a handoff targets a
region's ENTRY port, never its interior, so a jump into the middle of a loop is
unrepresentable rather than merely caught.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from neograph._modifier_base import Modifier
from neograph.errors import ConfigurationError

if TYPE_CHECKING:
    from neograph._ir_protocols import ConstructItem

# The sentinel a Portal member returns to end the mesh run.
HANDOFF_END = "__end__"

# The route value marking a Portal in runtime-dispatch (mode b) form.
DISPATCH_ROUTE = "decide"


class Portal(Modifier, frozen=True):
    """Dynamic-handoff modifier — one modifier with two modes (design §2.1).

    Mode (a) — peer routing (``to=[...]``): a node picks its successor at
    runtime from a declared peer set. Lowers to ``Command(goto=<peer>)`` (T2).
    Mode (b) — dynamic flow definition (``route="decide"``): the node emits the
    spec of the next flow; neograph validates -> compiles -> dispatches it.

    Usage:
        # peer routing (typed swarm):
        Node("billing", ...) | Portal(to=["triage", "technical"], max_hops=6)

        # dynamic flow definition:
        Node("planner", ...) | Portal(route="decide", spec_field="spec",
                                        input_field="dispatch_input", output=Summary)

    The mode is discriminated in ``model_post_init`` (mirrors ``Loop``):
    ``to`` set => peer mode (``route`` must not be ``"decide"``);
    ``route == "decide"`` => dispatch mode (requires ``spec_field`` /
    ``input_field`` / ``output``; forbids the peer-mode knobs). Neither or both
    => ``ConfigurationError``. Excludes every other modifier (Each/Oracle/Loop/
    Operator) — Portal owns the node's outgoing edge (D-NO-OPERATOR-COMBO).
    """

    # arbitrary_types_allowed: for the ``output`` class field and the
    # ``scripted`` / ``conditions`` callable registries (mode b).
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # -- mode (a): peer routing --
    to: list[str] | None = None  # declared successor names (directed, per-node)
    route: str = "goto"  # mode (a): routing FIELD on the payload model; mode (b): literal "decide"
    trigger: Literal["output", "tool"] = "output"  # peer mode only: how the member decides its goto
    max_hops: int = 10  # mesh budget; settable ONLY on the entry member
    on_exhaust: Literal["error", "exit"] = "error"
    name: str | None = None  # mesh group name (peer mode only); None = the implicit default group

    # -- mode (b): dynamic flow definition (route="decide") --
    spec_field: str | None = None  # output-model field holding the emitted Spec dict
    input_field: str | None = None  # output-model field holding the dispatch input dict
    output: type[BaseModel] | str | None = None  # REQUIRED in mode (b): dispatched-flow output type
    scripted: dict[str, Callable] | None = None  # building-block registry for the emitted flow
    conditions: dict[str, Callable] | None = None  # condition registry for the emitted flow
    on_invalid: Literal["raise", "route_to_error"] = "raise"  # emitted-spec validation-gate failure handling
    error_handler: str | None = None  # sibling Node to route to when on_invalid='route_to_error'
    max_depth: int | None = None  # REQUIRED in dispatch mode (self-extending-flow budget); forbidden in peer mode

    @property
    def is_dispatch(self) -> bool:
        """Mode discriminator — the SINGLE SOURCE OF TRUTH for peer vs dispatch.

        True for dynamic-flow-definition mode (``route == DISPATCH_ROUTE``), False
        for peer routing. EVERY layer (validator, wiring collector, compiler walk,
        state builder, producer registration) discriminates the mode ONLY through
        this property — never an inline ``route == "decide"`` string check, so a new
        call site cannot forget the rule (anti-band-aid; pinned by a structural
        guard that bans the inline literal outside this module).
        """
        return self.route == DISPATCH_ROUTE

    @property
    def is_tool_triggered(self) -> bool:
        """Peer-mode sub-mode discriminator (design portal-tool-triggered-handoff).

        True when a PEER-mode member decides its handoff by emitting a
        synthesized ``transfer_to_<peer>`` tool call from its ReAct turn
        (``trigger == "tool"``), rather than by writing a routing field on its
        typed output (``trigger == "output"``, the default). The direct sibling
        of ``is_dispatch``: the SINGLE SOURCE OF TRUTH every layer (validator,
        wiring, factory) reads to distinguish the trigger sub-mode — never an
        inline ``trigger == "tool"`` check. Always False in dispatch mode (a
        ``route="decide"`` Portal is not a peer member and cannot trigger a
        tool handoff), guarded by ``not self.is_dispatch``.
        """
        return not self.is_dispatch and self.trigger == "tool"

    def model_post_init(self, __context: Any) -> None:
        is_peer = self.to is not None
        is_dispatch = self.is_dispatch
        if is_peer and is_dispatch:
            raise ConfigurationError.build(
                "Portal cannot be both peer mode and dispatch mode",
                expected="to=[...] (peer routing) XOR route='decide' (dynamic flow)",
                found="to set AND route=='decide'",
            )
        if not is_peer and not is_dispatch:
            raise ConfigurationError.build(
                "Portal requires a mode",
                expected="to=[...] (peer routing) or route='decide' (dynamic flow)",
                found="neither to nor route='decide' provided",
            )
        if is_peer:
            if self.max_hops < 1:
                raise ConfigurationError.build(
                    "Portal max_hops must be >= 1",
                    found=str(self.max_hops),
                )
            if "max_depth" in self.model_fields_set:
                raise ConfigurationError.build(
                    "Portal peer mode forbids max_depth — max_depth is a dispatch-mode-only "
                    "self-extending-flow budget (the mirror image of max_hops/on_exhaust being "
                    "forbidden in dispatch mode)",
                    expected="no max_depth in peer mode",
                    found="max_depth set",
                )
            if "error_handler" in self.model_fields_set:
                raise ConfigurationError.build(
                    "Portal peer mode forbids error_handler — it is a dispatch-mode-only knob",
                    expected="no error_handler in peer mode",
                    found="error_handler set",
                )
        else:  # dispatch mode (route == "decide")
            missing = [f for f in ("spec_field", "input_field", "output") if getattr(self, f) is None]
            if self.max_depth is None:
                missing.append("max_depth")
            if missing:
                raise ConfigurationError.build(
                    "Portal dispatch mode requires spec_field, input_field, output, and max_depth "
                    "(no numeric default — every self-extending Portal must declare its own bound)",
                    expected="spec_field=, input_field=, output=, max_depth=",
                    found=f"missing: {missing}",
                )
            forbidden = [k for k in ("max_hops", "on_exhaust", "name", "trigger") if k in self.model_fields_set]
            if forbidden:
                raise ConfigurationError.build(
                    "Portal dispatch mode forbids peer-mode knobs",
                    expected="no max_hops/on_exhaust/name/trigger in dispatch mode",
                    found=f"peer-mode knobs set: {forbidden}",
                )
            if self.on_invalid == "route_to_error" and self.error_handler is None:
                raise ConfigurationError.build(
                    "Portal on_invalid='route_to_error' requires error_handler=",
                    expected="error_handler=<sibling Node name>",
                    found="error_handler not set",
                )
            if self.on_invalid == "raise" and self.error_handler is not None:
                raise ConfigurationError.build(
                    "Portal error_handler is only meaningful with on_invalid='route_to_error'",
                    expected="on_invalid='route_to_error' when error_handler is set",
                    found=f"on_invalid={self.on_invalid!r} with error_handler set",
                )


def _group_portal_members(items: Sequence[ConstructItem]) -> dict[str | None, list[ConstructItem]]:
    """Partition a construct level's PEER-mode Portal members into named mesh
    groups (neograph-fefar, D-SINGLE-MESH's named-mesh extension).

    The SINGLE source of truth for "which mesh does this member belong to" --
    the validator (``_validation_portal._check_portal_mesh``), the IR
    normalizer (``_ir_normalize.py``'s ``handoff_channel`` writer), and the
    compiler's contiguous-mesh collector (``_wiring.py:_contiguous_portal_mesh``)
    all group members via THIS function, never a re-derived inline grouping --
    a construct that validates must lower exactly as validated.

    Grouping key is ``Portal.name`` -- ``None`` is the implicit DEFAULT group
    (unchanged pre-fefar behavior: all unnamed members are one mesh). Order is
    preserved: each group's member list is in construct order, and groups
    themselves appear in first-occurrence order (dict insertion order).

    ``items`` must already be filtered to PEER-mode Portal members (dispatch-
    mode Portals excluded by the caller, mirroring every other Portal-mesh
    call site's convention).
    """
    groups: dict[str | None, list[ConstructItem]] = {}
    for item in items:
        portal = item.modifier_set.portal
        assert portal is not None, "caller must pre-filter to Portal-modified members"
        groups.setdefault(portal.name, []).append(item)
    return groups
