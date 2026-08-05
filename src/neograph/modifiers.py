"""Modifiers — composable pipeline behaviors applied via the | operator.

node | Oracle(n=3, merge_prompt="rw/merge")
node | Each(over="clusters.clusters", key="label")
node | Operator(when="has_open_questions")
"""

from __future__ import annotations

import itertools

# --- typing names modifiers.py imported and RE-EXPORTED before the split.
from collections.abc import (
    Callable,
    Iterable,
    Sequence,  # noqa: E402,F401
)
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, Self, runtime_checkable  # noqa: E402,F401

from pydantic import BaseModel, ConfigDict, field_validator
from typing_extensions import TypeVar

from neograph._dev_warnings import dev_warn

# --- extracted clusters (neograph-3ffdg.5), re-exported so every existing
# --- `from neograph.modifiers import ...` call site keeps resolving unchanged.
from neograph._each import Each, EachFailure, split_each_path  # noqa: E402,F401
from neograph._modifier_base import Modifier  # noqa: E402,F401
from neograph._oracle_protocols import (  # noqa: E402,F401
    MergeFallback,
    MergePostProcess,
    MergePreProcess,
)
from neograph._portal import (  # noqa: E402,F401
    DISPATCH_ROUTE,
    HANDOFF_END,
    Portal,
    _group_portal_members,
)
from neograph.errors import ConfigurationError, ConstructError

if TYPE_CHECKING:
    from neograph._ir_protocols import ConstructItem

# ═══════════════════════════════════════════════════════════════════════════
# Oracle merge-hook Protocols
# ═══════════════════════════════════════════════════════════════════════════

# PEP 696 TypeVar defaults: variant element / result types are declared by
# node.oracle_gen_type / node.outputs. Defaulting to Any preserves existing
# un-parameterized call sites; subscription is optional for richer typing.
# `_Variant` appears inside `list[...]` (invariant container), so it stays
# invariant. `_FallbackResult` is output-only in MergeFallback, hence
# covariant. `_PostResult` is both input and output in MergePostProcess,
# so it must remain invariant.
_Variant = TypeVar("_Variant", default=Any)
_FallbackResult = TypeVar("_FallbackResult", covariant=True, default=Any)
_PostResult = TypeVar("_PostResult", default=Any)


class ModifierCombo(Enum):
    """Exhaustive enumeration of valid modifier combinations.

    Every dispatch site (compiler, state, factory) matches on this enum
    instead of ad-hoc has_modifier() chains. Adding a new combo forces
    handling at every site.
    """

    BARE = auto()  # no modifiers
    EACH = auto()  # Each only
    ORACLE = auto()  # Oracle only
    LOOP = auto()  # Loop only
    OPERATOR = auto()  # Operator only
    PORTAL = auto()  # Portal only (dynamic handoff; excludes Each/Oracle/Loop)
    EACH_ORACLE = auto()  # Each + Oracle (fusion)
    EACH_OPERATOR = auto()  # Each + Operator
    ORACLE_OPERATOR = auto()  # Oracle + Operator
    LOOP_OPERATOR = auto()  # Loop + Operator
    EACH_ORACLE_OPERATOR = auto()  # Each + Oracle + Operator
    PORTAL_OPERATOR = auto()  # Portal (PEER mode only) + Operator — human-approval gate on the dynamic path


# Single source of truth: modifier-name frozenset -> ModifierCombo. Both
# classify_modifiers() and ModifierSet.combo read this ONE map so a new combo
# is added exactly once. A structural guard (test_guards_*) bans a re-planted
# inline copy — prior byte-for-byte duplication silently diverged classification.
_COMBO_MAP: dict[frozenset[str], ModifierCombo] = {
    frozenset(): ModifierCombo.BARE,
    frozenset({"each"}): ModifierCombo.EACH,
    frozenset({"oracle"}): ModifierCombo.ORACLE,
    frozenset({"loop"}): ModifierCombo.LOOP,
    frozenset({"operator"}): ModifierCombo.OPERATOR,
    frozenset({"each", "oracle"}): ModifierCombo.EACH_ORACLE,
    frozenset({"each", "operator"}): ModifierCombo.EACH_OPERATOR,
    frozenset({"oracle", "operator"}): ModifierCombo.ORACLE_OPERATOR,
    frozenset({"loop", "operator"}): ModifierCombo.LOOP_OPERATOR,
    frozenset({"each", "oracle", "operator"}): ModifierCombo.EACH_ORACLE_OPERATOR,
    frozenset({"portal"}): ModifierCombo.PORTAL,
    frozenset({"portal", "operator"}): ModifierCombo.PORTAL_OPERATOR,
}


def modifier_names_for_combo(combo: ModifierCombo) -> frozenset[str]:
    """The modifier names a combo is composed of -- the inverse of
    ``_COMBO_MAP``, and the ONLY sanctioned way to ask "does this combo carry
    an Each / an Oracle / a Loop?".

    Exists so consumers that need MEMBERSHIP (rather than the decomposed
    primary shape) read the table instead of hand-typing a combo list.
    Also the source ``COMBO_DECOMPOSITION``'s ``fused`` column is derived
    from at table-definition time (neograph-jtawq.2) -- hoisted above the
    table for exactly that reason; it depends on ``_COMBO_MAP`` only.
    """
    for names, value in _COMBO_MAP.items():
        if value is combo:
            return names
    raise ConstructError.build(  # pragma: no cover - unreachable while _COMBO_MAP is total
        "ModifierCombo missing from _COMBO_MAP",
        found=combo.name,
        hint="every ModifierCombo value must appear exactly once in _COMBO_MAP",
    )


class PrimaryShape(Enum):
    """The primary body-shape a ModifierCombo decomposes to, orthogonal to
    the Operator wrapper. Every combo reduces to exactly one of these five."""

    BARE = auto()
    EACH = auto()
    ORACLE = auto()
    LOOP = auto()
    PORTAL = auto()


class ComboDecomposition(NamedTuple):
    """How a ModifierCombo decomposes into a primary body-shape plus an
    optional orthogonal Operator wrapper, plus whether it is the fused
    Each x Oracle combo. This is the ONE place that answers "what does this
    combo mean"; consumers (compiler.py, _agent_spec.py, and the other
    combo-dispatch sites) consult it instead of re-deriving it.
    """

    primary: PrimaryShape  # BARE | EACH | ORACLE | LOOP | PORTAL
    has_operator: bool  # True for every *_OPERATOR combo
    fused: bool  # True for EACH_ORACLE / EACH_ORACLE_OPERATOR (neograph-jtawq.2)


# Single source of truth for combo *meaning* (decomposition), complementing
# _COMBO_MAP's single source of truth for combo *classification*. A total
# function over ModifierCombo: every enum value has exactly one entry, pinned
# by a partition guard the same shape as _COMBO_MAP's own exhaustiveness.
# EACH_ORACLE/EACH_ORACLE_OPERATOR are primary=EACH because compiler.py fuses
# them (a single Node's map_over/ensemble_n M x N Send topology) rather than
# nesting -- the fusion, not a table gap. `fused` is DERIVED from
# modifier_names_for_combo (never hand-set) so it can never drift from what
# the table's own membership authority says (neograph-jtawq.2).
_COMBO_PRIMARY_OPERATOR: dict[ModifierCombo, tuple[PrimaryShape, bool]] = {
    ModifierCombo.BARE: (PrimaryShape.BARE, False),
    ModifierCombo.EACH: (PrimaryShape.EACH, False),
    ModifierCombo.ORACLE: (PrimaryShape.ORACLE, False),
    ModifierCombo.LOOP: (PrimaryShape.LOOP, False),
    ModifierCombo.OPERATOR: (PrimaryShape.BARE, True),
    ModifierCombo.PORTAL: (PrimaryShape.PORTAL, False),
    ModifierCombo.EACH_ORACLE: (PrimaryShape.EACH, False),  # fused
    ModifierCombo.EACH_OPERATOR: (PrimaryShape.EACH, True),
    ModifierCombo.ORACLE_OPERATOR: (PrimaryShape.ORACLE, True),
    ModifierCombo.LOOP_OPERATOR: (PrimaryShape.LOOP, True),
    ModifierCombo.EACH_ORACLE_OPERATOR: (PrimaryShape.EACH, True),  # fused + operator
    ModifierCombo.PORTAL_OPERATOR: (PrimaryShape.PORTAL, True),
}

COMBO_DECOMPOSITION: dict[ModifierCombo, ComboDecomposition] = {
    combo: ComboDecomposition(
        primary,
        has_operator,
        {"each", "oracle"} <= modifier_names_for_combo(combo),
    )
    for combo, (primary, has_operator) in _COMBO_PRIMARY_OPERATOR.items()
}


SUB_CONSTRUCT_UNSUPPORTED_COMBOS: frozenset[ModifierCombo] = frozenset(
    {ModifierCombo.EACH_ORACLE, ModifierCombo.EACH_ORACLE_OPERATOR}
)
"""ModifierCombo values that are meaningful on a bare Node but have no defined
lowering when the SAME combo is attached to a Construct used as one item inside
another Construct. Each x Oracle fusion is defined entirely in terms of a single
Node's map_over/ensemble_n fields (an M x N Send topology), which a multi-node
Construct structurally lacks -- a pre-existing compiler restriction
(compiler.py's Each x Oracle sub-construct CompileError), not an Agent Spec one.
Consulted FIRST, unconditionally, by both compiler.py's _add_subgraph and
_agent_spec.py's Construct-item handling before any Construct-level lowering.

Portal combos (PORTAL/PORTAL_OPERATOR) are deliberately EXCLUDED from this set:
not because a Construct Portal mesh member is impossible (it is not -- a Construct
CAN be a non-entry mesh member today), but because such a member routes through
the dedicated mesh path (_contiguous_portal_mesh/_add_portal_mesh in _wiring.py),
not the generic Construct-item modifier-check path this frozenset governs. The
mesh path has its own eligibility rules; this set is not the place they live.
"""


def classify_modifiers(item: ConstructItem) -> tuple[ModifierCombo, dict]:
    """Classify an item's modifiers into a ModifierCombo enum value.

    Returns (combo, modifiers_dict) where modifiers_dict has keys like
    'each', 'oracle', 'loop', 'operator' mapping to the modifier instances.

    Fast path: when item has a modifier_set attribute (Node/Construct/
    _BranchNode), reads directly from typed slots. Fallback path: uses
    get_modifier() for any remaining duck-typed items.
    """
    ms = getattr(item, "modifier_set", None)
    if ms is not None and isinstance(ms, ModifierSet):
        mods: dict[str, Any] = {r.slot: getattr(ms, r.slot) for r in _SLOT_RULES if getattr(ms, r.slot) is not None}
        return ms.combo, mods

    # Fallback for duck-typed items (e.g. _BranchNode)
    get_mod = getattr(item, "get_modifier", None)
    if get_mod is None:
        return ModifierCombo.BARE, {}

    mods = {r.slot: get_mod(r.mod_type) for r in _SLOT_RULES if get_mod(r.mod_type)}

    # Map to enum
    has = frozenset(mods.keys())
    combo = _COMBO_MAP.get(has)
    if combo is None:
        raise ConstructError.build(
            "Invalid modifier combination",
            found=str(sorted(has)),
            hint="This combination is not supported",
            node=getattr(item, "name", "?"),
        )
    return combo, mods


def combo_for_modifier_names(names: Iterable[str], *, context: str = "?") -> ModifierCombo:
    """Classify a set of modifier NAMES (not modifier instances) into a combo.

    The structural-recognition twin of ``classify_modifiers``: the Agent Spec
    loader recognizes which modifiers a Flow's node grouping encodes, and needs
    the same frozenset -> ModifierCombo answer without a ``Node`` to inspect.
    Reads the SAME ``_COMBO_MAP`` (no second mapping -- a re-planted copy is
    what ``TestComboMapMonopoly`` exists to ban), and fails loud on a miss for
    the same reason ``classify_modifiers`` does: a foreign or hand-edited spec
    can present a combination neograph has no meaning for, and a bare KeyError
    would surface as an internal error instead of a diagnosable one.
    """
    has = frozenset(names)
    combo = _COMBO_MAP.get(has)
    if combo is None:
        raise ConstructError.build(
            "Invalid modifier combination",
            found=str(sorted(has)),
            hint="This combination is not supported",
            node=context,
        )
    return combo


def primary_shape(item: ConstructItem) -> PrimaryShape:
    """The primary body-shape `item`'s modifier combo decomposes to.

    Convenience over ``COMBO_DECOMPOSITION[classify_modifiers(item)[0]].primary``
    for the many consumers that ask a pure shape question ("is this Portal-shaped?",
    "is this Loop-shaped?") and need neither the combo value nor the modifier dict.

    Call this only where the caller does NOT already hold a classify_modifiers
    result; where it does, index COMBO_DECOMPOSITION with the combo it already has
    rather than re-classifying.
    """
    return COMBO_DECOMPOSITION[classify_modifiers(item)[0]].primary


class _PathRecorder:
    """Proxy that records attribute-access chains for .map() lambda introspection.

    Passed into a user lambda to resolve a dotted state path at definition time:

        recorder = _PathRecorder()
        result = (lambda s: s.make_clusters.groups)(recorder)
        result._neo_path  # ('make_clusters', 'groups')

    Any attribute access returns a fresh recorder whose path extends the parent's,
    so chained access walks the tree without ever materializing a real value.
    """

    __slots__ = ("_neo_path",)

    def __init__(self, path: tuple[str, ...] = ()) -> None:
        # Plain assignment is fine — __slots__ without a custom __setattr__
        # stores through the slot descriptor directly. `_neo_path` is always
        # resolved via __getattribute__ (slot lookup), never __getattr__,
        # so future attribute access on the recorder records only user names.
        self._neo_path = path

    def __getattr__(self, name: str) -> _PathRecorder:
        # __getattr__ only fires for attrs not found by normal lookup; _neo_path
        # lives in __slots__ so it returns via __getattribute__ and never hits here.
        # Reject leading-underscore names (dunders, privates) so that
        # `lambda s: s.__dict__.foo` or `lambda s: s._private` can't silently
        # produce Each(over="__dict__.foo", ...) paths that would fail at runtime.
        if name.startswith("_"):
            raise AttributeError(name)
        return _PathRecorder(self._neo_path + (name,))


class Modifiable:
    """Mixin for objects that accept modifiers via the | operator.

    Both Node and Construct inherit this. Provides has_modifier(),
    get_modifier(), and __or__() — the pipe composition syntax.
    Uses modifier_set: ModifierSet for type-safe modifier storage.
    """

    # Every concrete subclass (Node, Construct, _BranchNode) assigns a
    # non-empty string name. Declared here so union narrowing (Node | Modifiable)
    # keeps .name access type-checkable.
    name: str
    modifier_set: ModifierSet

    @property
    def modifiers(self) -> list[Modifier]:
        """Backward compat bridge: returns modifier_set contents as a list."""
        return self.modifier_set.to_list()

    def __or__(self, modifier: Modifier) -> Self:
        """Compose modifiers via pipe: obj | Oracle(n=3) | Operator(when=...)"""

        # ModifierSet.with_modifier handles duplicate and illegal-combo
        # rejection (Each+Loop, Oracle+Loop). The typed slots make
        # duplicates structurally impossible.
        new_ms = self.modifier_set.with_modifier(modifier)

        # Dev-mode warnings for ambiguous-but-valid patterns

        if isinstance(modifier, Oracle):
            if modifier.n == 1:
                dev_warn(
                    f"Oracle(n=1) on '{getattr(self, 'name', '?')}' — "
                    f"an ensemble of 1 is equivalent to no ensemble. "
                    f"Did you mean n=3?"
                )
            if modifier.models and modifier.n % len(modifier.models) != 0:
                dev_warn(
                    f"Oracle(n={modifier.n}, models={modifier.models}) on "
                    f"'{getattr(self, 'name', '?')}' — uneven distribution: "
                    f"{modifier.n} generators across {len(modifier.models)} "
                    f"models means some models run more than others."
                )

        if isinstance(modifier, Loop) and modifier.max_iterations == 1:
            dev_warn(
                f"Loop(max_iterations=1) on '{getattr(self, 'name', '?')}' — "
                f"a loop that runs at most once is equivalent to a conditional. "
                f"Did you mean max_iterations=3?"
            )

        result = self.model_copy(update={"modifier_set": new_ms})  # type: ignore[attr-defined]
        # Loop validation at | time: check type compatibility immediately.
        if isinstance(modifier, Loop):
            # Discriminate Node vs Construct via isinstance, not hand-rolled
            # hasattr(.,'outputs')/(.,'output') probes. Lazy imports: node.py and
            # construct.py both import modifiers (Modifiable base), so a top-level
            # import here would cycle.
            from neograph.construct import Construct
            from neograph.node import Node

            if isinstance(result, Node):
                # Node: validate output compat with inputs
                from neograph._construct_validation import validate_loop_self_edge

                validate_loop_self_edge(result)
            elif isinstance(result, Construct):
                # Construct: validate output compat with input
                from neograph._construct_validation import validate_loop_construct

                validate_loop_construct(result)
        return result

    def has_modifier(self, modifier_type: type[Modifier]) -> bool:
        """Check if a specific modifier is applied."""
        rule = next((r for r in _SLOT_RULES if r.mod_type is modifier_type), None)
        return rule is not None and getattr(self.modifier_set, rule.slot) is not None

    def get_modifier(self, modifier_type: type[Modifier]) -> Modifier | None:
        """Get the modifier of a given type, or None."""
        rule = next((r for r in _SLOT_RULES if r.mod_type is modifier_type), None)
        return getattr(self.modifier_set, rule.slot) if rule is not None else None

    def map(
        self,
        source: str | Callable[[Any], Any],
        *,
        key: str,
        on_error: Literal["raise", "collect"] = "raise",
    ) -> Self:
        """Fan-out over a collection — sugar over `| Each(over=..., key=...)`.

        Usage:
            # Lambda form (refactor-safe, mypy-friendly):
            verify.map(lambda s: s.make_clusters.groups, key="label")

            # String form (escape hatch, equivalent to | Each(...)):
            verify.map("make_clusters.groups", key="label")

        The lambda is introspected once at definition time via a recording
        proxy. Pyright/Pylance catch typos in `.make_clusters.groups`, and
        renaming the upstream node surfaces as a red squiggle — the
        refactor-safety win over string paths.

        Args:
            source: Either a string dotted path (equivalent to `Each.over`)
                or a lambda taking the state proxy and returning an attribute
                chain. The lambda must be a pure attribute-access chain;
                indexing, arithmetic, or underscore-prefixed attributes raise
                TypeError.
            key: Field on each iterated item used as the dispatch key
                (same semantics as `Each.key`).
            on_error: Per-item fault handling, forwarded to `Each.on_error`.
                `'raise'` (default) aborts the run on a thrown item; `'collect'`
                keys a typed `EachFailure` into the barrier instead.

        Returns:
            A new instance of the same type with an `Each` modifier
            appended — fully equivalent to `self | Each(over=..., key=key)`.
            The compiler, state builder, and factory all see the same Each
            modifier as before.
        """
        if isinstance(source, str):
            over = source
        elif callable(source):
            recorder = _PathRecorder()
            try:
                result = source(recorder)
            except (TypeError, AttributeError) as exc:
                # Only these two error shapes indicate "not a pure attribute-
                # access chain" — indexing/subscript → TypeError, underscore-
                # prefixed attrs → AttributeError (see _PathRecorder). Any
                # other exception (ValueError, ZeroDivisionError, etc.) is a
                # genuine bug in the user lambda and should propagate unchanged
                # so they see their own error, not our wrapper.
                raise ConstructError.build(
                    "Node.map() lambda must be a pure attribute-access chain",
                    expected="lambda s: s.upstream_node.field",
                    found=f"error when introspecting: {exc}",
                ) from exc
            if not isinstance(result, _PathRecorder):
                raise ConstructError.build(
                    "Node.map() lambda must return an attribute-access chain",
                    expected="s.upstream_node.field",
                    found=type(result).__name__,
                )
            path = result._neo_path
            if not path:
                raise ConstructError.build(
                    "Node.map() lambda must access at least one attribute",
                    expected="lambda s: s.make_clusters.groups",
                    found="lambda returned the recorder unchanged",
                )
            over = ".".join(path)
        else:
            raise ConstructError.build(
                "Node.map() source must be a string path or a lambda",
                expected="str | Callable[[state], path]",
                found=type(source).__name__,
            )

        return self | Each(over=over, key=key, on_error=on_error)


class Oracle(Modifier, frozen=True):
    """Ensemble modifier: N parallel generators + judge-merge.

    The compiler expands this into:
    1. Fan-out: Send(node, payload) x N with different generator IDs
    2. Barrier: merge node with defer=True
    3. Merge: LLM judge (merge_prompt) or scripted function (merge_fn)

    Exactly one of merge_prompt or merge_fn must be provided.

    Usage:
        # Same model, N copies:
        node | Oracle(n=3, merge_fn="combine_variants")

        # Multi-model ensemble (one per model):
        node | Oracle(models=["reason", "fast", "creative"], merge_fn="pick_best")

        # LLM merge with hooks:
        node | Oracle(n=3, merge_prompt="rw/merge",
                       merge_pre_process=transform_variants,
                       merge_post_process=validate_result,
                       merge_fallback=deterministic_merge)

    Merge hooks (merge_prompt path only):
        merge_pre_process(variants: list[T]) -> dict
            Transform raw variants into the input_data dict for the prompt.
            Replaces the default ``{"variants": variants, ...upstream}`` construction.
        merge_post_process(result: T, variants: list[T]) -> T
            Transform the parsed LLM result before writing to state.
            Only runs on LLM success, NOT on fallback results.
        merge_fallback(variants: list[T], error: Exception) -> T
            Called when invoke_structured raises. Returns a deterministic result.
    """

    # arbitrary_types_allowed: required for the runtime_checkable Protocol
    # callback fields ``merge_pre_process``, ``merge_post_process``,
    # ``merge_fallback`` (Callables exposed by name; not Pydantic models).
    model_config = ConfigDict(arbitrary_types_allowed=True)

    n: int = 3
    models: list[str] | None = None  # per-generator model tiers (round-robin)
    merge_prompt: str | None = None
    merge_model: str = "reason"
    merge_fn: str | None = None  # registered scripted function name

    # Optional hooks for merge_prompt path
    merge_pre_process: MergePreProcess | None = None  # fn(variants) -> input_data
    merge_post_process: MergePostProcess | None = None  # fn(result, variants) -> result
    merge_fallback: MergeFallback | None = None  # fn(variants, error) -> result

    @field_validator("n")
    @classmethod
    def _validate_n(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Oracle n must be >= 1")
        return v

    def model_post_init(self, __context: Any) -> None:
        if not self.merge_prompt and not self.merge_fn:
            raise ConfigurationError.build(
                "Oracle requires a merge strategy",
                expected="merge_prompt (LLM judge) or merge_fn (scripted function)",
                found="neither provided",
            )
        if self.merge_prompt and self.merge_fn:
            raise ConfigurationError.build(
                "Oracle accepts merge_prompt or merge_fn, not both",
                found="both merge_prompt and merge_fn provided",
                hint="Remove one of the two merge strategies",
            )
        # Hooks are only valid with merge_prompt, not merge_fn
        if self.merge_fn and (self.merge_pre_process or self.merge_post_process or self.merge_fallback):
            raise ConfigurationError.build(
                "merge hooks (merge_pre_process, merge_post_process, merge_fallback) "
                "are only valid with merge_prompt, not merge_fn",
                found="merge_fn with merge hooks",
                hint="Use merge_prompt with hooks, or handle pre/post logic inside merge_fn",
            )
        # Empty models list is a user mistake — reject early
        if self.models is not None and len(self.models) == 0:
            raise ConfigurationError.build(
                "Oracle models= must not be empty",
                expected="at least one model tier",
                found="empty list",
            )
        # Infer n from models length when n wasn't explicitly set
        if self.models is not None and len(self.models) > 0:
            if "n" not in self.model_fields_set:
                object.__setattr__(self, "n", len(self.models))


class Operator(Modifier, frozen=True):
    """Human-in-the-loop modifier: pause graph for human review.

    The compiler inserts a check node after the modified node.
    If the condition is truthy, LangGraph interrupt() is called.
    The graph checkpoints and stops. Resume with:

        run(graph, resume={"approved": True, ...}, config=config)

    Usage:
        validate = Node(...) | Operator(when="any_test_failed")
    """

    when: str  # registered condition function name


class Loop(Modifier, frozen=True):
    """Cycle modifier: repeat a node or sub-construct until a condition is met.

    On a Node: self-loop (output feeds back as input).
    On a Construct: the sub-construct re-runs with its output as the next input.
    Multi-node loop bodies should be expressed as sub-constructs with Loop.

    The ``when`` callable receives the node's latest output and returns True
    to continue looping. On the first iteration, the output may be ``None``
    (the node hasn't produced a value yet), so the callable **must be
    None-safe**::

        Loop(when=lambda d: d is None or d.score < 0.8, max_iterations=5)

    Usage:
        # Self-loop on a node:
        node | Loop(when=lambda d: d is None or d.score < 0.8, max_iterations=5)

        # Multi-node loop body as sub-construct:
        body = construct_from_functions("refine", [review, revise], input=Draft, output=Draft)
        body | Loop(when=lambda d: d is None or d.score < 0.8, max_iterations=10)

        # @node sugar:
        @node(outputs=Draft, loop_when=lambda d: d is None or d.score < 0.8, max_iterations=5)
        def refine(draft: Draft) -> Draft: ...
    """

    when: str | Callable[[Any], bool]  # str (registered condition name) or predicate. True = continue looping.
    max_iterations: int = 10
    on_exhaust: str = "error"  # "error" raises ExecutionError, "last" returns last result

    def model_post_init(self, __context: Any) -> None:
        if self.on_exhaust not in ("error", "last"):
            raise ConfigurationError.build(
                "Invalid Loop on_exhaust value",
                expected="'error' or 'last'",
                found=repr(self.on_exhaust),
            )
        if self.max_iterations < 1:
            raise ConfigurationError.build(
                "Loop max_iterations must be >= 1",
                found=str(self.max_iterations),
            )


# The `route` sentinel that selects Portal's dynamic-flow-definition (dispatch)
# mode. The literal lives HERE ONLY — every layer discriminates the mode through
# `Portal.is_dispatch`, never an inline `route == "decide"` string check.
"""Route-field value meaning "leave the mesh" (design §2.1). Public sentinel."""


class _SlotRule(NamedTuple):
    """One row of the modifier -> ModifierSet-slot mapping.

    Pair legality (which slots may not coexist) is NOT here -- it lives in
    _CONFLICT_DIAGNOSTICS / _DYNAMIC_RULES below, the one gate every
    construction path calls (neograph-jtawq.3). This row is purely the
    modifier-type -> slot -> human-label mapping.
    """

    mod_type: type[Modifier]
    slot: str  # ModifierSet field name to populate
    label: str  # human-facing modifier name for duplicate/unknown-type errors


# The roster: every modifier type this mixin dispatches over. Adding a new
# modifier means adding ONE row here, not a sixth isinstance branch across
# classify_modifiers / combo / to_list / has_modifier / get_modifier.
_SLOT_RULES: tuple[_SlotRule, ...] = (
    _SlotRule(Each, "each", "Each"),
    _SlotRule(Oracle, "oracle", "Oracle"),
    _SlotRule(Loop, "loop", "Loop"),
    _SlotRule(Operator, "operator", "Operator"),
    _SlotRule(Portal, "portal", "Portal"),
)

# Portal excludes EVERY other modifier: it owns the node's outgoing edge, so
# no other modifier's edge/postlude can compose with a Command-returning member
# (D-NO-OPERATOR-COMBO).
_PORTAL_HINT = (
    "Portal owns the node's outgoing edge; place the other modifier on the node "
    "before the mesh entry or after the mesh exit"
)

# Single source of truth for illegal modifier PAIRS (neograph-jtawq.3): every
# 2-subset of _SLOT_RULES' slots is either a legal _COMBO_MAP key or a key
# here, with the pair-specific (message, hint) -- G-SLOT(i) pins this
# totality. The canonical phrasing is fixed PER PAIR (not per landing order),
# so the pipe path's old "whichever modifier landed second" wording collapses
# onto the direct-construction order for free.
_CONFLICT_DIAGNOSTICS: dict[frozenset[str], tuple[str, str]] = {
    frozenset({"each", "loop"}): (
        "Cannot combine Each and Loop on the same item",
        "Use a sub-construct with Loop inside an Each fan-out instead",
    ),
    frozenset({"oracle", "loop"}): (
        "Cannot combine Oracle and Loop on the same item",
        "Use a sub-construct: nest the Loop body inside an Oracle ensemble, or vice versa",
    ),
    frozenset({"each", "portal"}): (
        "Cannot combine Portal and Each on the same item",
        _PORTAL_HINT,
    ),
    frozenset({"oracle", "portal"}): (
        "Cannot combine Portal and Oracle on the same item",
        _PORTAL_HINT,
    ),
    frozenset({"loop", "portal"}): (
        "Cannot combine Portal and Loop on the same item",
        _PORTAL_HINT,
    ),
}

# The ONE instance-dependent rule: Portal(dispatch mode) + Operator depends on
# the Portal INSTANCE's is_dispatch, so it cannot be a static _CONFLICT_DIAGNOSTICS
# entry (which only sees slot NAMES). Kept as DATA (a predicate over a
# ModifierSet-shaped object, plus the message/hint) so the message literal
# still lives in a table, not a raise buried in a function body.
_DYNAMIC_RULES: tuple[tuple[Callable[[Any], bool], str, str], ...] = (
    (
        lambda ms: ms.portal is not None and ms.operator is not None and ms.portal.is_dispatch,
        "Cannot combine Portal (dispatch mode) and Operator on the same item",
        "Operator+Portal approval gate is defined for PEER mode (to=[...]) only",
    ),
)


def _validate_slot_set(slots: frozenset[str]) -> None:
    """The ONE gate deciding whether a set of occupied ModifierSet slots is
    legal -- both ``model_post_init`` (direct construction) and
    ``with_modifier`` (the pipe path) call this before anything else.

    Roster-ordered pair scan, NOT hash/set order: a 3+-member superset can
    contain more than one illegal pair (e.g. {each, oracle, loop} contains
    both {each, loop} and {oracle, loop}); which is reported first must be
    deterministic, not coin-flipped by CPython's per-process str-hash
    randomization. Iterating in ``_SLOT_RULES`` order reproduces the
    precedence the old hand-coded arms had (each+loop, then oracle+loop,
    then the portal pairs).
    """
    ordered = [r.slot for r in _SLOT_RULES if r.slot in slots]
    for a, b in itertools.combinations(ordered, 2):
        diagnosis = _CONFLICT_DIAGNOSTICS.get(frozenset({a, b}))
        if diagnosis is not None:
            message, hint = diagnosis
            raise ConstructError.build(message, hint=hint)


def _check_dynamic_rules(ms: ModifierSet) -> None:
    """Run the one instance-dependent rule against an already-built ModifierSet."""
    for predicate, message, hint in _DYNAMIC_RULES:
        if predicate(ms):
            raise ConstructError.build(message, hint=hint)


class ModifierSet(BaseModel, frozen=True):
    """Validated, typed modifier configuration.

    Cannot be constructed with an invalid combination -- pydantic
    model_post_init rejects it. Replaces list[Modifier] everywhere.

    Each slot is a single optional value, so duplicate modifiers are
    structurally impossible.
    """

    each: Each | None = None
    oracle: Oracle | None = None
    loop: Loop | None = None
    operator: Operator | None = None
    portal: Portal | None = None

    @property
    def combo(self) -> ModifierCombo:
        """Classify this set into a ModifierCombo enum value."""
        has = frozenset(r.slot for r in _SLOT_RULES if getattr(self, r.slot) is not None)
        return _COMBO_MAP[has]

    def model_post_init(self, __context: Any) -> None:
        # Direct construction: model_copy SKIPS model_post_init (pydantic v2),
        # so with_modifier (the pipe path) cannot reuse this method and instead
        # calls the same two functions itself, before/after its own model_copy.
        # Both paths route through ONE gate for slot-pair legality
        # (_validate_slot_set) and ONE gate for the instance-dependent rule
        # (_check_dynamic_rules) -- neograph-jtawq.3.
        occupied = frozenset(r.slot for r in _SLOT_RULES if getattr(self, r.slot) is not None)
        _validate_slot_set(occupied)
        _check_dynamic_rules(self)

    def with_modifier(self, mod: Modifier) -> ModifierSet:
        """Return a new ModifierSet with the given modifier added.

        Raises ConstructError for duplicate modifiers (slot already occupied)
        and for illegal combinations. The modifier-type -> slot mapping comes
        from ``_SLOT_RULES``; pair legality from ``_validate_slot_set`` /
        ``_check_dynamic_rules`` -- the SAME gates ``model_post_init`` calls,
        so a new modifier is described once, not open-coded across two paths.
        """
        rule = next((r for r in _SLOT_RULES if isinstance(mod, r.mod_type)), None)
        if rule is None:
            raise ConstructError.build(
                "Unknown modifier type",
                expected=", ".join(r.label for r in _SLOT_RULES),
                found=type(mod).__name__,
            )

        # Duplicate: this slot is already occupied. A set union can't detect
        # this (the duplicate arm cannot move into _validate_slot_set).
        if getattr(self, rule.slot) is not None:
            raise ConstructError.build(
                f"Duplicate {rule.label} modifier",
                found=f"A{'n' if rule.label[0] in 'AEIOU' else ''} {rule.label} is already applied to this item",
                hint="Use a sub-construct if you need nested composition",
            )

        prospective = frozenset(r.slot for r in _SLOT_RULES if getattr(self, r.slot) is not None) | {rule.slot}
        _validate_slot_set(prospective)

        result = self.model_copy(update={rule.slot: mod})
        # Dynamic check runs AFTER model_copy -- it needs the actual Portal
        # instance's is_dispatch, which a slot-NAME set cannot express.
        _check_dynamic_rules(result)
        return result

    def to_list(self) -> list[Modifier]:
        """Return modifiers as a list (backward compat bridge)."""
        return [getattr(self, r.slot) for r in _SLOT_RULES if getattr(self, r.slot) is not None]
