"""Modifiers — composable pipeline behaviors applied via the | operator.

node | Oracle(n=3, merge_prompt="rw/merge")
node | Each(over="clusters.clusters", key="label")
node | Operator(when="has_open_questions")
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, field_validator
from typing_extensions import TypeVar

from neograph._dev_warnings import dev_warn
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


@runtime_checkable
class MergePreProcess(Protocol[_Variant]):
    """Replaces the default ``{variants: ..., **upstream}`` input_data
    construction for the ``merge_prompt`` path. Returns the data passed
    verbatim to ``invoke_structured`` -- which accepts ``BaseModel | dict | str``.
    """

    def __call__(self, variants: list[_Variant]) -> BaseModel | dict[str, Any] | str: ...


@runtime_checkable
class MergePostProcess(Protocol[_PostResult, _Variant]):
    """Transforms the parsed LLM merge result before it is written to state."""

    def __call__(self, result: _PostResult, variants: list[_Variant]) -> _PostResult: ...


@runtime_checkable
class MergeFallback(Protocol[_Variant, _FallbackResult]):
    """Catches errors from ``invoke_structured`` during merge. Returns a
    deterministic fallback result instead of propagating the exception.
    """

    def __call__(self, variants: list[_Variant], error: Exception) -> _FallbackResult: ...


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
    KEYMAKER = auto()  # Keymaker only (dynamic handoff; excludes all others — D-NO-OPERATOR-COMBO)
    EACH_ORACLE = auto()  # Each + Oracle (fusion)
    EACH_OPERATOR = auto()  # Each + Operator
    ORACLE_OPERATOR = auto()  # Oracle + Operator
    LOOP_OPERATOR = auto()  # Loop + Operator
    EACH_ORACLE_OPERATOR = auto()  # Each + Oracle + Operator


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
    frozenset({"keymaker"}): ModifierCombo.KEYMAKER,
}


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
        mods: dict[str, Any] = {}
        if ms.each is not None:
            mods["each"] = ms.each
        if ms.oracle is not None:
            mods["oracle"] = ms.oracle
        if ms.loop is not None:
            mods["loop"] = ms.loop
        if ms.operator is not None:
            mods["operator"] = ms.operator
        if ms.keymaker is not None:
            mods["keymaker"] = ms.keymaker
        return ms.combo, mods

    # Fallback for duck-typed items (e.g. _BranchNode)
    get_mod = getattr(item, "get_modifier", None)
    if get_mod is None:
        return ModifierCombo.BARE, {}

    each = get_mod(Each)
    oracle = get_mod(Oracle)
    loop = get_mod(Loop)
    operator = get_mod(Operator)
    keymaker = get_mod(Keymaker)

    mods = {}
    if each:
        mods["each"] = each
    if oracle:
        mods["oracle"] = oracle
    if loop:
        mods["loop"] = loop
    if operator:
        mods["operator"] = operator
    if keymaker:
        mods["keymaker"] = keymaker

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


class Modifier(BaseModel, frozen=True):
    """Base class for node modifiers. Applied via Node.__or__."""


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
        if modifier_type is Each:
            return self.modifier_set.each is not None
        if modifier_type is Oracle:
            return self.modifier_set.oracle is not None
        if modifier_type is Loop:
            return self.modifier_set.loop is not None
        if modifier_type is Operator:
            return self.modifier_set.operator is not None
        if modifier_type is Keymaker:
            return self.modifier_set.keymaker is not None
        return False

    def get_modifier(self, modifier_type: type[Modifier]) -> Modifier | None:
        """Get the modifier of a given type, or None."""
        if modifier_type is Each:
            return self.modifier_set.each
        if modifier_type is Oracle:
            return self.modifier_set.oracle
        if modifier_type is Loop:
            return self.modifier_set.loop
        if modifier_type is Operator:
            return self.modifier_set.operator
        if modifier_type is Keymaker:
            return self.modifier_set.keymaker
        return None

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


class EachFailure(BaseModel, frozen=True):
    """Typed per-item failure written into an Each barrier under ``on_error='collect'``.

    Replaces a thrown item's result in the keyed barrier dict so the barrier
    always completes with one entry per planned key. Consumers assert
    set-equality over planned keys and branch on ``isinstance(v, EachFailure)``.
    """

    key: str  # the Each dispatch key of the item that failed
    error_type: str  # exception class name (e.g., "RuntimeError")
    message: str  # str() of the caught exception


class Each(Modifier, frozen=True):
    """Fan-out modifier: dispatch parallel instances over a collection.

    The compiler expands this into:
    1. Router node that iterates over the collection field in state
    2. Send() per item with the item as payload
    3. Barrier node with defer=True that collects results

    Usage:
        match_verify = Node(...) | Each(over="clusters.clusters", key="label")

    ``on_error`` controls per-item fault handling:
    - ``'raise'`` (default): a thrown item aborts the whole fan-out run.
    - ``'collect'``: a thrown item is caught and keyed into the barrier as a
      typed ``EachFailure`` instead of aborting; the barrier always completes.
    """

    over: str  # dotted path to collection in state (e.g., "clusters.clusters")
    key: str  # field on each item used as the dispatch key
    on_error: Literal["raise", "collect"] = "raise"

    @field_validator("over")
    @classmethod
    def _validate_over(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Each.over must not be empty")
        return v


def split_each_path(over: str) -> tuple[str, tuple[str, ...]]:
    """Parse an `Each.over` dotted path into (root_field, remaining_segments).

    Single point of truth for the path grammar. Both the assembly-time
    type walker in `construct.py` and the runtime value walker in
    `compiler.py` consume this so future extensions to the syntax (indexing,
    wildcards, escaping) land in one place.
    """
    parts = over.split(".")
    return parts[0], tuple(parts[1:])


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


HANDOFF_END = "__end__"

# The `route` sentinel that selects Keymaker's dynamic-flow-definition (dispatch)
# mode. The literal lives HERE ONLY — every layer discriminates the mode through
# `Keymaker.is_dispatch`, never an inline `route == "decide"` string check.
DISPATCH_ROUTE = "decide"
"""Route-field value meaning "leave the mesh" (design §2.1). Public sentinel."""


class Keymaker(Modifier, frozen=True):
    """Dynamic-handoff modifier — one modifier with two modes (design §2.1).

    Mode (a) — peer routing (``peers=[...]``): a node picks its successor at
    runtime from a declared peer set. Lowers to ``Command(goto=<peer>)`` (T2).
    Mode (b) — dynamic flow definition (``route="decide"``): the node emits the
    spec of the next flow; neograph validates -> compiles -> dispatches it.

    Usage:
        # peer routing (typed swarm):
        Node("billing", ...) | Keymaker(peers=["triage", "technical"], max_hops=6)

        # dynamic flow definition:
        Node("planner", ...) | Keymaker(route="decide", spec_field="spec",
                                        input_field="dispatch_input", output=Summary)

    The mode is discriminated in ``model_post_init`` (mirrors ``Loop``):
    ``peers`` set => peer mode (``route`` must not be ``"decide"``);
    ``route == "decide"`` => dispatch mode (requires ``spec_field`` /
    ``input_field`` / ``output``; forbids the peer-mode knobs). Neither or both
    => ``ConfigurationError``. Excludes every other modifier (Each/Oracle/Loop/
    Operator) — Keymaker owns the node's outgoing edge (D-NO-OPERATOR-COMBO).
    """

    # arbitrary_types_allowed: for the ``output`` class field and the
    # ``scripted`` / ``conditions`` callable registries (mode b).
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # -- mode (a): peer routing --
    peers: list[str] | None = None  # declared successor names (directed, per-node)
    route: str = "goto"  # mode (a): routing FIELD on the payload model; mode (b): literal "decide"
    max_hops: int = 10  # mesh budget; settable ONLY on the entry member
    on_exhaust: Literal["error", "exit"] = "error"

    # -- mode (b): dynamic flow definition (route="decide") --
    spec_field: str | None = None  # output-model field holding the emitted Spec dict
    input_field: str | None = None  # output-model field holding the dispatch input dict
    output: type[BaseModel] | str | None = None  # REQUIRED in mode (b): dispatched-flow output type
    scripted: dict[str, Callable] | None = None  # building-block registry for the emitted flow
    conditions: dict[str, Callable] | None = None  # condition registry for the emitted flow
    on_invalid: Literal["raise"] = "raise"  # v1: raise only (kwarg reserved)

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

    def model_post_init(self, __context: Any) -> None:
        is_peer = self.peers is not None
        is_dispatch = self.is_dispatch
        if is_peer and is_dispatch:
            raise ConfigurationError.build(
                "Keymaker cannot be both peer mode and dispatch mode",
                expected="peers=[...] (peer routing) XOR route='decide' (dynamic flow)",
                found="peers set AND route=='decide'",
            )
        if not is_peer and not is_dispatch:
            raise ConfigurationError.build(
                "Keymaker requires a mode",
                expected="peers=[...] (peer routing) or route='decide' (dynamic flow)",
                found="neither peers nor route='decide' provided",
            )
        if is_peer:
            if self.max_hops < 1:
                raise ConfigurationError.build(
                    "Keymaker max_hops must be >= 1",
                    found=str(self.max_hops),
                )
        else:  # dispatch mode (route == "decide")
            missing = [f for f in ("spec_field", "input_field", "output") if getattr(self, f) is None]
            if missing:
                raise ConfigurationError.build(
                    "Keymaker dispatch mode requires spec_field, input_field, and output",
                    expected="spec_field=, input_field=, output=",
                    found=f"missing: {missing}",
                )
            forbidden = [k for k in ("max_hops", "on_exhaust") if k in self.model_fields_set]
            if forbidden:
                raise ConfigurationError.build(
                    "Keymaker dispatch mode forbids peer-mode knobs",
                    expected="no max_hops/on_exhaust in dispatch mode",
                    found=f"peer-mode knobs set: {forbidden}",
                )


class _SlotRule(NamedTuple):
    """One row of the modifier -> ModifierSet-slot mapping.

    ``excludes`` lists mutual-exclusion conflicts as (conflicting_slot,
    error_message, hint) triples, checked when the incoming modifier lands.
    """

    mod_type: type[Modifier]
    slot: str  # ModifierSet field name to populate
    label: str  # human-facing modifier name for duplicate errors
    excludes: tuple[tuple[str, str, str], ...]


# Single source of truth for ModifierSet.with_modifier: which slot each
# modifier occupies and which sibling slots it may not coexist with. Adding a
# new modifier means adding ONE row here, not a fifth isinstance branch.
_EACH_LOOP_CONFLICT = (
    "Cannot combine Each and Loop on the same item",
    "Use a sub-construct with Loop inside an Each fan-out instead",
)
_ORACLE_LOOP_CONFLICT = (
    "Cannot combine Oracle and Loop on the same item",
    "Use a sub-construct: nest the Loop body inside an Oracle ensemble, or vice versa",
)
# Keymaker excludes EVERY other modifier: it owns the node's outgoing edge, so
# no other modifier's edge/postlude can compose with a Command-returning member
# (D-NO-OPERATOR-COMBO). The excludes are RECIPROCAL — listed on BOTH the
# Keymaker row and each sibling row — so a conflict is rejected with a clean
# ConstructError regardless of pipe order (review MEDIUM-2). Without the sibling-
# side entries, ``node | Keymaker() | Each()`` (Keymaker landing first) would
# slip past ``with_modifier`` and raise a raw KeyError in ``ModifierSet.combo``.
_KEYMAKER_HINT = (
    "Keymaker owns the node's outgoing edge; place the other modifier on the node "
    "before the mesh entry or after the mesh exit"
)


def _km_conflict(this_label: str) -> tuple[str, str]:
    """(message, hint) for a ``this_label`` slot row rejecting a Keymaker peer.

    Names ``this_label`` so the fixture/error regex (e.g. ``[Ll]oop`` /
    ``[Oo]perator``) matches whichever modifier landed second.
    """
    return (f"Cannot combine {this_label} and Keymaker on the same item", _KEYMAKER_HINT)


_SLOT_RULES: tuple[_SlotRule, ...] = (
    _SlotRule(Each, "each", "Each", (("loop", *_EACH_LOOP_CONFLICT), ("keymaker", *_km_conflict("Each")))),
    _SlotRule(Oracle, "oracle", "Oracle", (("loop", *_ORACLE_LOOP_CONFLICT), ("keymaker", *_km_conflict("Oracle")))),
    _SlotRule(
        Loop,
        "loop",
        "Loop",
        (("each", *_EACH_LOOP_CONFLICT), ("oracle", *_ORACLE_LOOP_CONFLICT), ("keymaker", *_km_conflict("Loop"))),
    ),
    _SlotRule(Operator, "operator", "Operator", (("keymaker", *_km_conflict("Operator")),)),
    _SlotRule(
        Keymaker,
        "keymaker",
        "Keymaker",
        (
            ("each", "Cannot combine Keymaker and Each on the same item", _KEYMAKER_HINT),
            ("oracle", "Cannot combine Keymaker and Oracle on the same item", _KEYMAKER_HINT),
            ("loop", "Cannot combine Keymaker and Loop on the same item", _KEYMAKER_HINT),
            ("operator", "Cannot combine Keymaker and Operator on the same item", _KEYMAKER_HINT),
        ),
    ),
)


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
    keymaker: Keymaker | None = None

    @property
    def combo(self) -> ModifierCombo:
        """Classify this set into a ModifierCombo enum value."""
        has: set[str] = set()
        if self.each is not None:
            has.add("each")
        if self.oracle is not None:
            has.add("oracle")
        if self.loop is not None:
            has.add("loop")
        if self.operator is not None:
            has.add("operator")
        if self.keymaker is not None:
            has.add("keymaker")
        return _COMBO_MAP[frozenset(has)]

    def model_post_init(self, __context: Any) -> None:
        # Each + Loop mutual exclusion
        if self.each is not None and self.loop is not None:
            raise ConstructError.build(
                "Cannot combine Each and Loop on the same item",
                hint="Use a sub-construct with Loop inside an Each fan-out instead",
            )
        # Oracle + Loop mutual exclusion
        if self.oracle is not None and self.loop is not None:
            raise ConstructError.build(
                "Cannot combine Oracle and Loop on the same item",
                hint="Use a sub-construct: nest the Loop body inside an Oracle ensemble, or vice versa",
            )
        # Keymaker excludes every other modifier (review M2: this direct-
        # construct path uses hard-coded pairwise arms — it does NOT read
        # _SLOT_RULES, so without these arms a direct ModifierSet(keymaker=...,
        # loop=...) would silently pass while the pipe path rejects, the exact
        # parity hazard). Mirrors the D-NO-OPERATOR-COMBO reciprocal excludes.
        if self.keymaker is not None and self.each is not None:
            raise ConstructError.build("Cannot combine Keymaker and Each on the same item", hint=_KEYMAKER_HINT)
        if self.keymaker is not None and self.oracle is not None:
            raise ConstructError.build("Cannot combine Keymaker and Oracle on the same item", hint=_KEYMAKER_HINT)
        if self.keymaker is not None and self.loop is not None:
            raise ConstructError.build("Cannot combine Keymaker and Loop on the same item", hint=_KEYMAKER_HINT)
        if self.keymaker is not None and self.operator is not None:
            raise ConstructError.build("Cannot combine Keymaker and Operator on the same item", hint=_KEYMAKER_HINT)

    def with_modifier(self, mod: Modifier) -> ModifierSet:
        """Return a new ModifierSet with the given modifier added.

        Raises ConstructError for duplicate modifiers (slot already occupied)
        and for illegal combinations (Each+Loop, Oracle+Loop).

        The modifier-type -> slot mapping and its mutual-exclusion rules are
        driven from the single ``_SLOT_RULES`` table so a new modifier is
        described once, not open-coded across four isinstance branches.
        """

        rule = next((r for r in _SLOT_RULES if isinstance(mod, r.mod_type)), None)
        if rule is None:
            raise ConstructError.build(
                "Unknown modifier type",
                expected="Each, Oracle, Loop, or Operator",
                found=type(mod).__name__,
            )

        # Duplicate: this slot is already occupied.
        if getattr(self, rule.slot) is not None:
            raise ConstructError.build(
                f"Duplicate {rule.label} modifier",
                found=f"A{'n' if rule.label[0] in 'AEIOU' else ''} {rule.label} is already applied to this item",
                hint="Use a sub-construct if you need nested composition",
            )

        # Mutual-exclusion: any conflicting slot already occupied.
        for conflict_slot, message, hint in rule.excludes:
            if getattr(self, conflict_slot) is not None:
                raise ConstructError.build(message, hint=hint)

        return self.model_copy(update={rule.slot: mod})

    def to_list(self) -> list[Modifier]:
        """Return modifiers as a list (backward compat bridge)."""
        result: list[Modifier] = []
        if self.each is not None:
            result.append(self.each)
        if self.oracle is not None:
            result.append(self.oracle)
        if self.loop is not None:
            result.append(self.loop)
        if self.operator is not None:
            result.append(self.operator)
        if self.keymaker is not None:
            result.append(self.keymaker)
        return result
