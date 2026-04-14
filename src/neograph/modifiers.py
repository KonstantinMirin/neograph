"""Modifiers — composable pipeline behaviors applied via the | operator.

    node | Oracle(n=3, merge_prompt="rw/merge")
    node | Each(over="clusters.clusters", key="label")
    node | Operator(when="has_open_questions")
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any, Self

from neograph._dev_warnings import dev_warn

from pydantic import BaseModel, field_validator

from neograph.errors import ConfigurationError, ConstructError


class ModifierCombo(Enum):
    """Exhaustive enumeration of valid modifier combinations.

    Every dispatch site (compiler, state, factory) matches on this enum
    instead of ad-hoc has_modifier() chains. Adding a new combo forces
    handling at every site.
    """
    BARE = auto()              # no modifiers
    EACH = auto()              # Each only
    ORACLE = auto()            # Oracle only
    LOOP = auto()              # Loop only
    OPERATOR = auto()          # Operator only
    EACH_ORACLE = auto()       # Each + Oracle (fusion)
    EACH_OPERATOR = auto()     # Each + Operator
    ORACLE_OPERATOR = auto()   # Oracle + Operator
    LOOP_OPERATOR = auto()     # Loop + Operator
    EACH_ORACLE_OPERATOR = auto()  # Each + Oracle + Operator


def classify_modifiers(item: Any) -> tuple[ModifierCombo, dict]:
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
        return ms.combo, mods

    # Fallback for duck-typed items (e.g. _BranchNode)
    get_mod = getattr(item, "get_modifier", None)
    if get_mod is None:
        return ModifierCombo.BARE, {}

    each = get_mod(Each)
    oracle = get_mod(Oracle)
    loop = get_mod(Loop)
    operator = get_mod(Operator)

    mods = {}
    if each:
        mods["each"] = each
    if oracle:
        mods["oracle"] = oracle
    if loop:
        mods["loop"] = loop
    if operator:
        mods["operator"] = operator

    # Map to enum
    has = frozenset(mods.keys())
    combo_map = {
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
    }

    combo = combo_map.get(has)
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
            if hasattr(result, 'outputs'):
                # Node: validate output compat with inputs
                from neograph._construct_validation import validate_loop_self_edge
                validate_loop_self_edge(result)
            elif hasattr(result, 'output') and hasattr(result, 'input'):
                # Construct-level Loop with history=True — not supported yet
                if modifier.history:
                    raise ConstructError.build(
                        "Loop(history=True) is not supported on Constructs",
                        hint="history tracking is only available on Node-level Loops",
                    )
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
        return None

    def map(self, source: Any, *, key: str) -> Self:
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
                msg = (
                    "Node.map() lambda must be a pure attribute-access chain "
                    "like `lambda s: s.upstream_node.field`; "
                    f"got error when introspecting: {exc}."
                )
                raise TypeError(msg) from exc
            if not isinstance(result, _PathRecorder):
                msg = (
                    "Node.map() lambda must return an attribute-access chain "
                    f"like `s.upstream_node.field`; got {type(result).__name__}."
                )
                raise TypeError(msg)
            path = result._neo_path
            if not path:
                msg = (
                    "Node.map() lambda must access at least one attribute, "
                    "e.g. `lambda s: s.make_clusters.groups`."
                )
                raise TypeError(msg)
            over = ".".join(path)
        else:
            msg = (
                "Node.map() source must be a string path or a lambda; "
                f"got {type(source).__name__}."
            )
            raise TypeError(msg)

        return self | Each(over=over, key=key)


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

        # Multi-model with redundancy (round-robin):
        node | Oracle(n=9, models=["reason", "fast", "creative"], merge_fn="pick_best")
    """

    n: int = 3
    models: list[str] | None = None  # per-generator model tiers (round-robin)
    merge_prompt: str | None = None
    merge_model: str = "reason"
    merge_fn: str | None = None  # registered scripted function name

    @field_validator('n')
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
        # Empty models list is a user mistake — reject early
        if self.models is not None and len(self.models) == 0:
            raise ConfigurationError.build(
                "Oracle models= must not be empty",
                expected="at least one model tier",
                found="empty list",
            )
        # Infer n from models length when n wasn't explicitly set
        if self.models is not None and len(self.models) > 0:
            if 'n' not in self.model_fields_set:
                object.__setattr__(self, 'n', len(self.models))


class Each(Modifier, frozen=True):
    """Fan-out modifier: dispatch parallel instances over a collection.

    The compiler expands this into:
    1. Router node that iterates over the collection field in state
    2. Send() per item with the item as payload
    3. Barrier node with defer=True that collects results

    Usage:
        match_verify = Node(...) | Each(over="clusters.clusters", key="label")
    """

    over: str       # dotted path to collection in state (e.g., "clusters.clusters")
    key: str        # field on each item used as the dispatch key

    @field_validator('over')
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

    when: str       # registered condition function name


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

    when: Any           # str (registered condition name) or Callable. True = continue looping.
    max_iterations: int = 10
    on_exhaust: str = "error"           # "error" raises ExecutionError, "last" returns last result
    history: bool = False               # collect each iteration's output in state

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
        combo_map = {
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
        }
        return combo_map[frozenset(has)]

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

    def with_modifier(self, mod: Modifier) -> ModifierSet:
        """Return a new ModifierSet with the given modifier added.

        Raises ConstructError for duplicate modifiers (slot already occupied)
        and for illegal combinations (Each+Loop, Oracle+Loop).
        """


        if isinstance(mod, Each):
            if self.each is not None:
                raise ConstructError.build(
                    "Duplicate Each modifier",
                    found="An Each is already applied to this item",
                    hint="Use a sub-construct if you need nested composition",
                )
            # Each + Loop mutual exclusion
            if self.loop is not None:
                raise ConstructError.build(
                    "Cannot combine Each and Loop on the same item",
                    hint="Use a sub-construct with Loop inside an Each fan-out instead",
                )
            return self.model_copy(update={"each": mod})
        elif isinstance(mod, Oracle):
            if self.oracle is not None:
                raise ConstructError.build(
                    "Duplicate Oracle modifier",
                    found="An Oracle is already applied to this item",
                    hint="Use a sub-construct if you need nested composition",
                )
            # Oracle + Loop mutual exclusion
            if self.loop is not None:
                raise ConstructError.build(
                    "Cannot combine Oracle and Loop on the same item",
                    hint="Use a sub-construct: nest the Loop body inside an Oracle ensemble, or vice versa",
                )
            return self.model_copy(update={"oracle": mod})
        elif isinstance(mod, Loop):
            if self.loop is not None:
                raise ConstructError.build(
                    "Duplicate Loop modifier",
                    found="A Loop is already applied to this item",
                    hint="Use a sub-construct if you need nested composition",
                )
            # Loop + Each mutual exclusion
            if self.each is not None:
                raise ConstructError.build(
                    "Cannot combine Each and Loop on the same item",
                    hint="Use a sub-construct with Loop inside an Each fan-out instead",
                )
            # Loop + Oracle mutual exclusion
            if self.oracle is not None:
                raise ConstructError.build(
                    "Cannot combine Oracle and Loop on the same item",
                    hint="Use a sub-construct: nest the Loop body inside an Oracle ensemble, or vice versa",
                )
            return self.model_copy(update={"loop": mod})
        elif isinstance(mod, Operator):
            if self.operator is not None:
                raise ConstructError.build(
                    "Duplicate Operator modifier",
                    found="An Operator is already applied to this item",
                    hint="Use a sub-construct if you need nested composition",
                )
            return self.model_copy(update={"operator": mod})
        else:
            raise ConstructError.build(
                "Unknown modifier type",
                expected="Each, Oracle, Loop, or Operator",
                found=type(mod).__name__,
            )

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
        return result
