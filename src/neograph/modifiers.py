"""Modifiers — composable pipeline behaviors applied via the | operator.

    node | Oracle(n=3, merge_prompt="rw/merge")
    node | Each(over="clusters.clusters", key="label")
    node | Operator(when="has_open_questions")
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel

from neograph.errors import ConfigurationError


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

    def __getattr__(self, name: str) -> "_PathRecorder":
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
    Requires a `modifiers: list[Modifier]` field on the subclass.
    """

    modifiers: list[Modifier]

    def __or__(self, modifier: Modifier) -> Self:
        """Compose modifiers via pipe: obj | Oracle(n=3) | Operator(when=...)"""
        from neograph.errors import ConstructError

        # Each + Loop mutual exclusion
        if isinstance(modifier, Loop) and self.has_modifier(Each):
            msg = (
                "Cannot combine Each and Loop on the same item. "
                "Use a sub-construct with Loop inside an Each fan-out instead."
            )
            raise ConstructError(msg)
        if isinstance(modifier, Each) and self.has_modifier(Loop):
            msg = (
                "Cannot combine Each and Loop on the same item. "
                "Use a sub-construct with Loop inside an Each fan-out instead."
            )
            raise ConstructError(msg)

        result = self.model_copy(update={"modifiers": [*self.modifiers, modifier]})
        # Loop validation at | time: check type compatibility immediately.
        if isinstance(modifier, Loop):
            if hasattr(result, 'outputs'):
                # Node: validate output compat with inputs
                from neograph._construct_validation import validate_loop_self_edge
                validate_loop_self_edge(result)
            elif hasattr(result, 'output') and hasattr(result, 'input'):
                # Construct: validate output compat with input
                from neograph._construct_validation import validate_loop_construct
                validate_loop_construct(result)
        return result

    def has_modifier(self, modifier_type: type[Modifier]) -> bool:
        """Check if a specific modifier is applied."""
        return any(isinstance(m, modifier_type) for m in self.modifiers)

    def get_modifier(self, modifier_type: type[Modifier]) -> Modifier | None:
        """Get the first modifier of a given type, or None."""
        for m in self.modifiers:
            if isinstance(m, modifier_type):
                return m
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


class Oracle(Modifier):
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

    def model_post_init(self, __context: Any) -> None:
        if not self.merge_prompt and not self.merge_fn:
            msg = "Oracle requires either merge_prompt (LLM judge) or merge_fn (scripted function)."
            raise ConfigurationError(msg)
        if self.merge_prompt and self.merge_fn:
            msg = "Oracle accepts merge_prompt or merge_fn, not both."
            raise ConfigurationError(msg)
        # Infer n from models length when n wasn't explicitly set
        if self.models is not None and len(self.models) > 0:
            if 'n' not in self.model_fields_set:
                object.__setattr__(self, 'n', len(self.models))


class Each(Modifier):
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


def split_each_path(over: str) -> tuple[str, tuple[str, ...]]:
    """Parse an `Each.over` dotted path into (root_field, remaining_segments).

    Single point of truth for the path grammar. Both the assembly-time
    type walker in `construct.py` and the runtime value walker in
    `compiler.py` consume this so future extensions to the syntax (indexing,
    wildcards, escaping) land in one place.
    """
    parts = over.split(".")
    return parts[0], tuple(parts[1:])


class Operator(Modifier):
    """Human-in-the-loop modifier: pause graph for human review.

    The compiler inserts a check node after the modified node.
    If the condition is truthy, LangGraph interrupt() is called.
    The graph checkpoints and stops. Resume with:

        run(graph, resume={"approved": True, ...}, config=config)

    Usage:
        validate = Node(...) | Operator(when="any_test_failed")
    """

    when: str       # registered condition function name


class Loop(Modifier):
    """Cycle modifier: repeat a node or sub-construct until a condition is met.

    On a Node: self-loop (output feeds back as input).
    On a Construct: the sub-construct re-runs with its output as the next input.
    Multi-node loop bodies should be expressed as sub-constructs with Loop.

    Usage:
        # Self-loop on a node:
        node | Loop(when=lambda d: d.score < 0.8, max_iterations=5)

        # Multi-node loop body as sub-construct:
        body = construct_from_functions("refine", [review, revise], input=Draft, output=Draft)
        body | Loop(when=lambda d: d.score < 0.8, max_iterations=10)

        # @node sugar:
        @node(outputs=Draft, loop_when=lambda d: d.score < 0.8, max_iterations=5)
        def refine(draft: Draft) -> Draft: ...
    """

    when: Any           # str (registered condition name) or Callable. True = continue looping.
    max_iterations: int = 10
    on_exhaust: str = "error"           # "error" raises ExecutionError, "last" returns last result
    history: bool = False               # collect each iteration's output in state

    def model_post_init(self, __context: Any) -> None:
        if self.max_iterations < 1:
            msg = f"Loop max_iterations must be >= 1, got {self.max_iterations}."
            raise ConfigurationError(msg)
