"""Modifiers — composable pipeline behaviors applied via the | operator.

    node | Oracle(n=3, merge_prompt="rw/merge")
    node | Each(over="clusters.clusters", key="label")
    node | Operator(when="has_open_questions")
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


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
        object.__setattr__(self, "_neo_path", path)

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

    def __or__(self, modifier: Modifier):
        """Compose modifiers via pipe: obj | Oracle(n=3) | Operator(when=...)"""
        return self.model_copy(update={"modifiers": [*self.modifiers, modifier]})

    def has_modifier(self, modifier_type: type[Modifier]) -> bool:
        """Check if a specific modifier is applied."""
        return any(isinstance(m, modifier_type) for m in self.modifiers)

    def get_modifier(self, modifier_type: type[Modifier]) -> Modifier | None:
        """Get the first modifier of a given type, or None."""
        for m in self.modifiers:
            if isinstance(m, modifier_type):
                return m
        return None

    def map(self, source: Any, *, key: str):
        """Fan-out over a collection — sugar over `| Each(over=..., key=...)`.

        `source` can be:

        1. A lambda taking the state and returning an attribute chain::

               verify.map(lambda s: s.make_clusters.groups, key="label")

           The lambda is introspected once at definition time via a recording
           proxy. Pyright/Pylance catch typos in `.make_clusters.groups`, and
           renaming the upstream node surfaces as a red squiggle — the
           refactor-safety win over string paths.

        2. A string path (escape hatch, equivalent to `| Each(...)`)::

               verify.map("make_clusters.groups", key="label")

        Returns a new instance with an `Each` modifier appended. Fully
        equivalent to `self | Each(over=..., key=key)` — the compiler, state
        builder, and factory all see the same Each modifier as before.
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
                    f"got error when introspecting: {exc}"
                )
                raise TypeError(msg) from exc
            if not isinstance(result, _PathRecorder):
                msg = (
                    "Node.map() lambda must return an attribute-access chain "
                    f"like `s.upstream_node.field`; got {type(result).__name__}"
                )
                raise TypeError(msg)
            path = result._neo_path
            if not path:
                msg = (
                    "Node.map() lambda must access at least one attribute, "
                    "e.g. `lambda s: s.make_clusters.groups`"
                )
                raise TypeError(msg)
            over = ".".join(path)
        else:
            msg = (
                "Node.map() source must be a string path or a lambda; "
                f"got {type(source).__name__}"
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
        # LLM merge:
        node | Oracle(n=3, merge_prompt="rw/decompose-merge")

        # Scripted merge:
        node | Oracle(n=3, merge_fn="combine_variants")
    """

    n: int = 3
    merge_prompt: str | None = None
    merge_model: str = "reason"
    merge_fn: str | None = None  # registered scripted function name

    def model_post_init(self, __context: Any) -> None:
        if not self.merge_prompt and not self.merge_fn:
            msg = "Oracle requires either merge_prompt (LLM judge) or merge_fn (scripted function)."
            raise ValueError(msg)
        if self.merge_prompt and self.merge_fn:
            msg = "Oracle accepts merge_prompt or merge_fn, not both."
            raise ValueError(msg)


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
