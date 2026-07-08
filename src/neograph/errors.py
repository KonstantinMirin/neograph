"""Error hierarchy for neograph.

All neograph-originated errors inherit from ``NeographError``.
Downstream code that wants a broad catch can use ``except NeographError``;
code that needs to distinguish failure modes catches the specific subclass.

THE RULE (parentage by lifecycle phase — do NOT parent a new error ad hoc)::

    a failure raised while the graph is EXECUTING (a run)   -> ExecutionError
    a failure during Construct assembly / validation         -> ConstructError
    a failure during compile()                               -> CompileError
    a bad/missing configuration or resume-time precondition  -> ConfigurationError

The motivating incident (review PAT-01 / HIGH-01): error types were parented
one ticket at a time, so ``ResourceExpiredError`` and ``NonIdempotentReplayError``
— both raised DURING execution — sat under bare ``NeographError`` while the
equally-runtime ``PromptVarMissing`` sat under ``ExecutionError``. A consumer
writing ``except ExecutionError`` around ``run()`` silently missed the first two.
The fix is this rule + a table-driven parentage guard in
``tests/test_error_hierarchy.py`` (every public error type is pinned to its
category; a new type must be added there or the guard fails).

Hierarchy::

    NeographError (Exception)
        ConstructError (+ ValueError for backward compat)
        CompileError
        ConfigurationError
            CheckpointSchemaError      (resume-time precondition mismatch)
        ExecutionError
            PromptVarMissing
            StateMissingError
            NodeOutputError
            NonIdempotentReplayError
            ResourceExpiredError

``ConstructError`` keeps ``ValueError`` as a second parent so existing
``pytest.raises(ValueError)`` patterns in downstream tests continue to
work. This is a deliberate migration bridge -- new test code should catch
``ConstructError`` directly.
"""

from __future__ import annotations


class NeographError(Exception):
    """Base for all neograph-originated errors."""

    @classmethod
    def build(
        cls,
        what: str,
        *,
        expected: str | None = None,
        found: str | None = None,
        hint: str | None = None,
        location: str | None = None,
        node: str | None = None,
        construct: str | None = None,
    ) -> NeographError:
        """Structured error builder. All neograph errors go through here.

        Format::

            [Node 'X' in 'Y'] what
              expected: ...
              found: ...
              hint: ...
              at file.py:42
        """
        parts: list[str] = []
        if node and construct:
            parts.append(f"[Node '{node}' in construct '{construct}']")
        elif node:
            parts.append(f"[Node '{node}']")
        elif construct:
            parts.append(f"[Construct '{construct}']")
        parts.append(what)

        msg = " ".join(parts)
        if expected:
            msg += f"\n  expected: {expected}"
        if found:
            msg += f"\n  found: {found}"
        if hint:
            msg += f"\n  hint: {hint}"
        if location:
            msg += f"\n  at {location}"

        return cls(msg)


class ConstructError(NeographError, ValueError):
    """Assembly-time validation errors during Construct creation.

    Raised when the node chain fails type or topology validation.
    Inherits from ValueError as a backward-compatibility bridge so
    existing ``pytest.raises(ValueError)`` patterns still catch it.
    """


class CompileError(NeographError):
    """Errors during ``compile()`` -- graph construction from a Construct.

    Examples: missing checkpointer for Operator nodes, sub-construct
    without declared input/output types, node without output type.
    """


class ConfigurationError(NeographError):
    """Bad or missing configuration -- unregistered functions, missing LLM factory.

    Raised when neograph cannot find a required registration (scripted
    function, condition, tool factory, LLM factory, prompt compiler).
    """


class ExecutionError(NeographError):
    """Runtime errors during graph execution.

    Examples: duplicate fan-out keys, unknown output strategy,
    state reducer conflicts, LLM response parse failures.
    """

    def __init__(self, *args: object, validation_errors: str | None = None) -> None:
        super().__init__(*args)
        self.validation_errors = validation_errors

    @classmethod
    def build(
        cls,
        what: str,
        *,
        expected: str | None = None,
        found: str | None = None,
        hint: str | None = None,
        location: str | None = None,
        node: str | None = None,
        construct: str | None = None,
        validation_errors: str | None = None,
    ) -> ExecutionError:
        """Override to pass validation_errors to ExecutionError.__init__."""
        parts: list[str] = []
        if node and construct:
            parts.append(f"[Node '{node}' in construct '{construct}']")
        elif node:
            parts.append(f"[Node '{node}']")
        elif construct:
            parts.append(f"[Construct '{construct}']")
        parts.append(what)

        msg = " ".join(parts)
        if expected:
            msg += f"\n  expected: {expected}"
        if found:
            msg += f"\n  found: {found}"
        if hint:
            msg += f"\n  hint: {hint}"
        if location:
            msg += f"\n  at {location}"

        return cls(msg, validation_errors=validation_errors)


class PromptVarMissing(ExecutionError):
    """A ``strict`` ``substitute()`` found a placeholder with no matching variable.

    The fail-loud counter to the silent ``{domain}``-reaches-the-model class:
    under ``strict=True`` an unfilled placeholder raises here instead of leaving
    the literal token in the prompt and shipping it to the LLM. Carries the
    offending ``var`` name and the sorted list of ``available`` variables so the
    caller can see exactly what was on offer.
    """

    def __init__(
        self,
        *args: object,
        var: str = "",
        available: list[str] | None = None,
    ) -> None:
        super().__init__(*args)
        self.var = var
        self.available = available if available is not None else []

    @classmethod
    def of(cls, var: str, available: list[str]) -> PromptVarMissing:
        """Build a PromptVarMissing for *var*, listing the *available* keys."""
        avail = ", ".join(available) if available else "(none)"
        msg = NeographError.build(
            f"prompt variable '{var}' has no value",
            hint=f"available variables: {avail}",
        )
        return cls(str(msg), var=var, available=list(available))


class StateMissingError(ExecutionError):
    """Raised when ``StateBus.get_required()`` finds a missing key.

    §7: required reads raise; optional reads return None with a documented
    justification. Silent-None reads of required fields are bugs.

    Parentage (THE RULE): a required state read that misses happens WHILE a node
    is executing, so this is an execution-time failure -> ``ExecutionError``.
    """

    @classmethod
    def build(  # type: ignore[override]
        cls,
        *,
        key: str,
        node_label: str | None = None,
    ) -> StateMissingError:
        if node_label:
            msg = f"[Node '{node_label}'] required state key '{key}' not found"
        else:
            msg = f"Required state key '{key}' not found"
        return cls(msg)


class NodeOutputError(ExecutionError):
    """A node RAN and produced None against its declared ``outputs=`` type.

    Fail-loud backstop at the state-write boundary (``_build_state_update``).
    A node that executed and returned ``None`` (or a dict whose primary output
    key is ``None``) against a declared output contract is a silent-swallow
    footgun: the field never reaches the state bus and the ``None`` surfaces far
    from its source (e.g. as a ``TypeError`` in a downstream fan-out router).

    This is distinct from never-ran / legitimately-absent fields (untaken branch
    arms, ``skip_when`` without ``skip_value``) which never reach the write
    boundary with a ran result and stay tolerant.

    Parentage (THE RULE): the node ran, so this is an execution-time failure at
    the state-write boundary -> ``ExecutionError``.
    """


class CheckpointSchemaError(ConfigurationError):
    """Checkpoint state schema does not match the current graph.

    Raised when resuming from a checkpoint whose state model has a different
    fingerprint than the current compiled graph. This prevents silent coercion
    where Pydantic fills defaults for missing fields and ignores extras,
    producing ghost state that looks "complete" but contains stale data.

    Parentage (THE RULE, decided WITH the rule — neograph-12dc): this is the one
    public error raised DURING a run that is NOT an execution failure. It fires
    at resume, BEFORE any node body re-executes, as a precondition check that the
    persisted checkpoint (a configuration input to the run, keyed by
    ``thread_id``) still matches the current compiled graph's schema. Grouping it
    with ``ConfigurationError`` matches consumer intent: the resolution is a
    configuration change (new ``thread_id`` or a schema migration), not a retry
    of node logic. The ``auto_resume=False`` contract reinforces this — it hands
    the caller ``invalidated_nodes`` as an actionable setup signal, not a crash.
    Contrast ``StateMissingError``: a required read that misses WHILE a node runs
    is an execution failure -> ``ExecutionError``.
    """

    def __init__(self, *args: object, invalidated_nodes: set[str] | None = None) -> None:
        super().__init__(*args)
        self.invalidated_nodes = invalidated_nodes or set()


class NonIdempotentReplayError(ExecutionError):
    """Refused to replay a non-idempotent producing tool call. See neograph-lhc6.

    Re-deriving an expired resource by replaying the tool call that produced it
    is only safe when that tool is idempotent (read-only, or a replay-safe
    mutation like an HTTP PUT). Replaying a non-idempotent producer -- an
    act-mode mutation -- would double-apply the side effect, so hydration replay
    (manifest-driven re-derivation, neograph-a5nh) raises this instead, refusing
    the unsafe replay rather than silently double-mutating.

    Carries the offending ``tool_name`` and the optional ``node`` it ran in so
    the caller can pinpoint which producer must be marked ``idempotent=True`` (if
    replay is genuinely safe) or re-fetched from source instead of replayed.
    """

    def __init__(self, *args: object, tool_name: str | None = None, node: str | None = None) -> None:
        super().__init__(*args)
        self.tool_name = tool_name
        self.node = node

    @classmethod
    def of(cls, tool_name: str, *, node: str | None = None) -> NonIdempotentReplayError:
        """Build a NonIdempotentReplayError for the producing *tool_name*."""
        where = f" in node '{node}'" if node else ""
        msg = NeographError.build(
            f"refusing to replay non-idempotent producing tool '{tool_name}'{where}",
            hint="mark the tool idempotent=True only if replay is side-effect-safe "
            "(read-only, or an idempotent mutation); otherwise re-derivation must "
            "fail loud rather than double-apply the side effect",
        )
        return cls(str(msg), tool_name=tool_name, node=node)


class ResourceExpiredError(ExecutionError):
    """A manifest-driven ``ResourceRef`` could not be hydrated. See neograph-a5nh.

    Raised at the END of the layered-expiry fallback (read -> replay producing_call
    -> fail loud): the direct read of ``ref.uri`` failed AND re-derivation by
    replaying the producing call was either impossible (no replayer configured) or
    itself failed. Silent staleness is worse than a loud failure — a stale/absent
    resource that flows into a prompt corrupts the run invisibly, so hydration
    fails loud here instead.

    Carries the offending ``ref`` (its uri + producing call pinpoint what to
    re-acquire), the optional ``node`` it was hydrated in, and the underlying
    ``cause`` (the last read/replay exception) for diagnosis.
    """

    def __init__(
        self,
        *args: object,
        ref: object | None = None,
        node: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(*args)
        self.ref = ref
        self.node = node
        self.cause = cause

    @classmethod
    def of(
        cls,
        ref: object,
        *,
        node: str | None = None,
        detail: str | None = None,
        cause: BaseException | None = None,
    ) -> ResourceExpiredError:
        """Build a ResourceExpiredError for an unrecoverable *ref*."""
        where = f" in node '{node}'" if node else ""
        uri = getattr(ref, "uri", "?")
        kind = getattr(ref, "kind", "?")
        reason = f" ({detail})" if detail else ""
        msg = NeographError.build(
            f"resource ref kind='{kind}' uri='{uri}' expired and could not be re-derived{where}{reason}",
            hint="the producing tool must be replay-eligible (idempotent=True) and "
            "the consumer must supply config['configurable']['mcp_resource_replayer']; "
            "otherwise re-acquire the resource from source",
        )
        return cls(str(msg), ref=ref, node=node, cause=cause)
