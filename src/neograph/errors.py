"""Error hierarchy for neograph.

All neograph-originated errors inherit from ``NeographError``.
Downstream code that wants a broad catch can use ``except NeographError``;
code that needs to distinguish failure modes catches the specific subclass.

Hierarchy::

    NeographError (Exception)
        ConstructError (+ ValueError for backward compat)
        CompileError
        ConfigurationError
        ExecutionError

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
