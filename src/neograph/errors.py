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
    state reducer conflicts.
    """
