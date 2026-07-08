"""Error-hierarchy parentage: THE RULE, pinned by category (neograph-12dc).

Two guards live here:

1. A consumer-facing E2E: ``except ExecutionError`` around a run must catch
   EVERY failure raised during graph execution — including the resource-expiry
   and non-idempotent-replay failures that historically sat under bare
   ``NeographError`` and silently escaped that catch (review HIGH-01 / PAT-01).

2. A table-driven parentage guard: every public neograph error type is pinned
   to its lifecycle category (execution / assembly / compile / configuration).
   A new error type that is not added to ``PARENTAGE`` fails the completeness
   test, and a type parented against the rule fails the parametrized test — so
   the hierarchy cannot re-drift ad hoc, one ticket at a time.
"""

from __future__ import annotations

import pytest

import neograph.errors as errmod
from neograph.errors import (
    CheckpointSchemaError,
    CompileError,
    ConfigurationError,
    ConstructError,
    ExecutionError,
    NeographError,
    NodeOutputError,
    NonIdempotentReplayError,
    PromptVarMissing,
    ResourceExpiredError,
    StateMissingError,
)


class _FakeRef:
    """Minimal stand-in for a manifest ResourceRef (uri/kind for the message)."""

    uri = "mcp://deal/42/doc"
    kind = "deal_doc"


# THE RULE, as a table: public error type -> the lifecycle-category class it
# must be a subclass of. The four category roots map to themselves; the base
# maps to itself. Everything raised DURING graph execution -> ExecutionError.
EXECUTION = ExecutionError
ASSEMBLY = ConstructError
COMPILE = CompileError
CONFIG = ConfigurationError
BASE = NeographError

PARENTAGE: dict[type, type] = {
    # category roots (direct children of NeographError; the base is itself)
    NeographError: BASE,
    ConstructError: ASSEMBLY,
    CompileError: COMPILE,
    ConfigurationError: CONFIG,
    ExecutionError: EXECUTION,
    # execution-time failures — all reachable via ``except ExecutionError``
    PromptVarMissing: EXECUTION,
    StateMissingError: EXECUTION,
    NodeOutputError: EXECUTION,
    NonIdempotentReplayError: EXECUTION,
    ResourceExpiredError: EXECUTION,
    # resume-time precondition mismatch (checkpoint schema vs current graph),
    # raised before any node re-executes -> a setup/config problem
    CheckpointSchemaError: CONFIG,
}


class TestExecutionErrorCatchesRuntimeFailures:
    """A consumer's ``except ExecutionError`` must catch every runtime failure."""

    def test_execution_error_catches_resource_expiry(self) -> None:
        with pytest.raises(ExecutionError):
            raise ResourceExpiredError.of(_FakeRef(), node="hydrate")

    def test_execution_error_catches_non_idempotent_replay(self) -> None:
        with pytest.raises(ExecutionError):
            raise NonIdempotentReplayError.of("mutate_deal", node="hydrate")

    def test_execution_error_catches_state_miss(self) -> None:
        with pytest.raises(ExecutionError):
            raise StateMissingError.build(key="claims", node_label="score")

    def test_execution_error_catches_node_output_none(self) -> None:
        with pytest.raises(ExecutionError):
            raise NodeOutputError("node produced None against its declared output")


class TestErrorParentageByCategory:
    """Table-driven guard pinning parentage-by-category (anti-drift)."""

    @pytest.mark.parametrize(
        "err_type, category",
        list(PARENTAGE.items()),
        ids=lambda x: getattr(x, "__name__", str(x)),
    )
    def test_error_is_parented_under_its_category(
        self, err_type: type, category: type
    ) -> None:
        assert issubclass(err_type, category), (
            f"{err_type.__name__} must subclass {category.__name__} per THE RULE "
            "in errors.py; see neograph-12dc / review PAT-01"
        )

    def test_every_public_error_type_is_pinned_in_the_table(self) -> None:
        """A new error type MUST be added to PARENTAGE or this fails.

        Discovers every ``NeographError`` subclass defined in ``errors.py`` and
        asserts it is pinned. This is what stops the next ticket from adding an
        error type with an ad-hoc parent nobody reviewed against the rule.
        """
        defined = {
            obj
            for obj in vars(errmod).values()
            if isinstance(obj, type) and issubclass(obj, NeographError)
        }
        missing = defined - set(PARENTAGE)
        assert not missing, (
            "new neograph error types are not pinned in tests/test_error_hierarchy.py "
            f"PARENTAGE: {sorted(t.__name__ for t in missing)}. Add each with its "
            "lifecycle category (execution/assembly/compile/configuration)."
        )
