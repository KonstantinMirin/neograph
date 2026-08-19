"""Structural guard: a caller-supplied annotation never reaches ``.model_fields``.

neograph-vduhp (GH issue #8). ``describe_type`` declared ``model: type[BaseModel]``
and enforced nothing, so a caller walking a node's dict-form ``outputs`` -- which
routinely mixes a BaseModel with a ``list[X]`` sibling -- got

    AttributeError: type object 'list' has no attribute 'model_fields'

a Pydantic internal raised several frames below the mistake, naming a builtin
rather than the annotation they passed. The scan that accompanied the fix found a
second PUBLIC face of the same root (``inject_schema`` forwards straight to
``describe_type``), which is the reason this guard exists at all: a shared root is
a CLAIM, and an unasserted claim is how the second face silently diverges later.

The ratchet is behavioural, not textual -- each public entry point is CALLED with
a deliberately wrong annotation and its escaping exception inspected. A guard that
grepped for ``.model_fields`` would pass the moment someone moved the access one
function deeper.

Known limit, stated rather than papered over: the completeness half below derives
its population from parameter ANNOTATIONS, and ``describe_type`` itself no longer
appears there (the fix widened its parameter to ``Any``, which is honest -- it now
accepts containers). Annotation-derived population is therefore a floor, not a
ceiling, and ROSTER is hand-maintained on top of it.
"""

from __future__ import annotations

import inspect

import pytest
from pydantic import BaseModel

import neograph
from neograph.tool import ToolInteraction

# A deliberately wrong annotation for a slot that wants a model class: the exact
# shape from the GH issue (a dict-form `tool_log` output).
WRONG = list[ToolInteraction]


class _Probe(BaseModel):
    field: str


def _leak(exc: BaseException) -> bool:
    """True when *exc* is the Pydantic-internal leak this guard bans."""
    return isinstance(exc, AttributeError) and "model_fields" in str(exc)


def _call_describe_type() -> None:
    neograph.describe_type(WRONG)


def _call_inject_schema() -> None:
    neograph.inject_schema({}, WRONG)


def _call_compile_prompt() -> None:
    neograph.compile_prompt(
        "tmpl",
        {"a": 1},
        output_model=WRONG,
        prompt_compiler=lambda template, data, **kw: [{"role": "user", "content": str(data)}],
    )


def _call_construct_from_functions() -> None:
    neograph.construct_from_functions("c", [], input=WRONG, output=WRONG)


def _call_construct_from_module() -> None:
    import types as _types

    neograph.construct_from_module(_types.ModuleType("guard_probe_mod"), "c", input=WRONG, output=WRONG)


def _call_register_type() -> None:
    neograph.register_type("guard_probe_bad", WRONG)


def _call_resource_reader() -> None:
    neograph.resource_reader(
        "r", uri_template="u://{a}", output_model=WRONG, description="d"
    )


# Public entry points that take a caller-supplied model annotation. A probe means
# "called with WRONG, and whatever escapes must not be the model_fields leak".
ROSTER = {
    "describe_type": _call_describe_type,
    "inject_schema": _call_inject_schema,
    "compile_prompt": _call_compile_prompt,
    "construct_from_functions": _call_construct_from_functions,
    "construct_from_module": _call_construct_from_module,
    "register_type": _call_register_type,
    "resource_reader": _call_resource_reader,
}

# Public functions whose model-typed parameter takes an INSTANCE, not a class, so
# there is no annotation to mis-supply and no .model_fields walk to reach.
INSTANCE_TAKERS = {
    "ask_human": "payload is a BaseModel INSTANCE (the HITL value), not a class to introspect",
    "emit_progress": "event is a BaseModel INSTANCE (the progress payload), not a class to introspect",
}


class TestModelAnnotationBoundary:
    """No public entry point may leak a Pydantic ``model_fields`` AttributeError
    for a caller-supplied annotation (neograph-vduhp / GH issue #8)."""

    @pytest.mark.parametrize("name", sorted(ROSTER))
    def test_public_entry_point_does_not_leak_model_fields(self, name: str):
        try:
            ROSTER[name]()
        except Exception as exc:  # noqa: BLE001 - the leak IS the thing under test
            assert not _leak(exc), (
                f"neograph.{name}() leaked a Pydantic internal for a non-model "
                f"annotation: {type(exc).__name__}: {exc}. Validate at the boundary "
                "(render the container, or refuse naming the annotation) instead of "
                "letting the value reach .model_fields."
            )

    def test_every_model_taking_public_function_is_rostered_or_exempted(self):
        """Completeness: a NEW public function with a model-typed parameter must
        be given a probe (or a written exemption) before it can be merged.

        This is the half that makes the guard a ratchet rather than a snapshot --
        without it the roster silently goes stale as the public API grows.
        """
        population = set()
        for name in neograph.__all__:
            obj = getattr(neograph, name, None)
            if not inspect.isfunction(obj):
                continue
            try:
                sig = inspect.signature(obj)
            except (ValueError, TypeError):  # pragma: no cover - builtins
                continue
            if any("BaseModel" in str(p.annotation) for p in sig.parameters.values()):
                population.add(name)

        uncovered = sorted(population - set(ROSTER) - set(INSTANCE_TAKERS))
        assert uncovered == [], (
            f"public function(s) {uncovered} declare a BaseModel-typed parameter but "
            "have no probe in ROSTER and no reason in INSTANCE_TAKERS. Add one of the "
            "two -- see neograph-vduhp / GH issue #8 for what goes wrong when a "
            "caller-supplied annotation reaches .model_fields unchecked."
        )

    # --- meta-tests: prove the detector actually detects ---

    def test_leak_detector_flags_the_pre_fix_exception(self):
        """Negative meta-test: the exact pre-fix exception must be recognised."""
        exc = AttributeError("type object 'list' has no attribute 'model_fields'")
        assert _leak(exc)

    def test_leak_detector_accepts_a_diagnosable_refusal(self):
        """Positive meta-test: a typed refusal naming the annotation is fine."""
        from neograph.errors import ConfigurationError

        exc = ConfigurationError.build("no notation", found=repr(WRONG))
        assert not _leak(exc)

    def test_leak_detector_does_not_flag_an_unrelated_attributeerror(self):
        """'Would-be-missed' inverse: the detector keys on the Pydantic attribute,
        not on the exception type, so an ordinary AttributeError from user code
        is not mistaken for this disease."""
        assert not _leak(AttributeError("'Foo' object has no attribute 'bar'"))

    def test_probe_actually_exercises_the_fixed_path(self):
        """Anti-vacuity: the roster's motivating probe must reach real rendering,
        not silently no-op. ``describe_type(list[M])`` renders the container."""
        rendered = neograph.describe_type(WRONG, prefix="")
        assert rendered.startswith("[{")
        assert "tool_name: string" in rendered
