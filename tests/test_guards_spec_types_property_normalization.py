"""Structural guard (neograph-s7zt3.16): no Agent-Spec ``Property`` may be
CLASS-dispatched before it has been normalized.

Disease, as diagnosed on develop @ 82b8836. ``Flow.from_dict()`` erases typed
``Property`` subclasses down to the base ``Property`` (``json_schema`` intact,
class gone), and ``spec_types`` dispatches on the class. Both class-dispatch
sites were defeated, and — the part that made the symptom confusing — they were
defeated DIFFERENTLY:

  * ``_property_to_field_type`` had a base-``Property`` fallback that routed the
    surviving ``json_schema`` into ``_resolve_field_type``, whose object branch
    mints an ad-hoc ``create_model(schema['title'])`` class instead of the
    canonical ``_structural_type_name`` + ``register_type`` path. A SECOND
    WALKER producing a structurally-equal but NON-IDENTICAL class.
  * ``_property_type_signature`` had no fallback at all and collapsed an erased
    property to its bare top-level keyword.

So the real defect is not "a branch was missing" — it is that the codebase held
two ways to turn a shape into a type, and which one you got depended on whether
your Flow had been through JSON. The behavioural tests in
``tests/test_agent_spec_property_erasure.py`` pin the SYMPTOM. This guard pins
the STRUCTURE, so a future change cannot re-fix the symptom while quietly
restoring the two-walker state.

Three checks:

1. Every function in ``spec_types.py`` that class-dispatches on a pyagentspec
   Property subclass normalizes FIRST.
2. The set of such functions is exactly the registered set — a NEW dispatcher
   added later fails here rather than silently inheriting the hole. This is the
   anti-vacuity check: without it the guard would keep passing over a shrinking
   surface.
3. The removed base-``Property`` fallback stays removed — nothing outside the
   normalizer may test ``type(prop) is pas.Property``.

Pure AST, no ``re``, so this module is exempt-by-construction from
``test_guards_meta.py`` Discipline 1.
"""

from __future__ import annotations

import ast
import pathlib

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

# The Property bridge lives in _agent_spec_types.py (split out of spec_types.py
# in this same ticket). BOTH are scanned, not just the current home: a dispatcher
# re-inlined back into spec_types.py must fail here too, or the guard silently
# narrows to whichever file happens to hold the code today.
SCANNED = [_SRC / "_agent_spec_types.py", _SRC / "spec_types.py"]

# The canonical normalization helper. Any class-dispatching function must call
# this before its first isinstance branch.
NORMALIZER = "_normalize_erased_property"

# The functions permitted to class-dispatch on a Property subclass. Check 2
# asserts the discovered set equals this one, so this is a REGISTRY, not an
# allowlist -- adding a dispatcher without registering it is a failure.
DISPATCH_SITES = frozenset({"_property_type_signature", "_property_to_field_type"})

# pyagentspec Property subclasses that carry the erasure risk (from_dict cannot
# resolve them: Property has no component_type discriminator).
ERASED_PROPERTY_CLASSES = frozenset(
    {
        "ListProperty",
        "DictProperty",
        "ObjectProperty",
        "UnionProperty",
        "NullProperty",
        "StringProperty",
        "IntegerProperty",
        "NumberProperty",
        "FloatProperty",
        "BooleanProperty",
    }
)


def _functions() -> dict[str, ast.FunctionDef]:
    """Every top-level function across BOTH scanned modules."""
    found: dict[str, ast.FunctionDef] = {}
    for path in SCANNED:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef):
                found[n.name] = n
    return found


def _is_property_class_dispatch(node: ast.AST) -> bool:
    """True for ``isinstance(x, pas.SomeProperty)`` / ``isinstance(x, SomeProperty)``."""
    if not isinstance(node, ast.Call):
        return False
    if not (isinstance(node.func, ast.Name) and node.func.id == "isinstance"):
        return False
    if len(node.args) < 2:
        return False
    target = node.args[1]
    if isinstance(target, ast.Attribute):
        return target.attr in ERASED_PROPERTY_CLASSES
    if isinstance(target, ast.Name):
        return target.id in ERASED_PROPERTY_CLASSES
    return False


def _first_dispatch_line(fn: ast.FunctionDef) -> int | None:
    lines = [n.lineno for n in ast.walk(fn) if _is_property_class_dispatch(n)]
    return min(lines) if lines else None


def _normalizer_call_lines(fn: ast.FunctionDef) -> list[int]:
    return [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == NORMALIZER
    ]


def _identity_check_against_base_property(node: ast.AST) -> bool:
    """True for ``type(x) is pas.Property`` / ``type(x) is Property``.

    This is the shape of the removed fallback guard. It is legitimate INSIDE
    the normalizer (that is exactly what the normalizer must test) and nowhere
    else -- anywhere else it means a second, competing erasure path.
    """
    if not isinstance(node, ast.Compare):
        return False
    if not (len(node.ops) == 1 and isinstance(node.ops[0], ast.Is)):
        return False
    left = node.left
    if not (isinstance(left, ast.Call) and isinstance(left.func, ast.Name) and left.func.id == "type"):
        return False
    right = node.comparators[0]
    if isinstance(right, ast.Attribute):
        return right.attr == "Property"
    if isinstance(right, ast.Name):
        return right.id == "Property"
    return False


class TestPropertyNormalizationPrecedesClassDispatch:
    """The invariant: normalize, THEN dispatch."""

    def test_normalizer_exists(self):
        fns = _functions()
        assert NORMALIZER in fns, (
            f"{NORMALIZER} must exist in {[p.name for p in SCANNED]} -- it is the single seam that "
            "re-inflates a Property whose class Flow.from_dict() erased. Without it, "
            "class dispatch silently falls through to a second walker (neograph-s7zt3.16)."
        )

    def test_every_dispatch_site_normalizes_before_its_first_isinstance(self):
        fns = _functions()
        offenders: list[str] = []

        for name in sorted(DISPATCH_SITES):
            fn = fns.get(name)
            assert fn is not None, f"registered dispatch site {name} not found in {[p.name for p in SCANNED]}"

            dispatch_line = _first_dispatch_line(fn)
            if dispatch_line is None:
                continue  # covered by the anti-vacuity check below

            calls = _normalizer_call_lines(fn)
            if not calls:
                offenders.append(f"{name}: class-dispatches at line {dispatch_line} but never calls {NORMALIZER}")
            elif min(calls) > dispatch_line:
                offenders.append(
                    f"{name}: calls {NORMALIZER} at line {min(calls)}, AFTER its first "
                    f"class dispatch at line {dispatch_line}"
                )

        assert not offenders, (
            "Every Property class-dispatch site must normalize BEFORE dispatching. An "
            "erased Property (Flow.from_dict drops the subclass, keeps json_schema) "
            "misses every isinstance branch, and the fallback that used to catch it was "
            "a SECOND walker minting non-identical classes (neograph-s7zt3.16).\n" + "\n".join(offenders)
        )

    def test_dispatch_site_registry_is_exhaustive(self):
        """Anti-vacuity: a NEW class-dispatching function must be registered.

        Without this the guard would keep passing while the real dispatch
        surface grew underneath it -- the exact 'shrinking surface' failure the
        markers guard nearly shipped with.
        """
        fns = _functions()
        discovered = {name for name, fn in fns.items() if _first_dispatch_line(fn) is not None}

        unregistered = discovered - DISPATCH_SITES
        assert not unregistered, (
            f"these functions class-dispatch on a Property subclass but are "
            f"not in DISPATCH_SITES: {sorted(unregistered)}. Add them to the registry AND "
            f"make them call {NORMALIZER} first."
        )

        stale = DISPATCH_SITES - discovered
        assert not stale, (
            f"these registered dispatch sites no longer class-dispatch: {sorted(stale)}. "
            "Remove them from DISPATCH_SITES so the registry does not grant cover to a "
            "function that moved or was deleted."
        )

    def test_base_property_identity_check_is_confined_to_the_normalizer(self):
        """The removed fallback stays removed.

        ``type(prop) is pas.Property`` is the signature of the erased-Property
        branch. It belongs in exactly one place; anywhere else it is a second
        competing path that can diverge again.
        """
        fns = _functions()
        offenders: list[str] = []

        for name, fn in fns.items():
            if name == NORMALIZER:
                continue
            for node in ast.walk(fn):
                if _identity_check_against_base_property(node):
                    offenders.append(f"{name}:{node.lineno}")

        assert not offenders, (
            f"only {NORMALIZER} may test 'type(prop) is Property'. A second such check is a "
            "competing erased-Property path -- exactly the two-walker split that made "
            "neograph-s7zt3.16 present as a type MISMATCH rather than a clean failure.\n" + "\n".join(offenders)
        )


class TestDetectorSlips:
    """Slip meta-tests (PROC-2) for the AST detectors above -- they carry the
    guard's precision, so each is pinned against the boundary a naiver matcher
    gets wrong."""

    @staticmethod
    def _expr(src: str) -> ast.AST:
        return ast.parse(src, mode="eval").body

    def test_slip_property_class_dispatch_detector(self):
        # Both spellings of the dispatch are caught.
        assert _is_property_class_dispatch(self._expr("isinstance(prop, pas.ListProperty)"))
        assert _is_property_class_dispatch(self._expr("isinstance(prop, ObjectProperty)"))
        # A non-Property isinstance is NOT a dispatch site -- spec_types is full
        # of these (isinstance(source, dict) etc.) and flagging them would make
        # the registry check unusable.
        assert not _is_property_class_dispatch(self._expr("isinstance(source, dict)"))
        assert not _is_property_class_dispatch(self._expr("isinstance(additional, dict)"))
        # The boundary: a bare `Property` is NOT a subclass dispatch. It is the
        # base class, and testing it is the normalizer's own job.
        assert not _is_property_class_dispatch(self._expr("isinstance(prop, pas.Property)"))

    def test_slip_base_property_identity_detector(self):
        assert _identity_check_against_base_property(self._expr("type(prop) is pas.Property"))
        assert _identity_check_against_base_property(self._expr("type(prop) is Property"))
        # A SUBCLASS identity check is a different statement and not this rule's business.
        assert not _identity_check_against_base_property(self._expr("type(prop) is pas.ListProperty"))
        # isinstance is not an identity check.
        assert not _identity_check_against_base_property(self._expr("isinstance(prop, pas.Property)"))
        # `==` is not `is`; the fallback used `is`, and widening to `==` here
        # would start flagging ordinary equality comparisons.
        assert not _identity_check_against_base_property(self._expr("type(prop) == pas.Property"))
