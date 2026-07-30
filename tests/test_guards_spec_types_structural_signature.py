"""Structural guard: ``spec_types._structural_type_name`` must hash a
RECURSIVE Property-type signature, never the bare top-level JSON-schema
type keyword alone (neograph-qtfof.4's codebase-scan MIGRATE row).

Disease pattern: ``_structural_type_name``'s per-field signature stops at
``str(getattr(p, "type", None))`` -- for a ``ListProperty`` that is just
the bare keyword ``'array'``, never the nested ``item_type`` schema. Two
structurally DIFFERENT models sharing a field name (``list[str]`` vs
``list[SomeModel]``) then hash to the IDENTICAL registry key, and
``register_type``'s idempotency check silently reuses the FIRST-registered
(wrong-shaped) class for the second reconstruction -- a real
``pydantic.ValidationError`` at data-validation time, far from the actual
bug site.
"""

from __future__ import annotations

import ast
import pathlib

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

# neograph-s7zt3.16 moved the Agent Spec Property bridge out of spec_types.py
# into _agent_spec_types.py. This list is EXTENDED, not repointed: scanning only
# the moved-TO file would leave the guard blind to a re-inlined bare-signature
# expression reappearing in the moved-FROM one, which is exactly how a guard
# starts passing vacuously over a shrinking surface.
SCANNED_FILES = [_SRC / "spec_types.py", _SRC / "_agent_spec_types.py"]


def _function_source(name: str) -> str:
    for path in SCANNED_FILES:
        text = path.read_text()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.get_source_segment(text, node) or ""
    raise AssertionError(f"{name} not found in any of {[p.name for p in SCANNED_FILES]}")


class TestStructuralTypeNameUsesRecursiveSignature:
    def test_structural_type_name_delegates_to_recursive_signature_helper(self):
        source = _function_source("_structural_type_name")
        assert "_property_type_signature(p)" in source, (
            "_structural_type_name must build its per-field signature via "
            "_property_type_signature (recursive), not a bare top-level "
            "str(getattr(p, 'type', None)) expression"
        )

    def test_signature_helper_recurses_into_list_and_dict_item_types(self):
        source = _function_source("_property_type_signature")
        assert "item_type" in source, (
            "_property_type_signature must recurse into ListProperty.item_type "
            "-- a bare 'array' keyword cannot distinguish list[str] from "
            "list[SomeModel]"
        )
        assert "value_type" in source, (
            "_property_type_signature must recurse into DictProperty.value_type for the same reason"
        )

    def test_meta_guard_catches_the_disease_pattern_if_reintroduced(self):
        """Meta-test (positive+negative pair, not regex-based -- plain
        substring checks have no regex-slip failure mode): prove the
        assertions above actually flag a pre-fix, bare-signature version of
        _structural_type_name, so this guard isn't vacuously passing only
        because the current source happens to be fixed."""
        buggy_source = (
            "def _structural_type_name(props):\n"
            "    sig = tuple(sorted((p.title, str(getattr(p, 'type', None))) for p in props))\n"
            "    digest = hashlib.sha256(repr(sig).encode()).hexdigest()[:16]\n"
            "    return f'AgentSpecType_{digest}'\n"
        )
        assert "_property_type_signature(p)" not in buggy_source
