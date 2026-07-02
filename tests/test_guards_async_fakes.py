"""Structural guard: async LLM/tool fakes mirror sync via BARE DELEGATION.

neograph-w74k.1 (Phase 0 scaffolding). Core Invariant: one compiled Construct,
many drivers — every behavioral assertion must hold identically under run() and
(future) arun(). That is guaranteed structurally by making sync ``invoke()`` the
SINGLE source of truth: each shared fake's ``ainvoke()`` is a bare delegation
that computes nothing (``return self.invoke(*a, **k)``), so sync and async fakes
CANNOT drift. Fakes that expose ``bind_tools`` likewise get an ``abind_tools``
that bare-delegates to ``bind_tools``; ``StringArgsFake`` additionally mirrors
``_generate`` via ``_agenerate``.

This is the enforcement of the Core Invariant and is state-safe: it inspects the
source/AST of ``tests/fakes.py`` (never calls the mutable-state scripted fakes,
so no false drift from advancing ``_call_idx``/``_call_counter``). It matches the
``tests/test_guards_*`` convention.

TDD red: this guard FAILS today because ``ainvoke``/``abind_tools``/``_agenerate``
do not exist on the fakes yet — the async mirrors are the work of neograph-w74k.1.
"""

from __future__ import annotations

import ast
import pathlib

FAKES_PATH = pathlib.Path(__file__).resolve().parent / "fakes.py"

# The 8 shared deterministic fakes that MUST each gain an async ``ainvoke``
# mirror (disease-scan MIGRATE rows 1-8 in neograph-w74k.1 notes).
FAKES_REQUIRING_AINVOKE = frozenset(
    {
        "StructuredFake",
        "StructuredFakeWithRaw",
        "ReActFake",
        "StringArgsFake",
        "TextFake",
        "GuardFake",
        "StubbornFake",
        "FakeTool",
    }
)

# Subset that exposes ``bind_tools`` and MUST also gain ``abind_tools``.
FAKES_REQUIRING_ABIND_TOOLS = frozenset(
    {
        "ReActFake",
        "StringArgsFake",
        "GuardFake",
        "StubbornFake",
    }
)

# StringArgsFake additionally mirrors its ``_generate`` fallback.
FAKES_REQUIRING_AGENERATE = frozenset({"StringArgsFake"})


def _fake_class_methods() -> dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Map every top-level class in tests/fakes.py -> {method_name: def_node}."""
    tree = ast.parse(FAKES_PATH.read_text())
    out: dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    methods[item.name] = item
            out[node.name] = methods
    return out


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _is_bare_delegation(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, target_method: str
) -> bool:
    """True iff ``fn`` body is exactly ``return self.<target_method>(...)``.

    Anything else (extra statements, a different call target, response logic)
    is NOT a bare delegation and re-opens the sync/async drift surface.
    """
    body = _strip_docstring(fn.body)
    if len(body) != 1:
        return False
    stmt = body[0]
    if not isinstance(stmt, ast.Return) or not isinstance(stmt.value, ast.Call):
        return False
    func = stmt.value.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == target_method
        and isinstance(func.value, ast.Name)
        and func.value.id == "self"
    )


class TestAsyncFakeDelegationGuard:
    """Every shared fake's async surface is a bare delegation to its sync twin.

    Enforces neograph-w74k.1's Core Invariant structurally so sync and async
    fakes cannot drift.
    """

    def test_every_shared_fake_has_bare_delegation_ainvoke(self):
        """Each of the 8 shared fakes must define ``async def ainvoke`` that is
        a bare ``return self.invoke(...)`` — no response logic."""
        classes = _fake_class_methods()
        missing: list[str] = []
        not_delegation: list[str] = []
        for name in sorted(FAKES_REQUIRING_AINVOKE):
            fn = classes.get(name, {}).get("ainvoke")
            if fn is None:
                missing.append(name)
                continue
            if not isinstance(fn, ast.AsyncFunctionDef):
                not_delegation.append(f"{name}.ainvoke must be `async def`")
            elif not _is_bare_delegation(fn, "invoke"):
                not_delegation.append(
                    f"{name}.ainvoke must be a bare `return self.invoke(*a, **k)` "
                    "delegation (no response logic)"
                )
        assert not missing and not not_delegation, (
            "async ainvoke mirror missing or not a bare delegation:\n"
            + "".join(f"  MISSING ainvoke: {n}\n" for n in missing)
            + "".join(f"  {m}\n" for m in not_delegation)
            + "\nCore Invariant (neograph-w74k.1): invoke() is the single source "
            "of truth. Add `async def ainvoke(self, *a, **k): return "
            "self.invoke(*a, **k)` — computing anything in ainvoke re-opens the "
            "sync/async drift surface."
        )

    def test_bind_tools_fakes_have_bare_delegation_abind_tools(self):
        """Fakes exposing ``bind_tools`` must define ``abind_tools`` that bare-
        delegates to ``bind_tools``."""
        classes = _fake_class_methods()
        missing: list[str] = []
        not_delegation: list[str] = []
        for name in sorted(FAKES_REQUIRING_ABIND_TOOLS):
            fn = classes.get(name, {}).get("abind_tools")
            if fn is None:
                missing.append(name)
                continue
            if not _is_bare_delegation(fn, "bind_tools"):
                not_delegation.append(
                    f"{name}.abind_tools must be a bare `return self.bind_tools(*a, **k)` "
                    "delegation"
                )
        assert not missing and not not_delegation, (
            "abind_tools mirror missing or not a bare delegation:\n"
            + "".join(f"  MISSING abind_tools: {n}\n" for n in missing)
            + "".join(f"  {m}\n" for m in not_delegation)
            + "\nAdd `def abind_tools(self, *a, **k): return self.bind_tools(*a, **k)`."
        )

    def test_stringargsfake_has_bare_delegation_agenerate(self):
        """StringArgsFake mirrors its ``_generate`` fallback via ``_agenerate``."""
        classes = _fake_class_methods()
        missing: list[str] = []
        not_delegation: list[str] = []
        for name in sorted(FAKES_REQUIRING_AGENERATE):
            fn = classes.get(name, {}).get("_agenerate")
            if fn is None:
                missing.append(name)
                continue
            if not isinstance(fn, ast.AsyncFunctionDef):
                not_delegation.append(f"{name}._agenerate must be `async def`")
            elif not _is_bare_delegation(fn, "_generate"):
                not_delegation.append(
                    f"{name}._agenerate must be a bare `return self._generate(*a, **k)` "
                    "delegation"
                )
        assert not missing and not not_delegation, (
            "_agenerate mirror missing or not a bare delegation:\n"
            + "".join(f"  MISSING _agenerate: {n}\n" for n in missing)
            + "".join(f"  {m}\n" for m in not_delegation)
            + "\nAdd `async def _agenerate(self, *a, **k): return self._generate(*a, **k)`."
        )

    def test_delegation_detector_is_not_vacuous(self):
        """Slip meta-test: the AST delegation detector accepts a true bare
        delegation and rejects extra logic / a different call target, so the
        guards above cannot pass vacuously."""
        good = ast.parse(
            "async def ainvoke(self, *a, **k):\n    return self.invoke(*a, **k)\n"
        ).body[0]
        assert _is_bare_delegation(good, "invoke")

        good_with_doc = ast.parse(
            'async def ainvoke(self, *a, **k):\n    "doc"\n    return self.invoke(*a, **k)\n'
        ).body[0]
        assert _is_bare_delegation(good_with_doc, "invoke")

        extra_logic = ast.parse(
            "async def ainvoke(self, *a, **k):\n"
            "    x = self.invoke(*a, **k)\n"
            "    return x\n"
        ).body[0]
        assert not _is_bare_delegation(extra_logic, "invoke")

        wrong_target = ast.parse(
            "async def ainvoke(self, *a, **k):\n    return self.respond(*a, **k)\n"
        ).body[0]
        assert not _is_bare_delegation(wrong_target, "invoke")
