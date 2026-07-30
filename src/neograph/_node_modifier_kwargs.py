"""Modifier-kwarg builders for the ``@node`` decorator's sugar chain.

Extracted from ``decorators.py`` (neograph-3ffdg.11) as a pure file split — the
functions below are unchanged, only their home moved. ``decorators.py``
re-exports them and ``node()`` (which stays there) is their only caller.

``_apply_eager_oracle_gen_type`` holds the ONE sanctioned decoration-time write
to the ``oracle_gen_type`` IR field — every other write belongs to the
normalizer. That single-writer carve-out is pinned by ``ALLOWED_PREPOP`` in
``tests/test_guards_llm_runtime.py``, which is keyed on THIS module now that the
write lives here.
"""

from __future__ import annotations

import ast
import secrets
import warnings
from collections.abc import Callable
from typing import Any

from neograph._construct_validation import ConstructError
from neograph._ir_normalize import oracle_gen_type_for
from neograph._runtime_registry import register_scripted
from neograph.node import Node


def _is_trivial_body(body: list[ast.stmt]) -> bool:
    """Check if a function body (docstring already stripped) is a placeholder.

    Trivial patterns: empty (docstring-only), single `...`, `pass`,
    bare constant, `return`, or `return None`.
    """
    if not body:
        return True
    if len(body) != 1:
        return False
    stmt = body[0]
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
        return True
    if isinstance(stmt, ast.Return):
        # `return` (no value) or `return None`
        if stmt.value is None:
            return True
        if isinstance(stmt.value, ast.Constant) and stmt.value.value is None:
            return True
    return False


def _apply_eager_oracle_gen_type(n: Node) -> Node:
    """Eagerly set ``oracle_gen_type`` at decoration time so a bare Node carries
    it before it is placed in a Construct.

    Returns a copy with the field set (IR-immutability — no in-place mutation),
    or the node unchanged when there is no inference. The inference rule lives
    in ``neograph._ir_normalize.oracle_gen_type_for``; ``normalize_ir`` owns the
    assembly-time write and is idempotent over this pre-population. Single
    named operation shared by the @node Each×Oracle and Oracle-only branches.
    """
    gen_type = oracle_gen_type_for(n)
    if gen_type is None:
        return n
    return n.model_copy(update={"oracle_gen_type": gen_type})


def _build_oracle_kwargs(
    *,
    node_label: str,
    f: Callable,
    merge_fn: str | None,
    merge_prompt: str | None,
    models: list[str] | None,
    ensemble_n: int | None,
    merge_pre_process: Callable | None = None,
    merge_post_process: Callable | None = None,
    merge_fallback: Callable | None = None,
    merge_model: str | None = None,
) -> dict[str, Any]:
    """Build and validate Oracle modifier kwargs from @node decorator arguments.

    Shared between Each+Oracle fusion and Oracle-only paths. Handles:
    - Body-as-merge detection + warning + shim registration
    - All validations (requires merge strategy, both set, ensemble_n >= 2)
    - Oracle kwargs dict construction
    """
    effective_merge_fn = merge_fn
    effective_merge_prompt = merge_prompt

    # Body-as-merge: models= set without merge_fn/merge_prompt
    if models is not None and merge_fn is None and merge_prompt is None:
        warnings.warn(
            f"@node '{node_label}': body used as both generator and merge function. "
            f"The first parameter receives list[OutputType] at merge time, not the "
            f"annotated upstream type. Consider adding an explicit merge_fn or merge_prompt.",
            UserWarning,
            stacklevel=4,
        )
        body_merge_name = f"_body_merge_{node_label}_{secrets.token_hex(8)}"

        def _make_body_merge(user_fn: Callable) -> Callable:
            def body_merge(variants: list, config: Any) -> Any:
                return user_fn(variants)

            return body_merge

        register_scripted(body_merge_name, _make_body_merge(f))
        effective_merge_fn = body_merge_name

    if effective_merge_fn is None and effective_merge_prompt is None:
        raise ConstructError.build(
            f"ensemble_n={ensemble_n} requires merge_fn or merge_prompt",
            node=node_label,
            hint="pass merge_fn='<name>' or merge_prompt='<template>'",
        )
    if effective_merge_fn is not None and effective_merge_prompt is not None:
        raise ConstructError.build(
            "both merge_fn and merge_prompt are set",
            node=node_label,
            hint="choose exactly one",
        )
    if ensemble_n is not None and ensemble_n < 2:
        raise ConstructError.build(
            "ensemble_n must be >= 2",
            node=node_label,
            found=str(ensemble_n),
        )

    oracle_kw: dict[str, Any] = {
        "merge_fn": effective_merge_fn,
        "merge_prompt": effective_merge_prompt,
    }
    if models is not None:
        oracle_kw["models"] = models
    if ensemble_n is not None:
        oracle_kw["n"] = ensemble_n
    if merge_pre_process is not None:
        oracle_kw["merge_pre_process"] = merge_pre_process
    if merge_post_process is not None:
        oracle_kw["merge_post_process"] = merge_post_process
    if merge_fallback is not None:
        oracle_kw["merge_fallback"] = merge_fallback
    if merge_model is not None:
        # Conditional-include so Oracle's 'reason' default stays authoritative.
        # Semantics are identical to programmatic Oracle(merge_model=...) —
        # including silent-ignore alongside merge_fn (pure-sugar invariant:
        # no decorator-only validation the modifier itself does not perform).
        oracle_kw["merge_model"] = merge_model
    return oracle_kw


def _build_each_kwargs(map_over: str | None, map_key: str | None, map_on_error: str) -> dict[str, Any]:
    """Each modifier kwargs from @node decorator arguments (both the fused
    Each×Oracle path and the plain fan-out path). Conditional-include for
    ``on_error`` so Each's ``'raise'`` default stays authoritative."""
    each_kw: dict[str, Any] = {"over": map_over, "key": map_key}
    if map_on_error != "raise":
        each_kw["on_error"] = map_on_error
    return each_kw


def _build_portal_kwargs(
    portal: list[str], route: str | None, max_hops: int | None, on_exhaust: str | None
) -> dict[str, Any]:
    """Portal (peer-mode) modifier kwargs from @node decorator arguments.

    Conditional-include for ``route`` / ``max_hops`` / ``on_exhaust`` so the
    modifier's own defaults (``route='goto'``, ``max_hops=10``,
    ``on_exhaust='error'``) stay authoritative AND ``model_fields_set`` matches
    the programmatic ``| Portal(...)`` form field-for-field. That identity is
    load-bearing, not cosmetic: ``_validation_portal`` reads
    ``Portal.model_fields_set`` to enforce the entry-only ``max_hops`` /
    ``on_exhaust`` knobs, so a non-entry member must NOT carry them set."""
    km_kw: dict[str, Any] = {"to": portal}
    if route is not None:
        km_kw["route"] = route
    if max_hops is not None:
        km_kw["max_hops"] = max_hops
    if on_exhaust is not None:
        km_kw["on_exhaust"] = on_exhaust
    return km_kw
