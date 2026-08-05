"""Modifier-kwarg builders for the ``@node`` decorator's sugar chain.

Extracted from ``decorators.py`` (neograph-3ffdg.11) as a pure file split — the
functions below are unchanged, only their home moved. ``decorators.py``
re-exports them and ``node()`` (which stays there) is their only caller.

``_apply_eager_oracle_gen_type`` holds the ONE sanctioned decoration-time write
to the ``oracle_gen_type`` IR field — every other write belongs to the
normalizer. That single-writer carve-out is pinned by ``ALLOWED_PREPOP`` in
``tests/test_guards_llm_runtime.py``, which is keyed on THIS module now that the
write lives here.

``MODIFIER_KWARGS`` / ``IDENTITY_KWARGS`` / ``valid_kwargs`` / ``derive_combo``
(neograph-jtawq.4, Phase 1) are the additive registry that replaces
``decorators.py``'s flat if/elif kwarg dispatch (Phase 2). The valid @node
kwarg contract for any modifier shape is DERIVED from ``ModifierCombo`` /
``_COMBO_MAP`` (``modifiers.py`` — the one existing IR-layer composition/
validity authority) via ``combo_for_modifier_names`` /
``modifier_names_for_combo``, never re-guessed post-hoc from which kwargs
happened to be non-None and never re-encoded as a second decorator-local
taxonomy. This module is DX-layer (see
``TestLowerLayersDoNotImportForwardDX.DX_MODULES`` in
``test_guards_assembly.py``), so importing ``modifiers.py`` from here is a
downward DX->IR edge, the same direction ``compiler.py`` already imports the
same combo vocabulary.
"""

from __future__ import annotations

import ast
import secrets
import warnings
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

from neograph._construct_validation import ConstructError
from neograph._ir_normalize import oracle_gen_type_for
from neograph._runtime_registry import register_condition, register_scripted
from neograph.modifiers import (
    Each,
    Loop,
    ModifierCombo,
    Operator,
    Oracle,
    Portal,
    combo_for_modifier_names,
    modifier_names_for_combo,
)
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


# ═══════════════════════════════════════════════════════════════════════════
# ModifierCombo-keyed kwarg registry (neograph-jtawq.4, Phase 1)
# ═══════════════════════════════════════════════════════════════════════════


class ModifierKwargs(NamedTuple):
    """One row: which ``@node`` kwargs trigger/configure a given modifier.

    ``triggers`` -- any ONE of these being non-None means the modifier is
    being requested (mirrors today's ``has_oracle_kwarg``-style checks).
    ``satellites`` -- kwargs that configure the modifier once triggered, but
    do not by themselves trigger it (Phase 3's strictness gate rejects a
    satellite with no owning trigger; Phases 0-2 leave that silently
    accepted, unchanged from today).
    ``field_map`` -- ``@node`` kwarg name -> modifier class field name, for
    the kwargs whose name DIFFERS from the field it becomes (e.g.
    ``ensemble_n`` -> ``Oracle.n``). Omits identically-named kwargs/fields —
    only renames are worth tracking, and the registry-integrity guard checks
    every value here is a real field on the row's modifier class.
    """

    name: str
    triggers: frozenset[str]
    satellites: frozenset[str]
    field_map: dict[str, str]


MODIFIER_KWARGS: tuple[ModifierKwargs, ...] = (
    ModifierKwargs(
        name="each",
        triggers=frozenset({"map_over"}),
        satellites=frozenset({"map_key", "map_on_error"}),
        field_map={"map_over": "over", "map_key": "key", "map_on_error": "on_error"},
    ),
    ModifierKwargs(
        name="oracle",
        triggers=frozenset({"ensemble_n", "models", "merge_fn", "merge_prompt"}),
        satellites=frozenset({"merge_pre_process", "merge_post_process", "merge_fallback", "merge_model"}),
        field_map={"ensemble_n": "n"},
    ),
    ModifierKwargs(
        name="operator",
        triggers=frozenset({"interrupt_when"}),
        satellites=frozenset(),
        field_map={"interrupt_when": "when"},
    ),
    ModifierKwargs(
        name="loop",
        triggers=frozenset({"loop_when"}),
        satellites=frozenset({"max_iterations", "on_exhaust"}),
        field_map={"loop_when": "when"},
    ),
    ModifierKwargs(
        name="portal",
        triggers=frozenset({"portal"}),
        satellites=frozenset({"route", "max_hops", "on_exhaust"}),
        field_map={"portal": "to"},
    ),
)

#: ``@node`` kwargs that carry no modifier-dispatch meaning at all -- always
#: valid regardless of which combo a node ends up with. Every one of node()'s
#: 31 keyword-only params (i.e. excluding the decorated function itself)
#: appears in EITHER this set OR >=1 ``MODIFIER_KWARGS`` row's
#: triggers/satellites (``on_exhaust`` is the one documented row that appears
#: in two rows at once -- a shared satellite of loop AND portal, never both
#: at once since {loop, portal} is not a valid combo). The anti-flat-explosion
#: ratchet in ``tests/test_guards_modifier_composition_completeness.py`` checks
#: this totality against ``inspect.signature(node)`` directly, so a 32nd kwarg
#: added without declaring its owning shape fails CI.
IDENTITY_KWARGS: frozenset[str] = frozenset(
    {
        "mode",
        "inputs",
        "outputs",
        "model",
        "prompt",
        "llm_config",
        "tools",
        "name",
        "renderer",
        "context",
        "skip_when",
        "skip_value",
        "gate_tools_when",
    }
)


def valid_kwargs(combo: ModifierCombo) -> frozenset[str]:
    """The full set of ``@node`` kwargs valid for a given ``ModifierCombo`` --
    every trigger + satellite of every modifier the combo carries, plus every
    ``IDENTITY_KWARGS`` name (always valid, combo-independent).

    Reads modifier membership via ``modifier_names_for_combo`` -- the one
    sanctioned reader of ``_COMBO_MAP`` for "does this combo carry an X" --
    never a hand-typed combo->modifiers table.
    """
    names = modifier_names_for_combo(combo)
    valid: set[str] = set(IDENTITY_KWARGS)
    for row in MODIFIER_KWARGS:
        if row.name in names:
            valid |= row.triggers | row.satellites
    return frozenset(valid)


def _owning_triggers(kwarg: str) -> frozenset[str]:
    """Every trigger kwarg that would make ``kwarg`` (a satellite) reachable.

    Unions across every ``MODIFIER_KWARGS`` row listing ``kwarg`` as a
    satellite -- ``on_exhaust`` is shared by both loop and portal, so its
    owning triggers are ``{loop_when, portal}``, not just one row's.
    """
    triggers: set[str] = set()
    for row in MODIFIER_KWARGS:
        if kwarg in row.satellites:
            triggers |= row.triggers
    return frozenset(triggers)


def _check_kwargs_against_shape(
    kwargs: Mapping[str, Any], combo: ModifierCombo, *, node_label: str, defaults: Mapping[str, Any]
) -> None:
    """Reject a passed ``@node`` kwarg that cannot reach any modifier ``combo``
    actually carries -- the Phase 3 strictness gate.

    Compares each kwarg's VALUE against its live signature default via
    ``defaults``, never ``is not None``: ``map_on_error`` is ``node()``'s ONLY
    non-``None`` default (``'raise'``), so an ``is not None`` test would
    reject ``map_on_error='raise'`` on every non-Each node in the codebase,
    since every call captures it via ``locals()`` whether the caller wrote it
    or not. An explicitly-default value is therefore indistinguishable from
    unset and stays accepted -- this is the only reason the value-vs-default
    distinction exists.

    ``defaults`` is a PARAMETER, not an internal ``inspect.signature(node)``
    call: this module cannot import ``decorators.node`` (``decorators.py``
    imports this module at module level, so that would cycle), and a
    function-local import back would grow
    ``FUNCTION_LOCAL_IMPORT_ALLOWLIST``, which must never grow. The caller
    derives ``defaults`` from the live signature; never a hand-written
    literal.
    """
    passed = {k for k, v in kwargs.items() if v != defaults.get(k, v)}
    invalid = sorted(passed - valid_kwargs(combo))
    if not invalid:
        return
    parts = [f"{kwarg}= requires one of: {', '.join(f'{t}=' for t in sorted(_owning_triggers(kwarg)))}" for kwarg in invalid]
    raise ConstructError.build(
        "; ".join(parts),
        node=node_label,
        found=", ".join(f"{k}={kwargs[k]!r}" for k in invalid),
        hint=f"a {combo.name} node accepts: {sorted(valid_kwargs(combo) - IDENTITY_KWARGS)}",
    )


def derive_combo(kwargs: Mapping[str, Any], *, node_label: str = "?") -> ModifierCombo:
    """Classify which modifiers a ``@node`` kwargs mapping implies.

    Tests each ``MODIFIER_KWARGS`` row's ``triggers`` for a non-None value
    (mirrors today's ``has_oracle_kwarg``-style checks, generalized to every
    modifier), then reads ``_COMBO_MAP`` -- the one validity authority -- via
    ``combo_for_modifier_names``. Raises ``ConstructError`` (the same error
    ``combo_for_modifier_names`` raises for any invalid frozenset) when the
    implied modifier-name set is not a recognized combo.
    """
    implied: set[str] = set()
    for row in MODIFIER_KWARGS:
        if any(kwargs.get(trigger) is not None for trigger in row.triggers):
            implied.add(row.name)
    return combo_for_modifier_names(implied, context=node_label)


# ═══════════════════════════════════════════════════════════════════════════
# Per-modifier build+pipe (neograph-jtawq.4, Phase 2)
# ═══════════════════════════════════════════════════════════════════════════
#
# One ``_build_<name>_node`` per modifier, each wrapping ``n | Modifier(**kw)``.
# ``decorators.py``'s dispatch (Phase 2) calls the five unconditionally, gated
# only on ``"<name>" in members`` -- fixed order oracle/each/operator/loop/
# portal, reproducing today's exact pipe sequence bit-for-bit (ModifierSet
# stores modifiers in typed slots, not an ordered list, so application order
# is inert for the final IR; see modifiers.py's ``ModifierSet``).


def _build_operator_kwargs(node_label: str, interrupt_when: str | Callable) -> dict[str, Any]:
    """Operator modifier kwargs from ``@node``'s ``interrupt_when=``. A string
    is used as-is (a registered condition name); a callable is auto-registered
    under a synthesized name."""
    if isinstance(interrupt_when, str):
        condition_name = interrupt_when
    elif callable(interrupt_when):
        condition_name = f"_node_interrupt_{node_label}_{secrets.token_hex(8)}"
        register_condition(condition_name, interrupt_when)
    else:
        raise ConstructError.build(
            "interrupt_when must be a string (registered condition name) or a callable",
            node=node_label,
            found=type(interrupt_when).__name__,
        )
    return {"when": condition_name}


def _build_loop_kwargs(
    node_label: str, loop_when: str | Callable, max_iterations: int | None, on_exhaust: str | None
) -> dict[str, Any]:
    """Loop modifier kwargs from ``@node`` decorator arguments. ``on_exhaust``
    is shared with the Portal sugar and routed by trigger -- Portal's
    ``'exit'`` value is the cross-modifier value here, rejected with a clean
    ``ConstructError`` (mirrors Portal's own ``on_exhaust='last'`` rejection
    in ``_build_portal_node`` below); any other invalid value falls through
    to ``Loop.model_post_init``'s own ``ConfigurationError``."""
    if on_exhaust == "exit":
        raise ConstructError.build(
            "loop_when= (Loop) does not accept on_exhaust='exit'",
            node=node_label,
            found=repr(on_exhaust),
            hint="'exit' is a Portal (portal=) value; use 'last' to return the last iteration",
        )
    loop_kw: dict[str, Any] = {"when": loop_when, "max_iterations": max_iterations or 10}
    if on_exhaust is not None:
        loop_kw["on_exhaust"] = on_exhaust
    return loop_kw


def _build_oracle_node(n: Node, *, node_label: str, f: Callable, kwargs: Mapping[str, Any]) -> Node:
    """Build+pipe Oracle from a ``@node`` kwargs mapping. All kwarg-shape
    validation stays in ``_build_oracle_kwargs`` -- unchanged from the
    pre-registry branches."""
    oracle_kw = _build_oracle_kwargs(
        node_label=node_label,
        f=f,
        merge_fn=kwargs.get("merge_fn"),
        merge_prompt=kwargs.get("merge_prompt"),
        models=kwargs.get("models"),
        ensemble_n=kwargs.get("ensemble_n"),
        merge_pre_process=kwargs.get("merge_pre_process"),
        merge_post_process=kwargs.get("merge_post_process"),
        merge_fallback=kwargs.get("merge_fallback"),
        merge_model=kwargs.get("merge_model"),
    )
    return n | Oracle(**oracle_kw)


def _build_each_node(n: Node, *, kwargs: Mapping[str, Any]) -> Node:
    """Build+pipe Each from a ``@node`` kwargs mapping."""
    each_kw = _build_each_kwargs(kwargs.get("map_over"), kwargs.get("map_key"), kwargs.get("map_on_error", "raise"))
    return n | Each(**each_kw)


def _build_operator_node(n: Node, *, node_label: str, kwargs: Mapping[str, Any]) -> Node:
    """Build+pipe Operator from a ``@node`` kwargs mapping."""
    return n | Operator(**_build_operator_kwargs(node_label, kwargs["interrupt_when"]))


def _build_loop_node(n: Node, *, node_label: str, kwargs: Mapping[str, Any]) -> Node:
    """Build+pipe Loop from a ``@node`` kwargs mapping."""
    loop_kw = _build_loop_kwargs(
        node_label, kwargs["loop_when"], kwargs.get("max_iterations"), kwargs.get("on_exhaust")
    )
    return n | Loop(**loop_kw)


def _build_portal_node(n: Node, *, node_label: str, kwargs: Mapping[str, Any]) -> Node:
    """Build+pipe Portal from a ``@node`` kwargs mapping. ``on_exhaust='last'``
    is the cross-modifier (Loop) value, rejected with a clean
    ``ConstructError`` (mirrors Loop's own ``on_exhaust='exit'`` rejection in
    ``_build_loop_kwargs`` above)."""
    on_exhaust = kwargs.get("on_exhaust")
    if on_exhaust == "last":
        raise ConstructError.build(
            "portal= (Portal) does not accept on_exhaust='last'",
            node=node_label,
            found=repr(on_exhaust),
            hint="'last' is a Loop (loop_when=) value; use 'exit' to leave the mesh on budget exhaustion",
        )
    portal_kw = _build_portal_kwargs(kwargs["portal"], kwargs.get("route"), kwargs.get("max_hops"), on_exhaust)
    return n | Portal(**portal_kw)
