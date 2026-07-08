"""Modifier-triggered validation rules: Loop back-edges and Oracle merge hooks.

These checks fire from a specific modifier rather than from the core
producer/consumer walk:
  - ``validate_loop_self_edge`` / ``validate_loop_construct`` run at ``|`` time
    from ``Modifiable.__or__`` when a ``Loop`` is applied to a Node / Construct.
  - ``_validate_merge_hooks`` runs during the chain walk when an Oracle declares
    ``merge_prompt`` — it type-checks the merge hook callables.

Imports the type-compat primitives from ``_validation_types``; imported only
from within the validation cluster (loop validators are re-exported through
``_construct_validation`` for ``modifiers.py``).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, get_args, get_origin, get_type_hints

from neograph._normalize import normalize_outputs
from neograph._validation_types import _fmt_type, _source_location, _types_compatible
from neograph.errors import ConstructError
from neograph.modifiers import Oracle
from neograph.node import Node

if TYPE_CHECKING:
    from neograph.construct import Construct


def validate_loop_self_edge(node: Node) -> None:
    """Check that a self-loop's output type is compatible with its input type.

    Called at ``|`` time (from ``Modifiable.__or__``) when a ``Loop`` modifier
    is applied to a Node. Loop on Node always means self-loop.
    """
    loop = node.modifier_set.loop
    if loop is None:
        return

    output_type = node.outputs
    input_type = node.inputs
    if output_type is None or input_type is None:
        return

    # Dict-form outputs: loop feeds back the PRIMARY key only (first key).
    # Secondary keys (e.g. tool_log) are per-iteration metadata.
    output_type = normalize_outputs(output_type).primary

    # Dict-form inputs: the back-edge feeds the node's output back as one
    # of its own inputs.  Check if the output is compatible with ANY input
    # value type — the compiler wires the specific slot.
    if isinstance(input_type, dict):
        for _key, expected in input_type.items():
            if _types_compatible(output_type, expected):
                return
        # No compatible input slot found.
        type_list = ", ".join(f"{k}={_fmt_type(v)}" for k, v in input_type.items())
        raise ConstructError.build(
            f"loop back-edge: output type {_fmt_type(output_type)} not compatible with any reentry input type",
            expected=f"one of ({type_list})",
            found=_fmt_type(output_type),
            hint="the loop's last node output must match the reentry node's input type",
            node=node.name,
            location=_source_location(),
        )
        return

    # Single-type inputs.
    if not _types_compatible(output_type, input_type):
        raise ConstructError.build(
            "loop back-edge: output type not compatible with reentry input type",
            expected=_fmt_type(input_type),
            found=_fmt_type(output_type),
            hint="the loop's last node output must match the reentry node's input type",
            node=node.name,
            location=_source_location(),
        )


def validate_loop_construct(construct: Construct) -> None:
    """Validate Loop on a Construct: both input and output must be declared.

    Called at ``|`` time from ``Modifiable.__or__`` when a Loop modifier
    is applied to a Construct.

    When output is compatible with input, the loop feeds each iteration's
    output back as the next iteration's input (classic refine pattern).

    When output differs from input (produce+validate pattern), the loop
    re-reads original inputs from parent state on each iteration instead
    of feeding output back.  The condition callable still receives the
    output type.
    """
    if construct.output is None or construct.input is None:
        raise ConstructError.build(
            "Loop requires both input= and output= declared",
            found=f"input={'declared' if construct.input is not None else 'None'}, "
            f"output={'declared' if construct.output is not None else 'None'}",
            hint="declare both input= and output= on the construct for Loop to wire the back-edge",
            construct=construct.name,
            location=_source_location(),
        )


def _validate_merge_hooks(oracle: Oracle, node: Node, construct_name: str) -> None:
    """Validate arity and type annotations of merge hook callables.

    Checks merge_pre_process, merge_post_process, merge_fallback signatures
    against the node's output type (or oracle_gen_type if set). Skips type
    checks when annotations are absent (lambdas). Never raises for missing
    annotations — only for provably wrong ones.
    """
    # Resolve the variant type (what the hooks receive as list elements)
    gen_type = normalize_outputs(getattr(node, "oracle_gen_type", None) or node.outputs).primary

    # The post-merge output type (what post_process/fallback must return)
    output_type = normalize_outputs(node.outputs).primary

    hooks: list[tuple[str, Callable | None, int]] = [
        ("merge_pre_process", oracle.merge_pre_process, 1),
        ("merge_post_process", oracle.merge_post_process, 2),
        ("merge_fallback", oracle.merge_fallback, 2),
    ]

    for hook_name, hook_fn, expected_arity in hooks:
        if hook_fn is None:
            continue

        # Arity check
        try:
            sig = inspect.signature(hook_fn)
        except (ValueError, TypeError):
            continue  # built-in or unresolvable — skip

        positional = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        required = [p for p in positional if p.default is inspect.Parameter.empty]

        if len(required) < expected_arity:
            raise ConstructError.build(
                f"{hook_name} requires {expected_arity} positional parameter(s), got {len(required)}",
                expected=f"{expected_arity} params",
                found=f"{len(required)} required params: {[p.name for p in required]}",
                node=node.name,
                construct=construct_name,
            )

        # Type annotation check (best-effort — lambdas have no annotations)
        try:
            hints = get_type_hints(hook_fn, include_extras=False)
        except (NameError, AttributeError, TypeError):
            continue  # unresolvable annotations (lambdas, closures) — skip type check

        if not hints:
            continue

        param_names = [p.name for p in positional]

        # Check variants param:
        # pre_process: first param is variants
        # post_process: second param is variants (first is result)
        # fallback: first param is variants
        variants_idx = 1 if hook_name == "merge_post_process" else 0
        if len(param_names) > variants_idx:
            variants_hint = hints.get(param_names[variants_idx])
            if variants_hint is not None:
                # Expect list[T] where T is compatible with gen_type
                origin = get_origin(variants_hint)
                if origin is list:
                    args = get_args(variants_hint)
                    if args and args[0] is not Any:
                        elem_type = args[0]
                        if isinstance(elem_type, type) and isinstance(gen_type, type):
                            if not _types_compatible(gen_type, elem_type):
                                raise ConstructError.build(
                                    f"{hook_name} variants param '{param_names[0]}' "
                                    f"type mismatch: declared list[{_fmt_type(elem_type)}] "
                                    f"but Oracle generates {_fmt_type(gen_type)}",
                                    expected=f"list[{_fmt_type(gen_type)}]",
                                    found=f"list[{_fmt_type(elem_type)}]",
                                    node=node.name,
                                    construct=construct_name,
                                )

        # Check return type (post_process and fallback must return output_type)
        if hook_name in ("merge_post_process", "merge_fallback"):
            return_hint = hints.get("return")
            if (
                return_hint is not None
                and return_hint is not Any
                and isinstance(return_hint, type)
                and isinstance(output_type, type)
                and not _types_compatible(output_type, return_hint)
                and not _types_compatible(return_hint, output_type)
            ):
                raise ConstructError.build(
                    f"{hook_name} return type mismatch: declared "
                    f"{_fmt_type(return_hint)} but node outputs {_fmt_type(output_type)}",
                    expected=_fmt_type(output_type),
                    found=_fmt_type(return_hint),
                    node=node.name,
                    construct=construct_name,
                )
