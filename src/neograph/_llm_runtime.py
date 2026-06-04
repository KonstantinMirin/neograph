"""LlmRuntime — frozen, closure-captured LLM configuration bundle.

Per docs/design/architecture-decisions.md §2: `compile()` reads runtime
configuration from keyword arguments and closes them over into factory
closures. The bundle below is the internal aggregator that flows through
those closures — it is NOT exported from `neograph.__init__`.

The dataclass is frozen so once `compile()` builds an instance, no
downstream call can mutate the shared runtime. Two `compile()` calls
produce two independent `LlmRuntime` instances; their compiled graphs
do not share state through this object.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neograph._llm import CostCallback, LlmFactory, PromptCompiler
    from neograph.renderers import Renderer


_ACCEPT_ALL: frozenset[str] = frozenset({"__all__"})


def _accepted_params(fn: Callable) -> frozenset[str]:
    """Inspect a callable and return the set of parameter names it accepts.

    Returns the `_ACCEPT_ALL` sentinel for functions that accept `**kwargs`.
    Returns an empty frozenset for builtins/C extensions whose signature
    cannot be introspected.
    """
    try:
        sig = inspect.signature(fn)
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return _ACCEPT_ALL
        return frozenset(sig.parameters.keys())
    except (ValueError, TypeError):
        return frozenset()


@dataclass(frozen=True)
class LlmRuntime:
    """Immutable bundle of runtime LLM configuration captured at compile time.

    `compile()` constructs this from its kwargs and threads the instance
    through every factory closure that needs LLM access. The dataclass is
    frozen so no closure can mutate the shared runtime.
    """

    llm_factory: LlmFactory | None = None
    prompt_compiler: PromptCompiler | None = None
    renderer: Renderer | None = None
    cost_callback: CostCallback | None = None
    llm_factory_params: frozenset[str] = field(default_factory=frozenset)
    prompt_compiler_params: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def build(
        cls,
        *,
        llm_factory: LlmFactory | None = None,
        prompt_compiler: PromptCompiler | None = None,
        renderer: Renderer | None = None,
        cost_callback: CostCallback | None = None,
    ) -> LlmRuntime:
        """Construct an LlmRuntime, inspecting callable signatures upfront.

        Inspection happens once at compile time so per-invocation calls do
        not pay the `inspect.signature` cost.
        """
        return cls(
            llm_factory=llm_factory,
            prompt_compiler=prompt_compiler,
            renderer=renderer,
            cost_callback=cost_callback,
            llm_factory_params=(
                _accepted_params(llm_factory) if llm_factory is not None else frozenset()
            ),
            prompt_compiler_params=(
                _accepted_params(prompt_compiler) if prompt_compiler is not None else frozenset()
            ),
        )


# Sentinel for "no runtime supplied" — distinct from a runtime with all-None
# fields. Used by helpers that fall back to a compile-time default if a
# closure passed nothing. Frozen and shared.
EMPTY_RUNTIME: LlmRuntime = LlmRuntime()


def collect_llm_nodes(construct: Any) -> list[str]:
    """Return the names of all LLM-mode (think/agent/act) nodes in ``construct``.

    Single source of truth for the LLM-node walk shared by
    ``check_llm_kwargs_or_raise`` (raises) and ``lint`` (reports). Recurses
    into sub-constructs via ``iter_nodes``.
    """
    # Function-local import keeps this leaf module dependency-free.
    from neograph.construct import iter_nodes

    return [n.name for n in iter_nodes(construct) if n.mode in ("think", "agent", "act")]


def missing_runtime_kwargs(llm_factory: Any, prompt_compiler: Any) -> list[str]:
    """Return which required LLM runtime kwargs are absent (``[]`` when all set)."""
    missing: list[str] = []
    if llm_factory is None:
        missing.append("llm_factory")
    if prompt_compiler is None:
        missing.append("prompt_compiler")
    return missing


def check_llm_kwargs_or_raise(
    construct: Any,
    llm_factory: Any,
    prompt_compiler: Any,
    *,
    source: str,
) -> None:
    """Fail-loud helper shared by `compile()` / `lint()` / `Node.run_isolated()`.

    Walks the construct's nodes (recursing into sub-constructs). If any
    LLM-mode node is found and either kwarg is missing, raises CompileError
    with the offending node names verbatim.
    """
    from neograph.errors import CompileError

    llm_nodes = collect_llm_nodes(construct)
    if not llm_nodes:
        return
    missing = missing_runtime_kwargs(llm_factory, prompt_compiler)
    if not missing:
        return
    raise CompileError.build(
        f"{source}: LLM-mode nodes ({', '.join(llm_nodes)}) require runtime configuration",
        expected="llm_factory= and prompt_compiler= passed to " + source,
        found=f"{' and '.join(missing)} not set",
        hint=f"Pass llm_factory= and prompt_compiler= to {source}.",
        construct=construct.name,
    )
