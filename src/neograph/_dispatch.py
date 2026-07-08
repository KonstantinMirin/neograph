"""Mode dispatch protocol and implementations for node execution.

Extracted from factory.py — these types define the mode-specific execution
strategies injected into the unified ``_execute_node`` path:

    ScriptedDispatch — deterministic Python function
    ThinkDispatch    — single LLM call, structured JSON output

Agent/act modes do NOT dispatch here — they compile to a multi-node inline ReAct
cycle (``_agent_cycle`` via ``_wiring._add_agent_cycle``). ``_shape_tool_output``
/ ``_render_input`` / ``_resolve_primary_output`` remain here because the cycle's
parse node reuses them.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol, cast

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from neograph import _llm
from neograph._config_carrier import _with_configurable
from neograph._llm_render import _is_inline_prompt
from neograph._llm_runtime import EMPTY_RUNTIME, LlmRuntime
from neograph._normalize import NormalizedOutputs, normalize_outputs
from neograph._run_cache import aget_or_build
from neograph._sidecar import _get_param_res
from neograph._state_keys import StateKeys
from neograph.di import DI_TEMPLATE_KINDS, DIKind
from neograph.errors import ConfigurationError, ExecutionError
from neograph.node import Node, TypeSpecStatic
from neograph.renderers import build_rendered_input


def _inject_di_inputs(node: Node, config: RunnableConfig) -> RunnableConfig:
    """Resolve a node's FromInput/FromConfig params and stash them on config.

    LLM-mode nodes never run their body, so — unlike scripted nodes, whose shim
    resolves DI — their FromInput/FromConfig params are otherwise dropped. That
    is the production bug where a ``domain: Annotated[str, FromInput]`` declared
    on a think node never reached its ``{domain}`` template placeholder. This
    resolves those params ONCE through the canonical ``DIBinding.resolve`` and
    injects the ``{param_name: value}`` map under ``StateKeys.DI_INPUTS``
    (copy-not-mutate, mirroring ``_inject_oracle_config``), so ``_compile_prompt``
    can offer it to a ``di_inputs``-aware prompt_compiler. Returns config
    unchanged when the node declares no template-usable DI params.
    """
    param_res = _get_param_res(node)
    if not param_res:
        return config
    # FROM_RESOURCE on an LLM-mode node is only ever a template var (the body
    # never runs), and its value is an AWAITED fetch — the sync driver cannot
    # serve it. Fail loud (naming param + node) rather than silently dropping the
    # template var (the R2 silent-no-op hole, neograph-3q6j). arun() routes to
    # `_ainject_di_inputs`, which awaits the fetch.
    resource_params = sorted(name for name, b in param_res.items() if b.kind is DIKind.FROM_RESOURCE)
    if resource_params:
        raise ConfigurationError.build(
            f"resource DI parameter(s) {resource_params} cannot resolve on the "
            f"sync run() driver (a FromResource fetch is awaited)",
            node=node.name,
            hint="drive the graph with arun() so the async di_inputs twin can "
            "await the fetch and feed the fetched text to the prompt template.",
        )
    di_inputs = {
        name: binding.resolve(config) for name, binding in param_res.items() if binding.kind in DI_TEMPLATE_KINDS
    }
    if not di_inputs:
        return config
    return _with_configurable(config, **{StateKeys.DI_INPUTS: di_inputs})


async def _ainject_di_inputs(node: Node, config: RunnableConfig) -> RunnableConfig:
    """Async twin of :func:`_inject_di_inputs`. See neograph-3q6j.

    Same contract and DI_INPUTS side-channel, but AWAITS FROM_RESOURCE bindings
    (the fetch) in addition to the synchronous DI_TEMPLATE_KINDS — so a fetched
    resource's text can serve as an LLM-mode prompt template var (e.g. ``{history}``)
    on the arun() path. Reuses the canonical ``DIBinding.aresolve`` (no second
    resolver): it awaits FROM_RESOURCE and delegates every other kind to the sync
    ``resolve``. Gate is DI_TEMPLATE_KINDS-or-FROM_RESOURCE (the sync twin stays
    DI_TEMPLATE_KINDS only and fails loud on FROM_RESOURCE). Copy-not-mutate, so
    per-superstep re-injection is idempotent. Returns config unchanged when the
    node declares no template-usable DI params.
    """
    param_res = _get_param_res(node)
    if not param_res:
        return config
    di_inputs = {}
    for name, binding in param_res.items():
        if binding.kind in DI_TEMPLATE_KINDS:
            di_inputs[name] = binding.resolve(config)
        elif binding.kind is DIKind.FROM_RESOURCE:
            # Per-run fetch cache: the agent cycle re-runs this
            # injector every superstep, so a FROM_RESOURCE fetch would hit the
            # consumer's fetcher each turn. Cache the awaited value on the
            # framework-minted RUN_ID (config-only, re-minted on resume -> refetch).
            # Copy-not-mutate idempotence is preserved: the resolved value is
            # stashed into a fresh config copy below exactly as before.
            di_inputs[name] = await aget_or_build(
                config,
                f"resource:{node.name}:{name}",
                partial(binding.aresolve, config),
            )
    if not di_inputs:
        return config
    return _with_configurable(config, **{StateKeys.DI_INPUTS: di_inputs})


# ── Typed dispatch containers (architecture-v2 section 1) ────────────────
#
# These replace bare ``Any`` at the factory boundary. Values are NOT
# constrained to BaseModel — real inputs include str, int, list, dict,
# BaseModel. The shape discriminator (single vs fan_in / multi) tells the
# dispatch which form it received.

# Concrete union of values that flow through the node execution pipeline.
NodeValue = BaseModel | dict | str | int | float | list | None


@dataclass
class NodeInput:
    """Typed container for extracted node input — replaces bare Any.

    ``single`` holds a single upstream value (any type).
    ``fan_in`` holds a dict-form fan-in mapping (upstream_name -> value).
    When both are set, ``fan_in`` takes precedence (the dict form is the
    more specific representation).
    """

    single: NodeValue = None
    fan_in: dict[str, NodeValue] | None = None

    @property
    def value(self) -> NodeValue | dict[str, NodeValue]:
        """The input value in the shape the mode dispatch expects."""
        if self.fan_in is not None:
            return self.fan_in
        return self.single


@dataclass
class NodeOutput:
    """Typed container for node output — replaces bare Any.

    ``single`` holds a single-type output (any type).
    ``multi`` holds a dict-form multi-output mapping (key -> value).
    When both are set, ``multi`` takes precedence.
    """

    single: NodeValue = None
    multi: dict[str, NodeValue] | None = None

    @property
    def value(self) -> NodeValue | dict[str, NodeValue]:
        """The output value in the shape _build_state_update expects."""
        if self.multi is not None:
            return self.multi
        return self.single


class ModeDispatch(Protocol):
    """Mode-specific execution logic, injected into the unified wrapper.

    Input and output are typed containers, not Any. The dispatch receives
    exactly what _extract_input produced (NodeInput) and returns exactly
    what _build_state_update expects (NodeOutput). No type erasure.
    """

    def execute(
        self,
        node: Node,
        input_data: NodeInput,
        config: RunnableConfig,
        context_data: dict[str, str] | None,
    ) -> NodeOutput: ...

    async def aexecute(
        self,
        node: Node,
        input_data: NodeInput,
        config: RunnableConfig,
        context_data: dict[str, str] | None,
    ) -> NodeOutput: ...


class ScriptedDispatch:
    """Dispatch for scripted (deterministic Python) nodes.

    Delegates to the registered scripted function. Receives ``context_data``
    in its ``execute()`` signature for protocol conformance but does NOT
    pass it to the function — scripted nodes have no LLM context needs.
    """

    def __init__(self, fn: Callable) -> None:
        self.fn = fn

    def execute(
        self,
        node: Node,
        input_data: NodeInput,
        config: RunnableConfig,
        context_data: dict[str, str] | None,
    ) -> NodeOutput:
        # context_data intentionally unused — scripted functions don't need LLM context
        result = self.fn(input_data.value, config)
        if inspect.isawaitable(result):
            # An `async def` body under the SYNC driver: we cannot await here, and
            # returning the coroutine would flow un-awaited into state (silent wrong
            # behavior). Fail loud — the async twin aexecute() awaits correctly.
            if hasattr(result, "close"):
                result.close()  # suppress the "never awaited" RuntimeWarning
            raise ExecutionError.build(
                "async node body invoked under sync run(); use arun()",
                node=node.name,
                hint="An `async def` scripted body requires the async driver. "
                "Call arun(graph, ...) / graph.ainvoke instead of run() / graph.invoke.",
            )
        return NodeOutput(single=result)

    async def aexecute(
        self,
        node: Node,
        input_data: NodeInput,
        config: RunnableConfig,
        context_data: dict[str, str] | None,
    ) -> NodeOutput:
        # The user's body may be `async def`, but for @node it is hidden behind
        # a synchronous shim (scripted_shim `return fn(*args)`), so
        # iscoroutinefunction(self.fn) is blind to it. Detect the awaitable at
        # the call boundary instead — this works identically across @node,
        # declarative, and programmatic surfaces. A sync body returns a plain
        # value (isawaitable False) and is not awaited.
        result = self.fn(input_data.value, config)
        if inspect.isawaitable(result):
            result = await result
        return NodeOutput(single=result)


class ThinkDispatch:
    """Dispatch for think mode — single LLM call, structured JSON output.

    Handles: rendering, output model resolution, Oracle model override,
    invoke_structured call, dict-form primary key wrapping.
    """

    def __init__(self, runtime: LlmRuntime = EMPTY_RUNTIME) -> None:
        self.runtime = runtime

    def execute(
        self,
        node: Node,
        input_data: NodeInput,
        config: RunnableConfig,
        context_data: dict[str, str] | None,
    ) -> NodeOutput:
        config = _inject_di_inputs(node, config)
        rendered = _render_input(node, input_data.value, runtime=self.runtime)
        output_model, primary_key = _resolve_primary_output(node)
        effective_model = config.get("configurable", {}).get("_oracle_model", node.model) or ""

        # think mode always resolves to a concrete BaseModel class
        # (TypeSpecStatic includes dict-form for multi-output Nodes; the
        # primary key resolution above unwraps it).
        result = _llm.invoke_structured(
            self.runtime,
            model_tier=effective_model,
            prompt_template=node.prompt or "",
            input_data=rendered,
            output_model=cast(type[BaseModel], output_model),
            config=config,
            node_name=node.name,
            llm_config=node.llm_config,
            context=context_data,
        )

        if primary_key is not None and result is not None:
            return NodeOutput(multi={primary_key: result})
        return NodeOutput(single=result)

    async def aexecute(
        self,
        node: Node,
        input_data: NodeInput,
        config: RunnableConfig,
        context_data: dict[str, str] | None,
    ) -> NodeOutput:
        # Async twin of execute(): same pure preamble/postamble, awaits the LLM
        # vertical (_llm.ainvoke_structured) instead of blocking it. The async
        # injector twin additionally awaits FROM_RESOURCE bindings. See neograph-3q6j.
        config = await _ainject_di_inputs(node, config)
        rendered = _render_input(node, input_data.value, runtime=self.runtime)
        output_model, primary_key = _resolve_primary_output(node)
        effective_model = config.get("configurable", {}).get("_oracle_model", node.model) or ""

        result = await _llm.ainvoke_structured(
            self.runtime,
            model_tier=effective_model,
            prompt_template=node.prompt or "",
            input_data=rendered,
            output_model=cast(type[BaseModel], output_model),
            config=config,
            node_name=node.name,
            llm_config=node.llm_config,
            context=context_data,
        )

        if primary_key is not None and result is not None:
            return NodeOutput(multi={primary_key: result})
        return NodeOutput(single=result)


def _shape_tool_output(
    result: BaseModel | None,
    tool_interactions: list,
    no: NormalizedOutputs,
    primary_key: str | None,
) -> NodeOutput:
    """Pure postamble reused by the agent cycle's parse node: shape the ReAct
    result + tool_log into a NodeOutput (single or dict-form)."""
    if primary_key is not None and result is not None:
        result_dict: dict[str, Any] = {primary_key: result}
        if no.is_dict_form and "tool_log" in no.all_keys:
            result_dict["tool_log"] = tool_interactions
        return NodeOutput(multi=result_dict)
    elif no.is_dict_form:
        pk = no.primary_key
        assert pk is not None  # is_dict_form implies primary_key is set
        result_dict = {pk: result} if result is not None else {}
        if result is not None and "tool_log" in no.all_keys:
            result_dict["tool_log"] = tool_interactions
        return NodeOutput(multi=result_dict) if result_dict else NodeOutput()
    return NodeOutput(single=result)


def _render_input(
    node: Node,
    input_data: Any,
    *,
    runtime: LlmRuntime = EMPTY_RUNTIME,
) -> Any:
    """Apply renderer dispatch chain via RenderedInput.

    Builds a RenderedInput carrying both raw and rendered views, then
    returns the appropriate view based on prompt type:
    - Inline prompts: raw data (for ${var.field} dotted access)
    - Template-ref prompts: rendered + flattened (for prompt_compiler)
    """
    effective_renderer = node.renderer or runtime.renderer

    ri = build_rendered_input(input_data, renderer=effective_renderer)

    prompt = node.prompt or ""
    if _is_inline_prompt(prompt):
        return ri.raw
    return ri.for_template_ref


def _resolve_primary_output(node: Node) -> tuple[TypeSpecStatic, str | None]:
    """Resolve the LLM output model and primary key for dict-form outputs.

    For dict-form outputs, the LLM produces the primary type (first key).
    Secondary outputs (e.g. tool_log) are framework-collected.

    When ``node.oracle_gen_type`` is set (Oracle with type-transforming merge_fn),
    the generator type overrides ``node.outputs`` — the LLM should produce the
    per-variant type, not the post-merge type.

    Returns (output_model, primary_key) where primary_key is None for
    single-type outputs.
    """
    # Oracle generator type override: merge_fn transforms A -> B, generators produce A.
    if node.oracle_gen_type is not None:
        return node.oracle_gen_type, None

    no = normalize_outputs(node.outputs)
    return no.primary, no.primary_key


def _dispatch_for_mode(
    node: Node,
    *,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    scripted_lookup: dict[str, Callable] | None = None,
    tool_factory_lookup: dict[str, Callable] | None = None,
) -> ModeDispatch:
    """Resolve the ModeDispatch for a node based on its mode.

    Scripted: delegates to the registered function via ScriptedDispatch.
    The function is looked up in ``scripted_lookup`` (per-compile dict).
    Think: single LLM call via ThinkDispatch (captures runtime).

    Agent/Act do NOT resolve here — they compile to a multi-node inline ReAct
    cycle (``_wiring._add_agent_cycle`` / ``_agent_cycle``), not a single
    ``_execute_node`` dispatch. Reaching this function with an agent/act node is a
    wiring bug (the compiler routes them to the cycle before ``make_node_fn``).
    """
    if node.mode == "scripted":
        if node.scripted_fn is None:
            raise ConfigurationError.build(
                "Scripted node has no scripted_fn registered",
                node=node.name,
            )
        per_compile = scripted_lookup or {}
        fn = per_compile.get(node.scripted_fn)
        if fn is None:
            raise ConfigurationError.build(
                f"Scripted function '{node.scripted_fn}' not registered",
                hint=f"Pass scripted={{'{node.scripted_fn}': fn}} to compile().",
                node=node.name,
            )
        return ScriptedDispatch(fn)
    if node.mode == "think":
        return ThinkDispatch(runtime)
    if node.mode in ("agent", "act"):
        raise ConfigurationError.build(
            f"agent/act node '{node.name}' cannot use single-node dispatch",
            expected="compile via the inline agent cycle (_wiring._add_agent_cycle)",
            found=f"_dispatch_for_mode reached with mode='{node.mode}'",
            hint="This is an internal wiring error — agent/act nodes are expanded "
            "to an agent/tools/parse cycle, not make_node_fn.",
            node=node.name,
        )
    raise ConfigurationError.build(
        f"Unknown mode '{node.mode}'",
        expected="scripted, think, agent, or act",
        found=node.mode,
        node=node.name,
    )
