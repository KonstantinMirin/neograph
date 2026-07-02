"""Mode dispatch protocol and implementations for node execution.

Extracted from factory.py — these types define the mode-specific execution
strategies injected into the unified ``_execute_node`` path:

    ScriptedDispatch — deterministic Python function
    ThinkDispatch    — single LLM call, structured JSON output
    ToolDispatch     — ReAct tool loop with tools (read-only or mutation)
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from neograph import _llm, _tool_loop
from neograph._llm_render import _is_inline_prompt
from neograph._llm_runtime import EMPTY_RUNTIME, LlmRuntime
from neograph._normalize import normalize_outputs
from neograph.errors import ConfigurationError
from neograph.node import Node, TypeSpecStatic
from neograph.renderers import build_rendered_input
from neograph.tool import ToolBudgetTracker

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
        # Fail loud rather than silently threadpool the sync LLM vertical
        # (review H2 — a sync-delegate here blocks the event loop invisibly).
        # The awaiting async LLM path lands in Phase 1c.
        raise NotImplementedError("async LLM/tool dispatch lands in Phase 1c")


class ToolDispatch:
    """Dispatch for agent/act modes — ReAct tool loop.

    Handles: rendering, output model resolution, ToolBudgetTracker creation,
    dual-path renderer resolution (node.renderer > runtime.renderer), Oracle
    model override, oracle_gen_type resolution for dict-form outputs with
    tools, invoke_with_tools call, tool_log wiring.
    """

    def __init__(
        self,
        runtime: LlmRuntime = EMPTY_RUNTIME,
        tool_factory_lookup: dict[str, Callable] | None = None,
    ) -> None:
        self.runtime = runtime
        self.tool_factory_lookup = tool_factory_lookup or {}

    def execute(
        self,
        node: Node,
        input_data: NodeInput,
        config: RunnableConfig,
        context_data: dict[str, str] | None,
    ) -> NodeOutput:
        rendered = _render_input(node, input_data.value, runtime=self.runtime)
        output_model, primary_key = _resolve_primary_output(node)
        no = normalize_outputs(node.outputs)

        budget_tracker = ToolBudgetTracker(node.tools)

        # Renderer resolution: node-level renderer takes priority, then runtime
        effective_renderer = node.renderer or self.runtime.renderer

        effective_model = config.get("configurable", {}).get("_oracle_model", node.model) or ""

        # Resolve oracle_gen_type for dict-form outputs with tools
        oracle_gen_type = output_model
        if no.is_dict_form and primary_key is not None:
            oracle_gen_type = no.all_keys[primary_key]

        result, tool_interactions = _tool_loop.invoke_with_tools(
            self.runtime,
            model_tier=effective_model,
            prompt_template=node.prompt or "",
            input_data=rendered,
            output_model=cast(type[BaseModel], oracle_gen_type),
            tools=node.tools,
            budget_tracker=budget_tracker,
            config=config,
            node_name=node.name,
            llm_config=node.llm_config,
            renderer=effective_renderer,
            context=context_data,
            tool_factory_lookup=self.tool_factory_lookup,
        )

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

    async def aexecute(
        self,
        node: Node,
        input_data: NodeInput,
        config: RunnableConfig,
        context_data: dict[str, str] | None,
    ) -> NodeOutput:
        # Fail loud rather than silently threadpool the sync tool loop
        # (review H2). The awaiting async tool loop lands in Phase 1c.
        raise NotImplementedError("async LLM/tool dispatch lands in Phase 1c")


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
    Agent/Act: ReAct tool loop via ToolDispatch (captures runtime).
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
        return ToolDispatch(runtime, tool_factory_lookup=tool_factory_lookup)
    raise ConfigurationError.build(
        f"Unknown mode '{node.mode}'",
        expected="scripted, think, agent, or act",
        found=node.mode,
        node=node.name,
    )
