"""Mode dispatch protocol and implementations for node execution.

Extracted from factory.py — these types define the mode-specific execution
strategies injected into the unified ``_execute_node`` path:

    ScriptedDispatch — deterministic Python function
    ThinkDispatch    — single LLM call, structured JSON output
    ToolDispatch     — ReAct tool loop with tools (read-only or mutation)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from neograph._llm import _is_inline_prompt
from neograph.errors import ConfigurationError
from neograph.node import Node

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
        context_data: dict[str, Any] | None,
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
        context_data: dict[str, Any] | None,
    ) -> NodeOutput:
        # context_data intentionally unused — scripted functions don't need LLM context
        result = self.fn(input_data.value, config)
        return NodeOutput(single=result)


class ThinkDispatch:
    """Dispatch for think mode — single LLM call, structured JSON output.

    Handles: rendering, output model resolution, Oracle model override,
    invoke_structured call, dict-form primary key wrapping.
    """

    def execute(
        self,
        node: Node,
        input_data: NodeInput,
        config: RunnableConfig,
        context_data: dict[str, Any] | None,
    ) -> NodeOutput:
        from neograph._llm import invoke_structured

        rendered = _render_input(node, input_data.value)
        output_model, primary_key = _resolve_primary_output(node)
        effective_model = config.get("configurable", {}).get("_oracle_model", node.model) or ""

        result = invoke_structured(
            model_tier=effective_model,
            prompt_template=node.prompt or "",
            input_data=rendered,
            output_model=output_model,
            config=config,
            node_name=node.name,
            llm_config=node.llm_config,
            context=context_data,
        )

        if primary_key is not None and result is not None:
            return NodeOutput(multi={primary_key: result})
        return NodeOutput(single=result)


class ToolDispatch:
    """Dispatch for agent/act modes — ReAct tool loop.

    Handles: rendering, output model resolution, ToolBudgetTracker creation,
    dual-path renderer resolution (node.renderer > global), Oracle model
    override, oracle_gen_type resolution for dict-form outputs with tools,
    invoke_with_tools call, tool_log wiring.
    """

    def execute(
        self,
        node: Node,
        input_data: NodeInput,
        config: RunnableConfig,
        context_data: dict[str, Any] | None,
    ) -> NodeOutput:
        from neograph._llm import invoke_with_tools
        from neograph.tool import ToolBudgetTracker

        rendered = _render_input(node, input_data.value)
        output_model, primary_key = _resolve_primary_output(node)

        budget_tracker = ToolBudgetTracker(node.tools)

        # Renderer resolution: node-level renderer takes priority, then global
        try:
            from neograph._llm import _get_global_renderer
            effective_renderer = node.renderer or _get_global_renderer()
        except (ImportError, AttributeError):
            effective_renderer = node.renderer

        effective_model = config.get("configurable", {}).get("_oracle_model", node.model) or ""

        # Resolve oracle_gen_type for dict-form outputs with tools
        oracle_gen_type = output_model
        if isinstance(node.outputs, dict) and primary_key is not None:
            oracle_gen_type = node.outputs[primary_key]

        result, tool_interactions = invoke_with_tools(
            model_tier=effective_model,
            prompt_template=node.prompt or "",
            input_data=rendered,
            output_model=oracle_gen_type,
            tools=node.tools,
            budget_tracker=budget_tracker,
            config=config,
            node_name=node.name,
            llm_config=node.llm_config,
            renderer=effective_renderer,
            context=context_data,
        )

        if primary_key is not None and result is not None:
            result_dict: dict[str, Any] = {primary_key: result}
            if isinstance(node.outputs, dict) and "tool_log" in node.outputs:
                result_dict["tool_log"] = tool_interactions
            return NodeOutput(multi=result_dict)
        elif isinstance(node.outputs, dict):
            pk = next(iter(node.outputs))
            result_dict = {pk: result} if result is not None else {}
            if result is not None and "tool_log" in node.outputs:
                result_dict["tool_log"] = tool_interactions
            return NodeOutput(multi=result_dict) if result_dict else NodeOutput()
        return NodeOutput(single=result)


def _render_input(node: Node, input_data: Any) -> Any:
    """Apply renderer dispatch chain: node renderer > global renderer > BAML default.

    For inline prompts (containing spaces or ${} markers), returns raw input
    unchanged — inline var substitution needs raw model access for dotted paths
    like ${claim.text}. The _resolve_var function handles BAML rendering of
    resolved BaseModel values individually.

    For template-ref prompts, always renders via render_input — the prompt
    compiler receives rendered (BAML/XML/JSON) strings.
    """
    # Inline prompts need raw data for dotted var access
    prompt = node.prompt or ""
    if _is_inline_prompt(prompt):
        return input_data

    from neograph.renderers import render_input
    try:
        from neograph._llm import _get_global_renderer
        effective_renderer = node.renderer or _get_global_renderer()
    except ImportError:
        effective_renderer = node.renderer
    return render_input(input_data, renderer=effective_renderer)


def _resolve_primary_output(node: Node) -> tuple[Any, str | None]:
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

    if isinstance(node.outputs, dict):
        primary_key = next(iter(node.outputs))
        return node.outputs[primary_key], primary_key
    return node.outputs, None


def _dispatch_for_mode(node: Node) -> ModeDispatch:
    """Resolve the ModeDispatch for a node based on its mode.

    Scripted: delegates to registered function via ScriptedDispatch.
    Think: single LLM call via ThinkDispatch.
    Agent/Act: ReAct tool loop via ToolDispatch.
    """
    from neograph._registry import registry

    if node.mode == "scripted":
        if node.scripted_fn is None:
            raise ConfigurationError.build(
                "Scripted node has no scripted_fn registered",
                node=node.name,
            )
        fn = registry.scripted[node.scripted_fn]
        return ScriptedDispatch(fn)
    if node.mode == "think":
        return ThinkDispatch()
    if node.mode in ("agent", "act"):
        return ToolDispatch()
    raise ConfigurationError.build(
        f"Unknown mode '{node.mode}'",
        expected="scripted, think, agent, or act",
        found=node.mode,
        node=node.name,
    )
