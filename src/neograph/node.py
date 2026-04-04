"""Node — typed processing block that compiles to a LangGraph node.

    # Declarative: framework generates the LangGraph node function
    classify = Node(
        "classify",
        mode="produce",
        input=DecompositionResult,
        output=ClassificationResult,
        model="reason",
        prompt="rw/classify",
    )

    # Scripted: deterministic Python, no LLM
    build_catalog = Node.scripted("build-catalog", fn="build_catalog", output=str)

    # Raw: classic LangGraph escape hatch
    @raw_node(input=SomeInput, output=SomeOutput)
    def custom_logic(state, config):
        ...
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from pydantic import BaseModel

from neograph.modifiers import Modifiable, Modifier
from neograph.tool import Tool


class Node(Modifiable, BaseModel):
    """A typed processing block. The unit of graph specification.

    mode= determines execution mechanics:
        "produce"   — single LLM call, structured JSON output, no tools
        "gather"    — ReAct tool loop (exploration, read-only)
        "execute"   — ReAct tool loop (mutations, side effects)

    Node.scripted() creates a deterministic node (no LLM).
    """

    name: str
    mode: Literal["produce", "gather", "execute", "scripted"] = "produce"

    # Typed contracts — specific Pydantic models, not BaseModel
    input: Any = None   # type[BaseModel] | dict[str, type] | None
    output: Any = None  # type[BaseModel] | None

    # LLM configuration
    model: str | None = None        # "fast", "reason", "large"
    prompt: str | None = None       # template name in prompt registry
    llm_config: dict[str, Any] = {} # consumer-specific LLM settings (temperature, max_tokens, etc.)

    # Tools with per-tool budgets
    tools: list[Tool] = []

    # Deterministic implementation (scripted mode only)
    scripted_fn: str | None = None

    # Raw node function (raw_node only)
    raw_fn: Callable | None = None

    # Modifiers applied via | operator
    modifiers: list[Modifier] = []

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def scripted(
        cls,
        name: str,
        fn: str,
        input: Any = None,
        output: Any = None,
    ) -> Node:
        """Create a deterministic node — pure Python, no LLM.

        Usage:
            build_catalog = Node.scripted("build-catalog", fn="build_catalog", output=str)
        """
        return cls(
            name=name,
            mode="scripted",
            input=input,
            output=output,
            scripted_fn=fn,
        )

    # has_modifier, get_modifier, __or__ inherited from Modifiable
        return None


def raw_node(
    input: Any = None,
    output: Any = None,
) -> Callable:
    """Decorator: register a classic LangGraph function as a NeoGraph node.

    The framework wires edges, observability, and state around it,
    but the function body is yours — full control.

    Usage:
        @raw_node(input=ConsolidatedDisposition, output=ConsolidatedDisposition)
        def custom_resolution(state, config):
            ...
    """

    def decorator(fn: Callable) -> Node:
        return Node(
            name=fn.__name__.replace("_", "-"),
            mode="scripted",  # raw nodes are dispatched directly
            input=input,
            output=output,
            raw_fn=fn,
        )

    return decorator
