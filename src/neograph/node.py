"""Node — typed processing block that compiles to a LangGraph node.

    # Declarative: framework generates the LangGraph node function
    classify = Node(
        "classify",
        mode="produce",
        inputs=DecompositionResult,
        output=ClassificationResult,
        model="reason",
        prompt="rw/classify",
    )

    # Scripted: deterministic Python, no LLM
    build_catalog = Node.scripted("build-catalog", fn="build_catalog", output=str)

    # Raw: classic LangGraph escape hatch
    @node(mode='raw', inputs=SomeInput, output=SomeOutput)
    def custom_logic(state, config):
        ...
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

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
    inputs: Any = None   # type[BaseModel] | dict[str, type] | None
    output: Any = None  # type[BaseModel] | None

    # LLM configuration
    model: str | None = None        # "fast", "reason", "large"
    prompt: str | None = None       # template name in prompt registry
    llm_config: dict[str, Any] = Field(default_factory=dict)  # temperature, max_tokens, output_strategy, etc.

    # Tools with per-tool budgets
    tools: list[Tool] = []

    # Deterministic implementation (scripted mode only)
    scripted_fn: str | None = None

    # Raw node function — explicit mode='raw' escape hatch only.
    raw_fn: Callable | None = None

    # Which inputs key receives the Each fan-out item (neo_each_item) instead
    # of reading from the named upstream state field. Set by @node decoration
    # when map_over= is used. Used by factory._extract_input and by the
    # validator to skip upstream-name validation for this key.
    fan_out_param: str | None = None

    # Pluggable prompt-input renderer. When set, the factory layer renders
    # input data through this renderer before prompt insertion. Dispatch
    # hierarchy: model.render_for_prompt() > node.renderer > global > None.
    renderer: Any = None

    # Conditional produce: skip the LLM call when the predicate returns True.
    # skip_when receives the extracted input_data (after _extract_input, before
    # renderer). skip_value produces the output when skipped; if None, the node
    # returns an empty state update.
    skip_when: Callable | None = None
    skip_value: Callable | None = None

    # Modifiers applied via | operator
    modifiers: list[Modifier] = []

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, name_: str | None = None, /, **kwargs):
        """Node accepts name positionally or as a keyword argument."""
        if name_ is not None:
            kwargs["name"] = name_
        super().__init__(**kwargs)

    @classmethod
    def scripted(
        cls,
        name: str,
        fn: str,
        inputs: Any = None,
        output: Any = None,
    ) -> Node:
        """Create a deterministic node — pure Python, no LLM.

        Usage:
            build_catalog = Node.scripted("build-catalog", fn="build_catalog", output=str)
        """
        return cls(
            name=name,
            mode="scripted",
            inputs=inputs,
            output=output,
            scripted_fn=fn,
        )

    # has_modifier, get_modifier, __or__ inherited from Modifiable

    def run_isolated(
        self,
        input: Any = None,
        *,
        config: dict | None = None,
    ) -> Any:
        """Execute this node in isolation — for unit testing.

        Bypasses compile() and run(). Creates the node function via the
        factory, builds a minimal state with the provided input, and invokes
        it directly. Returns the node's output (the Pydantic model instance),
        not a state dict.

        Usage:

            # Unit test a scripted node
            result = extract.run_isolated(input={"raw": "hello"})
            assert result.text == "hello"

            # Unit test a produce node (with configure_llm already set)
            configure_llm(llm_factory=lambda tier: FakeLLM(), prompt_compiler=...)
            result = classify.run_isolated(input=Claims(items=["x"]))
            assert isinstance(result, Classified)

        Args:
            input: Either the typed input instance (e.g. a Claims(...) object)
                   or a dict of field-value pairs to seed the state.
            config: Optional RunnableConfig. Pipeline metadata goes in
                    config["configurable"]. Defaults to an empty configurable.
        """
        from neograph.factory import make_node_fn

        node_fn = make_node_fn(self)

        # Build a minimal state dict the node function can read
        state: dict[str, Any] = {}
        if isinstance(input, dict):
            state.update(input)
        elif input is not None:
            # Typed instance — place it under the node name so _extract_input finds it by type
            state["_neo_isolated_input"] = input

        config = config or {"configurable": {}}
        if "configurable" not in config:
            config["configurable"] = {}

        result = node_fn(state, config)

        # node_fn returns a state update dict — extract the output field
        field_name = self.name.replace("-", "_")
        return result.get(field_name)
