"""Node — typed processing block that compiles to a LangGraph node.

    # Declarative: framework generates the LangGraph node function
    classify = Node(
        "classify",
        mode="think",
        inputs=DecompositionResult,
        outputs=ClassificationResult,
        model="reason",
        prompt="rw/classify",
    )

    # Scripted: deterministic Python, no LLM
    build_catalog = Node.scripted("build-catalog", fn="build_catalog", outputs=str)

    # Raw: classic LangGraph escape hatch
    @node(mode='raw', inputs=SomeInput, outputs=SomeOutput)
    def custom_logic(state, config):
        ...
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, PlainValidator, PrivateAttr

from neograph.errors import ConstructError


def _validate_type_spec(v: Any) -> Any:
    """Accept type objects, generic aliases, and dict-form type specs.

    Rejects ints and other non-type garbage that would silently pass through
    to compile() and produce confusing errors downstream.

    Valid forms: None, concrete type, generic alias (list[X], dict[str,X],
    Optional[X], X|None), dict[str, type|str|GenericAlias].  Dict values may
    be strings (loader path uses type names before resolution, and the decorator
    fallback path produces string annotations when get_type_hints fails).
    """
    if v is None:
        return None
    if isinstance(v, dict):
        for key, val in v.items():
            if not isinstance(key, str):
                raise TypeError(f"dict-form type spec keys must be strings, got {type(key).__name__}")
            if not (isinstance(val, (type, str)) or _is_type_like(val)):
                raise TypeError(f"dict-form type spec value for '{key}' must be a type or type name, got {type(val).__name__}: {val!r}")
        return v
    if isinstance(v, type) or _is_type_like(v):
        return v
    raise TypeError(f"inputs/outputs must be a type, dict[str, type], or None — got {type(v).__name__}: {v!r}")


def _is_type_like(v: Any) -> bool:
    """Check if v is a generic alias (list[X], dict[str, X], Optional[X], X | None)."""
    import types as _types
    import typing
    return (
        hasattr(v, "__origin__")
        or isinstance(v, (typing._GenericAlias, typing._SpecialForm))  # type: ignore[attr-defined]
        or isinstance(v, _types.UnionType)
    )


# Valid forms: None | type | GenericAlias (list[X], dict[str,X], X|None) |
# dict[str, type|str|GenericAlias].  Static annotation is Any because Python
# has no TypeForm (PEP 747). PlainValidator is the real enforcement point.
TypeSpec = Annotated[Any, PlainValidator(_validate_type_spec)]

from neograph.modifiers import Modifiable, ModifierSet
from neograph.naming import field_name_for
from neograph.tool import Tool


class Node(Modifiable, BaseModel):
    """A typed processing block. The unit of graph specification.

    mode= determines execution mechanics:
        "think"     — single LLM call, structured JSON output, no tools
        "agent"     — ReAct tool loop (exploration, read-only)
        "act"       — ReAct tool loop (mutations, side effects)

    Node.scripted() creates a deterministic node (no LLM).
    """

    name: str
    mode: Literal["think", "agent", "act", "scripted"] = "think"

    # Typed contracts — specific Pydantic models, not BaseModel
    inputs: TypeSpec = None
    outputs: TypeSpec = None

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

    # Verbatim state fields injected into the prompt alongside typed input.
    # Values are passed as-is (not BAML-rendered). Use for pre-formatted
    # context like graph catalogs or domain briefings.
    context: list[str] | None = None

    # Conditional produce: skip the LLM call when the predicate returns True.
    # skip_when receives the extracted input_data (after _extract_input, before
    # renderer). skip_value produces the output when skipped; if None, the node
    # returns an empty state update.
    skip_when: Callable | None = None
    skip_value: Callable | None = None

    # Oracle generator output type — when merge_fn transforms types (A → B),
    # this is A (per-variant type). The LLM produces this type, the merge_fn
    # converts list[A] → B (= node.outputs). Inferred from merge_fn signature.
    oracle_gen_type: Any = None

    # Modifiers applied via | operator (typed slots, not a list)
    modifier_set: ModifierSet = Field(default_factory=ModifierSet)

    # Sidecar metadata — lives on the Node via PrivateAttr, not in global dicts.
    # Preserved by model_copy (Pydantic v2 copies __pydantic_private__).
    # _sidecar: (original_fn, param_names_tuple) from @node decoration.
    # _param_res: DI bindings from _classify_di_params.
    _sidecar: tuple[Callable, tuple[str, ...]] | None = PrivateAttr(default=None)
    _param_res: dict | None = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, name_: str | None = None, /, **kwargs):
        """Node accepts name positionally or as a keyword argument."""
        if name_ is not None:
            kwargs["name"] = name_
        # Reject legacy modifiers=[...] constructor form.
        # modifier_set: ModifierSet replaces the old list field. Passing
        # modifiers= would be silently ignored — fail loudly instead.
        if "modifiers" in kwargs:

            raise ConstructError.build(
                "Node(modifiers=[...]) is no longer supported",
                hint="Use the pipe syntax instead: node | Oracle(...) | Each(...). "
                "See AGENTS.md 'Three API surfaces' for details.",
            )
        super().__init__(**kwargs)
        self._validate_skip_callables()

    def _validate_skip_callables(self) -> None:
        """Check skip_when/skip_value accept at least 1 positional arg."""
        from neograph.errors import ConstructError
        for attr_name in ("skip_when", "skip_value"):
            fn = getattr(self, attr_name, None)
            if fn is None:
                continue
            sig = inspect.signature(fn)
            positional = [
                p for p in sig.parameters.values()
                if p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                )
                and p.default is inspect.Parameter.empty
            ]
            if len(positional) < 1:
                raise ConstructError.build(
                    f"{attr_name} must accept at least 1 positional argument (the input data)",
                    node=self.name,
                    hint=f"define {attr_name} as: lambda data: ...",
                )

    @classmethod
    def scripted(
        cls,
        name: str,
        fn: str,
        inputs: TypeSpec = None,
        outputs: TypeSpec = None,
    ) -> Node:
        """Create a deterministic node — pure Python, no LLM.

        Usage:
            build_catalog = Node.scripted("build-catalog", fn="build_catalog", outputs=str)
        """
        return cls(
            name=name,
            mode="scripted",
            inputs=inputs,
            outputs=outputs,
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

        Note:
            This uses dict-form state internally (not a compiled Pydantic model).
            Modifier-bearing nodes (Each, Loop, Oracle) require state fields
            (neo_each_item, neo_loop_count_*, neo_oracle_*) that run_isolated
            does not populate. Use compile() + run() for modified nodes.

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
        field_name = field_name_for(self.name)
        return result.get(field_name)
