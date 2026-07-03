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
from typing import Annotated, Any, Literal, Protocol, cast, runtime_checkable

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PlainValidator, PrivateAttr, field_validator
from typing_extensions import TypeVar

from neograph._llm_config import LlmConfig
from neograph._llm_runtime import EMPTY_RUNTIME, LlmRuntime
from neograph._state_keys import StateKeys
from neograph.errors import ConstructError, NeographError
from neograph.renderers import Renderer

# ═══════════════════════════════════════════════════════════════════════════
# Node lifecycle Protocols
# ═══════════════════════════════════════════════════════════════════════════

# PEP 696 TypeVar defaults: the input/output types of these Protocols are declared
# elsewhere (node.inputs / node.outputs); defaulting to Any preserves the prior
# un-parameterized call sites without forcing users to subscript at every callsite.
# typing.TypeVar gained `default=` support in Python 3.13; typing_extensions
# backports it to 3.11+. Inputs are contravariant, outputs are covariant —
# matches Callable's variance contract.
_SkipIn = TypeVar("_SkipIn", contravariant=True, default=Any)
_SkipOut = TypeVar("_SkipOut", covariant=True, default=Any)


@runtime_checkable
class SkipPredicate(Protocol[_SkipIn]):
    """Returns True to bypass the LLM call. Receives extracted ``input_data``
    (after ``_extract_input``, before renderer dispatch).
    """

    def __call__(self, input_data: _SkipIn) -> bool: ...


@runtime_checkable
class SkipValueFactory(Protocol[_SkipIn, _SkipOut]):
    """Produces the output value when ``skip_when`` fires. Receives the same
    ``input_data`` shape as ``skip_when``. If absent, the node returns an
    empty state update.
    """

    def __call__(self, input_data: _SkipIn) -> _SkipOut: ...


@runtime_checkable
class RawNodeFn(Protocol):
    """Raw escape hatch for ``mode='raw'``. Direct LangGraph node signature."""

    def __call__(self, state: BaseModel, config: RunnableConfig) -> dict[str, Any]: ...


@runtime_checkable
class HasName(Protocol):
    """Anything that carries a user-facing declaration name.

    Both ``Node`` and ``Construct`` satisfy this structurally (each declares
    ``name: str``). Redirect-closure factories (`_oracle.py`) capture a
    ``HasName`` and read ``.name`` for error/observability labels — the label
    concern is sourced from the IR object (Information Expert), never threaded
    as a string kwarg nor scraped from a wrapper's ``__name__``.
    """

    name: str


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

# Static-annotation alias for user-declared type values flowing through the
# framework. Distinct from the Pydantic-validator-bearing TypeSpec field type,
# which carries _validate_type_spec on top of the same union. Use this on
# parameter and return annotations of helpers that introspect user-declared
# types (closes the PEP 747 gap until TypeForm lands). See
# docs/design/architecture-decisions.md §5 and §8.
import types as _types_mod

TypeSpecStatic = type | dict[str, type] | _types_mod.GenericAlias | _types_mod.UnionType | None

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
    llm_config: LlmConfig = Field(default_factory=LlmConfig)  # framework knobs + provider_kwargs (typed)

    # Tools with per-tool budgets. A raw LangChain BaseTool may be passed
    # directly; the validator below normalizes it to a Tool spec (name from
    # tool.name) carrying the tool on Tool._bound_tool. The compile seam then
    # auto-registers a factory — no register_tool_factory boilerplate needed.
    tools: list[Tool | BaseTool] = []

    @field_validator("tools", mode="before")
    @classmethod
    def _normalize_raw_base_tools(cls, value: Any) -> Any:
        """Convert any raw LangChain BaseTool in tools= to a Tool spec.

        Pure normalization (no registration side effect — that lives at the
        compile assembly seam). Runs before pydantic union validation so a
        StructuredTool is never coerced into a Tool by field-shape matching.
        """
        if not isinstance(value, list):
            return value
        return [
            Tool.from_base_tool(item) if isinstance(item, BaseTool) else item
            for item in value
        ]

    # Deterministic implementation (scripted mode only)
    scripted_fn: str | None = None

    # Raw node function — explicit mode='raw' escape hatch only.
    raw_fn: RawNodeFn | None = None

    # Which inputs key receives the Each fan-out item (neo_each_item) instead
    # of reading from the named upstream state field. Set by @node decoration
    # when map_over= is used. Used by factory._extract_input and by the
    # validator to skip upstream-name validation for this key.
    fan_out_param: str | None = None

    # Pluggable prompt-input renderer. When set, the factory layer renders
    # input data through this renderer before prompt insertion. Dispatch
    # hierarchy: model.render_for_prompt() > node.renderer > global > None.
    renderer: Renderer | None = None

    # Verbatim state fields injected into the prompt alongside typed input.
    # Values are passed as-is (not BAML-rendered). Use for pre-formatted
    # context like graph catalogs or domain briefings.
    context: list[str] | None = None

    # Conditional produce: skip the LLM call when the predicate returns True.
    # skip_when receives the extracted input_data (after _extract_input, before
    # renderer). skip_value produces the output when skipped; if None, the node
    # returns an empty state update.
    skip_when: SkipPredicate | None = None
    skip_value: SkipValueFactory | None = None

    # Oracle generator output type — when merge_fn transforms types (A → B),
    # this is A (per-variant type). The LLM produces this type, the merge_fn
    # converts list[A] → B (= node.outputs). Inferred from merge_fn signature.
    oracle_gen_type: type[BaseModel] | None = None

    # Modifiers applied via | operator (typed slots, not a list)
    modifier_set: ModifierSet = Field(default_factory=ModifierSet)

    # Sidecar metadata — lives on the Node via PrivateAttr, not in global dicts.
    # Preserved by model_copy (Pydantic v2 copies __pydantic_private__).
    # _sidecar: (original_fn, param_names_tuple) from @node decoration.
    # _param_res: DI bindings from _classify_di_params.
    # _scripted_shim: the closure built at construct-build time. compile()
    #   reads it and inserts the entry into the per-compile scripted dict.
    _sidecar: tuple[Callable, tuple[str, ...]] | None = PrivateAttr(default=None)
    _param_res: dict | None = PrivateAttr(default=None)
    _scripted_shim: Callable | None = PrivateAttr(default=None)

    # arbitrary_types_allowed: required for the runtime_checkable Protocol
    # fields ``raw_fn``, ``renderer``, ``skip_when``, ``skip_value`` (none of
    # which are Pydantic models) and the ``tools`` list of ``Tool`` runnables.
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
        llm_factory: Callable | None = None,
        prompt_compiler: Callable | None = None,
        scripted: dict[str, Callable] | None = None,
        conditions: dict[str, Callable] | None = None,
        tool_factories: dict[str, Callable] | None = None,
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

            # Unit test a produce node
            result = classify.run_isolated(
                input=Claims(items=["x"]),
                llm_factory=lambda tier: FakeLLM(),
                prompt_compiler=lambda t, d, **kw: [{"role": "user", "content": "x"}],
            )
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

        # Modifier-bearing nodes need state fields (neo_each_item,
        # neo_loop_count_*, neo_oracle_*) that run_isolated does not populate.
        # Refuse at entry with a clear message rather than silently returning
        # None from an unpopulated state field downstream.
        ms = self.modifier_set
        active = [
            kind for kind, mod in (
                ("Each", ms.each),
                ("Oracle", ms.oracle),
                ("Loop", ms.loop),
            )
            if mod is not None
        ]
        if active:
            kinds = "/".join(active)
            raise NeographError.build(
                f"Node '{self.name}' carries modifiers ({kinds}); "
                "run_isolated does not support modifier-bearing nodes",
                hint="use compile(construct, ...) + run(graph, ...) instead",
                node=self.name,
            )

        # Fail-loud check (§2): LLM-mode nodes require llm_factory +
        # prompt_compiler kwargs.
        if self.mode in ("think", "agent", "act"):
            need_factory = llm_factory is None
            need_compiler = prompt_compiler is None
            if need_factory or need_compiler:
                missing = []
                if need_factory:
                    missing.append("llm_factory")
                if need_compiler:
                    missing.append("prompt_compiler")
                raise NeographError.build(
                    f"Node '{self.name}' (mode={self.mode}) requires runtime configuration",
                    expected="llm_factory= and prompt_compiler= passed to run_isolated()",
                    found=f"{' and '.join(missing)} not set",
                    hint=f"Pass llm_factory= and prompt_compiler= to {self.name}.run_isolated().",
                    node=self.name,
                )

        if llm_factory is not None or prompt_compiler is not None:
            runtime = LlmRuntime.build(
                llm_factory=llm_factory,
                prompt_compiler=prompt_compiler,
            )
        else:
            runtime = EMPTY_RUNTIME

        # Collect a scripted_lookup for the node. Start with this Node's
        # own `_scripted_shim` (if any), merge in caller-supplied `scripted=`,
        # then merge in decoration-time shims (for @merge_fn / @tool /
        # interrupt_when shims registered at decoration time in the
        # _runtime_registry leaf).
        from neograph._runtime_registry import _decoration_registry
        scripted_lookup: dict[str, Callable] = {}
        own_shim = getattr(self, "_scripted_shim", None)
        if own_shim is not None and self.scripted_fn:
            scripted_lookup[self.scripted_fn] = own_shim
        scripted_lookup.update(_decoration_registry.scripted)
        if scripted:
            scripted_lookup.update(scripted)

        tool_factory_lookup: dict[str, Callable] = dict(_decoration_registry.tool_factory)
        if tool_factories:
            tool_factory_lookup.update(tool_factories)

        node_fn = make_node_fn(
            self,
            runtime=runtime,
            scripted_lookup=scripted_lookup,
            tool_factory_lookup=tool_factory_lookup,
        )

        # Build a minimal state dict the node function can read
        state: dict[str, Any] = {}
        if isinstance(input, dict):
            state.update(input)
        elif input is not None:
            # Typed instance — place it under the node name so _extract_input finds it by type
            state[StateKeys.ISOLATED_INPUT] = input

        config = config or {"configurable": {}}
        if "configurable" not in config:
            config["configurable"] = {}

        result = node_fn.invoke(state, cast(RunnableConfig, config))

        # node_fn returns a state update dict — extract the output field.
        # If the field is missing or None, raise rather than silently returning
        # None: run_isolated is a testing/inspection tool, and a silent None
        # masks the underlying cause (body returned None, return-type annotation
        # mismatch, dict-form output missing the primary key).
        field_name = field_name_for(self.name)
        if field_name not in result or result[field_name] is None:
            raise NeographError.build(
                f"Node '{self.name}' did not produce output field '{field_name}'",
                hint=(
                    "the node body returned None or the @node return-type "
                    "annotation does not match the actual return; "
                    "for dict-form outputs, the primary key must be populated"
                ),
                node=self.name,
            )
        return result[field_name]
