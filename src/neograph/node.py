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

# ═══════════════════════════════════════════════════════════════════════════
# Node lifecycle Protocols
# ═══════════════════════════════════════════════════════════════════════════
# PEP 696 TypeVar defaults: the input/output types of these Protocols are declared
# elsewhere (node.inputs / node.outputs); defaulting to Any preserves the prior
# un-parameterized call sites without forcing users to subscript at every callsite.
# typing.TypeVar gained `default=` support in Python 3.13; typing_extensions
# backports it to 3.11+. Inputs are contravariant, outputs are covariant —
# matches Callable's variance contract.
# --- extracted clusters (neograph-3ffdg.18), re-exported so existing
# --- `from neograph.node import ...` call sites keep resolving unchanged.
# --- names node.py imported and RE-EXPORTED before the split; the extracted
# --- protocol and type-spec clusters were their only local consumers here.
import types as _types_mod  # noqa: E402,F401
from collections.abc import Callable
from typing import Annotated, Any, Literal, Protocol, cast, runtime_checkable  # noqa: E402,F401

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainValidator,  # noqa: E402,F401
    PrivateAttr,
    field_validator,
)
from typing_extensions import TypeVar  # noqa: E402,F401

from neograph._llm_config import LlmConfig
from neograph._llm_runtime import EMPTY_RUNTIME, LlmRuntime
from neograph._node_protocols import (  # noqa: E402,F401
    HasName,
    RawNodeFn,
    SkipPredicate,
    SkipValueFactory,
    _SkipIn,
    _SkipOut,
)
from neograph._state_keys import StateKeys
from neograph._type_spec import (  # noqa: E402,F401
    TypeSpec,
    TypeSpecStatic,
    _is_type_like,
    _validate_type_spec,
)
from neograph.errors import ConstructError, NeographError
from neograph.modifiers import Modifiable, ModifierSet
from neograph.naming import field_name_for
from neograph.renderers import Renderer
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
    model: str | None = None  # "fast", "reason", "large"
    prompt: str | None = None  # template name in prompt registry
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
        return [Tool.from_base_tool(item) if isinstance(item, BaseTool) else item for item in value]

    # Deterministic implementation (scripted mode only)
    scripted_fn: str | None = None

    # Raw node function — explicit mode='raw' escape hatch only.
    raw_fn: RawNodeFn | None = None

    # Which inputs key receives the Each fan-out item (neo_each_item) instead
    # of reading from the named upstream state field. Set by @node decoration
    # when map_over= is used. Used by factory._extract_input and by the
    # validator to skip upstream-name validation for this key.
    fan_out_param: str | None = None

    # Which inputs key reads the Portal mesh channel (neo_handoff_<entry>)
    # instead of a named upstream state field — the reserved "handoff" key
    # (design §3.3). Written ONLY by the IR normalizer (_ir_normalize.py),
    # keyed off the presence of the reserved "handoff" inputs key on a
    # Portal-modified node — the exact fan_out_param single-writer ownership
    # rule (neograph-k7bg, review H2). No assembly path may write it.
    handoff_param: str | None = None

    # The resolved entry-keyed mesh-channel field name (neo_handoff_<entry_field>)
    # a Portal member reads its `handoff` payload from — the fan_out_param
    # precedent applied to the READ side (decision D10): a node-self-contained IR
    # field so _extract_input resolves the channel WITHOUT threading a key through
    # _execute_node. The channel key is entry-keyed (one mesh per level), so only
    # the construct-level normalizer knows it — hence, like handoff_param, this is
    # written ONLY by _ir_normalize.py (single-writer, review H2 / neograph-k7bg).
    handoff_channel: str | None = None

    # Which state field satisfies this node's SINGLE-TYPE ``inputs=X`` binding
    #. Resolved once at assembly from the declared producers,
    # so the runtime reads a NAME instead of scanning the state bag for a type
    # match, and the Agent Spec export reads the SAME name instead of running a
    # second scan in the opposite direction.
    #
    # Written ONLY by the IR normalizer (_ir_normalize.py) -- the same
    # single-writer ownership as fan_out_param / handoff_param / handoff_channel
    # (neograph-k7bg, review H2). No assembly path may write it.
    #
    # ``None`` means "no single-type source to resolve" -- dict-form inputs, no
    # inputs at all, or no compatible producer and no compatible port. It NEVER
    # means "ambiguous": two eligible producers raise ConstructError at
    # assembly, so ambiguity cannot reach the runtime. A None here must not fall
    # back to a type scan; that would leave every resolved site a silent bypass.
    input_source_field: str | None = None

    #: The port this node's single-type ``inputs=`` reads, spelled ``"member"`` or
    #: ``"member.output"``. USER-DECLARED, the input-side twin of
    #: ``Construct.output_from``.
    #:
    #: Naming is the disambiguator, not the default: declare as you do today and
    #: reach for this when neograph tells you two producers are eligible. It exists
    #: because the two directions were asymmetric -- disambiguating an output meant
    #: adding a name, while disambiguating an input meant REWRITING the declaration
    #: from ``inputs=Claims`` to ``inputs={"settle": Claims}``. Those are different
    #: asks, and an author reading an error does the cheaper one.
    input_from: str | None = None

    # Pluggable prompt-input renderer. When set, the factory layer renders
    # input data through this renderer before prompt insertion. Dispatch
    # hierarchy: model.render_for_prompt() > node.renderer > global > None.
    renderer: Renderer | None = None

    # Values produced EARLIER IN THE RUN that this node reads without them
    # being threaded through the intervening node shapes. The back-reference is
    # DECLARED, so `_construct_validation` fails at assembly when no upstream
    # produces the field -- it is not an ambient global.
    #
    # This is the answer to "my step needs run identity or session context that
    # its input port does not carry". It works inside a fan-out, where the port
    # carries WHICH ITEM and context carries WHICH RUN: the two were never
    # competing. And it reads STATE, not config, so a value that changes during
    # the run -- a session restored after a human-in-the-loop gate fires hours
    # later -- is expressible, where static config cannot express it.
    #
    # Values are passed as-is (not BAML-rendered), which makes it right for a
    # pre-formatted catalog or briefing too. That is a PROPERTY, not the
    # purpose; documenting it as the purpose is what made the capability
    # undiscoverable (GH #15).
    #
    # Limit: LLM-mode nodes only. `_execute` gates it on
    # `node.mode != "scripted"`.
    context: list[str] | None = None

    # Conditional produce: skip the LLM call when the predicate returns True.
    # skip_when receives the extracted input_data (after _extract_input, before
    # renderer). skip_value produces the output when skipped; if None, the node
    # returns an empty state update.
    skip_when: SkipPredicate | None = None
    skip_value: SkipValueFactory | None = None

    # Tool-gating HITL (agent/act only): pause the ReAct cycle for human
    # approval BEFORE the {node}__tools body executes — i.e. before any tool
    # side effect runs. A callable predicate (or registered condition name)
    # receiving the full state; a truthy return is the interrupt payload shown
    # to the human, a falsy return lets tools run without pausing. Mirrors the
    # Operator `when` contract, but targets the synthesized tools boundary the
    # user cannot name. Only meaningful where a {node}__tools node exists, so
    # setting it on a non-agent/act node raises at construction (see below).
    gate_tools_when: Callable | str | None = None

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
    # _remote_agent_endpoint: (agent_class_name, {attr_name: value}) stashed by
    #   _agent_spec_node_import.py's best-effort AgentNode reconstruction so a
    #   future export-side lowering can pick the correct RemoteAgent subclass
    #   and endpoint config to reconstruct. None for every other Node.
    _sidecar: tuple[Callable, tuple[str, ...]] | None = PrivateAttr(default=None)
    _param_res: dict | None = PrivateAttr(default=None)
    _scripted_shim: Callable | None = PrivateAttr(default=None)
    _remote_agent_endpoint: tuple[str, dict[str, Any]] | None = PrivateAttr(default=None)

    # arbitrary_types_allowed: required for the runtime_checkable Protocol
    # fields ``raw_fn``, ``renderer``, ``skip_when``, ``skip_value`` (none of
    # which are Pydantic models) and the ``tools`` list of ``Tool`` runnables.
    model_config = ConfigDict(arbitrary_types_allowed=True)

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
        self._validate_gate_tools_when()

    def _validate_gate_tools_when(self) -> None:
        """gate_tools_when only makes sense where a {node}__tools node exists —
        i.e. agent/act mode. Reject it on any other mode at construction time."""
        if self.gate_tools_when is not None and self.mode not in ("agent", "act"):
            raise ConstructError.build(
                "gate_tools_when requires an agent/act node",
                expected="mode='agent' or mode='act' (a node with a tools boundary)",
                found=f"mode={self.mode!r}",
                hint="Tool-gating pauses before the {node}__tools superstep, which "
                "only exists for agent/act nodes. Remove gate_tools_when or set "
                "mode='agent'/'act'.",
                node=self.name,
            )

    def _validate_skip_callables(self) -> None:
        """Check skip_when/skip_value accept at least 1 positional arg."""
        from neograph.errors import ConstructError

        for attr_name in ("skip_when", "skip_value"):
            fn = getattr(self, attr_name, None)
            if fn is None:
                continue
            sig = inspect.signature(fn)
            positional = [
                p
                for p in sig.parameters.values()
                if p.kind
                in (
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
            from neograph.testing import FakeLLM
            result = classify.run_isolated(
                input=Claims(items=["x"]),
                llm_factory=FakeLLM({"classify": Classified(...)}),
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
            kind
            for kind, mod in (
                ("Each", ms.each),
                ("Oracle", ms.oracle),
                ("Loop", ms.loop),
            )
            if mod is not None
        ]
        if active:
            kinds = "/".join(active)
            raise NeographError.build(
                f"Node '{self.name}' carries modifiers ({kinds}); run_isolated does not support modifier-bearing nodes",
                hint="use compile(construct, ...) + run(graph, ...) instead",
                node=self.name,
            )

        # Agent/act nodes compile to a multi-node inline ReAct cycle (agent/tools/
        # parse + checkpointer for turn-boundary interrupts), so they cannot run as
        # a single isolated node. Same rationale as the modifier restriction above.
        if self.mode in ("agent", "act"):
            raise NeographError.build(
                f"Node '{self.name}' (mode={self.mode}) is a ReAct cycle; "
                "run_isolated does not support agent/act nodes",
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
