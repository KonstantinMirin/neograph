"""Tool — LLM-callable tool with per-tool budget.

Two ways to define a tool:

    # 1. Declarative: Tool class + register_tool_factory
    search = Tool("search_code", budget=5)
    register_tool_factory("search_code", lambda config, tool_config: MySearchTool())

    # 2. Decorator: @tool wraps a function, auto-registers the factory
    @tool(budget=5)
    def search_code(query: str) -> list[str]:
        '''Search the codebase for the given query.'''
        return _do_search(query)

    # In both cases, pass to a Node via tools=[...]
    research = Node("research", mode="agent", tools=[search_code], ...)
"""

from __future__ import annotations

import base64
import re
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

# Module-level so get_type_hints() on the nested resource-reader coroutines can
# resolve the stringified `config: RunnableConfig` annotation (under
# `from __future__ import annotations`) — LangChain's _get_runnable_config_param
# resolves against the function's __globals__, i.e. THIS module's namespace. A
# function-local import would leave the hint unresolved and config injection
# would silently not fire.
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, PrivateAttr

from neograph.di import RESOURCE_FETCHER_KEY, parse_resource_content
from neograph.errors import ConfigurationError


class Tool(BaseModel, frozen=True):
    """A tool the LLM can call, with a per-tool call budget.

    Usage:
        search = Tool("search_nodes", budget=5)
        read   = Tool("read_artifact", budget=10, config={"max_chars": 6000})
    """

    name: str
    budget: int = 0  # max calls for this tool (0 = unlimited)
    config: dict[str, Any] = Field(default_factory=dict)

    # Replay-safety gate; see neograph-lhc6. True only when re-invoking this tool is
    # side-effect-safe -- read-only, or an idempotent mutation (e.g. an HTTP PUT).
    # Default is the CONSERVATIVE non-idempotent: a bare Tool must not be replayed
    # to re-derive an expired resource, because replaying an act-mode producer
    # would double-apply the mutation. Hydration replay (out-of-repo, neograph-a5nh)
    # gates on this and raises NonIdempotentReplayError for a non-idempotent
    # producer; lint warns when an act-mode node's tools are ALL idempotent.
    idempotent: bool = False

    # Set when this spec was synthesized from a raw LangChain BaseTool passed
    # directly in Node(tools=[...]). Carries the original tool so the compile
    # seam can auto-register a factory and lint can introspect it (async-only
    # detection) without instantiating anything. PrivateAttr survives
    # model_copy (pipe/deepcopy), so it round-trips through modifier chains.
    _bound_tool: Any = PrivateAttr(default=None)

    def __init__(self, name_: str | None = None, /, **kwargs):
        """Tool accepts name positionally or as a keyword argument."""
        if name_ is not None:
            kwargs["name"] = name_
        super().__init__(**kwargs)

    @classmethod
    def from_base_tool(cls, base_tool: Any) -> Tool:
        """Synthesize a Tool spec from a raw LangChain BaseTool.

        Name is taken from ``base_tool.name``; the original tool is carried on
        the ``_bound_tool`` private attribute for later factory registration.
        A per-tool budget hint stashed under
        ``base_tool.metadata['ng_tool_budget']`` (e.g. by
        ``resource_reader(budget=...)``) is lifted onto the spec so the existing
        budget/tracker path applies without the user re-wrapping.
        """
        budget = 0
        idempotent = False
        meta = getattr(base_tool, "metadata", None)
        if isinstance(meta, dict):
            budget = meta.get("ng_tool_budget", 0) or 0
            # resource_reader() stashes ng_idempotent=True (read-only by nature);
            # a bare BaseTool carries no hint and stays conservatively non-idempotent.
            idempotent = bool(meta.get("ng_idempotent", False))
        spec = cls(name=base_tool.name, budget=budget, idempotent=idempotent)
        # Tool is frozen=True; set the PrivateAttr via object.__setattr__ (the
        # canonical frozen-model mutation path) so the assignment is honest to
        # both the runtime and the type checker.
        object.__setattr__(spec, "_bound_tool", base_tool)
        return spec


def is_async_only_tool(tool: Any) -> bool:
    """True when a LangChain tool supports only async invocation.

    MCP tools loaded via langchain-mcp-adapters are ``StructuredTool`` instances
    with a coroutine but no sync ``func`` — calling ``.invoke()`` raises
    ``NotImplementedError``. Such a tool requires the async driver (``arun()``);
    a sync ``run()`` cannot execute it. A generic ``BaseTool`` subclass counts
    as async-only when it overrides ``_arun`` but leaves ``_run`` as the base
    (which raises).
    """
    try:
        from langchain_core.tools import BaseTool, StructuredTool
    except ImportError:  # pragma: no cover - langchain always present in practice
        return False

    if isinstance(tool, StructuredTool):
        return tool.coroutine is not None and tool.func is None
    if isinstance(tool, BaseTool):
        cls = type(tool)
        run_overridden = cls._run is not BaseTool._run
        arun_overridden = cls._arun is not BaseTool._arun
        return arun_overridden and not run_overridden
    return False


def register_bound_tool_factories(
    construct: Any, tool_factory_lookup: dict[str, Callable]
) -> None:
    """Auto-register factories for raw BaseTools passed via ``Node(tools=[...])``.

    A ``Tool`` spec synthesized from a raw LangChain BaseTool carries the
    original tool on ``Tool._bound_tool``. Register a factory returning it so
    the ReAct loop can instantiate the tool without the user calling
    ``register_tool_factory``. Explicit ``tool_factories=`` entries win
    (``setdefault``). Called from the compile assembly seam over the same node
    traversal as tool-factory verification (``iter_with_arms``).
    """
    from neograph._ir_branch import iter_with_arms
    from neograph.node import Node

    for item in iter_with_arms(construct):
        if isinstance(item, Node) and item.tools:
            for spec in item.tools:
                bound = getattr(spec, "_bound_tool", None)
                if bound is not None:
                    tool_factory_lookup.setdefault(
                        spec.name,
                        lambda config, tool_config, _t=bound: _t,
                    )


class ToolInteraction(BaseModel, frozen=True):
    """Record of a single tool call during a ReAct loop.

    Collected by the agent cycle (_agent_cycle) and exposed as a secondary output
    when the agent/act node declares it in dict-form outputs.

    ``result`` is the rendered string form (backward compat — existing code
    reads this). ``typed_result`` holds the original object returned by the
    tool (Pydantic model, list, dict, etc.) so downstream nodes receive
    structured data, not repr strings.
    """

    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    typed_result: Any = None
    duration_ms: int = 0


class ProducingCall(BaseModel, frozen=True):
    """The tool call that emitted a resource_link — the replay path for a ref.

    Frozen: a manifest entry's provenance must not mutate. Re-derivation of an
    expired resource is exactly ``(tool_name, args)`` — the only path an MCP
    ``resource_link`` (which carries no lifetime contract) reliably allows.
    """

    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ResourceRef(BaseModel, frozen=True):
    """A typed, self-healing reference to an MCP resource — the OPPOSITE of a
    stringly ``file_id``.

    Lifted from a ``resource_link`` tool-result block (``_agent_cycle.
    _lift_resource_refs``) co-located with the ``ToolInteraction`` it corresponds
    to, and parked in the checkpointed resource-manifest channel (``StateKeys.
    resource_manifest`` — the HITL-surviving tier). It carries its ``producing_
    call`` so an expired ref can be re-derived by replay.

    Read/hydration/expiry (read -> replay -> fail loud) is a SEPARATE downstream
    concern, neograph-a5nh; this model + its lift are the manifest tier only.
    ``ttl_ms`` is reserved for the future SEP-2549 layered-expiry layer.
    """

    uri: str                       # the resource_link uri (server-defined stability)
    kind: str                      # domain KIND: "email-history", "activity-history"
    server: str                    # which MCP server (for the consumer's fetcher routing)
    producing_call: ProducingCall  # THE re-derivation path
    mime: str | None = None        # hint from the resource_link block
    size: int | None = None        # hint from the resource_link block
    fetched_at: str | None = None  # ISO ts when last hydrated (provenance; set at hydration)
    ttl_ms: int | None = None      # future SEP-2549 layered expiry


class ToolBudgetTracker:
    """Tracks per-tool call counts and enforces budgets at runtime.

    Created by the node factory for gather/execute mode nodes.
    """

    def __init__(self, tools: list[Tool]) -> None:
        self._budgets: dict[str, int] = {}
        self._counts: dict[str, int] = {}
        for tool in tools:
            self._budgets[tool.name] = tool.budget
            self._counts[tool.name] = 0

    def can_call(self, tool_name: str) -> bool:
        """Check if a tool still has budget remaining."""
        budget = self._budgets.get(tool_name, 0)
        if budget == 0:
            return True  # unlimited
        return self._counts.get(tool_name, 0) < budget

    def record_call(self, tool_name: str) -> None:
        """Record a tool call."""
        self._counts[tool_name] = self._counts.get(tool_name, 0) + 1

    def exhausted_tools(self) -> set[str]:
        """Return names of tools that have hit their budget."""
        return {
            name
            for name, budget in self._budgets.items()
            if budget > 0 and self._counts.get(name, 0) >= budget
        }

    def all_exhausted(self) -> bool:
        """True if every budgeted tool is spent. Unlimited tools (budget=0) never exhaust."""
        if not self._budgets:
            return False  # no tools → nothing to exhaust
        for name, budget in self._budgets.items():
            if budget == 0:
                return False
            if self._counts.get(name, 0) < budget:
                return False
        return True


def tool(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    budget: int = 0,
    config: dict[str, Any] | None = None,
) -> Any:
    """Decorator that turns a function into a Tool and auto-registers its factory.

    The tool's name defaults to the function name. The description comes from
    the docstring. Arguments are introspected from the function signature.

    Usage:

        @tool(budget=5)
        def search_code(query: str) -> list[str]:
            '''Search the codebase for the given query.'''
            return _do_search(query)

        research = Node("research", mode="agent", tools=[search_code], ...)

    The decorated function IS a Tool instance (with the original function
    attached as .fn), so it can be passed directly to Node's tools= list.
    The factory is auto-registered under the tool's name.
    """
    def decorator(f: Callable) -> Tool:
        from neograph._runtime_registry import register_tool_factory

        tool_name = name or f.__name__

        # Build a LangChain-compatible tool from the function
        from langchain_core.tools import tool as lc_tool
        lc_tool_instance = lc_tool(f)

        # Register the factory so the ReAct loop can instantiate it
        register_tool_factory(tool_name, lambda config, tool_config: lc_tool_instance)

        # Return a Tool spec that Node accepts in tools=[...]
        return Tool(tool_name, budget=budget, config=config or {})

    # Support both @tool and @tool(budget=5)
    if fn is not None:
        return decorator(fn)
    return decorator


# ── Typed resource readers — neograph-2dtk ─────────────────────────────────────
#
# resource_reader() turns (uri template + output model + name) into a properly
# TYPED async BaseTool — the antidote to the untyped read_resource(uri)->bytes
# trap. It emits a StructuredTool with a coroutine and NO func (async-only, so
# is_async_only_tool()/the tool_requires_async_driver lint fire for free), plugs
# into Node(tools=[...]) unchanged (register_bound_tool_factories auto-registers
# it), and its return flows through ToolInteraction.typed_result as the parsed
# model. ZERO new tool infrastructure.


class BlobResult(BaseModel, frozen=True):
    """Typed escape hatch for genuinely opaque resource content — the exception,
    not the default. Prefer a typed ``resource_reader`` unless the content has no
    schema. Keeps ``typed_result`` honest (a ``BlobResult``, not raw bytes)."""

    uri: str
    mime: str | None = None
    text: str | None = None
    bytes_b64: str | None = None
    size: int | None = None


# RFC 6570 (subset): {var}, {+var}, {?a,b}, {&a}. group 1 = operator, group 2 =
# comma-separated var list; a trailing '*' explode modifier is stripped.
_URI_VAR_RE = re.compile(r"\{([+#./;?&]?)([^{}]+)\}")


def _extract_uri_vars(uri_template: str) -> list[str]:
    """Ordered, de-duplicated variable names from an RFC 6570 uri template."""
    names: list[str] = []
    for _op, body in _URI_VAR_RE.findall(uri_template):
        for raw in body.split(","):
            nm = raw.strip().rstrip("*")
            if nm and nm not in names:
                names.append(nm)
    return names


def _expand_uri(uri_template: str, values: dict[str, Any]) -> str:
    """Interpolate an RFC 6570 (subset) uri template from ``values``.

    Supports simple ``{var}`` / reserved ``{+var}`` string expansion and
    form-query ``{?a,b}`` / ``{&a}`` expansion — enough for the static and
    templated resource URIs v1 emits. Missing values are omitted.
    """
    def _sub(match: re.Match[str]) -> str:
        op, body = match.group(1), match.group(2)
        varnames = [v.strip().rstrip("*") for v in body.split(",")]
        if op in ("?", "&"):
            pairs = [
                f"{vn}={quote(str(values[vn]), safe='')}"
                for vn in varnames
                if values.get(vn) is not None
            ]
            return (op + "&".join(pairs)) if pairs else ""
        out = [
            str(values[vn]) if op == "+" else quote(str(values[vn]), safe="")
            for vn in varnames
            if values.get(vn) is not None
        ]
        return ",".join(out)

    return _URI_VAR_RE.sub(_sub, uri_template)


def _resolve_fetcher(config: Any, tool_name: str) -> Callable:
    """Read the consumer-owned async resource fetcher from config; fail loud."""
    cfg = (config or {}).get("configurable", {}) or {}
    fetcher = cfg.get(RESOURCE_FETCHER_KEY)
    if fetcher is None:
        raise ConfigurationError.build(
            f"resource tool '{tool_name}' has no resource fetcher to call",
            hint=f"provide config['configurable']['{RESOURCE_FETCHER_KEY}'] = "
                 "an async 'fetch(uri) -> (content, mime)' callable",
        )
    return fetcher


def resource_reader(
    name: str,
    *,
    uri_template: str,
    output_model: type[BaseModel],
    description: str,
    parse: Callable[[Any, str | None], BaseModel] | None = None,
    budget: int = 0,
    idempotent: bool = True,
) -> Any:
    """Emit a typed, async LangChain BaseTool that reads ONE known resource KIND.

    The tool's args schema is derived from ``uri_template``'s RFC 6570 vars, so
    the LLM (agent mode) or the caller (scripted/DI) supplies typed parameters.
    At call time it resolves the fetcher from
    ``config['configurable']['mcp_resource_fetcher']`` (consumer-owned, async),
    reads the interpolated URI, and parses the blob into ``output_model`` (via
    ``parse`` if given, else ``application/json`` -> ``model_validate_json``).
    Returns the typed model instance, so ``ToolInteraction.typed_result`` carries
    the Pydantic model, not a repr string.

    ``idempotent`` (default True) marks the reader replay-safe -- readers are
    read-only by nature, so re-invoking one to re-derive an expired resource does
    not double-apply a side effect. Lifted onto the ``Tool`` spec via
    ``Tool.from_base_tool`` (metadata key ``ng_idempotent``).
    """
    from langchain_core.tools import StructuredTool
    from pydantic import create_model

    var_names = _extract_uri_vars(uri_template)
    field_defs: dict[str, Any] = dict.fromkeys(var_names, (str, ...))
    args_schema = create_model(f"{name}_Args", **field_defs)

    async def _read(config: RunnableConfig, **kwargs: Any) -> Any:
        uri = _expand_uri(uri_template, kwargs)
        fetcher = _resolve_fetcher(config, name)
        content, mime = await fetcher(uri)
        return parse_resource_content(content, mime, output_model, parse)

    metadata: dict[str, Any] = {"ng_idempotent": idempotent}
    if budget:
        metadata["ng_tool_budget"] = budget

    return StructuredTool(
        name=name,
        description=description,
        args_schema=args_schema,
        coroutine=_read,
        func=None,
        metadata=metadata,
    )


def _build_read_blob() -> Any:
    """Construct the singleton ``read_blob`` async tool (the escape hatch)."""
    from langchain_core.tools import StructuredTool
    from pydantic import create_model

    args_schema = create_model("read_blob_Args", uri=(str, ...))

    async def _read(uri: str, config: RunnableConfig) -> BlobResult:
        fetcher = _resolve_fetcher(config, "read_blob")
        content, mime = await fetcher(uri)
        if isinstance(content, bytes):
            return BlobResult(
                uri=uri, mime=mime,
                bytes_b64=base64.b64encode(content).decode("ascii"),
                size=len(content),
            )
        return BlobResult(uri=uri, mime=mime, text=content, size=len(content))

    return StructuredTool(
        name="read_blob",
        description=(
            "Read an opaque MCP resource as a typed BlobResult (bytes/text + mime "
            "+ size). Use a typed resource_reader unless the content has no schema."
        ),
        args_schema=args_schema,
        coroutine=_read,
        func=None,
    )


# The escape hatch is a ready-to-use async-only tool: Node(tools=[read_blob]).
read_blob = _build_read_blob()

