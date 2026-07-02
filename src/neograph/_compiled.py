"""The typed facade returned by ``compile()``.

``compile()`` used to return LangGraph's ``CompiledStateGraph`` with eight
``_neo_*`` attributes monkey-patched on (each needing ``# type: ignore``), and
the runner/verify layers read them back with ``getattr(graph, "_neo_*", None)``.

``CompiledNeograph`` replaces that stitch-on pattern with a frozen dataclass:
the LangGraph graph plus the framework metadata as typed fields. Execution
methods the runner needs (``invoke`` / ``get_state`` / ``get_state_history`` /
``get_graph`` / ``checkpointer``) are delegated EXPLICITLY — no blanket
``__getattr__`` — so the facade stays a real type boundary and never silently
leaks an untyped LangGraph attribute.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from neograph._llm_runtime import LlmRuntime
    from neograph.construct import Construct


@dataclass(frozen=True)
class CompiledNeograph:
    """Typed result of :func:`neograph.compiler.compile`.

    Holds the compiled LangGraph graph plus the framework metadata that the
    runner and ``verify_compiled`` consume. Replaces the prior pattern of
    stashing ``_neo_*`` attributes on the LangGraph object.
    """

    graph: CompiledStateGraph
    required_di: dict[str, set[str]]
    schema_fingerprint: str
    node_fingerprints: dict[str, str]
    construct: Construct
    runtime: LlmRuntime
    scripted: dict[str, Callable]
    conditions: dict[str, Callable]
    tool_factories: dict[str, Callable]

    # --- Explicit delegation to the wrapped LangGraph graph -----------------
    # Only the methods/attributes the runner actually uses are exposed, so the
    # facade is a closed surface rather than a transparent proxy.

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self.graph.invoke(*args, **kwargs)

    def get_state(self, *args: Any, **kwargs: Any) -> Any:
        return self.graph.get_state(*args, **kwargs)

    def get_state_history(self, *args: Any, **kwargs: Any) -> Any:
        return self.graph.get_state_history(*args, **kwargs)

    # Async delegations (Phase 1d). ainvoke/aget_state are coroutines (awaited);
    # aget_state_history/astream are async GENERATORS returned un-awaited (the
    # caller drives them with `async for`) — do NOT `async def`/await them or the
    # returned object becomes a coroutine yielding a generator (double-wrap).
    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        return await self.graph.ainvoke(*args, **kwargs)

    async def aget_state(self, *args: Any, **kwargs: Any) -> Any:
        return await self.graph.aget_state(*args, **kwargs)

    def aget_state_history(self, *args: Any, **kwargs: Any) -> Any:
        return self.graph.aget_state_history(*args, **kwargs)

    def astream(self, *args: Any, **kwargs: Any) -> Any:
        return self.graph.astream(*args, **kwargs)

    def get_graph(self, *args: Any, **kwargs: Any) -> Any:
        return self.graph.get_graph(*args, **kwargs)

    @property
    def checkpointer(self) -> Any:
        return getattr(self.graph, "checkpointer", None)

    @property
    def builder(self) -> Any:
        """The graph builder — exposes ``state_schema`` for verify_compiled."""
        return getattr(self.graph, "builder", None)
