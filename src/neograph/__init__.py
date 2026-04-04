"""NeoGraph — Declarative LLM Graph Compiler.

Define typed Nodes, compose them into Constructs, compile to LangGraph.

    from neograph import Node, Tool, Construct, Oracle, Each, Operator, compile, run
"""

from neograph._llm import configure_llm
from neograph.compiler import compile
from neograph.construct import Construct, ConstructError
from neograph.factory import register_condition, register_scripted, register_tool_factory
from neograph.modifiers import Operator, Oracle, Each
from neograph.node import Node, raw_node
from neograph.runner import run
from neograph.tool import Tool, tool

__all__ = [
    "Node",
    "Tool",
    "Construct",
    "ConstructError",
    "Oracle",
    "Each",
    "Operator",
    "compile",
    "run",
    "raw_node",
    "tool",
    "configure_llm",
    "register_scripted",
    "register_condition",
    "register_tool_factory",
]

__version__ = "0.1.0"
