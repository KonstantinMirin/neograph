"""NeoGraph — Declarative LLM Graph Compiler.

Define typed Nodes, compose them into Constructs, compile to LangGraph.

    from neograph import Node, Tool, Construct, Oracle, Replicate, Operator, compile, run
"""

from neograph.compiler import compile
from neograph.construct import Construct
from neograph.modifiers import Operator, Oracle, Replicate
from neograph.node import Node, raw_node
from neograph.runner import run
from neograph.tool import Tool

__all__ = [
    "Node",
    "Tool",
    "Construct",
    "Oracle",
    "Replicate",
    "Operator",
    "compile",
    "run",
    "raw_node",
]

__version__ = "0.1.0"
