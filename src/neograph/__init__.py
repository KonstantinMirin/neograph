"""NeoGraph — Declarative LLM Graph Compiler.

Declare pipeline nodes with @node and assemble from function signatures:

    from neograph import node, construct_from_module, compile, run

    @node(outputs=Claims, prompt='rw/decompose', model='reason')
    def decompose(topic: RawText) -> Claims: ...

    @node(outputs=Classified, prompt='rw/classify', model='fast')
    def classify(decompose: Claims) -> Classified: ...

    pipeline = construct_from_module(sys.modules[__name__])
    graph = compile(pipeline)
    result = run(graph, input={'node_id': 'doc-001'})

For advanced use (IR-level tests, programmatic construction from config,
sub-constructs), see Node and Construct directly.
"""

from neograph._llm import configure_llm, render_prompt
from neograph.compiler import compile, describe_graph
from neograph.conditions import parse_condition
from neograph.construct import Construct
from neograph.decorators import (
    FromConfig,
    FromInput,
    construct_from_functions,
    construct_from_module,
    merge_fn,
    node,
)
from neograph.describe_type import ExcludeFromOutput, describe_type, describe_value
from neograph.errors import (
    CheckpointSchemaError,
    CompileError,
    ConfigurationError,
    ConstructError,
    ExecutionError,
    NeographError,
)
from neograph.factory import register_condition, register_scripted, register_tool_factory
from neograph.forward import ForwardConstruct
from neograph.lint import LintIssue, lint
from neograph.loader import load_spec
from neograph.modifiers import Each, Loop, ModifierSet, Operator, Oracle
from neograph.node import Node
from neograph.renderers import (
    DelimitedRenderer,
    JsonRenderer,
    Renderer,
    XmlRenderer,
    render_input,
)
from neograph.runner import run
from neograph.spec_types import lookup_type, register_type
from neograph.tool import Tool, ToolInteraction, tool

__all__ = [
    # Primary API — @node decorator + module assembly
    "node",
    "construct_from_module",
    "construct_from_functions",
    "FromInput",
    "FromConfig",
    "compile",
    "run",
    "tool",
    "merge_fn",
    # Modifiers (used as @node kwargs; also available standalone)
    "Oracle",
    "Each",
    "Loop",
    "Operator",
    # Low-level IR (advanced use: programmatic construction, IR tests)
    "Node",
    "Tool",
    "Construct",
    "ForwardConstruct",
    # Error hierarchy
    "NeographError",
    "ConstructError",
    "CompileError",
    "ConfigurationError",
    "ExecutionError",
    # Configuration
    "configure_llm",
    # Schema rendering
    "describe_type",
    "describe_value",
    "ExcludeFromOutput",
    # Prompt inspection
    "render_prompt",
    # Renderers
    "Renderer",
    "XmlRenderer",
    "DelimitedRenderer",
    "JsonRenderer",
    "render_input",
    "register_scripted",
    "register_condition",
    "register_tool_factory",
    # Type registry (spec-based lookup)
    "register_type",
    "lookup_type",
    # Spec loader
    "load_spec",
    # Lint
    "lint",
    "LintIssue",
]

__version__ = "0.4.0"
