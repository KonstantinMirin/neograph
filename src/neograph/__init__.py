"""neograph — Declarative LLM Graph Compiler.

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

from neograph._image import configure_image, resolve_image
from neograph._llm import CostCallback, LlmFactory, PromptCompiler, render_prompt
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
from neograph.describe_type import (
    ExcludeFromOutput,
    describe_type,
    describe_value,
    type_display_name,
)
from neograph.errors import (
    CheckpointSchemaError,
    CompileError,
    ConfigurationError,
    ConstructError,
    ExecutionError,
    NeographError,
)
from neograph.forward import ForwardConstruct
from neograph.hitl import ask_human
from neograph.lint import LintIssue, lint
from neograph.loader import load_spec
from neograph.modifiers import (
    Each,
    EachFailure,
    Loop,
    MergeFallback,
    MergePostProcess,
    MergePreProcess,
    ModifierSet,
    Operator,
    Oracle,
)
from neograph.node import Node, RawNodeFn, SkipPredicate, SkipValueFactory, TypeSpecStatic
from neograph.progress import emit_progress
from neograph.renderers import (
    DelimitedRenderer,
    JsonRenderer,
    Renderer,
    XmlRenderer,
    render_input,
)
from neograph.runner import arun, astream, run, stream
from neograph.spec_types import lookup_type, register_type
from neograph.tool import Tool, ToolInteraction, tool
from neograph.verify import VerifyIssue, verify_compiled

__all__ = [
    # Primary API — @node decorator + module assembly
    "node",
    "construct_from_module",
    "construct_from_functions",
    "FromInput",
    "FromConfig",
    "compile",
    "run",
    "arun",
    "stream",
    "astream",
    "emit_progress",
    "tool",
    "merge_fn",
    # Human-in-the-loop sugar (agent/act tool bodies)
    "ask_human",
    # Modifiers (used as @node kwargs; also available standalone)
    "Oracle",
    "Each",
    "EachFailure",
    "Loop",
    "Operator",
    # Low-level IR (advanced use: programmatic construction, IR tests)
    "Node",
    "Tool",
    "Construct",
    "ForwardConstruct",
    "TypeSpecStatic",
    # Error hierarchy
    "NeographError",
    "ConstructError",
    "CompileError",
    "ConfigurationError",
    "ExecutionError",
    # Schema rendering
    "describe_type",
    "describe_value",
    "type_display_name",
    "ExcludeFromOutput",
    # Prompt inspection
    "render_prompt",
    # Multimodal utilities
    "configure_image",
    "resolve_image",
    # Renderers
    "Renderer",
    "XmlRenderer",
    "DelimitedRenderer",
    "JsonRenderer",
    "render_input",
    # Type registry (spec-based lookup)
    "register_type",
    "lookup_type",
    # Spec loader
    "load_spec",
    # Lint
    "lint",
    "LintIssue",
    # Verify
    "verify_compiled",
    "VerifyIssue",
    # Callback Protocols (typed user-supplied callback contracts)
    "LlmFactory",
    "PromptCompiler",
    "CostCallback",
    "MergePreProcess",
    "MergePostProcess",
    "MergeFallback",
    "SkipPredicate",
    "SkipValueFactory",
    "RawNodeFn",
]

__version__ = "0.5.0"
