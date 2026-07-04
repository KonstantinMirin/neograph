"""Structural guards: IR typing, compiler wiring, node mutation, branch nodes,
build-construct body size, registry dicts."""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

# Error classes that must use .build() instead of direct construction.
ERROR_CLASSES = frozenset({
    "ConstructError",
    "ExecutionError",
    "CompileError",
    "ConfigurationError",
    "NeographError",
})


class TestNoFrameDepthParam:
    """Frame-stack walking must be replaced with explicit namespace passing.

    The frame_depth parameter forces fragile arithmetic (frame_depth=2,
    +1 for nesting). Explicit caller_ns dict is stable regardless of
    call depth.
    """

    def test_no_frame_depth_in_source(self):
        """No function should accept or pass frame_depth."""
        violations = []
        for py_file in sorted(SRC_DIR.glob("*.py")):
            text = py_file.read_text()
            for i, line in enumerate(text.splitlines(), 1):
                stripped = line.lstrip()
                if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                    continue
                if "frame_depth" in line:
                    violations.append(f"  {py_file.name}:{i}: {stripped[:80]}")
        assert violations == [], (
            f"\n{len(violations)} frame_depth reference(s) remain:\n"
            + "\n".join(violations)
            + "\n\nReplace with explicit caller_ns parameter."
        )


class TestNoTicketIdsInComments:
    """Ticket IDs (neograph-xxxx) belong in git blame, not in source comments.

    Parenthesized ticket refs like (neograph-26ih) clutter code for zero value
    — the commit message already has the context. This guard prevents them
    from creeping back after cleanup.
    """

    # Named so the regex carries a slip meta-test (PROC-2). The id body is
    # {3,5} chars; the boundary is where a naiver regex slips.
    _TICKET_ID_RE = re.compile(r"\(neograph-[a-z0-9]{3,5}\)")

    def test_no_parenthesized_ticket_ids(self):
        """No (neograph-xxxx) patterns in source code."""
        violations = []
        for py_file in sorted(SRC_DIR.glob("*.py")):
            for i, line in enumerate(py_file.read_text().splitlines(), 1):
                if self._TICKET_ID_RE.search(line):
                    stripped = line.lstrip()[:80]
                    violations.append(f"  {py_file.name}:{i}: {stripped}")
        assert violations == [], (
            f"\n{len(violations)} ticket ID reference(s) in source:\n"
            + "\n".join(violations[:20])
            + ("\n  ..." if len(violations) > 20 else "")
            + "\n\nRemove the (neograph-xxxx) part. Keep the comment text."
        )

    def test_slip_ticket_id_re(self):
        """Regex-slip: the {3,5} id body and the required parentheses are the
        boundaries where a regression slips. Prove the live forms match and the
        near-misses do not."""
        # Must MATCH: real parenthesized ids (3-5 char bodies).
        assert self._TICKET_ID_RE.search("see (neograph-abc) here")
        assert self._TICKET_ID_RE.search("fixed (neograph-26ih)")
        # Must NOT match: no parens (bare id in prose is allowed) ...
        assert not self._TICKET_ID_RE.search("see neograph-26ih here")
        # ... and an over-long body outside {3,5} (e.g. a 6+ char tail) at the
        # paren boundary: the closing paren cannot follow a 6-char body directly.
        assert not self._TICKET_ID_RE.search("(neograph-abcdef)")


class TestNodeIRTyping:
    """Node.inputs and Node.outputs must not be typed as bare Any.

    Any at the IR core defeats the typed compiler claim. 22 isinstance checks
    across the codebase branch on the shape — a Union annotation makes
    the shapes explicit at the type level.
    """

    def test_node_inputs_not_any(self):
        """Node.inputs field annotation must not be bare Any."""
        node_file = SRC_DIR / "node.py"
        tree = ast.parse(node_file.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name != "Node":
                continue
            for item in node.body:
                if not isinstance(item, ast.AnnAssign):
                    continue
                if item.target is None or not isinstance(item.target, ast.Name):
                    continue
                if item.target.id == "inputs":
                    ann = item.annotation
                    assert not (isinstance(ann, ast.Name) and ann.id == "Any"), (
                        f"Node.inputs is typed as bare Any (line {item.lineno}). "
                        "Use: inputs: type[BaseModel] | dict[str, type] | None = None"
                    )

    def test_node_outputs_not_any(self):
        """Node.outputs field annotation must not be bare Any."""
        node_file = SRC_DIR / "node.py"
        tree = ast.parse(node_file.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name != "Node":
                continue
            for item in node.body:
                if not isinstance(item, ast.AnnAssign):
                    continue
                if item.target is None or not isinstance(item.target, ast.Name):
                    continue
                if item.target.id == "outputs":
                    ann = item.annotation
                    assert not (isinstance(ann, ast.Name) and ann.id == "Any"), (
                        f"Node.outputs is typed as bare Any (line {item.lineno}). "
                        "Use: outputs: type[BaseModel] | dict[str, type] | None = None"
                    )


class TestVersionSync:
    """__init__.py __version__ must match pyproject.toml version."""

    def test_version_matches_pyproject(self):
        import tomllib
        pyproject = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        pyproject_version = data["project"]["version"]

        import neograph
        assert neograph.__version__ == pyproject_version, (
            f"__init__.py __version__={neograph.__version__!r} != "
            f"pyproject.toml version={pyproject_version!r}. Keep them in sync."
        )


class TestFieldNameContract:
    """The hyphen-to-underscore naming contract must use a central utility.

    36+ copies of name.replace('-', '_') is a typo waiting to happen.
    Once field_name_for() exists, this guard enforces its use.
    """

    def test_no_inline_hyphen_replace(self):
        """Source code should use field_name_for(), not inline .replace('-', '_')."""
        violations = []
        for py_file in sorted(SRC_DIR.glob("*.py")):
            text = py_file.read_text()
            for i, line in enumerate(text.splitlines(), 1):
                stripped = line.lstrip()
                if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                    continue
                if '.replace("-", "_")' in line or ".replace('-', '_')" in line:
                    # Allow the definition of field_name_for itself
                    if "def field_name_for" in line:
                        continue
                    violations.append(f"  {py_file.name}:{i}: {stripped[:80]}")
        assert violations == [], (
            f"\n{len(violations)} inline .replace('-', '_') call(s):\n"
            + "\n".join(violations[:20])
            + ("\n  ..." if len(violations) > 20 else "")
            + "\n\nUse field_name_for(name) from neograph.naming instead."
        )


class TestOracleAutoMergeWarning:
    """Oracle with models= but no merge_fn should warn about body dual-purpose."""

    def test_body_merge_emits_warning(self):
        """@node with models= and no merge_fn/merge_prompt emits UserWarning."""
        import warnings

        from neograph import node
        from tests.schemas import Claims, RawText

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @node(
                outputs=Claims,
                model="fast",
                prompt="test",
                models=["fast", "reason"],
            )
            def dual_purpose(topic: RawText) -> Claims:
                return Claims(items=["merged"])

            body_warnings = [
                x for x in w
                if "body used as both generator and merge" in str(x.message)
            ]
            assert len(body_warnings) >= 1, (
                "@node with models= but no merge_fn should emit UserWarning "
                "about body serving dual purpose (generator + merge)."
            )


class TestConstructFromModuleDetectsPlainNodes:
    """construct_from_module should detect plain Node() instances, not just @node."""

    def test_plain_node_instance_detected(self):
        """Module-scope Node(...) should be picked up by construct_from_module."""
        import types as _types

        from neograph import Node, construct_from_module, node
        from tests.schemas import Claims, RawText

        mod = _types.ModuleType("test_plain_node_mod")

        @node
        def source() -> RawText:
            return RawText(text="hi")

        plain = Node.scripted("transform", fn="identity", inputs=RawText, outputs=Claims)

        mod.source = source
        mod.plain = plain

        pipeline = construct_from_module(mod)
        node_names = {n.name for n in pipeline.nodes}
        assert "transform" in node_names, (
            "construct_from_module should detect plain Node() instances. "
            f"Found nodes: {node_names}"
        )


class TestExtractInputDispatch:
    """_extract_input must be a pure dispatch function (neograph-b9pm).

    The main function body should be < 30 lines — all logic lives in
    named helper functions dispatched via InputShape classification.
    """

    def test_extract_input_body_is_short(self):
        """_extract_input body must be < 30 lines (pure dispatch)."""
        source = (SRC_DIR / "_input_shape.py").read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_extract_input":
                body_lines = node.end_lineno - node.lineno
                assert body_lines < 30, (
                    f"_extract_input is {body_lines} lines — should be < 30 "
                    f"(pure dispatch to named helpers, not inline logic)."
                )
                return

        pytest.fail("_extract_input function not found in _input_shape.py")

    def test_input_shape_enum_exists(self):
        """InputShape enum must exist for exhaustive dispatch."""
        source = (SRC_DIR / "_input_shape.py").read_text()
        tree = ast.parse(source)
        class_names = {
            n.name for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef)
        }
        assert "InputShape" in class_names, (
            "InputShape enum must exist in _input_shape.py for exhaustive "
            "_extract_input dispatch."
        )


class TestNoNodeMutationInAssembly:
    """Assembly code must not mutate Node instances in place (neograph-n573).

    _construct_builder.py iterates the `ordered` list and must use model_copy()
    instead of direct attribute assignment on Node instances.

    Scope note: this guard polices *in-place attribute mutation* in
    _construct_builder.py ONLY (any field, `n.<attr> = ...`). It is NOT the
    authority on where IR-inferred fields may be written — that is
    TestNormalizeIrIsSoleIrFieldWriter, which scans all of src/neograph for
    every write form (attr/subscript/dict-literal/dict-kwarg/setattr/...) of
    fan_out_param/oracle_gen_type against an allowlist. The two overlap on
    field names but enforce different invariants; do not conflate them.
    """

    def test_no_direct_attr_assignment_on_nodes_in_ordered_loops(self):
        """No `.inputs =`, `.fan_out_param =`, `.scripted_fn =`, `.oracle_gen_type =`
        direct assignments on loop variables in _construct_builder.py."""
        source = (SRC_DIR / "_construct_builder.py").read_text()
        tree = ast.parse(source)

        # Collect all attribute assignments (x.attr = value) that target
        # known Node fields being mutated during assembly.
        forbidden_attrs = {"inputs", "fan_out_param", "scripted_fn", "oracle_gen_type"}
        violations = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr in forbidden_attrs
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "n"
                ):
                    violations.append(
                        f"line {node.lineno}: n.{target.attr} = ..."
                    )

        assert violations == [], (
            "Direct attribute assignment on Node instances in _construct_builder.py "
            "violates immutable IR principle. Use model_copy() instead.\n"
            "Violations:\n" + "\n".join(f"  {v}" for v in violations)
        )


class TestDeadBodyDocstringStrip:
    """Dead-body check must strip docstrings before triviality check (neograph-hn8e).

    The _is_trivial_body helper and the dead-body warning code must handle
    docstring + placeholder patterns like `'''doc''' + ...` without false warnings.
    """

    def test_is_trivial_body_exists(self):
        """_is_trivial_body must exist as a named function in decorators.py."""
        source = (SRC_DIR / "decorators.py").read_text()
        tree = ast.parse(source)
        func_names = {
            n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        }
        assert "_is_trivial_body" in func_names, (
            "_is_trivial_body helper must exist in decorators.py "
            "to handle docstring + placeholder patterns."
        )

    def test_dead_body_check_strips_docstring(self):
        """The dead-body warning code must strip docstrings before the trivial check.

        Looks for: isinstance check on body[0] for str constant (docstring detection)
        followed by body slice (body = body[1:] or equivalent).
        """
        source = (SRC_DIR / "decorators.py").read_text()
        # Must contain docstring detection (isinstance + str) AND body re-slicing
        assert "isinstance(body[0].value.value, str)" in source, (
            "Dead-body check must detect docstrings by checking "
            "isinstance(body[0].value.value, str)"
        )
        assert "body[1:]" in source, (
            "Dead-body check must strip the docstring via body = body[1:]"
        )


class TestBranchNodeIsNotDuckTyped:
    """_BranchNode should inherit from a shared base, not duck-type."""

    def test_branch_node_has_no_stub_methods(self):
        """_BranchNode should not define its own has_modifier/get_modifier stubs."""

        # If _BranchNode inherits properly, has_modifier/get_modifier come
        # from the parent (Modifiable mixin on Node or a Protocol).
        # If it duck-types, the methods are defined directly on _BranchNode.
        src = pathlib.Path(SRC_DIR / "forward.py").read_text()
        tree = ast.parse(src)

        for cls_node in ast.walk(tree):
            if not isinstance(cls_node, ast.ClassDef) or cls_node.name != "_BranchNode":
                continue
            method_names = {
                item.name for item in cls_node.body
                if isinstance(item, ast.FunctionDef)
            }
            stubs = method_names & {"has_modifier", "get_modifier"}
            assert stubs == set(), (
                f"_BranchNode has duck-typed stub methods: {stubs}. "
                "It should inherit these from a shared base (Node or Protocol)."
            )


class TestIterNodesCoversBranchArms:
    """iter_nodes -- the single source of truth for the IR node-tree walk --
    MUST descend into _BranchNode arm contents.

    Regression pin for neograph-tdbb: iter_nodes previously skipped _BranchNode
    sentinels, so nodes inside branch arms were invisible to every
    iter_nodes-based walk (scripted-shim collection, required-DI collection,
    LLM-node discovery). A scripted @node in a branch arm then failed to
    compile with 'Scripted function ... not registered'.

    Behavioral guard (not regex): positive asserts the invariant holds; the
    negative meta-test re-implements the OLD arm-skipping walk and proves it
    would miss the arm node -- i.e. the arm descent is load-bearing, not
    incidental.
    """

    @staticmethod
    def _parent_with_arm_node():
        from neograph import Node
        from neograph._ir_branch import _BranchMeta, _BranchNode, _ConditionSpec
        from neograph.construct import Construct

        seed = Node.scripted("seed", fn="seed_fn", outputs=int)
        arm = Node.scripted("arm-node", fn="arm_fn", inputs=int, outputs=int)
        cond = _ConditionSpec(
            source_node=seed, attr_chain=["x"],
            op_fn=lambda v, _t: bool(v), op_str="route", threshold=None,
        )
        meta = _BranchMeta(condition_spec=cond, true_arm_nodes=[arm], false_arm_nodes=[])
        return Construct("parent", nodes=[seed, _BranchNode(meta, 0)])

    def test_iter_nodes_yields_a_node_inside_a_branch_arm(self):
        from neograph.construct import iter_nodes

        parent = self._parent_with_arm_node()
        names = {n.name for n in iter_nodes(parent)}
        assert "arm-node" in names, (
            "iter_nodes must descend into _BranchNode arms; a node inside a "
            "branch arm was not yielded (neograph-tdbb regression)."
        )

    def test_old_arm_skipping_walk_would_miss_the_arm_node(self):
        """Meta-test: the pre-fix walk (Node/Construct dispatch only, no arm
        descent) misses the arm node -- proving this guard distinguishes the
        fixed walk from the broken one."""
        from neograph import Node
        from neograph.construct import Construct

        def _old_iter(construct):
            for item in construct.nodes:
                if isinstance(item, Construct):
                    yield from _old_iter(item)
                elif isinstance(item, Node):
                    yield item

        parent = self._parent_with_arm_node()
        old_names = {n.name for n in _old_iter(parent)}
        assert "arm-node" not in old_names, (
            "meta-test invariant broken: the old arm-skipping walk should NOT "
            "see the arm node; if it does, this guard no longer proves the fix."
        )


class TestBuildConstructBodySize:
    """_build_construct_from_decorated must be a thin orchestrator, not a monolith.

    The body should delegate to named step helpers (_build_decorated_dict,
    _identify_port_params, _build_adjacency, etc.) and stay under 50 lines.
    """

    def test_build_construct_body_under_50_lines(self):
        """_build_construct_from_decorated body must be < 50 lines."""
        source = (SRC_DIR / "_construct_builder.py").read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_build_construct_from_decorated":
                # Body starts after the docstring (if any)
                body_start = node.body[0].end_lineno if (
                    isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ) else node.lineno
                body_lines = node.end_lineno - body_start
                assert body_lines < 50, (
                    f"_build_construct_from_decorated body is {body_lines} lines — "
                    f"should be < 50 (delegate to named step helpers)."
                )
                return

        pytest.fail("_build_construct_from_decorated not found in _construct_builder.py")


class TestTypesCompatibleUnionForms:
    """_types_compatible must handle all Union representations."""

    def test_optional_x_compatible_with_x(self):
        """Optional[X] (Union[X, None]) is compatible with X."""
        from typing import Optional

        from neograph._construct_validation import _types_compatible
        from tests.schemas import RawText

        assert _types_compatible(RawText, Optional[RawText])  # noqa: UP045

    def test_pipe_none_compatible_with_x(self):
        """X | None is compatible with X (Python 3.10+ union syntax)."""
        from neograph._construct_validation import _types_compatible
        from tests.schemas import RawText

        assert _types_compatible(RawText, RawText | None)

    def test_union_xy_compatible_with_x(self):
        """Union[X, Y] is compatible with X."""
        from typing import Union

        from neograph._construct_validation import _types_compatible
        from tests.schemas import Claims, RawText

        assert _types_compatible(RawText, Union[RawText, Claims])  # noqa: UP007

    def test_nested_optional_in_dict(self):
        """dict[str, Optional[X]] producer compatible with dict[str, X] consumer."""
        from neograph._construct_validation import _types_compatible
        from tests.schemas import RawText

        # This tests that the validator doesn't crash on nested Optional
        # within parameterized generics
        assert _types_compatible(dict[str, RawText], dict[str, RawText | None])


class TestCompilerWiringSplit:
    """compiler.py wiring helpers must live in _wiring.py.

    Wiring helpers (_wire_each, _wire_oracle, _add_each_oracle_fused,
    _add_subgraph_loop, loop/branch/operator helpers) were extracted to
    _wiring.py to keep compiler.py focused on the compile() entry point.
    """

    def test_compiler_under_600_lines(self):
        """compiler.py must be < 600 lines after wiring extraction."""
        compiler = SRC_DIR / "compiler.py"
        line_count = len(compiler.read_text().splitlines())
        assert line_count < 600, (
            f"compiler.py is {line_count} lines — must be < 600. "
            "Move wiring helpers to _wiring.py."
        )

    def test_wiring_module_exists(self):
        """_wiring.py must exist in the neograph package."""
        wiring = SRC_DIR / "_wiring.py"
        assert wiring.exists(), (
            "_wiring.py does not exist. Create it with the extracted wiring helpers."
        )

    WIRING_FUNCTIONS = [
        "def _wire_oracle",
        "def _wire_each",
        "def _add_each_oracle_fused",
        "def _merge_one_group",
        "def _make_loop_router",
        "def _node_loop_unwrap",
        "def _construct_loop_unwrap",
        "def _add_loop_back_edge",
        "def _add_subgraph_loop",
        "def _add_branch_to_graph",
        "def _add_operator_check",
    ]

    def test_wiring_functions_in_correct_module(self):
        """Extracted wiring functions must be in _wiring.py, not compiler.py."""
        wiring = SRC_DIR / "_wiring.py"
        compiler = SRC_DIR / "compiler.py"

        if not wiring.exists():
            pytest.skip("_wiring.py does not exist yet")

        wiring_text = wiring.read_text()
        compiler_text = compiler.read_text()
        violations = []

        for sig in self.WIRING_FUNCTIONS:
            if sig not in wiring_text:
                violations.append(f"  MISSING: '{sig}' not found in _wiring.py")
            if sig in compiler_text:
                violations.append(
                    f"  DRIFTED: '{sig}' found in compiler.py "
                    f"(should only be in _wiring.py)"
                )

        assert violations == [], (
            f"\n{len(violations)} wiring split violation(s):\n"
            + "\n".join(violations)
        )


class TestNoModuleLevelRegistryDicts:
    """Module-level registry dicts in factory.py must be replaced by a Registry class.

    Three dicts (_scripted_registry, _condition_registry, _tool_factory_registry)
    were module-level globals. They cause cross-pipeline collisions and have no
    thread safety. All must live inside a Registry class now.
    """

    def test_no_module_level_registry_dicts(self):
        """AST-scan factory.py for module-level dict assignments named *_registry."""
        source = (SRC_DIR / "factory.py").read_text()
        tree = ast.parse(source)

        violations = []
        for node in ast.iter_child_nodes(tree):
            # Module-level assignments: x: type = {} or x = {}
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id.endswith("_registry"):
                    violations.append(
                        f"  factory.py:{node.lineno}: {node.target.id}"
                    )
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.endswith("_registry"):
                        violations.append(
                            f"  factory.py:{node.lineno}: {target.id}"
                        )

        assert violations == [], (
            f"\n{len(violations)} module-level *_registry dict(s) in factory.py:\n"
            + "\n".join(violations)
            + "\n\nAll registries must live inside the Registry class."
        )




class TestToolGateRoutesConditionally:
    """Structural guard for neograph-whq0: a ``gate_tools_when`` tool-gate must
    route on the human's decision — a conditional edge with BOTH a proceed arm
    (``{node}__tools``) and a deny arm (``{node}__agent``). The original bug was
    an UNCONDITIONAL ``gate -> tools`` edge that ran the tool regardless of the
    resume value. This guard inspects the COMPILED graph topology (not source
    text), so any future refactor that reintroduces an unconditional gate->tools
    edge — or drops the deny arm — is caught even if the behavioral deny test is
    deleted.
    """

    @staticmethod
    def _gate_edges(compiled):
        dg = compiled.get_graph()
        return [
            (e.source, e.target, bool(getattr(e, "conditional", False)))
            for e in dg.edges
            if e.source.endswith("__tools_gate")
        ]

    @classmethod
    def _gate_is_diseased(cls, compiled) -> bool:
        """True when a tool-gate cannot honor a deny: it reaches tools but has no
        conditional deny (``__agent``) arm. No gate present -> not diseased (N/A)."""
        edges = cls._gate_edges(compiled)
        if not edges:
            return False
        reaches_tools = any(t.endswith("__tools") for (_, t, _) in edges)
        has_conditional_deny_arm = any(
            t.endswith("__agent") and cond for (_, t, cond) in edges
        )
        return reaches_tools and not has_conditional_deny_arm

    def _build_real_gated_graph(self):
        from typing import Any

        from langchain_core.messages import AIMessage, ToolMessage
        from langgraph.checkpoint.memory import MemorySaver
        from pydantic import BaseModel

        from neograph import Tool, compile, construct_from_functions, node
        from tests.fakes import (
            build_fake_llm_kwargs,
            build_test_compile_kwargs,
            register_tool_factory,
        )

        class GResult(BaseModel, frozen=True):
            items: list[str]

        class _Fake:
            def bind_tools(self, tools):
                return self

            def abind_tools(self, *a, **k):
                return self

            def invoke(self, messages, **k):
                n = sum(isinstance(m, ToolMessage) for m in messages)
                if n == 0:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "record", "args": {}, "id": "r1"}]
                    return msg
                return AIMessage(content='{"items": ["done"]}')

            async def ainvoke(self, *a, **k):
                return self.invoke(*a, **k)

            def with_structured_output(self, model, **k):
                inst = _Fake()
                inst._m = model
                inst.invoke = lambda messages, **kk: model(items=["done"])  # type: ignore[assignment]
                return inst

        class _Rec:
            name = "record"

            def invoke(self, args):
                return "ok"

            async def ainvoke(self, *a, **k):
                return "ok"

        register_tool_factory("record", lambda config, tool_config: _Rec())

        @node(
            mode="agent",
            outputs=GResult,
            model="reason",
            prompt="test/explore",
            tools=[Tool(name="record", budget=3)],
            gate_tools_when=lambda state: {"reason": "approve?"},
        )
        def research() -> GResult: ...

        return compile(
            construct_from_functions("guard-gating", [research]),
            checkpointer=MemorySaver(),
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(lambda tier: _Fake()),
        )

    def test_real_gated_graph_routes_conditionally_with_deny_arm(self):
        """POSITIVE: the real compiled gate routes to BOTH tools and agent, all
        conditional — never an unconditional gate->tools edge."""
        compiled = self._build_real_gated_graph()
        edges = self._gate_edges(compiled)
        assert edges, "no __tools_gate node found in the compiled gated graph"
        targets = {t for (_, t, _) in edges}
        assert any(t.endswith("__tools") for t in targets), (
            f"gate has no proceed (tools) arm: {edges}"
        )
        assert any(t.endswith("__agent") for t in targets), (
            f"gate has no deny (agent) arm — deny cannot be honored (whq0): {edges}"
        )
        assert all(cond for (_, _, cond) in edges), (
            f"gate has an UNCONDITIONAL edge — the whq0 disease: {edges}"
        )
        assert not self._gate_is_diseased(compiled)

    def test_guard_catches_unconditional_gate_to_tools(self):
        """NEGATIVE meta-test: a gate wired with an UNCONDITIONAL gate->tools edge
        (the original whq0 disease) must be flagged by the predicate."""
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        class S(TypedDict):
            x: int

        b = StateGraph(S)
        b.add_node("r__agent", lambda s: s)
        b.add_node("r__tools_gate", lambda s: s)
        b.add_node("r__tools", lambda s: s)
        b.add_edge(START, "r__agent")
        b.add_conditional_edges("r__agent", lambda s: "r__tools_gate", path_map=["r__tools_gate"])
        b.add_edge("r__tools_gate", "r__tools")  # DISEASE: unconditional, no deny arm
        b.add_edge("r__tools", END)
        compiled = b.compile()
        assert self._gate_is_diseased(compiled), (
            "guard failed to flag an unconditional gate->tools edge (whq0 disease)"
        )

    def test_guard_catches_conditional_gate_without_deny_arm(self):
        """NEGATIVE meta-test (would-be-missed variant): even a CONDITIONAL gate
        edge is diseased if it only ever reaches tools and has no deny (agent)
        arm — a conditional edge is not enough; the deny arm must exist."""
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        class S(TypedDict):
            x: int

        b = StateGraph(S)
        b.add_node("r__agent", lambda s: s)
        b.add_node("r__tools_gate", lambda s: s)
        b.add_node("r__tools", lambda s: s)
        b.add_edge(START, "r__agent")
        b.add_conditional_edges("r__agent", lambda s: "r__tools_gate", path_map=["r__tools_gate"])
        # Conditional, but the only reachable target is tools — no deny arm.
        b.add_conditional_edges("r__tools_gate", lambda s: "r__tools", path_map=["r__tools"])
        b.add_edge("r__tools", END)
        compiled = b.compile()
        assert self._gate_is_diseased(compiled), (
            "guard failed to flag a conditional gate that has no deny (agent) arm"
        )

    def test_predicate_is_na_when_no_gate(self):
        """A graph with no tool-gate is not diseased (N/A), so the guard never
        fires on ungated agent pipelines."""
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        class S(TypedDict):
            x: int

        b = StateGraph(S)
        b.add_node("r__agent", lambda s: s)
        b.add_edge(START, "r__agent")
        b.add_edge("r__agent", END)
        assert not self._gate_is_diseased(b.compile())
