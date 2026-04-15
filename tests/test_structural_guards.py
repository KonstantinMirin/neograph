"""Structural guards — AST-based invariants that run every build.

These tests enforce architectural rules by scanning source code structure,
not by running pipelines. They prevent erosion of patterns that were
established via migration sprints.

Each guard was written BEFORE the migration it enforces. If a guard fails,
fix the violation — do NOT add an allowlist.
"""

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


class TestErrorBuilderEnforcement:
    """Every raise of a neograph error class must use .build(), not direct construction.

    Good:  raise ConstructError.build("what", expected="X", found="Y")
    Bad:   raise ConstructError(f"Node '{name}' ...")
    Bad:   raise ConstructError(msg)
    Bad:   raise ConstructError("literal string")

    This guard prevents the error builder migration from eroding.
    If this test fails, convert the flagged raise to .build() — do NOT
    add an allowlist entry.
    """

    def test_all_error_raises_use_build(self):
        violations = []

        for py_file in sorted(SRC_DIR.glob("*.py")):
            source = py_file.read_text()
            try:
                tree = ast.parse(source, filename=str(py_file))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if not isinstance(node, ast.Raise) or node.exc is None:
                    continue

                exc = node.exc

                # raise SomeError(...) — direct call, should be .build()
                if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                    if exc.func.id in ERROR_CLASSES:
                        violations.append(
                            f"  {py_file.name}:{node.lineno}: "
                            f"raise {exc.func.id}(...) — use {exc.func.id}.build() instead"
                        )

        assert violations == [], (
            f"\n{len(violations)} error raise(s) bypass .build():\n"
            + "\n".join(violations)
            + "\n\nConvert each to .build(what=, expected=, found=, hint=). "
            "See src/neograph/errors.py for the API."
        )


class TestFileSplitEnforcement:
    """Functions extracted from decorators.py and factory.py must not drift back.

    Wave 1 split 4 groups of functions into dedicated modules. If someone
    adds a new function to the wrong file, these tests catch it.
    """

    # (function_or_class_name, must_be_in, must_NOT_be_in)
    SPLIT_RULES = [
        # decorators.py → _di_classify.py
        ("def _classify_di_params", "_di_classify.py", "decorators.py"),
        ("class FromInput", "_di_classify.py", "decorators.py"),
        ("class FromConfig", "_di_classify.py", "decorators.py"),
        ("def _resolve_di_args", "_di_classify.py", "decorators.py"),
        ("def _resolve_merge_args", "_di_classify.py", "decorators.py"),
        # decorators.py → _construct_builder.py
        ("def construct_from_module", "_construct_builder.py", "decorators.py"),
        ("def construct_from_functions", "_construct_builder.py", "decorators.py"),
        ("def _build_construct_from_decorated", "_construct_builder.py", "decorators.py"),
        ("def _register_node_scripted", "_construct_builder.py", "decorators.py"),
        # factory.py → _dispatch.py
        ("class ScriptedDispatch", "_dispatch.py", "factory.py"),
        ("class ThinkDispatch", "_dispatch.py", "factory.py"),
        ("class ToolDispatch", "_dispatch.py", "factory.py"),
        ("class ModeDispatch", "_dispatch.py", "factory.py"),
        ("def _dispatch_for_mode", "_dispatch.py", "factory.py"),
        # factory.py → _oracle.py
        ("def make_oracle_redirect_fn", "_oracle.py", "factory.py"),
        ("def make_eachoracle_redirect_fn", "_oracle.py", "factory.py"),
        ("def make_oracle_merge_fn", "_oracle.py", "factory.py"),
        ("def make_each_redirect_fn", "_oracle.py", "factory.py"),
    ]

    def test_extracted_functions_in_correct_module(self):
        violations = []
        for signature, must_be_in, must_not_be_in in self.SPLIT_RULES:
            good_file = SRC_DIR / must_be_in
            bad_file = SRC_DIR / must_not_be_in

            good_text = good_file.read_text() if good_file.exists() else ""
            bad_text = bad_file.read_text() if bad_file.exists() else ""

            if signature not in good_text:
                violations.append(
                    f"  MISSING: '{signature}' not found in {must_be_in}"
                )
            if signature in bad_text:
                violations.append(
                    f"  DRIFTED: '{signature}' found in {must_not_be_in} "
                    f"(should only be in {must_be_in})"
                )

        assert violations == [], (
            f"\n{len(violations)} file-split violation(s):\n"
            + "\n".join(violations)
        )


class TestDeadCodeRemoval:
    """Verify dead code that was deliberately removed stays removed."""

    def test_di_resolver_not_in_codebase(self):
        """DIResolver was dead code — removed in neograph-eq80."""
        for py_file in sorted(SRC_DIR.glob("*.py")):
            text = py_file.read_text()
            assert "class DIResolver" not in text, (
                f"DIResolver found in {py_file.name} — it was removed as dead code. "
                f"If you need a resolver wrapper, design a new one."
            )

    def test_no_raw_node_decorator(self):
        """@raw_node decorator was removed in favor of @node(mode='raw').

        Checks for the decorator definition (not local function names
        like raw_node_wrapper which are fine).
        """
        for py_file in sorted(SRC_DIR.glob("*.py")):
            tree = ast.parse(py_file.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "raw_node":
                    # Check if it has a decorator list (it's a decorator definition)
                    # or is a top-level function (public API)
                    assert False, (
                        f"raw_node() function found in {py_file.name}:{node.lineno} — "
                        f"use @node(mode='raw') instead."
                    )

    def test_no_old_from_input_subscription(self):
        """FromInput[T] generic subscription was removed in 0.2.0.

        Only checks actual code, not comments or docstrings.
        """
        for py_file in sorted(SRC_DIR.glob("*.py")):
            tree = ast.parse(py_file.read_text())
            for node in ast.walk(tree):
                # Look for FromInput[...] subscription in actual code
                if isinstance(node, ast.Subscript):
                    if isinstance(node.value, ast.Name) and node.value.id in ("FromInput", "FromConfig"):
                        assert False, (
                            f"{node.value.id}[T] subscription found in "
                            f"{py_file.name}:{node.lineno} — "
                            f"use Annotated[T, {node.value.id}] instead."
                        )


class TestOutputsInferenceGuard:
    """@node outputs= inference from return annotation (neograph-pcdp).

    The decorator infers outputs from -> T. When both outputs= and -> T
    are present and disagree, it must raise ConstructError at decoration
    time. This guard verifies the check exists in the decorator source.
    """

    def test_mismatch_check_exists_in_decorator(self):
        """The decorator must contain the mismatch check logic."""
        decorators = (SRC_DIR / "decorators.py").read_text()
        assert "differs from return annotation" in decorators, (
            "The outputs=/annotation mismatch check is missing from decorators.py. "
            "The @node decorator must raise ConstructError when outputs= and -> T disagree."
        )

    def test_mismatch_actually_raises(self):
        """Runtime verification: mismatched outputs= and -> T raises."""
        from pydantic import BaseModel

        from neograph import ConstructError, node

        class TypeA(BaseModel):
            x: int

        class TypeB(BaseModel):
            y: str

        with pytest.raises(ConstructError, match="differs from return annotation"):
            @node(outputs=TypeA)
            def bad_node(topic: str) -> TypeB:
                return TypeB(y="oops")


class TestNoAnyInTypeBoundaries:
    """NodeInput, NodeOutput, and DIBinding must not use bare Any for typed fields.

    Any at type boundaries defeats the "typed compiler" claim. These
    containers carry concrete values — their fields should say so.
    """

    def test_node_input_output_no_any_fields(self):
        """NodeInput and NodeOutput fields must not be annotated with bare Any."""
        dispatch_file = SRC_DIR / "_dispatch.py"
        tree = ast.parse(dispatch_file.read_text())
        violations = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name not in ("NodeInput", "NodeOutput"):
                continue
            for item in node.body:
                if not isinstance(item, ast.AnnAssign):
                    continue
                if item.target is None or not isinstance(item.target, ast.Name):
                    continue
                field_name = item.target.id
                # Check if annotation is bare "Any"
                ann = item.annotation
                if isinstance(ann, ast.Name) and ann.id == "Any":
                    violations.append(
                        f"  {node.name}.{field_name}: Any "
                        f"(line {item.lineno})"
                    )
                # Check dict[str, Any] — the value type should be concrete
                if isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name):
                    if ann.value.id == "dict":
                        # dict[str, Any] — check the second element
                        if isinstance(ann.slice, ast.Tuple) and len(ann.slice.elts) == 2:
                            val_type = ann.slice.elts[1]
                            if isinstance(val_type, ast.Name) and val_type.id == "Any":
                                violations.append(
                                    f"  {node.name}.{field_name}: dict[str, Any] "
                                    f"(line {item.lineno})"
                                )

        assert violations == [], (
            f"\n{len(violations)} bare Any field(s) in dispatch containers:\n"
            + "\n".join(violations)
            + "\n\nReplace with a concrete union type."
        )

    def test_di_binding_has_no_payload_field(self):
        """DIBinding must not have a 'payload' field (neograph-xklx).

        Replaced with typed fields: default_value (CONSTANT) and model_cls (MODEL).
        """
        di_file = SRC_DIR / "di.py"
        tree = ast.parse(di_file.read_text())

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name != "DIBinding":
                continue
            field_names = []
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    field_names.append(item.target.id)
            assert "payload" not in field_names, (
                "DIBinding still has 'payload' field — should be replaced with "
                "default_value (CONSTANT) and model_cls (MODEL)"
            )
            assert "default_value" in field_names, "DIBinding missing default_value field"
            assert "model_cls" in field_names, "DIBinding missing model_cls field"


class TestNoRedundantValidation:
    """Modifier mutual-exclusion checks must live in ModifierSet only.

    ModifierSet.model_post_init is the structural gate (catches all paths).
    _validate_node_chain should NOT duplicate those checks — it adds
    maintenance burden without catching anything model_post_init misses.
    """

    def test_no_each_loop_check_in_validate_node_chain(self):
        """_construct_validation.py must not check Each+Loop exclusion."""
        validation_file = SRC_DIR / "_construct_validation.py"
        text = validation_file.read_text()
        tree = ast.parse(text)

        # Find _validate_node_chain function
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "_validate_node_chain":
                continue
            # Check the function body source for Each+Loop or Oracle+Loop
            # mutual exclusion patterns
            func_source = ast.get_source_segment(text, node)
            if func_source is None:
                continue
            violations = []
            if "Each" in func_source and "Loop" in func_source and "mutual exclu" in func_source.lower():
                violations.append(
                    "  _validate_node_chain contains Each+Loop mutual exclusion check"
                )
            if "Oracle" in func_source and "Loop" in func_source and "mutual exclu" in func_source.lower():
                violations.append(
                    "  _validate_node_chain contains Oracle+Loop mutual exclusion check"
                )
            assert violations == [], (
                "\nRedundant modifier checks in _validate_node_chain:\n"
                + "\n".join(violations)
                + "\n\nModifierSet.model_post_init is the structural gate. "
                "Remove the belt-and-suspenders checks."
            )


class TestNoSidecarPattern:
    """Node metadata must live on the Node via PrivateAttr, not in global dicts.

    The sidecar pattern (_node_sidecar, _param_resolutions keyed by id(Node))
    is a hack around Pydantic immutability. PrivateAttr is the Pydantic-native
    solution — data travels with the object, no re-registration needed.
    """

    def test_no_node_sidecar_dict(self):
        """No global _node_sidecar dict should exist in the codebase."""
        violations = []
        for py_file in sorted(SRC_DIR.glob("*.py")):
            text = py_file.read_text()
            for i, line in enumerate(text.splitlines(), 1):
                # Match the dict definition, not comments or strings
                stripped = line.lstrip()
                if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                    continue
                if "_node_sidecar" in line and "import" not in line:
                    violations.append(f"  {py_file.name}:{i}: {stripped[:80]}")
        assert violations == [], (
            f"\n{len(violations)} _node_sidecar reference(s) remain:\n"
            + "\n".join(violations)
            + "\n\nReplace with PrivateAttr on Node."
        )

    def test_no_param_resolutions_dict(self):
        """No global _param_resolutions dict should exist in the codebase."""
        violations = []
        for py_file in sorted(SRC_DIR.glob("*.py")):
            text = py_file.read_text()
            for i, line in enumerate(text.splitlines(), 1):
                stripped = line.lstrip()
                if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                    continue
                # Match _param_resolutions as a dict or function, not as a PrivateAttr field name
                if "_param_resolutions" in line and "import" not in line:
                    # Allow: _param_res (the PrivateAttr field, shorter name)
                    # Reject: _param_resolutions (the old global dict pattern)
                    violations.append(f"  {py_file.name}:{i}: {stripped[:80]}")
        assert violations == [], (
            f"\n{len(violations)} _param_resolutions reference(s) remain:\n"
            + "\n".join(violations)
            + "\n\nReplace with PrivateAttr on Node."
        )

    def test_no_weakref_finalize_for_sidecar(self):
        """No weakref.finalize calls for sidecar cleanup should exist."""
        violations = []
        for py_file in sorted(SRC_DIR.glob("*.py")):
            text = py_file.read_text()
            for i, line in enumerate(text.splitlines(), 1):
                if "weakref.finalize" in line:
                    stripped = line.lstrip()
                    if not stripped.startswith("#"):
                        violations.append(f"  {py_file.name}:{i}: {stripped[:80]}")
        assert violations == [], (
            f"\n{len(violations)} weakref.finalize call(s) remain:\n"
            + "\n".join(violations)
            + "\n\nPrivateAttr data lives on the Node — no external cleanup needed."
        )

    def test_node_has_private_attrs(self):
        """Node must declare PrivateAttr fields for sidecar data."""
        from neograph.node import Node
        private_fields = getattr(Node, "__private_attributes__", {})
        assert "_sidecar" in private_fields or "_sidecar_fn" in private_fields, (
            "Node must have a _sidecar or _sidecar_fn PrivateAttr field. "
            "Add: _sidecar: tuple[Callable, tuple[str, ...]] | None = PrivateAttr(default=None)"
        )
        assert "_param_res" in private_fields, (
            "Node must have a _param_res PrivateAttr field. "
            "Add: _param_res: dict = PrivateAttr(default_factory=dict)"
        )


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

    def test_no_parenthesized_ticket_ids(self):
        """No (neograph-xxxx) patterns in source code."""
        violations = []
        for py_file in sorted(SRC_DIR.glob("*.py")):
            for i, line in enumerate(py_file.read_text().splitlines(), 1):
                if re.search(r"\(neograph-[a-z0-9]{3,5}\)", line):
                    stripped = line.lstrip()[:80]
                    violations.append(f"  {py_file.name}:{i}: {stripped}")
        assert violations == [], (
            f"\n{len(violations)} ticket ID reference(s) in source:\n"
            + "\n".join(violations[:20])
            + ("\n  ..." if len(violations) > 20 else "")
            + "\n\nRemove the (neograph-xxxx) part. Keep the comment text."
        )


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
        source = (SRC_DIR / "factory.py").read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_extract_input":
                body_lines = node.end_lineno - node.lineno
                assert body_lines < 30, (
                    f"_extract_input is {body_lines} lines — should be < 30 "
                    f"(pure dispatch to named helpers, not inline logic)."
                )
                return

        pytest.fail("_extract_input function not found in factory.py")

    def test_input_shape_enum_exists(self):
        """InputShape enum must exist for exhaustive dispatch."""
        source = (SRC_DIR / "factory.py").read_text()
        tree = ast.parse(source)
        class_names = {
            n.name for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef)
        }
        assert "InputShape" in class_names, (
            "InputShape enum must exist in factory.py for exhaustive "
            "_extract_input dispatch."
        )


class TestNoNodeMutationInAssembly:
    """Assembly code must not mutate Node instances in place (neograph-n573).

    _construct_builder.py iterates the `ordered` list and must use model_copy()
    instead of direct attribute assignment on Node instances.
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


class TestSidecarModule:
    """_sidecar.py must exist and _construct_builder.py must NOT import from decorators.py.

    TASK neograph-s0cu: extract sidecar storage to break circular import.
    """

    def test_sidecar_module_exists(self):
        """_sidecar.py must exist as a standalone module."""
        sidecar_path = SRC_DIR / "_sidecar.py"
        assert sidecar_path.exists(), (
            "_sidecar.py does not exist — sidecar storage functions "
            "must be extracted from decorators.py"
        )

    def test_construct_builder_does_not_import_from_decorators(self):
        """_construct_builder.py must import sidecar functions from _sidecar, not decorators."""
        builder_path = SRC_DIR / "_construct_builder.py"
        tree = ast.parse(builder_path.read_text())
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "neograph.decorators":
                names = [a.name for a in node.names]
                violations.append(
                    f"  _construct_builder.py:{node.lineno}: from neograph.decorators import {', '.join(names)}"
                )
        assert violations == [], (
            f"\n_construct_builder.py still imports from decorators.py:\n"
            + "\n".join(violations)
            + "\n\nThese should import from neograph._sidecar instead."
        )


class TestDeferredImportBudget:
    """Track deferred imports — they should decrease, not increase.

    TASK neograph-2cb2: move registry ops to _registry.py to eliminate deferred imports.
    """

    def test_deferred_import_count_within_budget(self):
        """Deferred imports across src/neograph/*.py must stay within budget."""
        count = 0
        for py_file in sorted(SRC_DIR.glob("*.py")):
            tree = ast.parse(py_file.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("neograph."):
                    # Check if this import is inside a function (deferred)
                    # by seeing if it has col_offset > 0 (indented)
                    if node.col_offset > 0:
                        count += 1
        # Budget: 41 (was 56→40, +1 for CLI test-scaffold lazy-load).
        assert count <= 41, (
            f"Deferred import count is {count}, budget is 41. "
            f"Move registry ops to _registry.py and promote leaf-module imports to reduce."
        )


class TestNoBareExceptException:
    """No bare 'except Exception' that swallows errors silently.

    TASK neograph-bjin: 13 bare except Exception handlers found.
    Every except must either:
    - Catch a specific exception tuple (NameError, TypeError, etc.)
    - Re-raise as a typed error (except Exception as exc: raise ... from exc)
    - Be explicitly marked with # noqa: bare-except (must be justified)
    """

    def test_no_bare_except_exception_pass(self):
        """No 'except Exception:' followed by pass/return None/empty dict."""
        violations = []
        for py_file in sorted(SRC_DIR.glob("*.py")):
            source = py_file.read_text()
            tree = ast.parse(source)
            lines = source.splitlines()
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                # Only flag bare "except Exception" (not "except Exception as exc")
                if (
                    node.type is not None
                    and isinstance(node.type, ast.Name)
                    and node.type.id == "Exception"
                    and node.name is None  # no "as exc"
                ):
                    # Check if the line has a noqa comment
                    line_text = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    if "noqa: bare-except" in line_text:
                        continue
                    violations.append(
                        f"  {py_file.name}:{node.lineno}: bare except Exception (no 'as exc', no specific types)"
                    )
        assert violations == [], (
            f"\n{len(violations)} bare 'except Exception:' handler(s) found.\n"
            + "\n".join(violations)
            + "\n\nFix: catch specific exceptions, or use 'except Exception as exc:' and re-raise/log."
        )
