"""Structural guards: no-Any in public IR APIs, arbitrary-types justification,
public functions raise NeographError."""

from __future__ import annotations

import ast
import pathlib
import re

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

# Error classes that must use .build() instead of direct construction.
ERROR_CLASSES = frozenset(
    {
        "ConstructError",
        "ExecutionError",
        "CompileError",
        "ConfigurationError",
        "NeographError",
    }
)


ANY_AUDIT_MODULES = (
    "node.py",
    "construct.py",
    "modifiers.py",
    "_construct_validation.py",
    # Validation cluster sub-modules (neograph-gig0): the no-Any audit follows
    # the moved public functions (effective_producer_type -> _validation_types;
    # validate_loop_self_edge / validate_loop_construct -> _validation_modifiers)
    # into their new homes. Zero new ANY_ALLOWLIST entries — the Cluster-5
    # functions were already migrated to TypeSpecStatic (neograph-86r1).
    "_validation_types.py",
    "_validation_inputs.py",
    "_validation_modifiers.py",
    "factory.py",
    "_dispatch.py",
    "_oracle.py",
    "_wiring.py",
    "_ir_normalize.py",
    # Note: _normalize.py (the sibling normalized-view helper) is intentionally
    # NOT in scope — it is a pre-existing input/output-shape adapter whose Any
    # uses carry user-declared types throughout; bringing it under the
    # discipline is a separate effort (neograph-y95c scoped only _ir_normalize).
)

# Allowlist of public functions/methods that may use ``Any`` because the value
# at that boundary is genuinely untyped from neograph's perspective. Each entry
# names the boundary and is keyed by ``module:qualname:param`` (param is the
# parameter name or ``return`` for the return annotation). The value is a
# one-line reason naming the boundary.
#
# ANY ADDITION TO THIS ALLOWLIST REQUIRES a one-line boundary reason. If you
# can replace the Any with a precise type (or the NormalizedInputs view, or
# NodeInput/NodeOutput, or a BaseModel), do that instead.
ANY_ALLOWLIST: dict[str, str] = {
    # ── node.py — Protocol signatures and TypeSpec validator boundaries ──
    "node.py:RawNodeFn.__call__:return": "user-supplied state-update dict; values typed by user node",
    "node.py:_validate_type_spec:v": "Pydantic BeforeValidator boundary; raw input is untyped",
    "node.py:_validate_type_spec:return": "Pydantic BeforeValidator boundary; returns type | dict[str, type] | None",
    "node.py:_is_type_like:v": "introspection helper called on arbitrary user-declared shapes",
    "node.py:Node.run_isolated:input": "user-supplied initial state (typed instance or dict) for isolated execution",
    "node.py:Node.run_isolated:return": "user-supplied output value; type declared by node.outputs",
    # ── construct.py — node list validator boundary and dynamic kwargs ──
    "construct.py:_validate_node_list:v": "Pydantic BeforeValidator boundary; raw input is untyped",
    "construct.py:Construct.__init__:kwargs": "Pydantic BaseModel kwargs passthrough boundary",
    # ── modifiers.py — Protocol signatures for user merge/fallback callbacks ──
    "modifiers.py:MergePreProcess.__call__:return": "invoke_structured accepts BaseModel | dict[str, Any] | str; dict-form retains Any value type",
    "modifiers.py:Oracle.model_post_init:__context": "Pydantic model_post_init context payload; framework-internal",
    "modifiers.py:Loop.model_post_init:__context": "Pydantic model_post_init context payload; framework-internal",
    "modifiers.py:Portal.model_post_init:__context": "Pydantic model_post_init context payload; framework-internal",
    "modifiers.py:ModifierSet.model_post_init:__context": "Pydantic model_post_init context payload; framework-internal",
    # ── _construct_validation.py — IR introspection over user-declared types ──
    # Cluster-5 entries (effective_producer_type:return, _check_item_input:input_type,
    # _check_fan_in_inputs:inputs_dict, _check_each_path:input_type,
    # _resolve_field_annotation:return, _types_compatible:producer/target,
    # _extract_list_element:tp/return, _fmt_type:tp, _build_no_producer_error:input_type,
    # _suggest_hint:input_type) migrated to TypeSpecStatic in Batch 1 (neograph-86r1).
    # ── factory.py — state-bus polymorphism resolved via StateBus protocol ──
    # The state union BaseModel | dict[str, Any] is now adapted into a StateBus
    # at the dispatch entry; helpers take ``state: StateBus`` and never see the
    # raw union. ``Any`` returns survive where the value is genuinely user-typed
    # (Cluster 5 / 6 boundaries — covered by their own allowlist entries).
    # Cluster 3 / 8 state entries migrated to StateBus in Batch 2 (neograph-036p).
    "factory.py:_extract_context:return": "context dict values resolved from user-declared state fields",
    # factory.py:_type_name:t migrated to TypeSpecStatic in Batch 1 (neograph-86r1).
    "factory.py:_apply_skip_when:input_data": "user-supplied extracted input; type declared by node.inputs",
    "factory.py:_apply_skip_when:return": "state update dict; values typed by user node outputs",
    "factory.py:_build_state_update:result": "user-supplied node result; type declared by node.outputs",
    "factory.py:_build_state_update:return": "state update dict; values typed by user node outputs",
    "factory.py:_execute_node:return": "state update dict; values typed by user node outputs",
    "factory.py:_extract_loop_reentry:return": "user-supplied loop value; type declared by node.outputs",
    "factory.py:_extract_each_item:return": "user-supplied Each item; element type from each.over collection",
    "factory.py:_extract_fan_in_dict:return": "dict of upstream values; element types declared by node.inputs",
    "factory.py:_extract_single_type:return": "user-supplied upstream value; type declared by node.inputs",
    "factory.py:_extract_input:return": "user-supplied extracted input; type declared by node.inputs",
    # ── factory.py — Portal routing decision (neograph-nnds9 extraction) ──
    "factory.py:_portal_route_to_command:update": "state update dict; values typed by user node outputs",
    "factory.py:make_portal_agent_cycle_fn:return": "ReAct-cycle body dict; callables typed by _agent_cycle, opaque here",
    # ── _wiring.py — shared agent-cycle wiring (neograph-nnds9) ──
    "_wiring.py:_wire_agent_cycle_body:parts": "ReAct-cycle body dict from make_agent_cycle_bodies/make_portal_agent_cycle_fn; opaque here",
    # ── _dispatch.py — render boundary; context_data is now precise (dict[str, str]) ──
    "_dispatch.py:_render_input:input_data": "user-supplied extracted input; type declared by node.inputs",
    "_dispatch.py:_render_input:return": "RenderedInput.raw or RenderedInput.for_template_ref; user-typed payload",
    # _dispatch.py:_resolve_primary_output:return migrated to TypeSpecStatic in Batch 1 (neograph-86r1).
    # ── _oracle.py — user-declared output models ──
    # _oracle.py:_unwrap_oracle_results:output_model migrated to TypeSpecStatic in Batch 1 (neograph-86r1).
    "_oracle.py:_build_oracle_merge_result:merged": "user-supplied merge result; type declared by node.outputs",
    # ── _oracle.py — canonical merge kernel (ARCH-1 / neograph-s0iz) ──
    # state is the dynamically-shaped LangGraph state object; upstream_context
    # values are heterogeneous upstream model instances; returns are the
    # user-declared merge output (type-erased at this layer, declared by node.outputs).
    "_oracle.py:_build_upstream_context:state": "dynamically-shaped LangGraph state object",
    "_oracle.py:_build_upstream_context:return": "heterogeneous upstream model instances keyed by input name",
    "_oracle.py:_run_merge_prompt:upstream_context": "heterogeneous upstream model instances keyed by input name",
    "_oracle.py:_run_merge_prompt:return": "user-supplied merge result; type declared by node.outputs",
    "_oracle.py:_run_merge_fn:state_for_di": "dynamically-shaped LangGraph state object (from_state DI source)",
    "_oracle.py:_run_merge_fn:return": "user-supplied merge result; type declared by node.outputs",
    "_oracle.py:_merge_variants:upstream_context": "heterogeneous upstream model instances keyed by input name",
    "_oracle.py:_merge_variants:state_for_di": "dynamically-shaped LangGraph state object (from_state DI source)",
    "_oracle.py:_merge_variants:return": "user-supplied merge result; type declared by node.outputs",
    # neograph-p3c7 — async merge twins + extracted pure helpers; same user-data
    # boundaries as their sync counterparts above.
    "_oracle.py:_merge_prompt_input:upstream_context": "heterogeneous upstream model instances keyed by input name",
    "_oracle.py:_merge_prompt_input:return": "merge LLM input_data (user models) + primary output model",
    "_oracle.py:_merge_prompt_post:merged": "user-supplied merge result; type declared by node.outputs",
    "_oracle.py:_merge_prompt_post:return": "user-supplied merge result; type declared by node.outputs",
    "_oracle.py:_merge_fallback_or_reraise:return": "user-supplied merge_fallback result; type declared by node.outputs",
    "_oracle.py:_arun_merge_prompt:upstream_context": "heterogeneous upstream model instances keyed by input name",
    "_oracle.py:_arun_merge_prompt:return": "user-supplied merge result; type declared by node.outputs",
    "_oracle.py:_amerge_variants:upstream_context": "heterogeneous upstream model instances keyed by input name",
    "_oracle.py:_amerge_variants:state_for_di": "dynamically-shaped LangGraph state object (from_state DI source)",
    "_oracle.py:_amerge_variants:return": "user-supplied merge result; type declared by node.outputs",
    # ── _wiring.py — Callable fn pointers ──
    # gen_fn / merge_fn / fan_fn / subgraph_fn are runtime-built closures whose
    # precise signatures are determined by the user's modifier configuration.
    "_wiring.py:_merge_one_group:return": "user-supplied merge result; type declared by node.outputs",
    "_wiring.py:_merge_one_group:upstream_context": "heterogeneous upstream model instances keyed by input name",
    "_wiring.py:_merge_one_group:state": "dynamically-shaped LangGraph state object (from_state DI source)",
    # neograph-p3c7 — async twin of _merge_one_group; same boundaries.
    "_wiring.py:_amerge_one_group:return": "user-supplied merge result; type declared by node.outputs",
    "_wiring.py:_amerge_one_group:upstream_context": "heterogeneous upstream model instances keyed by input name",
    "_wiring.py:_amerge_one_group:state": "dynamically-shaped LangGraph state object (from_state DI source)",
    "_wiring.py:_construct_loop_unwrap:return": "user-supplied loop value; type declared by the sub-construct output",
    "_wiring.py:_add_branch_to_graph:checkpointer": "LangGraph checkpointer (BaseCheckpointSaver | None) threaded opaquely into the arm sub-construct compile; mirrors _add_subgraph's checkpointer: Any (neograph-faf8)",
    "_wiring.py:_add_arm_nodes:checkpointer": "LangGraph checkpointer (BaseCheckpointSaver | None) threaded opaquely into the arm sub-construct compile; extracted verbatim from _add_branch_to_graph:checkpointer (DRY-07 dedup, neograph-7w0d)",
    "_wiring.py:_add_portal_mesh:checkpointer": "LangGraph checkpointer (BaseCheckpointSaver | None) threaded opaquely into a sub-construct Portal mesh member's compile; mirrors _add_subgraph's checkpointer: Any (do0d9)",
    "_wiring.py:_make_portal_subgraph_member_fn:checkpointer": "LangGraph checkpointer (BaseCheckpointSaver | None) threaded opaquely into the sub-construct mesh-member compile; mirrors _add_subgraph's checkpointer: Any (do0d9)",
    # ── _ir_normalize.py — IrNormalizer.apply update dict ──
    "_ir_normalize.py:IrNormalizer.apply:return": "model_copy update dict; heterogeneous IR field values (str fan_out_param, type[BaseModel] oracle_gen_type)",
}


class TestNoAnyInIRPublicAPIs:
    """Public functions/methods in IR + dispatch layers must not use bare ``Any``.

    Scope: ``node.py``, ``construct.py``, ``modifiers.py``, ``_construct_validation.py``,
    ``factory.py``, ``_dispatch.py``, ``_oracle.py``, ``_wiring.py``.

    Rule: every parameter or return annotation that resolves to ``typing.Any``
    on a public (non-underscore-prefixed) function or method must appear in
    ``ANY_ALLOWLIST`` with a one-line reason naming the user-data boundary it
    represents.

    Use precise types (incl. NormalizedInputs, NodeInput/NodeOutput, BaseModel,
    RunnableConfig) where possible. Bare ``Any`` is only acceptable where the
    value genuinely originates from user-declared types or user-supplied state.

    Sentinel for the mutation test: a parametrized fixture proves the scanner
    detects un-allowlisted ``Any`` annotations.
    """

    def test_no_unallowlisted_any_in_public_apis(self):
        offenders = self._scan_public_any_uses(SRC_DIR, ANY_AUDIT_MODULES, ANY_ALLOWLIST)
        assert offenders == [], (
            f"\n{len(offenders)} public function(s)/method(s) use bare typing.Any "
            "outside the allowlist:\n"
            + "\n".join(f"  {o}" for o in offenders)
            + "\n\nReplace Any with a precise type, or add an allowlist entry "
            "with a one-line reason naming the user-data boundary."
        )

    def test_scanner_detects_injected_any(self, tmp_path: pathlib.Path):
        """Mutation: a synthetic module with un-allowlisted Any must be flagged.

        Verifies the scanner is not silently passing because of a bug.
        """
        synthetic = tmp_path / "node.py"  # mimic an in-scope filename
        synthetic.write_text("from typing import Any\ndef public_fn(x: Any) -> Any:\n    return x\n")
        offenders = self._scan_public_any_uses(tmp_path, ("node.py",), allowlist={})
        assert any("public_fn" in o for o in offenders), f"scanner failed to detect injected Any; offenders={offenders}"

    def test_scanner_respects_allowlist(self, tmp_path: pathlib.Path):
        """Mutation: an allowlisted Any must pass the scanner."""
        synthetic = tmp_path / "node.py"
        synthetic.write_text("from typing import Any\ndef public_fn(x: Any) -> Any:\n    return x\n")
        offenders = self._scan_public_any_uses(
            tmp_path,
            ("node.py",),
            allowlist={
                "node.py:public_fn:x": "test boundary",
                "node.py:public_fn:return": "test boundary",
            },
        )
        assert offenders == [], f"allowlist not honored: {offenders}"

    @staticmethod
    def _scan_public_any_uses(
        src_dir: pathlib.Path,
        modules: tuple[str, ...],
        allowlist: dict[str, str],
    ) -> list[str]:
        offenders: list[str] = []
        for fname in modules:
            path = src_dir / fname
            if not path.exists():
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            for qualname, fn in _walk_public_functions(tree):
                # Parameter annotations
                for arg, ann in _iter_annotated_params(fn):
                    if _annotation_uses_any(ann):
                        key = f"{fname}:{qualname}:{arg}"
                        if key not in allowlist:
                            offenders.append(f"{fname}:{fn.lineno} {qualname}({arg}: Any)")
                # Return annotation
                if fn.returns is not None and _annotation_uses_any(fn.returns):
                    key = f"{fname}:{qualname}:return"
                    if key not in allowlist:
                        offenders.append(f"{fname}:{fn.lineno} {qualname} -> Any")
                # Class-level attribute annotations (AnnAssign in a class body)
                # are handled by _walk_public_functions returning a synthetic
                # marker — we collect those via the module walk below.
            # Collect public class-attribute annotations that resolve to Any.
            for qualname, target_name, ann in _iter_public_class_attr_annotations(tree):
                if _annotation_uses_any(ann):
                    key = f"{fname}:{qualname}.{target_name}"
                    if key not in allowlist:
                        offenders.append(f"{fname}:{ann.lineno} {qualname}.{target_name}: Any")
        return offenders


def _walk_public_functions(tree: ast.AST) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Yield (qualname, function-node) for every public function/method in tree.

    Public = not underscore-prefixed at any level, except dunder methods (which
    we include because Protocol __call__ is a real boundary).
    Methods inside private classes are excluded. Nested functions are excluded.
    """
    out: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Module-level function: include even if private — Any audit applies.
            out.append((node.name, node))
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_") and not (node.name.startswith("__") and node.name.endswith("__")):
                continue
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = sub.name
                    # Private helper methods on public classes are excluded; dunders included.
                    if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
                        continue
                    out.append((f"{node.name}.{name}", sub))
    return out


def _iter_annotated_params(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[tuple[str, ast.expr]]:
    """Yield (param_name, annotation) for every annotated parameter (skipping self/cls)."""
    out: list[tuple[str, ast.expr]] = []
    args = fn.args
    all_args = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    for a in all_args:
        if a.arg in ("self", "cls"):
            continue
        if a.annotation is not None:
            out.append((a.arg, a.annotation))
    if args.vararg is not None and args.vararg.annotation is not None:
        out.append((args.vararg.arg, args.vararg.annotation))
    if args.kwarg is not None and args.kwarg.annotation is not None:
        out.append((args.kwarg.arg, args.kwarg.annotation))
    return out


def _iter_public_class_attr_annotations(
    tree: ast.AST,
) -> list[tuple[str, str, ast.expr]]:
    """Yield (class_qualname, attr_name, annotation) for public class attrs."""
    out: list[tuple[str, str, ast.expr]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name.startswith("_") and not (node.name.startswith("__") and node.name.endswith("__")):
            continue
        for sub in node.body:
            if isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                name = sub.target.id
                if name.startswith("_") and not name.startswith("__"):
                    continue
                out.append((node.name, name, sub.annotation))
    return out


def _annotation_uses_any(ann: ast.expr) -> bool:
    """Return True if the annotation expression mentions ``typing.Any``.

    Catches ``Any``, ``Any | None``, ``dict[str, Any]``, ``list[Any]``,
    ``Annotated[Any, ...]``, etc. ``RunnableConfig``, ``BaseModel``, etc. do
    NOT count.

    ``Any`` appearing inside a ``Callable[[...], ...]`` subscript is exempt:
    the visible union (e.g. ``str | Callable[[Any], bool]``) is honest about
    the two-shape boundary, and the inner ``Any`` represents the user state
    value -- a §5 boundary that has no more precise type at the IR layer.
    """
    skip_nodes: set[int] = set()
    for node in ast.walk(ann):
        if isinstance(node, ast.Subscript):
            value = node.value
            is_callable = (isinstance(value, ast.Name) and value.id == "Callable") or (
                isinstance(value, ast.Attribute) and value.attr == "Callable"
            )
            if is_callable:
                for inner in ast.walk(node.slice):
                    skip_nodes.add(id(inner))
    for node in ast.walk(ann):
        if id(node) in skip_nodes:
            continue
        if isinstance(node, ast.Name) and node.id == "Any":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "Any":
            return True
    return False


# Models with arbitrary_types_allowed=True must appear here AND have an
# adjacent justification comment naming the field(s) that require it.
# Add a new entry only with sign-off — non-Pydantic-friendly types should
# usually live in a PrivateAttr sidecar (see Node._sidecar pattern).
ARBITRARY_TYPES_ALLOWLIST: frozenset[str] = frozenset(
    {
        "construct.py",
        "node.py",
        "forward.py",
        "modifiers.py",
    }
)


class TestArbitraryTypesJustified:
    """Every ``arbitrary_types_allowed=True`` in src/neograph must be justified.

    Two requirements per use:
      1. The host file is in ``ARBITRARY_TYPES_ALLOWLIST``.
      2. There is a justification comment within 3 lines above the config line
         (or trailing on the same line) starting with ``# arbitrary_types_allowed:``
         and naming the field(s) that require it.

    Habitual uses (where every field is already a Pydantic-friendly type) must
    be removed. If the only blocker is a Callable or sidecar value, prefer
    moving it to ``PrivateAttr`` (see ``Node._sidecar``).
    """

    JUSTIFICATION_PREFIX = "# arbitrary_types_allowed:"

    def test_all_uses_justified_and_allowlisted(self):
        offenders: list[str] = []
        for hit in _find_arbitrary_types_hits(SRC_DIR):
            fname, lineno, raw_line = hit
            if fname not in ARBITRARY_TYPES_ALLOWLIST:
                offenders.append(
                    f"{fname}:{lineno} uses arbitrary_types_allowed=True but file is not in ARBITRARY_TYPES_ALLOWLIST"
                )
                continue
            path = SRC_DIR / fname
            if not _has_arbitrary_types_justification(path, lineno, self.JUSTIFICATION_PREFIX):
                offenders.append(
                    f"{fname}:{lineno} has no '{self.JUSTIFICATION_PREFIX}' comment "
                    f"within 3 lines above (or trailing on the same line). Add a "
                    f"comment naming the field(s) that require arbitrary_types_allowed=True."
                )
        assert offenders == [], (
            f"\n{len(offenders)} arbitrary_types_allowed=True use(s) missing justification:\n"
            + "\n".join(f"  {o}" for o in offenders)
        )

    def test_scanner_detects_unjustified_arbitrary_types(self, tmp_path: pathlib.Path):
        """Mutation: file with arbitrary_types_allowed=True and no comment must fail."""
        synthetic = tmp_path / "construct.py"
        synthetic.write_text(
            "from pydantic import BaseModel\n"
            "class M(BaseModel):\n"
            "    model_config = {'arbitrary_types_allowed': True}\n"
        )
        hits = list(_find_arbitrary_types_hits(tmp_path))
        assert hits, "scanner failed to detect arbitrary_types_allowed=True"
        fname, lineno, _ = hits[0]
        assert not _has_arbitrary_types_justification(tmp_path / fname, lineno, "# arbitrary_types_allowed:"), (
            "scanner incorrectly accepted a missing justification"
        )

    def test_scanner_accepts_justified_use(self, tmp_path: pathlib.Path):
        """Mutation: file with a justification comment must pass."""
        synthetic = tmp_path / "node.py"
        synthetic.write_text(
            "from pydantic import BaseModel\n"
            "class M(BaseModel):\n"
            "    # arbitrary_types_allowed: 'renderer' field is a Renderer Protocol\n"
            "    model_config = {'arbitrary_types_allowed': True}\n"
        )
        hits = list(_find_arbitrary_types_hits(tmp_path))
        assert hits, "scanner failed to detect arbitrary_types_allowed=True"
        fname, lineno, _ = hits[0]
        assert _has_arbitrary_types_justification(tmp_path / fname, lineno, "# arbitrary_types_allowed:"), (
            "scanner failed to honor a present justification"
        )

    def test_slip_arbitrary_types_re(self):
        """Regex-slip: the BOTH-forms support is load-bearing. A naive
        ``arbitrary_types_allowed=True`` regex would slip the dict-literal form
        ``"arbitrary_types_allowed": True``. Prove both ``=`` and ``:`` (quoted)
        forms match, with whitespace variants, and a ``=False`` does not."""
        assert _ARBITRARY_TYPES_RE.search("model_config = ConfigDict(arbitrary_types_allowed=True)")
        assert _ARBITRARY_TYPES_RE.search('    "arbitrary_types_allowed": True,')
        assert _ARBITRARY_TYPES_RE.search("arbitrary_types_allowed = True")
        # Must NOT match the disabled form (the whole point is detecting True).
        assert not _ARBITRARY_TYPES_RE.search("arbitrary_types_allowed=False")


# Matches both ConfigDict(arbitrary_types_allowed=True) and
# {"arbitrary_types_allowed": True} forms. The optional trailing quote handles
# the dict-literal form where the key is quoted.
_ARBITRARY_TYPES_RE = re.compile(r'arbitrary_types_allowed["\']?\s*(?:=|:)\s*True')

# Dunder names embedded in an allowlist reason string (e.g. "__bool__ misuse"),
# used to cross-check that a reason naming a dunder sits in a method of that name.
_DUNDER_IN_REASON_RE = re.compile(r"__\w+__")


def _find_arbitrary_types_hits(src_dir: pathlib.Path) -> list[tuple[str, int, str]]:
    """Return (filename, lineno, raw_line) for every arbitrary_types_allowed=True use."""
    hits: list[tuple[str, int, str]] = []
    for py_file in sorted(src_dir.glob("*.py")):
        for lineno, line in enumerate(py_file.read_text().splitlines(), start=1):
            if _ARBITRARY_TYPES_RE.search(line):
                hits.append((py_file.name, lineno, line))
    return hits


def _has_arbitrary_types_justification(path: pathlib.Path, lineno: int, prefix: str) -> bool:
    """Look for `prefix` in the contiguous comment block immediately above.

    A justification comment may span multiple lines as long as those lines
    form an uninterrupted comment block ending immediately above the
    arbitrary_types_allowed=True line. Trailing comments on the same line
    are also accepted.
    """
    lines = path.read_text().splitlines()
    # Trailing inline comment on the same line.
    same_line = lines[lineno - 1]
    if prefix in same_line and "#" in same_line:
        return True
    # Walk up the contiguous comment block immediately above.
    i = lineno - 1  # 0-indexed line above
    while i >= 1:
        stripped = lines[i - 1].lstrip()
        if not stripped.startswith("#"):
            return False
        if stripped.startswith(prefix):
            return True
        i -= 1
    return False


# Stdlib exception types whose use is allowlisted at specific call sites where
# the stdlib semantics ARE the contract (Pydantic validators, parser grammar,
# attribute-protocol fallback for hasattr/getattr). Every entry must have a
# one-line reason naming why NeographError is wrong at that site.
#
# Key format: "{filename}:{lineno}" — pinned so a moved raise re-triggers
# review. Update the key when the line moves, and confirm the boundary
# reason still applies.
NEOGRAPH_ERROR_ALLOWLIST: dict[str, str] = {
    # ── conditions.py — string-grammar parser (stdlib parser contract) ──
    # parse_condition() implements a tiny expression grammar; ValueError is
    # the documented contract and tests depend on it. AttributeError raises
    # inside _resolve_field implement the Python attribute-protocol contract
    # for dotted-path lookup (caller may catch AttributeError to fall back).
    "conditions.py:66": "ValueError is the documented contract for parse_condition grammar errors; tests assert it",
    "conditions.py:81": "AttributeError is the Python attribute-protocol contract for dotted-path lookup",
    "conditions.py:86": "AttributeError is the Python attribute-protocol contract for dotted-path lookup",
    "conditions.py:74": "AttributeError is the Python attribute-protocol contract for dotted-path lookup",
    "conditions.py:121": "ValueError is the documented contract for parse_condition grammar errors; tests assert it",
    "conditions.py:110": "ValueError is the documented contract for parse_condition grammar errors; tests assert it",
    # ── construct.py — Pydantic BeforeValidator boundary ──
    # _validate_node_list runs inside Pydantic field validation; Pydantic
    # catches TypeError/ValueError and rolls them into ValidationError.
    "construct.py:46": "Pydantic BeforeValidator boundary; TypeError is rolled into ValidationError",
    "construct.py:49": "Pydantic BeforeValidator boundary; TypeError is rolled into ValidationError",
    # ── forward.py — proxy / tracer / abstract-method contracts ──
    # _Proxy.__getattr__ raises AttributeError per the Python attribute
    # protocol (hasattr depends on it). __bool__/__iter__ raise TypeError per
    # the Python protocol contract (a non-iterable used in `for` raises
    # TypeError, not NeographError). forward() raises NotImplementedError as
    # a Python abstract-method idiom.
    "forward.py:233": "NotImplementedError is the Python abstract-method idiom",
    "forward.py:262": "AttributeError is the Python attribute-protocol contract (hasattr depends on it)",
    "forward.py:291": "TypeError is the Python protocol contract for __bool__ misuse",
    "forward.py:297": "TypeError is the Python protocol contract for __iter__ misuse",
    "forward.py:324": "TypeError is the Python protocol contract for __bool__ misuse on _ConditionProxy",
    # ── modifiers.py — Pydantic field_validator + proxy attribute protocol ──
    # _PathRecorder.__getattr__ implements the attribute protocol. Pydantic
    # @field_validator boundaries catch ValueError into ValidationError.
    "modifiers.py:199": "AttributeError is the Python attribute-protocol contract (private-attr guard)",
    "modifiers.py:439": "Pydantic @field_validator boundary; ValueError is rolled into ValidationError",
    "modifiers.py:514": "Pydantic @field_validator boundary; ValueError is rolled into ValidationError",
    # ── node.py — Pydantic BeforeValidator boundary ──
    # _validate_type_spec runs inside Pydantic field validation; Pydantic
    # catches TypeError and rolls it into ValidationError.
    "node.py:109": "Pydantic BeforeValidator boundary; TypeError is rolled into ValidationError",
    "node.py:111": "Pydantic BeforeValidator boundary; TypeError is rolled into ValidationError",
    "node.py:117": "Pydantic BeforeValidator boundary; TypeError is rolled into ValidationError",
}


class TestPublicFunctionsRaiseNeographError:
    """All `raise <BareException>` in src/neograph/ must use a NeographError subclass.

    Rule: every `raise` statement in src/neograph/*.py must satisfy one of:

      - ``raise <NeographError-subclass>(...)`` (or via ``.build()``)
      - ``raise`` bare (re-raise of caught exception)
      - ``raise <NameLoaded>`` where the name binds an exception variable
        (e.g. ``raise exc``, ``raise e from None``)
      - allowlisted entry in NEOGRAPH_ERROR_ALLOWLIST, with the line documented
        as a stdlib-contract boundary

    Anything else — ``raise ValueError(...)``, ``raise TypeError(...)``,
    ``raise RuntimeError(...)`` — fails this guard. Convert to
    ``NeographError.build(...)`` (use ``ConfigurationError`` for configuration,
    ``ConstructError`` for assembly-time validation, ``ExecutionError`` for
    runtime failures, ``CompileError`` for graph-build failures).
    """

    NEOGRAPH_ERROR_NAMES = frozenset(
        {
            "NeographError",
            "ConstructError",
            "ExecutionError",
            "CompileError",
            "ConfigurationError",
            "CheckpointSchemaError",
            "StateMissingError",
            "NodeOutputError",
            "PromptVarMissing",
            "NonIdempotentReplayError",
            "ResourceExpiredError",
        }
    )

    def test_no_bare_stdlib_raises(self):
        offenders = self._scan_bare_raises(SRC_DIR, NEOGRAPH_ERROR_ALLOWLIST)
        assert offenders == [], (
            f"\n{len(offenders)} raise(s) use a stdlib exception class instead of NeographError:\n"
            + "\n".join(f"  {o}" for o in offenders)
            + "\n\nConvert each to NeographError.build(...) using the appropriate subclass "
            "(ConfigurationError, ConstructError, ExecutionError, CompileError), or add an "
            "allowlist entry with a one-line reason naming the stdlib-contract boundary."
        )

    def test_scanner_detects_injected_stdlib_raise(self, tmp_path: pathlib.Path):
        """Mutation: a file with `raise ValueError(...)` must be flagged."""
        synthetic = tmp_path / "fake_module.py"
        synthetic.write_text("def fn():\n    raise ValueError('boom')\n")
        offenders = self._scan_bare_raises(tmp_path, {})
        assert any("ValueError" in o for o in offenders), (
            f"scanner failed to detect injected ValueError; offenders={offenders}"
        )

    def test_scanner_accepts_neograph_error(self, tmp_path: pathlib.Path):
        """Mutation: a file with `raise ConstructError.build(...)` must pass."""
        synthetic = tmp_path / "fake_module.py"
        synthetic.write_text(
            "from neograph.errors import ConstructError\ndef fn():\n    raise ConstructError.build('boom')\n"
        )
        offenders = self._scan_bare_raises(tmp_path, {})
        assert offenders == [], f"scanner rejected a valid NeographError raise: {offenders}"

    def test_scanner_respects_allowlist(self, tmp_path: pathlib.Path):
        """Mutation: an allowlisted stdlib raise must pass."""
        synthetic = tmp_path / "fake_module.py"
        synthetic.write_text("def fn():\n    raise ValueError('boom')\n")
        offenders = self._scan_bare_raises(tmp_path, {"fake_module.py:2": "test boundary"})
        assert offenders == [], f"allowlist not honored: {offenders}"

    def test_allowlist_dunder_reason_matches_enclosing_method(self):
        """neograph-awor / CON-03: the pinned allowlist keys the file:line but
        historically DID NOT verify the human-written reason is TRUE of that
        line — so the __bool__/__iter__ reasons sat silently SWAPPED (line 234,
        which is __bool__, was documented as __iter__ misuse and vice versa). This
        extension closes that: for every allowlist entry whose reason names a
        dunder (``__bool__`` / ``__iter__`` / ...), the raise at that line MUST be
        lexically enclosed by a method of exactly that name. Reasons naming no
        dunder are unconstrained here.
        """
        mismatches: list[str] = []
        for key, reason in NEOGRAPH_ERROR_ALLOWLIST.items():
            dunders = _DUNDER_IN_REASON_RE.findall(reason)
            if not dunders:
                continue
            fname, _, lineno_s = key.partition(":")
            path = SRC_DIR / fname
            if not path.exists():
                mismatches.append(f"{key}: file missing")
                continue
            enclosing = _enclosing_def_name(path, int(lineno_s))
            for dunder in dunders:
                if enclosing != dunder:
                    mismatches.append(
                        f"{key}: reason names {dunder} but the raise is inside "
                        f"{enclosing!r} — swapped/stale reason string"
                    )
        assert mismatches == [], (
            "\nAllowlist reason strings disagree with the raise site's enclosing "
            "method:\n" + "\n".join(f"  {m}" for m in mismatches)
        )

    def test_meta_enclosing_def_name_resolves_method(self, tmp_path: pathlib.Path):
        """slip test: the enclosing-method resolver used above pins the line to
        its method, so a future swap of the __bool__/__iter__ reasons is caught."""
        f = tmp_path / "m.py"
        f.write_text(
            "class C:\n"
            "    def __bool__(self):\n"
            "        raise TypeError('x')\n"
            "    def __iter__(self):\n"
            "        raise TypeError('y')\n"
        )
        assert _enclosing_def_name(f, 3) == "__bool__"
        assert _enclosing_def_name(f, 5) == "__iter__"

    def test_slip_dunder_in_reason_re(self):
        """Regex-slip: the reason-vs-method cross-check only fires on reasons
        that actually name a dunder, so the extractor must match ``__bool__``
        forms and NOT plain words — else a swapped reason with no dunder token
        would be skipped, or a non-dunder reason would be falsely constrained."""
        assert _DUNDER_IN_REASON_RE.findall("TypeError for __bool__ misuse") == ["__bool__"]
        assert _DUNDER_IN_REASON_RE.findall("__bool__ and __iter__") == ["__bool__", "__iter__"]
        assert _DUNDER_IN_REASON_RE.findall("Pydantic BeforeValidator boundary") == []

    @classmethod
    def _scan_bare_raises(
        cls,
        src_dir: pathlib.Path,
        allowlist: dict[str, str],
    ) -> list[str]:
        offenders: list[str] = []
        for py_file in sorted(src_dir.glob("*.py")):
            try:
                tree = ast.parse(py_file.read_text(), filename=str(py_file))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Raise):
                    continue
                if node.exc is None:
                    # bare ``raise`` — re-raises caught exception, always OK
                    continue
                exc_name = _raised_exception_name(node.exc)
                if exc_name is None:
                    # ``raise exc`` or ``raise self.something()`` — name bound at
                    # runtime, can't statically verify; skip.
                    continue
                if exc_name in cls.NEOGRAPH_ERROR_NAMES:
                    continue
                if exc_name.startswith("_") and exc_name.endswith("Error"):
                    # ``_ExecutionError``, ``_ConfigurationError`` aliases used
                    # to break import cycles. Verify by stripping the leading
                    # underscore.
                    if exc_name[1:] in cls.NEOGRAPH_ERROR_NAMES:
                        continue
                key = f"{py_file.name}:{node.lineno}"
                if key in allowlist:
                    continue
                offenders.append(f"{key}: raise {exc_name}(...)")
        return offenders


def _enclosing_def_name(path: pathlib.Path, lineno: int) -> str | None:
    """Return the name of the innermost function/method enclosing ``lineno``.

    Used to verify that a pinned-allowlist reason string naming a dunder
    (``__bool__`` / ``__iter__``) actually sits inside a method of that name —
    the check that catches a swapped/stale reason (neograph-awor / CON-03).
    Picks the innermost (largest lineno) def whose body span contains the line.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    best: tuple[int, str] | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            if start <= lineno <= end and (best is None or start > best[0]):
                best = (start, node.name)
    return best[1] if best else None


def _raised_exception_name(exc_node: ast.expr) -> str | None:
    """Return the bare name of the raised exception class, or None if dynamic.

    Handles ``raise Foo(...)``, ``raise Foo`` (no call), and ``raise Foo.build(...)``.
    Returns None for ``raise some_var`` and ``raise mod.Foo(...)`` (attribute on a module).
    """
    # raise Foo(...) — Call with a Name as func
    if isinstance(exc_node, ast.Call):
        func = exc_node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            # raise Foo.build(...) — Attribute access; check the base name.
            base = func.value
            if isinstance(base, ast.Name):
                return base.id
            return None
        return None
    # raise Foo (no call) — Name directly
    if isinstance(exc_node, ast.Name):
        # Bare name: could be a class or a variable holding an exception instance.
        # Treat capitalized names as exception classes; lowercase as variables.
        if exc_node.id[:1].isupper():
            return exc_node.id
        return None  # ``raise exc`` — runtime-bound variable
    return None
