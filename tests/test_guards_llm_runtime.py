"""Structural guards: factory kwargs, LLM responsibility discipline, LLM
scenario/cohesion, StateBus.get discipline, runtime fan-out, normalize_ir
field writer, routing-key invariant."""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

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


class TestFactoryFunctionsTakeKwargs:
    """Factory functions must not import the removed LLM globals.

    Per docs/design/architecture-decisions.md §2: factory functions
    (`make_node_fn`, `make_subgraph_fn`, `make_oracle_*`) close over the
    `LlmRuntime` bundle passed at compile time instead of reading from
    module-level state in `_llm.py`. This guard AST-scans `factory.py`
    and `_oracle.py` for any import of the six forbidden names.
    """

    FORBIDDEN_NAMES = frozenset(
        {
            "_llm_factory",
            "_llm_factory_params",
            "_prompt_compiler",
            "_prompt_compiler_params",
            "_global_renderer",
            "_cost_callback",
            "_get_global_renderer",
        }
    )

    FACTORY_FILES = (
        "factory.py",
        "_oracle.py",
        "_dispatch.py",
        "_execute.py",
        "_state_write.py",
        "_input_shape.py",
        "_subconstruct.py",
        "_wiring.py",
    )

    def test_factory_files_do_not_import_forbidden_names(self):
        violations: list[str] = []
        for fname in self.FACTORY_FILES:
            path = SRC_DIR / fname
            if not path.exists():  # pragma: no cover
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and "neograph._llm" in node.module:
                        for alias in node.names:
                            if alias.name in self.FORBIDDEN_NAMES:
                                violations.append(f"  {fname}:{node.lineno}: from {node.module} import {alias.name}")

        assert violations == [], (
            f"\n{len(violations)} factory file(s) still import removed LLM globals:\n"
            + "\n".join(violations)
            + "\n\nClose over the LlmRuntime passed at compile time instead."
        )

    def test_factory_files_do_not_use_get_global_renderer(self):
        """No factory file should call `_get_global_renderer()` — that name is gone."""
        violations: list[str] = []
        for fname in self.FACTORY_FILES:
            path = SRC_DIR / fname
            if not path.exists():  # pragma: no cover
                continue
            text = path.read_text()
            if "_get_global_renderer" in text:
                for i, line in enumerate(text.splitlines(), 1):
                    if "_get_global_renderer" in line and not line.lstrip().startswith("#"):
                        violations.append(f"  {fname}:{i}: {line.strip()[:80]}")

        assert violations == [], (
            f"\n{len(violations)} reference(s) to `_get_global_renderer` remain:\n"
            + "\n".join(violations)
            + "\n\nUse `runtime.renderer` (closure-captured) instead."
        )

    def test_no_id_keying_in_construct_builder(self):
        """`_register_node_scripted` must NOT use `id(n)` to mint shim keys.

        `id(n)` is recycled by Python and silently shadows shims when two
        Nodes happen to land at the same memory address. Use a fresh
        `secrets.token_hex` value per shim (or a per-compile dict — which
        eliminates the collision risk entirely).
        """
        path = SRC_DIR / "_construct_builder.py"
        text = path.read_text()
        violations: list[str] = []
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            # Reject `id(n)` or `id(node)` patterns used to mint shim names.
            if "id(n)" in line or "id(node)" in line:
                violations.append(f"  _construct_builder.py:{i}: {stripped[:80]}")
        assert violations == [], (
            f"\n{len(violations)} `id()`-keyed shim name(s) remain:\n"
            + "\n".join(violations)
            + "\n\nUse a per-compile dict or `secrets.token_hex` for shim names."
        )

    def test_factory_files_do_not_reach_into_compat_slot(self):
        """No factory file should read from `_legacy_config.get_compat_runtime`.

        Factory functions must receive their runtime as a parameter, not
        consult the legacy compat slot directly. Reading from `get_compat_runtime`
        in a factory closure would re-introduce the same multi-tenant
        contamination risk the §2 work eliminated — even if the call sits
        downstream of a closure.

        Only `compile()` itself (in `compiler.py`) and the deprecated public
        wrappers (`Node.run_isolated`, `render_prompt`) may consult the slot —
        everything else should take a runtime parameter.
        """
        violations: list[str] = []
        for fname in self.FACTORY_FILES:
            path = SRC_DIR / fname
            if not path.exists():  # pragma: no cover
                continue
            text = path.read_text()
            for i, line in enumerate(text.splitlines(), 1):
                if "get_compat_runtime" in line and not line.lstrip().startswith("#"):
                    violations.append(f"  {fname}:{i}: {line.strip()[:80]}")

        assert violations == [], (
            f"\n{len(violations)} factory file(s) reach into the compat runtime "
            "slot:\n" + "\n".join(violations) + "\n\nThread the runtime as a parameter instead — the compat slot "
            "is reserved for the deprecated `configure_llm()` bridge."
        )


class TestLlmResponsibilityDiscipline:
    """neograph-8ne2 / §4: _llm.py split into responsibility-aligned modules.

    Per docs/design/architecture-decisions.md §4 (Module and function
    responsibilities): _llm.py mixed six responsibilities (cost callback,
    LLM resolution, prompt rendering, JSON parsing/retry, output-strategy
    dispatch, produce-mode orchestration). This guard pins the post-split
    layout so the god-module cannot re-form by accretion.

    Change-axis clusters (each scenario forces exactly one file):

    | Scenario                              | Owning file       |
    |---------------------------------------|-------------------|
    | Add new output_strategy               | _llm_dispatch.py  |
    | Add new JSON retry rule / repair      | _llm_retry.py     |
    | Add new prompt-render output (image,  | _llm_render.py    |
    |   streaming, etc.)                    |                   |
    | Add new produce-mode orchestration    | _llm.py           |
    |   step (cost, telemetry, LLM resolve) |                   |

    Mutation-verified: moving `_extract_json` back into `_llm.py` makes
    `test_llm_module_only_defines_allowed_names` fail naming `_extract_json`.
    """

    # Single-responsibility name allowlist per module. Names that appear here
    # MUST be defined at module top-level in the named file; names that don't
    # appear MUST NOT be top-level definitions there.
    ALLOWED_NAMES = {
        "_llm.py": frozenset(
            {
                # produce-mode orchestrator + runtime adapters it owns
                "invoke_structured",
                "_get_llm",
                "_notify_cost",
                # neograph-w74k.2.3 (Phase 1c): async twin + the pure preamble/
                # postamble helpers both orchestrators share (anti-drift extraction).
                "ainvoke_structured",
                "_prepare_structured_call",
                "_finish_structured_call",
                # neograph-15s2: native json_mode call staging bound at the shared
                # preamble (both twins inherit). Binding the provider-native
                # response_format is part of "how a produce-mode call is staged
                # (resolve LLM -> ...)" — the orchestrator's own change axis.
                "_NativeJsonModeLLM",
                "_is_response_format_rejection",
                "_ensure_json_instruction",
            }
        ),
        "_llm_dispatch.py": frozenset(
            {
                "_call_structured",
                # neograph-w74k.2.3: async twin of the strategy dispatch.
                "_acall_structured",
                # neograph-7wya: shared fail-loud for the structured path's
                # parsed=None silent variant (sync + async arms).
                "_raise_decoded_none",
                # neograph-ykun: the four dispatch fail-loud helpers, single-sited so
                # a message edit lands once across the sync/async twins.
                "_raise_markup_unrecoverable",
                "_raise_dispatch_failed",
                "_raise_unknown_strategy",
                "_raise_unexpected_variant",
                # neograph-zcxd: shared fail-loud for the structured re-prompt loop's
                # max_retries exhaustion on a validation failure (sync + async twins).
                "_raise_structured_retry_exhausted",
            }
        ),
        # neograph-ble3: DSML detection extracted to its own pure leaf module.
        "_dsml.py": frozenset(
            {
                "contains_dsml",
                "message_text",
            }
        ),
        # neograph-ble3: provider-quirk compat shim — StructuredResult tagged
        # union + Protocol-based adapter chain. New provider quirks are new
        # decorator classes HERE, never new branches in _call_structured.
        "_llm_structured_compat.py": frozenset(
            {
                "Parsed",
                "Raw",
                "Failed",
                "StructuredOutputAdapter",
                "LangChainStructuredAdapter",
                "IncludeRawCompatDecorator",
                "DsmlClassifierDecorator",
                "build_default_adapter",
                # neograph-w74k.2.3: pure classifiers shared by the sync/async adapters.
                "_classify_lc_result",
                "_reclassify_dsml",
            }
        ),
        "_llm_retry.py": frozenset(
            {
                "_extract_json",
                "_extract_balanced",
                "_is_list_annotation",
                "_apply_null_defaults",
                "_parse_json_response",
                "_build_retry_msg",
                "_invoke_json_with_retry",
                # neograph-ble3: DSML recovery (re-prompt + re-parse) is the RETRY
                # side of the DSML story; detection moved to _dsml. Renamed from
                # _attempt_dsml_recovery to recover_dsml.
                "recover_dsml",
                # neograph-w74k.2.3: async twins of the retry + DSML-recovery seams.
                "_ainvoke_json_with_retry",
                "arecover_dsml",
                # neograph-ykun: shared DSML re-prompt prep (detection + warning log +
                # targeted-retry message assembly) single-sited across the twins.
                "_dsml_recovery_messages",
                # neograph-zcxd: structured-strategy validation-retry helpers. The
                # pure repair-hint body (_repair_hint) + one-line detail digest
                # (_validation_error_details) are shared with the json_mode retry
                # builder; build_structured_repair_message / structured_retry_messages
                # are the structured-flavored re-prompt assembly the dispatch twins call.
                "_validation_error_details",
                "_repair_hint",
                "build_structured_repair_message",
                "structured_retry_messages",
                # neograph-8uoot: truncation-aware retry. _is_truncated reads the
                # provider's finish_reason/stop_reason; _build_continuation_msg is
                # the emit-only continuation directive for length-truncated
                # responses; _retry_msg_for_failure is the single-site chooser both
                # retry twins call (continuation vs generic repair hint).
                "_is_truncated",
                "_build_continuation_msg",
                "_retry_msg_for_failure",
                # stark-h46: stringly-null repair. GLM emits the STRING "null" for
                # Optional numeric/enum fields; _is_stringly_null (guarded by
                # _optional_inner_types so only nullable fields are touched)
                # normalizes the sentinel to None inside _apply_null_defaults.
                "_optional_inner_types",
                "_is_stringly_null",
                # neograph-zhwgh: shape-driven nested descent. _unwrap_optional
                # peels one Optional layer; _descend_null_defaults is the single
                # recursive classifier that reaches interiors of Optional-wrapped
                # models/lists, dict-of-models, and list-of-optional-models so
                # _apply_null_defaults no longer hand-enumerates container shapes.
                "_unwrap_optional",
                "_descend_null_defaults",
            }
        ),
        "_llm_render.py": frozenset(
            {
                "_is_inline_prompt",
                "_compile_multimodal_prompt",
                "_resolve_var",
                "_resolve_var_raw",
                # shared ${path} walker that _resolve_var (BAML-rendered) and
                # _resolve_var_raw (verbatim) both thin onto — the dedup'd spine,
                # not a second render path.
                "_walk_var_path",
                "_substitute_vars",
                "_compile_prompt",
                "render_prompt",
                # neograph-v569: the public standalone compile_prompt (eval-parity)
                # and its shared render-then-compile core + variant-source resolver.
                # Squarely this module's change axis ("how a prompt template is
                # turned into a message list") — the SAME seam render_prompt and the
                # runtime ThinkDispatch use, promoted to a public function. NOT a
                # second rendering/compile path (the anti-duplication invariant).
                "compile_prompt",
                "_render_and_compile",
                "_resolve_variant_compiler",
            }
        ),
    }

    # Coarse line-count budget. Not the load-bearing assertion (the name set
    # is); a proxy that catches accretion that escapes name-level review.
    # neograph-w74k.2.3 (Phase 1c): budgets raised to accommodate the async
    # twins across the LLM vertical. The twins are thin over shared pure
    # preamble/postamble/classify helpers (anti-drift), but adding an awaiting
    # mirror of each orchestrator is a real, reviewed line increase — not
    # accretion. The name-set assertion remains the load-bearing guard.
    LINE_BUDGETS = {
        # neograph-ykun: 290 -> 265. The async twins route through the shared
        # pure preamble/postamble; the raised budget is no longer needed.
        # neograph-15s2: 265 -> 385. Native json_mode call staging landed at the
        # shared preamble (attempt-bind-and-fall-back wrapper with sync+async
        # entrypoints + json-word guard + response_format rejection predicate).
        # The load-bearing assertion is the name allowlist above; this coarse
        # proxy is widened for the new, reviewed names.
        # neograph-dyy7: 400 -> 410. _notify_cost's arity probe changed from a
        # try/except-retry (which double-counted cost on a body TypeError) to
        # single-invocation _accepted_params introspection; correctness fix, +12
        # lines. The load-bearing check is the name allowlist above.
        "_llm.py": 410,  # 408 actual
        # neograph-ble3: tightened 130 -> 115. The 5-path include_raw try/except
        # ladder collapsed to a match on the StructuredResult variant; the
        # provider-quirk wiring moved to the compat shim. Locks the deletion.
        # neograph-ykun: the four ExecutionError builders are now single-site
        # module-level fns; the twin match arms collapsed to one-line raises.
        # neograph-zcxd: 199 -> 250. The structured branch of each twin gained a
        # bounded validation re-prompt loop (parity with json_mode) plus a shared
        # exhaustion-raise helper. Reviewed increase; the re-prompt message
        # assembly is single-sited in _llm_retry.structured_retry_messages.
        "_llm_dispatch.py": 285,  # m0tv format rewrap (273 actual)
        # neograph-ble3: tightened 365 -> 360. _DSML_PATTERN regex moved to
        # _dsml.py; recover_dsml is detection-free. Locks the deletion.
        # neograph-s1u4: 360 -> 375. _apply_null_defaults gained a guarded
        # default_factory coercion branch (a real fix, not accretion).
        # neograph-ykun: 480 -> 460. The usage-dict shape moved to _usage._usage_dict
        # and the DSML re-prompt prep is single-sited across the retry twins.
        # neograph-zcxd: 460 -> 510. The json_mode retry builder was refactored
        # into a shared pure repair-hint body (_repair_hint + _validation_error_details)
        # that the new structured re-prompt builders (build_structured_repair_message,
        # structured_retry_messages) reuse. Reviewed increase — shared, not duplicated.
        # neograph-8uoot: 510 -> 565. repair_json is now guarded (its blowups
        # become ExecutionError instead of escaping the retry loop) and
        # length-truncated responses get a continuation re-prompt via the
        # shared _retry_msg_for_failure chooser. Reviewed increase.
        # stark-h46: 565 -> 615. Stringly-null repair (_is_stringly_null +
        # _optional_inner_types) so the STRING "null" GLM emits for Optional
        # numeric/enum fields is normalized to None inside _apply_null_defaults
        # instead of crashing the node. Reviewed increase.
        # neograph-zhwgh: 615 -> 665. Shape-driven nested descent
        # (_unwrap_optional + _descend_null_defaults) replaces the 0.7.2
        # hand-enumerated (bare-model, bare-list-of-model) descent that silently
        # skipped every container shape it did not spell out -- Optional-wrapped
        # models/lists, dict-of-models, list-of-optional-models. One recursive
        # classifier now reaches every leaf model dict at any depth. Reviewed
        # increase (net: kills the whole missing-shape bug family, not one case).
        "_llm_retry.py": 665,
        # neograph-v569: 310 -> 445. The public standalone compile_prompt landed
        # here (its change axis) with a thorough public docstring, a shared
        # render-then-compile core (_render_and_compile, which render_prompt now
        # also routes through — a net dedup), and the variant-source resolver. The
        # load-bearing assertion is the name allowlist above; this proxy is widened
        # for the reviewed new names.
        "_llm_render.py": 445,
        # neograph-ble3: new pure-leaf detection module.
        "_dsml.py": 55,
        # neograph-ble3: compat shim — sum-type + Protocol + 3 adapters + factory.
        # neograph-zcxd: 220 -> 235. _classify_lc_result now discriminates a
        # ValidationError parsing_error (weak decode -> Failed, retryable) from the
        # silent parsed=None (stays Raw); IncludeRawCompat also catches ValidationError.
        "_llm_structured_compat.py": 235,
    }

    def _top_level_defs(self, path: pathlib.Path) -> set[str]:
        """Collect top-level function / class names. Module-level data
        assignments (regex constants, logger handles, sentinels) are not
        responsibilities — they're configuration that supports the named
        functions and may live anywhere those functions need them.
        """
        tree = ast.parse(path.read_text())
        names: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names.add(node.name)
        return names

    def test_all_split_modules_exist(self):
        missing = [name for name in self.ALLOWED_NAMES if not (SRC_DIR / name).exists()]
        assert not missing, (
            f"Split modules do not exist yet: {missing}. "
            "neograph-8ne2 requires each responsibility module to be created."
        )

    def test_llm_module_only_defines_allowed_names(self):
        for fname, allowed in self.ALLOWED_NAMES.items():
            path = SRC_DIR / fname
            if not path.exists():
                pytest.fail(f"{fname} does not exist (neograph-8ne2 split not complete).")
            top = self._top_level_defs(path)
            extra = top - allowed
            missing = allowed - top
            assert not extra, (
                f"{fname} defines unexpected top-level names: {sorted(extra)}. "
                f"Allowed: {sorted(allowed)}. "
                "Move them to their change-axis cluster module."
            )
            assert not missing, (
                f"{fname} is missing required top-level names: {sorted(missing)}. Required: {sorted(allowed)}."
            )

    def test_line_count_budgets(self):
        violations: list[str] = []
        for fname, budget in self.LINE_BUDGETS.items():
            path = SRC_DIR / fname
            if not path.exists():
                violations.append(f"  {fname}: file missing")
                continue
            n = len(path.read_text().splitlines())
            if n > budget:
                violations.append(f"  {fname}: {n} lines (budget {budget})")
        assert not violations, "Line-count budgets exceeded — coarse proxy for accretion:\n" + "\n".join(violations)

    def test_no_cycles_among_split_modules(self):
        """The five _llm* modules must form a DAG.

        Allowed edges (a -> b means a imports from b):
            _llm.py        -> _llm_dispatch, _llm_render, _llm_config,
                              _llm_protocols, _llm_runtime
            _llm_dispatch  -> _llm_retry, _llm_structured_compat
            _llm_structured_compat -> _dsml (pure leaf, not _llm*-prefixed)
            _llm_retry     -> _dsml (no other _llm* deps except possibly _llm_config)
            _llm_render    -> _llm_config, _llm_protocols, _llm_runtime
            _llm_protocols -> (no _llm* deps)
            _llm_runtime   -> (no _llm* deps; uses TYPE_CHECKING for protocols)
            _llm_config    -> (no _llm* deps)

        neograph-ble3: _dsml.py is a pure leaf (no neograph._llm* imports) so it
        cannot participate in any cycle; it is intentionally omitted from the
        _llm*-prefixed collector below. _llm_structured_compat.py IS tracked.
        """
        modules = [
            "_llm.py",
            "_llm_dispatch.py",
            "_llm_structured_compat.py",
            "_llm_retry.py",
            "_llm_render.py",
            "_llm_protocols.py",
            "_llm_runtime.py",
            "_llm_config.py",
        ]

        def _collect_runtime_imports(tree: ast.AST) -> set[str]:
            """Walk imports, skipping anything inside an `if TYPE_CHECKING:` block."""
            tc_blocks: list[ast.If] = []
            for n in ast.walk(tree):
                if isinstance(n, ast.If):
                    test = n.test
                    is_tc = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
                    )
                    if is_tc:
                        tc_blocks.append(n)

            def in_tc(node: ast.AST) -> bool:
                for blk in tc_blocks:
                    for sub in ast.walk(blk):
                        if sub is node:
                            return True
                return False

            deps: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if not node.module.startswith("neograph._llm"):
                        continue
                    if in_tc(node):
                        continue
                    leaf = node.module.split(".")[-1] + ".py"
                    deps.add(leaf)
            return deps

        graph: dict[str, set[str]] = {}
        for fname in modules:
            path = SRC_DIR / fname
            if not path.exists():
                continue
            tree = ast.parse(path.read_text())
            deps = {d for d in _collect_runtime_imports(tree) if d in modules and d != fname}
            graph[fname] = deps

        # Cycle detection via DFS
        WHITE, GRAY, BLACK = 0, 1, 2
        color = dict.fromkeys(graph, WHITE)
        cycle: list[str] = []

        def dfs(node: str, path: list[str]) -> bool:
            color[node] = GRAY
            for dep in graph.get(node, set()):
                if dep not in color:
                    continue
                if color[dep] == GRAY:
                    idx = path.index(dep) if dep in path else 0
                    cycle.extend(path[idx:] + [dep])
                    return True
                if color[dep] == WHITE and dfs(dep, path + [dep]):
                    return True
            color[node] = BLACK
            return False

        for m in graph:
            if color[m] == WHITE and dfs(m, [m]):
                break
        assert not cycle, f"Cycle detected among _llm* modules: {' -> '.join(cycle)}"

    def test_no_function_local_llm_split_imports(self):
        """No spin-off module may use function-local `from neograph._llm*` imports.

        Mutation-verified: adding `from neograph._llm_retry import x` inside
        a function in _llm_dispatch.py makes this guard fire — symptomatic
        of a hidden cycle the layout was supposed to prevent.
        """
        violations: list[str] = []
        for fname in (
            "_llm_dispatch.py",
            "_llm_retry.py",
            "_llm_render.py",
            "_llm_structured_compat.py",  # neograph-ble3
        ):
            path = SRC_DIR / fname
            if not path.exists():
                continue
            tree = ast.parse(path.read_text())
            for func in ast.walk(tree):
                if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                for sub in ast.walk(func):
                    if isinstance(sub, ast.ImportFrom) and sub is not func:
                        if sub.module and sub.module.startswith("neograph._llm"):
                            # The walk above includes the function's own toplevel
                            # imports; filter to ones inside the body specifically.
                            if sub in func.body:
                                continue  # toplevel of function body (rare)
                            violations.append(f"  {fname}:{sub.lineno}: from {sub.module} import ...")
        assert not violations, (
            "Function-local `from neograph._llm*` import detected in spin-off module:\n"
            + "\n".join(violations)
            + "\n\nResolve the underlying cycle by reshaping the layout, "
            "not by hiding the import inside a function body."
        )


class TestLlmScenarioTouchpoints:
    """neograph-8ne2 / §4: Cohesion property test — scenario walkthrough.

    Architectural teeth for the responsibility-cluster split: each scenario
    enumerates which files MUST change and which files MAY change. If a
    future change forces a file outside the may_touch set, the cluster has
    leaked responsibility.

    The point is not to forbid touching other files (tests, importers will
    always need updates for renames). The point is to bound the *source*
    files implementing the new behaviour.

    Mutation-verified by `test_mutation_excess_scenario_touchpoints`.
    """

    SCENARIO_TOUCHPOINTS = {
        # "Add new output_strategy (e.g., dsml, function_calling)"
        # → strategy dispatch only.
        "add_new_output_strategy": {
            "must_touch": {"_llm_dispatch.py"},
            "may_touch": {"_llm_config.py"},
            "max_touch": 2,
        },
        # "Add new retry rule (e.g., schema mismatch → reformat)"
        # → retry module only.
        "add_new_retry_rule": {
            "must_touch": {"_llm_retry.py"},
            "may_touch": {"_llm_config.py"},
            "max_touch": 2,
        },
        # "Add new prompt rendering output (e.g., audio block, streaming)"
        # → render module only.
        "add_new_prompt_render_output": {
            "must_touch": {"_llm_render.py"},
            "may_touch": set(),
            "max_touch": 1,
        },
        # "Add new produce-mode orchestration step (e.g., pre-call hook)"
        # → orchestrator only.
        "add_new_orchestration_step": {
            "must_touch": {"_llm.py"},
            "may_touch": set(),
            "max_touch": 1,
        },
        # neograph-ble3: "Add new provider quirk (e.g., provider returns empty
        # string on overload)" → a new StructuredOutputAdapter decorator in the
        # compat shim ONLY. This is the Open/Closed proof: a new quirk is a new
        # decorator class, never a new branch in _call_structured.
        "add_new_provider_quirk": {
            "must_touch": {"_llm_structured_compat.py"},
            "may_touch": set(),
            "max_touch": 1,
        },
    }

    def test_all_must_touch_files_exist(self):
        missing: list[str] = []
        for scenario, spec in self.SCENARIO_TOUCHPOINTS.items():
            for fname in spec["must_touch"] | spec["may_touch"]:
                if not (SRC_DIR / fname).exists():
                    missing.append(f"  {scenario}: {fname} does not exist")
        assert not missing, "\n".join(missing)

    def test_max_touch_bounded(self):
        for scenario, spec in self.SCENARIO_TOUCHPOINTS.items():
            total = len(spec["must_touch"]) + len(spec["may_touch"])
            assert total <= spec["max_touch"], f"Scenario '{scenario}' touches {total} files; max {spec['max_touch']}."

    def test_mutation_excess_scenario_touchpoints(self):
        """Mutation case — synthesize an oversized scenario, scanner detects."""
        bad = {
            "must_touch": {"a.py", "b.py", "c.py"},
            "may_touch": set(),
            "max_touch": 1,
        }
        total = len(bad["must_touch"]) + len(bad["may_touch"])
        assert total > bad["max_touch"]


class TestLlmCohesionFanOut:
    """neograph-8ne2 / §4: Cohesion fan-out for the _llm* spin-off modules.

    For each spin-off module, count distinct external src/neograph modules
    that import from it. Spin-off modules are leaf-ish helpers; they should
    have few importers. A bloated import set signals the spin-off has
    become a kitchen sink.

    Mutation-verified: synthesizing 4 importers of a fictitious leaf module
    exceeds a ceiling of 2.
    """

    FAN_OUT_CEILING = {
        # _llm_dispatch is a leaf strategy table; called by _llm + _tool_loop.
        "_llm_dispatch.py": 3,
        # _llm_retry is a leaf parser; called by _llm_dispatch + _tool_loop.
        "_llm_retry.py": 4,
        # _llm_render owns prompt compilation + public render_prompt;
        # called by _llm, _tool_loop, _dispatch (for _is_inline_prompt),
        # and re-exported via __init__.
        "_llm_render.py": 5,
        # neograph-ble3: _dsml is the shared detection leaf — imported by
        # _llm_dispatch, _llm_structured_compat, _llm_retry, _tool_loop.
        "_dsml.py": 5,
        # neograph-ble3: compat shim, called only by the dispatch table.
        "_llm_structured_compat.py": 2,
    }

    def _count_importers(self, target_module_basename: str) -> list[str]:
        target = f"neograph.{target_module_basename[:-3]}"
        importers: list[str] = []
        for py in sorted(SRC_DIR.glob("*.py")):
            if py.name == target_module_basename:
                continue
            text = py.read_text()
            if f"from {target} " in text or f"from {target}\n" in text or f"import {target}\n" in text:
                importers.append(py.name)
        return importers

    def test_fan_out_under_ceiling(self):
        violations: list[str] = []
        for mod, ceiling in self.FAN_OUT_CEILING.items():
            if not (SRC_DIR / mod).exists():
                continue
            importers = self._count_importers(mod)
            if len(importers) > ceiling:
                violations.append(f"  {mod}: {len(importers)} importers (ceiling {ceiling}): {importers}")
        assert not violations, "Spin-off module fan-out exceeded ceiling:\n" + "\n".join(violations)

    def test_mutation_excess_importers_detected(self, tmp_path):
        for i in range(4):
            (tmp_path / f"client_{i}.py").write_text("from neograph.leaf import x\n")
        count = sum(1 for p in tmp_path.glob("client_*.py") if "from neograph.leaf " in p.read_text())
        assert count > 2


class TestStateBusGetDiscipline:
    """Every .get(...) call on a StateBus receiver in src/neograph/ must
    either use ``.get_required(...)`` (§7 fail-loud) OR carry a
    ``# StateBus.get optional: <reason>`` justification comment on the
    same line or the immediately-preceding line.

    Per neograph-tzzi (izo1-D). Mutation tests verify the scanner.
    """

    # Filenames whose StateBus.get sites have been audited (izo1-B). The
    # scanner only enforces discipline on these files; new modules can be
    # added once their sites are classified per the audit doc.
    IN_SCOPE = frozenset(
        {
            "_input_shape.py",
            "_execute.py",
            "_oracle.py",
            "_subconstruct.py",
            "_wiring.py",
            "_state_write.py",
        }
    )

    # Variable names that we treat as "this expression is a StateBus".
    BUS_NAMES = frozenset({"bus", "state", "_state"})

    OPTIONAL_TAG = "StateBus.get optional:"

    @classmethod
    def _scan(cls, src_dir: pathlib.Path) -> list[str]:
        offenders: list[str] = []
        for py_file in sorted(src_dir.glob("*.py")):
            if py_file.name not in cls.IN_SCOPE:
                continue
            text = py_file.read_text()
            lines = text.splitlines()
            try:
                tree = ast.parse(text, filename=str(py_file))
            except SyntaxError:  # pragma: no cover — file must parse
                continue
            for ast_node in ast.walk(tree):
                if not isinstance(ast_node, ast.Call):
                    continue
                func = ast_node.func
                if not isinstance(func, ast.Attribute):
                    continue
                if func.attr != "get":
                    continue
                value = func.value
                # Receiver must be a bare Name matching the StateBus naming
                # convention used in the codebase.
                if not isinstance(value, ast.Name):
                    continue
                if value.id not in cls.BUS_NAMES:
                    continue
                lineno = ast_node.lineno
                if 0 < lineno <= len(lines) and cls.OPTIONAL_TAG in lines[lineno - 1]:
                    continue
                annotated = False
                cursor = lineno - 2
                while cursor >= 0 and lines[cursor].lstrip().startswith("#"):
                    if cls.OPTIONAL_TAG in lines[cursor]:
                        annotated = True
                        break
                    cursor -= 1
                if annotated:
                    continue
                offenders.append(f"{py_file.name}:{lineno}")
        return offenders

    def test_every_state_bus_get_is_annotated(self):
        offenders = self._scan(SRC_DIR)
        assert offenders == [], (
            f"\n{len(offenders)} unannotated StateBus.get site(s) found:\n"
            + "\n".join(f"  {o}" for o in offenders)
            + "\n\nEither switch to .get_required(...) (§7 required-getter) "
            "or add a `# StateBus.get optional: <reason>` comment on the "
            "call site or the immediately-preceding line."
        )

    def test_mutation_unannotated_get_detected(self, tmp_path: pathlib.Path):
        """Inject an unannotated `bus.get(k)` into a scoped temp file;
        scanner must flag it."""
        target = tmp_path / "_input_shape.py"
        target.write_text("def f(bus):\n    return bus.get('k')\n")
        offenders = self._scan(tmp_path)
        assert any(o.endswith(":2") for o in offenders), offenders

    def test_mutation_annotated_get_passes(self, tmp_path: pathlib.Path):
        """`bus.get(k, None)  # StateBus.get optional: test` → scanner skips."""
        target = tmp_path / "_input_shape.py"
        target.write_text("def f(bus):\n    return bus.get('k', None)  # StateBus.get optional: test reason\n")
        offenders = self._scan(tmp_path)
        assert offenders == [], offenders

    def test_mutation_get_required_passes(self, tmp_path: pathlib.Path):
        """`bus.get_required(k)` (not `.get`) is never flagged."""
        target = tmp_path / "_input_shape.py"
        target.write_text("def f(bus):\n    return bus.get_required('k')\n")
        offenders = self._scan(tmp_path)
        assert offenders == [], offenders

    def test_mutation_out_of_scope_file_skipped(self, tmp_path: pathlib.Path):
        """A file outside IN_SCOPE is not scanned."""
        target = tmp_path / "not_audited.py"
        target.write_text("def f(bus):\n    return bus.get('k')\n")
        offenders = self._scan(tmp_path)
        assert offenders == [], offenders


class TestNoRawStateAccessInRoutingModules:
    """ARCH-2 (neograph-aiiz): the routing/boundary modules must read pipeline
    state ONLY through the StateBus protocol. Three raw forms are structurally
    forbidden on a bare ``state`` name:

      1. ``getattr(state, ...)``               -- use ``bus.get`` / ``get_required``
      2. ``state.__class__.model_fields`` (any ``state.__class__`` access)
      3. ``state[...]`` subscript              -- use ``bus.get``

    The full-state snapshot lives in exactly one helper (``snapshot_state`` in
    ``_state_bus.py``, which goes through ``bus.keys()``/``bus.get``); the
    Each-router navigation+dedup lives in exactly one helper
    (``_collect_each_items`` in ``_wiring.py``). Neither re-derives raw access.

    Scope is the four routing modules ONLY -- ``_state_bus.py`` is the StateBus
    implementation and legitimately uses ``getattr(self._state, ...)`` and
    ``self._state.__class__.model_fields``; it is NOT in scope. ``adapt_state(state)``
    call-args are not a banned form (``state`` is an argument, not a receiver).

    Value-navigation (``getattr(item, key)``, ``getattr(value, attr)``) is
    excluded because the scanner keys on ``Name(id='state')`` only.
    """

    IN_SCOPE = frozenset(
        {
            "_wiring.py",
            "_oracle.py",
            "_subconstruct.py",
            "_state_write.py",
        }
    )

    @classmethod
    def _scan(cls, src_dir: pathlib.Path) -> list[str]:
        offenders: list[str] = []
        for py_file in sorted(src_dir.glob("*.py")):
            if py_file.name not in cls.IN_SCOPE:
                continue
            text = py_file.read_text()
            try:
                tree = ast.parse(text, filename=str(py_file))
            except SyntaxError:  # pragma: no cover — file must parse
                continue
            for ast_node in ast.walk(tree):
                # Form 1: getattr(state, ...)
                if (
                    isinstance(ast_node, ast.Call)
                    and isinstance(ast_node.func, ast.Name)
                    and ast_node.func.id == "getattr"
                    and ast_node.args
                    and isinstance(ast_node.args[0], ast.Name)
                    and ast_node.args[0].id == "state"
                ):
                    offenders.append(f"{py_file.name}:{ast_node.lineno} getattr(state,...)")
                # Form 2: state.__class__ (covers state.__class__.model_fields)
                if (
                    isinstance(ast_node, ast.Attribute)
                    and ast_node.attr == "__class__"
                    and isinstance(ast_node.value, ast.Name)
                    and ast_node.value.id == "state"
                ):
                    offenders.append(f"{py_file.name}:{ast_node.lineno} state.__class__")
                # Form 3: state[...] subscript
                if (
                    isinstance(ast_node, ast.Subscript)
                    and isinstance(ast_node.value, ast.Name)
                    and ast_node.value.id == "state"
                ):
                    offenders.append(f"{py_file.name}:{ast_node.lineno} state[...]")
        return offenders

    def test_no_raw_state_access_in_routing_modules(self):
        offenders = self._scan(SRC_DIR)
        assert offenders == [], (
            f"\n{len(offenders)} raw-state-access site(s) found in routing modules:\n"
            + "\n".join(f"  {o}" for o in offenders)
            + "\n\nRoute reads through StateBus: adapt_state(state) then bus.get / "
            "get_required / get_counter / keys. Use snapshot_state(bus) for the "
            "full snapshot and _collect_each_items(bus, each, fan_out=...) for "
            "Each-router navigation+dedup."
        )

    def test_mutation_getattr_state_detected(self, tmp_path: pathlib.Path):
        """A probe ``getattr(state, 'x')`` in a scoped temp file is flagged."""
        (tmp_path / "_wiring.py").write_text("def f(state):\n    return getattr(state, 'x')\n")
        offenders = self._scan(tmp_path)
        assert any("getattr(state,...)" in o for o in offenders), offenders

    def test_mutation_model_fields_enumeration_detected(self, tmp_path: pathlib.Path):
        """The regex-slip case: ``state.__class__.model_fields`` enumeration."""
        (tmp_path / "_oracle.py").write_text("def f(state):\n    return {k: 1 for k in state.__class__.model_fields}\n")
        offenders = self._scan(tmp_path)
        assert any("state.__class__" in o for o in offenders), offenders

    def test_mutation_subscript_detected(self, tmp_path: pathlib.Path):
        """A raw ``state[root]`` subscript is flagged."""
        (tmp_path / "_state_write.py").write_text("def f(state):\n    return state['root']\n")
        offenders = self._scan(tmp_path)
        assert any("state[...]" in o for o in offenders), offenders

    def test_mutation_statebus_calls_pass(self, tmp_path: pathlib.Path):
        """StateBus method calls and adapt_state(state) args are NOT flagged."""
        (tmp_path / "_subconstruct.py").write_text(
            "def f(state):\n"
            "    bus = adapt_state(state)\n"
            "    a = bus.get('k')\n"
            "    b = bus.get_required('k')\n"
            "    c = getattr(item, 'key', None)\n"
            "    return a, b, c\n"
        )
        offenders = self._scan(tmp_path)
        assert offenders == [], offenders

    def test_mutation_out_of_scope_file_skipped(self, tmp_path: pathlib.Path):
        """``_state_bus.py`` (the StateBus impl) is out of scope and skipped."""
        (tmp_path / "_state_bus.py").write_text("def f(state):\n    return getattr(state, 'x')\n")
        offenders = self._scan(tmp_path)
        assert offenders == [], offenders


class TestNoRuntimeFanOutDetection:
    """neograph-vgc1: ``neograph._ir_normalize.normalize_ir`` is the single
    source of truth for ``node.fan_out_param``. The runtime extractor at
    ``_input_shape._extract_fan_in_dict`` must NOT detect fan-out via
    ``state.keys()`` scanning — that band-aid existed during the izo1-B
    migration and was removed once the construct-time normalization landed.
    Reintroducing it splits the rule across two layers and re-creates the
    drift class (cf. neograph-8k3, neograph-ayq).
    """

    def test_input_shape_does_not_call_state_keys_for_fan_out_detection(self):
        source = (SRC_DIR / "_input_shape.py").read_text()
        tree = ast.parse(source)

        # Find _extract_fan_in_dict body.
        target_fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_extract_fan_in_dict":
                target_fn = node
                break
        assert target_fn is not None, "_extract_fan_in_dict must exist"

        # Within its body, no `state.keys()` call.
        for sub in ast.walk(target_fn):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "keys"
                and isinstance(sub.func.value, ast.Name)
                and sub.func.value.id in ("state", "bus")
            ):
                raise AssertionError(
                    f"_extract_fan_in_dict re-introduced state.keys() scan at "
                    f"line {sub.lineno}; fan-out detection belongs in "
                    f"neograph._ir_normalize.normalize_ir (single source of truth, "
                    f"neograph-vgc1)."
                )


class TestNormalizeIrIsSoleIrFieldWriter:
    """neograph-20xq: IR-level inferred fields (fan_out_param, oracle_gen_type)
    are written at exactly one ASSEMBLY-TIME site — neograph._ir_normalize,
    called once from Construct.__init__. This replaces the older
    TestConstructNormalizesEveryAtNodeOnlyIRField, which only policed
    _construct_builder's `updates` dict and required a parallel
    Construct._normalize_<field> method per field (the very drift mechanism
    the epic removed).

    The invariant is structural, not a snapshot: rather than enumerate the
    expected writes, the guard scans ALL of src/neograph for EVERY syntactic
    form that writes one of the IR fields — attribute assignment, subscript
    assignment, a key in a model_copy/dict-literal update, and
    setattr/object.__setattr__ — and asserts the (file, field) write set
    equals a tightly-justified allowlist:

      _ir_normalize.py  → {fan_out_param, oracle_gen_type}  (the canonical site;
            normalize_ir is the SOLE writer of fan_out_param after neograph-k7bg)
      decorators.py → {oracle_gen_type}   (@node DECORATION-time eager
            pre-population so a bare Node carries the type before assembly;
            normalize_ir owns the inference rule via oracle_gen_type_for and
            is idempotent over it)

    The remaining pre-population site (decorators.py) runs BEFORE the Construct
    object exists. Any other
    (file, field) pair — e.g. re-adding oracle_gen_type inference to
    _construct_builder, or a new Construct._normalize_* method — fails the
    guard. So does any write of a NEW IR field anywhere but _ir_normalize.
    """

    # IR-level inferred fields owned by normalize_ir.
    IR_FIELDS = frozenset({"fan_out_param", "oracle_gen_type"})

    # Sanctioned (file, field) pre-population writes outside _ir_normalize.
    # After neograph-k7bg, _construct_builder no longer writes fan_out_param —
    # the normalizer is its sole writer. Only the @node decoration-time eager
    # oracle_gen_type write remains.
    ALLOWED_PREPOP: dict[str, frozenset[str]] = {
        "decorators.py": frozenset({"oracle_gen_type"}),
    }

    @classmethod
    def _scan_ir_field_writes(cls, tree: ast.AST) -> set[str]:
        """Return the set of IR field names WRITTEN anywhere in ``tree``.

        Detects every write form for an IR field on an *instance*:
          - attribute-assign            ``x.field = ...``
          - subscript-assign            ``d["field"] = ...``
          - augmented-assign            ``x.field += ...`` / ``d["field"] += ...``
          - annotated-assign            ``x.field: T = ...`` / ``d["field"]: T = ...``
          - tuple/list-unpack targets   ``x.field, y = a, b``
          - dict-literal keys           ``{"field": ...}`` (covers
            ``model_copy(update={"field": ...})`` and a normalizer's
            ``return {"field": ...}``)
          - dict() keyword args         ``dict(field=...)`` (the kwarg form of
            the same ``model_copy(update=...)`` idiom)
          - ``setattr``/``object.__setattr__`` string-literal targets

        Deliberately NOT matched (not writes to an instance field):
          - bare-``Name`` annotated/plain assignments — these are class-body
            field DECLARATIONS (``fan_out_param: str | None = None`` in
            ``node.py``) or module constants, not instance writes.
          - read forms — ``getattr(x, "field")``, ``x.field`` access.
        """
        written: set[str] = set()

        def _str_const(node: ast.AST) -> str | None:
            return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None

        def _check_target(target: ast.AST) -> None:
            """Record an assignment target if it writes an IR field on an
            instance. Recurses tuple/list unpacking. Bare ``Name`` targets are
            ignored (declaration/constant, not an instance write)."""
            if isinstance(target, ast.Attribute):
                if target.attr in cls.IR_FIELDS:
                    written.add(target.attr)
            elif isinstance(target, ast.Subscript):
                key = _str_const(target.slice)
                if key in cls.IR_FIELDS:
                    written.add(key)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    _check_target(elt)

        for node in ast.walk(tree):
            # x.field = ... | d["field"] = ... | a, b = ... (tuple targets)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    _check_target(target)
            # x.field += ... | d["field"] += ...
            elif isinstance(node, ast.AugAssign):
                _check_target(node.target)
            # x.field: T = ... | d["field"]: T = ...  (Name targets ignored above)
            elif isinstance(node, ast.AnnAssign):
                _check_target(node.target)
            # {"field": ...} dict literals (model_copy update=, normalizer return)
            elif isinstance(node, ast.Dict):
                for key_node in node.keys:
                    if key_node is not None:
                        key = _str_const(key_node)
                        if key in cls.IR_FIELDS:
                            written.add(key)
            elif isinstance(node, ast.Call):
                # setattr(x, "field", ...) / object.__setattr__(x, "field", ...)
                if (isinstance(node.func, ast.Attribute) and node.func.attr == "__setattr__") or (
                    isinstance(node.func, ast.Name) and node.func.id == "setattr"
                ):
                    for arg in node.args:
                        key = _str_const(arg)
                        if key in cls.IR_FIELDS:
                            written.add(key)
                # dict(field=...) — kwarg form of model_copy(update=dict(...))
                elif isinstance(node.func, ast.Name) and node.func.id == "dict":
                    for kw in node.keywords:
                        if kw.arg in cls.IR_FIELDS:
                            written.add(kw.arg)
        return written

    def test_ir_fields_written_only_in_normalize_ir_and_sanctioned_prepop(self):
        offenders: list[str] = []
        for py_file in sorted(SRC_DIR.glob("*.py")):
            written = self._scan_ir_field_writes(ast.parse(py_file.read_text()))
            if not written:
                continue
            name = py_file.name
            if name == "_ir_normalize.py":
                continue  # the canonical site may write any IR field
            allowed = self.ALLOWED_PREPOP.get(name, frozenset())
            unexpected = written - allowed
            for field in sorted(unexpected):
                offenders.append(f"{name}: writes IR field '{field}'")

        assert offenders == [], (
            "IR-level inferred fields must be written only by "
            "neograph._ir_normalize (the single assembly-time normalization "
            "site) or by a sanctioned @node pre-population site listed in "
            "ALLOWED_PREPOP.\nUnsanctioned writes:\n"
            + "\n".join(f"  {o}" for o in offenders)
            + "\n\nIf this is a new IR inference, add it as an IrNormalizer in "
            "_ir_normalize.py — do NOT inline it in an assembly path (that "
            "re-creates the drift class of neograph-vgc1/aqau/20xq)."
        )

    # Named so the regex carries a slip meta-test (PROC-2). Matches the exact
    # set of removed per-field Construct methods.
    _NORMALIZE_FIELD_RE = re.compile(r"_normalize_(fan_out_params|oracle_gen_type)")

    def test_construct_has_no_normalize_field_methods(self):
        """The per-field Construct._normalize_<field> methods are GONE — the
        whole point of the epic. A new one would re-introduce parallel
        inference. Guards against regression to the old shape."""
        construct_src = (SRC_DIR / "construct.py").read_text()
        tree = ast.parse(construct_src)
        offenders = [
            n.name
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and self._NORMALIZE_FIELD_RE.fullmatch(n.name)
        ]
        assert offenders == [], (
            f"construct.py defines {offenders}; IR-field inference belongs in "
            f"_ir_normalize.IrNormalizer implementations, not per-field methods "
            f"on Construct (neograph-20xq)."
        )

    def test_slip_normalize_field_re(self):
        """Regex-slip: fullmatch (not search) is what makes this precise — a
        method merely CONTAINING the name (e.g. a helper or a longer name) must
        NOT be flagged, while the exact removed names must be."""
        assert self._NORMALIZE_FIELD_RE.fullmatch("_normalize_fan_out_params")
        assert self._NORMALIZE_FIELD_RE.fullmatch("_normalize_oracle_gen_type")
        # Slip guard: substring/longer forms would match re.search but NOT
        # fullmatch — proving the fullmatch anchoring is load-bearing.
        assert not self._NORMALIZE_FIELD_RE.fullmatch("_normalize_fan_out_params_v2")
        assert not self._NORMALIZE_FIELD_RE.fullmatch("_normalize_inputs")

    def test_construct_init_calls_normalize_ir(self):
        """Construct.__init__ must delegate to normalize_ir — proves the single
        site is actually wired in, not just present."""
        construct_src = (SRC_DIR / "construct.py").read_text()
        assert "normalize_ir(self)" in construct_src, (
            "Construct.__init__ must call normalize_ir(self) so all three API "
            "surfaces converge on identical IR before validation (neograph-20xq)."
        )

    def test_scanner_detects_every_write_form(self):
        """Mutation: the scanner must catch EVERY write form it claims to, so
        the guard is not a dead snapshot. Each form below must surface the
        field; if any line stops being detected, the real guard would silently
        pass a regression that writes an IR field via that form."""
        forms = {
            "attribute": 'node.fan_out_param = "x"',
            "subscript": 'd["fan_out_param"] = "x"',
            "aug-assign-attr": 'node.fan_out_param += "x"',
            "aug-assign-subscript": 'd["fan_out_param"] += "x"',
            "ann-assign-attr": 'node.fan_out_param: str = "x"',
            "ann-assign-subscript": 'd["fan_out_param"]: str = "x"',
            "tuple-unpack": "node.fan_out_param, y = a, b",
            "list-unpack": "[node.fan_out_param, y] = a, b",
            "dict-literal": 'node.model_copy(update={"fan_out_param": "y"})',
            "dict-kwarg": 'node.model_copy(update=dict(fan_out_param="y"))',
            "setattr": 'setattr(node, "fan_out_param", "x")',
            "object-setattr": 'object.__setattr__(node, "fan_out_param", T)',
        }
        for label, src in forms.items():
            written = self._scan_ir_field_writes(ast.parse(src))
            assert "fan_out_param" in written, (
                f"scanner missed write form {label!r} ({src!r}); detected "
                f"{sorted(written)}. The guard's docstring claims this form is "
                f"covered — a regression using it would slip past silently."
            )

    def test_scanner_ignores_class_body_field_declarations(self):
        """A class-body field declaration (``fan_out_param: str | None = None``)
        and a module-level annotated constant are NOT instance writes and must
        NOT be flagged — otherwise the guard would fail against node.py's own
        Node field definitions and _ir_normalize's _NORMALIZERS constant."""
        synthetic = (
            "class Node:\n"
            "    fan_out_param: str | None = None\n"
            "    oracle_gen_type: object = None\n"
            "_NORMALIZERS: list = []\n"
        )
        written = self._scan_ir_field_writes(ast.parse(synthetic))
        assert written == set(), (
            f"scanner flagged a class-body declaration as a write: {sorted(written)}; "
            f"bare-Name annotated/plain targets must be ignored."
        )

    def test_scanner_ignores_read_forms(self):
        """getattr/attribute-read of an IR field is NOT a write and must not
        be flagged (the validator legitimately reads fan_out_param via
        getattr)."""
        synthetic = (
            "def f(node):\n"
            '    a = getattr(node, "fan_out_param", None)\n'
            "    b = node.oracle_gen_type\n"
            "    return a, b\n"
        )
        written = self._scan_ir_field_writes(ast.parse(synthetic))
        assert written == set(), f"scanner flagged a read as a write: {sorted(written)}"


class TestRoutingKeyNotLabelInvariant:
    """neograph-y20i / 7df1: LangGraph routing identity is the explicit
    graph.add_node(name, fn) argument — never a wrapper/shim closure's
    __name__. No closure may override its __name__ to a mangled state
    field_name (``= field_name`` or ``= field_name_for(...)``). Such
    overrides conflate routing identity with the display label and are
    dead weight (routing never reads closure __name__).

    Legitimate __name__ assignments that preserve a real user-facing name
    (e.g. ``legacy_shim.__name__ = fn_name`` in decorators.py, which keeps
    the user's function name) are NOT flagged — only mangled-field_name
    overrides are.
    """

    @staticmethod
    def _scan_name_overrides(tree: ast.AST) -> list[int]:
        """Return line numbers of ``<x>.__name__ = field_name`` or
        ``<x>.__name__ = field_name_for(...)`` assignments."""
        offenders: list[int] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            # target must be an attribute access ending in `.__name__`
            if not (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and node.targets[0].attr == "__name__"
            ):
                continue
            val = node.value
            is_field_name = isinstance(val, ast.Name) and val.id == "field_name"
            is_field_name_for = (
                isinstance(val, ast.Call) and isinstance(val.func, ast.Name) and val.func.id == "field_name_for"
            )
            if is_field_name or is_field_name_for:
                offenders.append(node.lineno)
        return offenders

    def test_no_closure_name_mangled_to_field_name(self):
        offenders: list[str] = []
        for py in SRC_DIR.rglob("*.py"):
            tree = ast.parse(py.read_text())
            for lineno in self._scan_name_overrides(tree):
                offenders.append(f"{py.name}:{lineno}")
        assert offenders == [], (
            f"\n{len(offenders)} closure __name__ override(s) mangle routing "
            f"identity into a state field_name:\n"
            + "\n".join(f"  {o}" for o in offenders)
            + "\n\nRouting identity is the graph.add_node(name, fn) argument; "
            "do NOT override the closure __name__ to field_name/field_name_for(). "
            "Source display labels from node.name/sub.name via the captured IR "
            "object (see neograph-y20i)."
        )

    def test_scanner_detects_field_name_override(self):
        """Mutation: ``x.__name__ = field_name`` must be flagged."""
        tree = ast.parse("def f():\n    x.__name__ = field_name\n")
        assert self._scan_name_overrides(tree) == [2]

    def test_scanner_detects_field_name_for_override(self):
        """Mutation: ``x.__name__ = field_name_for(n.name)`` must be flagged."""
        tree = ast.parse("def f():\n    x.__name__ = field_name_for(n.name)\n")
        assert self._scan_name_overrides(tree) == [2]

    def test_scanner_ignores_legitimate_name_assignment(self):
        """Mutation: preserving a real user name (fn_name) is NOT flagged."""
        tree = ast.parse("def f():\n    shim.__name__ = fn_name\n")
        assert self._scan_name_overrides(tree) == []


class TestNoOrZeroCounterIdiom:
    """neograph-ylk9: the counter 'None-means-zero' rule lives ONLY in
    ``StateBus.get_counter``. No production site may re-derive it via
    ``.get(<count_field>) or 0`` or ``.get(<count_field>, 0)``.

    Same root cause as ``loop_condition_none_unsafe``: counter zero-default
    semantics must not leak across the codebase. This guard is scoped to the
    StateBus-audited files (TestStateBusGetDiscipline.IN_SCOPE).
    """

    # A .get(...) whose argument list mentions a counter field.
    _COUNTER_GET = re.compile(r"\.get\([^)]*count[^)]*\)")
    # The explicit-default form: .get(<...count...>, 0)
    _COUNTER_GET_DEFAULT = re.compile(r"\.get\([^)]*count[^)]*,\s*0\s*\)")

    @classmethod
    def _is_counter_zero_idiom(cls, line: str) -> bool:
        if not cls._COUNTER_GET.search(line):
            return False
        return ("or 0" in line) or bool(cls._COUNTER_GET_DEFAULT.search(line))

    @classmethod
    def _scan(cls, src_dir) -> list[str]:
        offenders: list[str] = []
        for name in sorted(TestStateBusGetDiscipline.IN_SCOPE):
            py = src_dir / name
            if not py.exists():
                continue
            for lineno, line in enumerate(py.read_text().splitlines(), start=1):
                if cls._is_counter_zero_idiom(line):
                    offenders.append(f"{name}:{lineno}: {line.strip()}")
        return offenders

    def test_no_or_zero_counter_idiom_in_statebus_files(self):
        offenders = self._scan(SRC_DIR)
        assert offenders == [], (
            f"\n{len(offenders)} counter 'or 0' / get(count,0) idiom(s) — "
            f"use StateBus.get_counter(key) instead:\n" + "\n".join(f"  {o}" for o in offenders)
        )

    def test_scanner_detects_or_zero_form(self):
        """Mutation: 'bus.get(count_field) or 0' must be flagged."""
        assert self._is_counter_zero_idiom("    current = bus.get(count_field) or 0")

    def test_scanner_detects_guarded_or_zero_form(self):
        """Mutation: the _state_write guarded form must be flagged."""
        line = "        current = (state.get(count_field) if state is not None else None) or 0"
        assert self._is_counter_zero_idiom(line)

    def test_scanner_detects_explicit_zero_default_form(self):
        """Mutation: 'bus.get(count_field, 0)' must be flagged."""
        assert self._is_counter_zero_idiom("        count = bus.get(count_field, 0)")

    def test_scanner_ignores_get_counter(self):
        """get_counter(key) is the sanctioned form — never flagged."""
        assert not self._is_counter_zero_idiom("        count = bus.get_counter(count_field)")

    def test_scanner_ignores_unrelated_or_zero(self):
        """'or 0' on a non-counter .get must NOT be flagged."""
        assert not self._is_counter_zero_idiom("        x = bus.get('score') or 0")
        assert not self._is_counter_zero_idiom("        total = some_list_len or 0")

    def test_slip_counter_get(self):
        """Regex-slip: _COUNTER_GET keys on the 'count' substring inside a .get
        arg list. Prove the boundary: a counter .get matches, a non-counter .get
        does not (the 'count'-substring scoping is load-bearing and documented)."""
        assert self._COUNTER_GET.search("bus.get(loop_count_field)")
        assert self._COUNTER_GET.search('bus.get("neo_loop_count")')
        assert not self._COUNTER_GET.search("bus.get(score_field)")

    def test_slip_counter_get_default(self):
        """Regex-slip: _COUNTER_GET_DEFAULT requires the explicit ', 0' default
        with optional surrounding whitespace. Prove whitespace variants match and
        a no-default counter .get does not (so it can't masquerade as the
        explicit-default form)."""
        assert self._COUNTER_GET_DEFAULT.search("bus.get(loop_count, 0)")
        assert self._COUNTER_GET_DEFAULT.search("bus.get(loop_count,0)")
        assert not self._COUNTER_GET_DEFAULT.search("bus.get(loop_count)")


class TestToolBudgetPreambleSingleSource:
    """The framework tool-budget preamble has exactly ONE producer.

    announced==enforced holds only while a single helper renders the budget
    prose from Tool.budget / cfg.max_iterations. A future PR that hand-rolls a
    second budget announcement in another module would silently reintroduce the
    drift this feature eliminated (the in-repo analogue of the piarch
    max_tool_calls=15 hardcode). This guard pins the single source.
    """

    _PRODUCER_MODULE = "_tool_budget_preamble.py"
    # Distinctive fragment of the locked plan-ahead directive. Only the
    # canonical helper may contain it.
    _DIRECTIVE_MARKER = "you need not use every call"

    @staticmethod
    def _modules_with_marker(corpus: dict[str, str], marker: str) -> list[str]:
        return sorted(name for name, text in corpus.items() if marker in text)

    def _real_corpus(self) -> dict[str, str]:
        return {py.name: py.read_text() for py in SRC_DIR.glob("*.py")}

    def test_directive_lives_in_exactly_one_module(self):
        """The live tree: the directive marker appears in only the producer."""
        hits = self._modules_with_marker(self._real_corpus(), self._DIRECTIVE_MARKER)
        assert hits == [self._PRODUCER_MODULE], (
            "Tool-budget preamble directive must live only in "
            f"{self._PRODUCER_MODULE}; found in: {hits}. Do not hand-roll a "
            "second budget announcement -- call render_tool_budget_preamble so "
            "announced==enforced by construction."
        )

    def test_tool_loop_delegates_to_helper(self):
        """The announce branch delegates to the helper -- numbers are derived
        from tools/cfg, never written as literals."""
        src = (SRC_DIR / "_tool_loop.py").read_text()
        assert "render_tool_budget_preamble(tools, cfg.max_iterations)" in src

    def test_scanner_accepts_single_producer(self):
        """Positive meta-test: one producer -> the sanctioned result."""
        corpus = {
            "_tool_budget_preamble.py": "... you need not use every call ...",
            "other.py": "unrelated",
        }
        assert self._modules_with_marker(corpus, self._DIRECTIVE_MARKER) == [self._PRODUCER_MODULE]

    def test_scanner_flags_second_producer(self):
        """Negative meta-test: a rogue second producer diverges from the
        sanctioned single-module result, so the guard fires."""
        corpus = {
            "_tool_budget_preamble.py": "... you need not use every call ...",
            "_rogue.py": "hand-rolled: you need not use every call",
        }
        hits = self._modules_with_marker(corpus, self._DIRECTIVE_MARKER)
        assert hits != [self._PRODUCER_MODULE]
        assert "_rogue.py" in hits


class TestAgentModeNoStructuredRegeneration:
    """Agent/act mode does not UNCONDITIONALLY re-generate its output.

    neograph-eoi8 (refining f7nt): the ReAct loop's final turn is parsed from
    messages[-1] on the happy path (0 extra calls). A constrained-decoding
    re-generation via _call_structured is allowed ONLY as a parse-failure
    fallback (output_strategy='structured'). So any _call_structured call inside
    _tool_loop.py must live inside an `except` block, never on the happy path.

    LIMITS (do not over-trust this AST scan): it is a lexical, single-file
    approximation. It cannot catch a happy-path re-gen introduced via a helper
    in ANOTHER module (the name scan is _tool_loop.py-only), and it treats ANY
    `except` as the parse-failure branch (it does not prove the except catches
    the parse error specifically). The REAL invariant pins are the behavioral
    tests: TestAgentStrategyAwareFallback (happy path = K+1, fallback only on
    parse failure) and the with_structured_output-absent assertion below.
    """

    @staticmethod
    def _call_ids_in_except(func_name: str, tree: ast.AST) -> set[int]:
        ids: set[int] = set()
        for handler in ast.walk(tree):
            if isinstance(handler, ast.ExceptHandler):
                for node in ast.walk(handler):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == func_name:
                        ids.add(id(node))
        return ids

    @staticmethod
    def _all_call_ids(func_name: str, tree: ast.AST) -> set[int]:
        return {
            id(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == func_name
        }

    @classmethod
    def _unconditional_calls(cls, func_name: str, src: str) -> int:
        """Count func_name calls that are NOT inside any except handler."""
        tree = ast.parse(src)
        return len(cls._all_call_ids(func_name, tree) - cls._call_ids_in_except(func_name, tree))

    @classmethod
    def _total_calls(cls, func_name: str, src: str) -> int:
        return len(cls._all_call_ids(func_name, ast.parse(src)))

    def _tool_loop_src(self) -> str:
        return (SRC_DIR / "_tool_loop.py").read_text()

    def test_call_structured_only_in_failure_branch_in_tool_loop(self):
        """Live tree: _call_structured appears in _tool_loop.py (the fallback)
        but every call is inside an except block — none on the happy path."""
        src = self._tool_loop_src()
        assert self._total_calls("_call_structured", src) >= 1, "expected a _call_structured fallback in _tool_loop.py"
        assert self._unconditional_calls("_call_structured", src) == 0, (
            "_call_structured is called OUTSIDE an except block in _tool_loop.py; "
            "agent mode must parse messages[-1] first and only fall back to "
            "constrained decoding ON parse failure (no unconditional re-generation)."
        )

    def test_no_with_structured_output_in_tool_loop(self):
        """AR-03: the loop must not bypass the fallback discipline by calling
        with_structured_output (the primitive _call_structured wraps) directly."""
        assert "with_structured_output" not in self._tool_loop_src()

    def test_scanner_flags_unconditional_call(self):
        """Negative meta-test: a top-level (non-except) call is flagged."""
        src = "def g():\n    return _call_structured(x, y)\n"
        assert self._unconditional_calls("_call_structured", src) == 1

    def test_scanner_accepts_except_only_call(self):
        """Positive meta-test: a call reachable only inside except is accepted."""
        src = (
            "def g():\n"
            "    try:\n"
            "        return parse(m)\n"
            "    except ExecutionError:\n"
            "        return _call_structured(x, y)\n"
        )
        assert self._total_calls("_call_structured", src) == 1
        assert self._unconditional_calls("_call_structured", src) == 0


class TestDefaultFactoryCoercionIsGuarded:
    """The default_factory null-coercion must call the factory inside a
    TypeError guard (neograph-s1u4).

    A bare ``default_factory()`` crashes on Pydantic 2.10+ data-accepting
    factories (``default_factory=lambda data: ...``); conversely, passing the
    data dict to a zero-arg factory like ``list`` silently returns the dict's
    keys instead of ``[]``. So the coercion must be zero-arg-first with an
    ``except TypeError -> factory(data)`` fallback. This guard stops a future PR
    from "simplifying" it back to a bare call.

    LIMITS: a single-function AST check on ``_apply_null_defaults`` — it proves a
    TypeError-guarded try wraps a factory call, not the exact call shape. The
    behavioral pins are TestParseJsonResponseLenientParsing (plain list factory
    AND data-accepting factory).
    """

    @staticmethod
    def _func_default_factory_is_typeerror_guarded(src: str, func_name: str) -> bool:
        tree = ast.parse(src)
        target = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func_name),
            None,
        )
        if target is None:
            return False
        references_factory = any(isinstance(n, ast.Attribute) and n.attr == "default_factory" for n in ast.walk(target))
        if not references_factory:
            return False
        for node in ast.walk(target):
            if isinstance(node, ast.Try) and any(
                isinstance(h.type, ast.Name) and h.type.id == "TypeError" for h in node.handlers
            ):
                body_has_call = any(isinstance(n, ast.Call) for b in node.body for n in ast.walk(b))
                handler_has_call = any(isinstance(n, ast.Call) for h in node.handlers for n in ast.walk(h))
                if body_has_call and handler_has_call:
                    return True
        return False

    def test_apply_null_defaults_guards_default_factory(self):
        """Live tree: the coercion in _apply_null_defaults is TypeError-guarded."""
        src = (SRC_DIR / "_llm_retry.py").read_text()
        assert self._func_default_factory_is_typeerror_guarded(src, "_apply_null_defaults")

    def test_scanner_flags_bare_factory_call(self):
        """Negative meta-test: a bare default_factory() call is NOT guarded."""
        src = "def _apply_null_defaults(data, model):\n    fi = model\n    data['x'] = fi.default_factory()\n"
        assert not self._func_default_factory_is_typeerror_guarded(src, "_apply_null_defaults")

    def test_scanner_accepts_guarded_call(self):
        """Positive meta-test: zero-arg-first + except TypeError fallback passes."""
        src = (
            "def _apply_null_defaults(data, model):\n"
            "    fi = model\n"
            "    factory = fi.default_factory\n"
            "    try:\n"
            "        data['x'] = factory()\n"
            "    except TypeError:\n"
            "        data['x'] = factory(data)\n"
        )
        assert self._func_default_factory_is_typeerror_guarded(src, "_apply_null_defaults")


class TestNoHandRolledExitStrip:
    """neo_* filtering must be DECLARED to the engine (output_schema), never
    hand-wrapped around invoke/ainvoke exits (neograph-pjqe).

    The disease: wrapping engine results in ``_strip_internals(...)`` at run()/
    arun()/sub-construct exits — a parallel mechanism for a concern the engine
    already owns via StateGraph(output_schema=...). After neograph-pjqe the ONLY
    sanctioned ``_strip_internals`` call sites are the two stream arms in
    runner.py's ``_finalize_by_mode`` (a cited engine gap: langgraph 1.2.4 does
    not filter streamed chunks). Any other call site re-introduces the disease.

    AST-walk guard (no regex-slip case): positive + negative meta-tests suffice.
    """

    # The one function permitted to call _strip_internals (the stream-arm residue).
    _ALLOWED_CALLERS = frozenset({"_finalize_by_mode"})

    @staticmethod
    def _functions_calling(source: str, callee: str) -> set[str]:
        """Names of def/async-def functions that CALL ``callee`` in ``source``."""
        tree = ast.parse(source)
        hits: set[str] = set()
        for func in ast.walk(tree):
            if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for node in ast.walk(func):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == callee:
                        hits.add(func.name)
        return hits

    def test_strip_internals_called_only_in_finalize_by_mode(self):
        """No source file may call _strip_internals outside the sanctioned
        stream-arm residue — else the hand-rolled exit-strip disease is back."""
        violations = []
        for py_file in sorted(SRC_DIR.glob("*.py")):
            callers = self._functions_calling(py_file.read_text(), "_strip_internals")
            stray = callers - self._ALLOWED_CALLERS
            for fn in sorted(stray):
                violations.append(f"  {py_file.name}: {fn}() calls _strip_internals")
        assert violations == [], (
            "\n_strip_internals called outside _finalize_by_mode "
            "(hand-rolled exit-strip re-introduced — declare output_schema instead, "
            "see neograph-pjqe):\n" + "\n".join(violations)
        )

    def test_compile_declares_output_schema(self):
        """compiler.py must pass output_schema to StateGraph — the engine-level
        replacement for the deleted exit strips. Pins the cure in place."""
        tree = ast.parse((SRC_DIR / "compiler.py").read_text())
        declared = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "StateGraph"
            and any(kw.arg == "output_schema" for kw in node.keywords)
            for node in ast.walk(tree)
        )
        assert declared, (
            "compiler.py no longer declares StateGraph(output_schema=...) — the "
            "engine-level neo_ filter is gone; exits would leak framework channels "
            "(neograph-pjqe)."
        )

    def test_compile_does_not_declare_input_schema(self):
        """compiler.py must NOT pass input_schema to StateGraph. R1 (docs/design/
        langgraph-output-schema-research-2026-07-03.md) proved a narrowed
        input_schema silently drops the schema/node fingerprints run() seeds
        through the initial input dict — breaking checkpoint auto-rewind. The
        state_model (first positional arg) IS the input schema; narrowing it must
        never be reintroduced. Pins the omission in place."""
        tree = ast.parse((SRC_DIR / "compiler.py").read_text())
        offenders = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "StateGraph"
            and any(kw.arg == "input_schema" for kw in node.keywords)
        ]
        assert offenders == [], (
            "compiler.py passes StateGraph(input_schema=...) — a narrowed input "
            "schema silently drops the fingerprints run() seeds through the initial "
            "input, breaking checkpoint auto-rewind (R1, langgraph-output-schema-"
            "research-2026-07-03.md). The full state_model must remain the input."
        )

    # ---- meta-tests: prove the guard actually discriminates ----

    def test_meta_positive_stray_call_is_caught(self):
        """A _strip_internals call in a non-sanctioned function is flagged."""
        src = "def run(graph, x):\n    return _strip_internals(graph.invoke(x))\n"
        stray = self._functions_calling(src, "_strip_internals") - self._ALLOWED_CALLERS
        assert stray == {"run"}

    def test_meta_negative_sanctioned_call_is_allowed(self):
        """The stream-arm residue in _finalize_by_mode is not flagged."""
        src = (
            "def _finalize_by_mode(payload, mode):\n"
            "    if mode == 'values':\n"
            "        return _strip_internals(payload)\n"
            "    return payload\n"
        )
        stray = self._functions_calling(src, "_strip_internals") - self._ALLOWED_CALLERS
        assert stray == set()

    @staticmethod
    def _stategraph_has_input_schema(source: str) -> bool:
        """True if any StateGraph(...) call in source passes input_schema=."""
        tree = ast.parse(source)
        return any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "StateGraph"
            and any(kw.arg == "input_schema" for kw in node.keywords)
            for node in ast.walk(tree)
        )

    def test_meta_positive_input_schema_kwarg_is_detected(self):
        """A StateGraph(..., input_schema=...) call is caught by the AST check."""
        src = "g = StateGraph(State, output_schema=Out, input_schema=In)\n"
        assert self._stategraph_has_input_schema(src) is True

    def test_meta_negative_output_schema_only_is_allowed(self):
        """StateGraph with only output_schema= (current code) is not flagged."""
        src = "g = StateGraph(State, output_schema=Out)\n"
        assert self._stategraph_has_input_schema(src) is False


class TestAgentCycleTurnsAreSupersteps:
    """An agent/act node's ReAct turns are REAL parent supersteps
    ({node}__agent / {node}__tools / {node}__parse), NOT a ``while`` loop inside
    one node (neograph-m6d3.4 Core Invariant, restating _agent_cycle.py's own
    docstring). The ReAct iteration is the graph's tools->agent edge; the
    tool-gating gate is a real inserted node. This guard pins that no in-node
    loop is ever reintroduced into the cycle-body module — which is what makes
    every payoff (turn-boundary interrupts, tool-gating, honest budgets across a
    checkpoint) possible.

    AST-walk guard (no regex-slip case): positive + negative meta-tests suffice.
    """

    @staticmethod
    def _has_while_loop(source: str) -> bool:
        """True if the source contains any ``while`` statement."""
        return any(isinstance(n, ast.While) for n in ast.walk(ast.parse(source)))

    def test_agent_cycle_module_has_no_while_loop(self):
        """_agent_cycle.py (the ReAct cycle bodies + router) must contain no
        ``while`` loop — turns are supersteps, not an in-node loop."""
        src = (SRC_DIR / "_agent_cycle.py").read_text()
        assert not self._has_while_loop(src), (
            "_agent_cycle.py contains a `while` loop — the ReAct turn loop must be "
            "the graph's {node}__tools -> {node}__agent edge (real supersteps), NOT "
            "an in-node while loop (neograph-m6d3.4 Core Invariant)."
        )

    # ---- meta-tests: prove the guard actually discriminates ----

    def test_meta_positive_while_loop_is_detected(self):
        src = "def agent_turn(state):\n    while True:\n        break\n    return {}\n"
        assert self._has_while_loop(src) is True

    def test_meta_negative_for_loop_is_allowed(self):
        # A `for` over tool_calls is fine — that is per-turn work, not the ReAct loop.
        src = "def tools_turn(state):\n    for tc in state['calls']:\n        run(tc)\n    return {}\n"
        assert self._has_while_loop(src) is False


class TestTwinThinness:
    """THINNESS guard for sync/async twin pairs across the LLM/tool vertical +
    runner (neograph-ykun, review PAT-01).

    ## The invariant
    A sync/async twin pair must route through shared pure helpers and differ
    ONLY at the ``await`` seam. In particular, every *value-builder block* — an
    ``ExecutionError``/``CheckpointSchemaError``/``ConfigurationError`` builder,
    a logger event (``log.info``/``.warning``/...), a ``ToolMessage`` builder,
    or the ``{input_tokens, output_tokens, total_tokens}`` usage dict — must be
    SINGLE-SITE: it may not appear word-for-word in BOTH twins of a pair. If it
    does, the two copies drift independently (a bug fix or message edit must be
    applied twice). The reference implementations ``_oracle.py`` and
    ``_subconstruct.py`` show the target shape: the awaiting control skeleton may
    stay in each twin (it is irreducible under async), but the *content* — error
    strings, log events, result dicts — lives in one shared helper.

    ## Why "builder blocks", not "body-size delta"
    The await seam is genuinely irreducible: a shared skeleton cannot straddle
    ``await`` (a match arm or retry loop that awaits in the async twin must be
    duplicated). So a naive body-length/line-delta budget would either license
    the drift (verbatim copies have delta≈0) or forbid the irreducible skeleton
    forever. Single-siting the value-builder blocks is the part of the invariant
    that IS reducible and IS the maintenance hazard the review found — so that is
    what this guard pins.

    ## The closure hole this closes
    The pre-existing co-location guard (``test_guards_async_dispatch.py``) checks
    only that twin *names* co-exist and CANNOT see the worst twins, which are
    nested closures inside ``make_agent_cycle_bodies`` (``agent_body`` /
    ``tools_body`` / ...). This walker descends closures via ``ast.walk`` +
    name lookup, so those twins are in scope.

    Non-vacuity is proven by the slip meta-tests: a synthetic twin pair sharing
    a builder block IS flagged; one that routes the builder through a shared
    helper is NOT.
    """

    # {module_filename: [(sync_name, async_name), ...]} — includes nested-closure
    # body twins (agent_body/tools_body/...) the co-location guard cannot see.
    TWIN_TABLE: dict[str, list[tuple[str, str]]] = {
        "_agent_cycle.py": [
            ("agent_body", "aagent_body"),
            ("tools_body", "atools_body"),
            ("parse_body", "aparse_body"),
            ("_build_turn_prep", "_abuild_turn_prep"),
        ],
        # The di_inputs injector twins (review PAT-02): both built the
        # config-carrier dict verbatim — the un-tabled twin blind spot this
        # guard was extended (carrier shape below) to close. Now single-sited on
        # `_with_configurable`; tabling the pair keeps a re-inline visible.
        "_dispatch.py": [("_inject_di_inputs", "_ainject_di_inputs")],
        "_tool_loop.py": [
            ("_parse_final_turn", "_aparse_final_turn"),
            ("_prepare_tool_loop", "_aprepare_tool_loop"),
        ],
        "_llm_dispatch.py": [("_call_structured", "_acall_structured")],
        "_llm_retry.py": [
            ("_invoke_json_with_retry", "_ainvoke_json_with_retry"),
            ("recover_dsml", "arecover_dsml"),
        ],
        "_llm.py": [("invoke_structured", "ainvoke_structured")],
        "runner.py": [
            ("_verify_checkpoint_schema", "_averify_checkpoint_schema"),
            ("_auto_resume_from_divergence", "_aauto_resume_from_divergence"),
        ],
    }

    # Error builders (``X.build(...)``) whose X is a known error class.
    _ERROR_BUILDER_CLASSES = ERROR_CLASSES
    # Direct-constructor value builders that carry user-facing content.
    _CONSTRUCTOR_BUILDERS = frozenset({"CheckpointSchemaError", "ToolMessage"})
    # Logger event methods (structlog): first positional arg is the event string.
    _LOG_METHODS = frozenset({"info", "warning", "error", "debug", "exception"})
    # A dict literal is treated as the usage shape iff it carries this key.
    _USAGE_DICT_MARKER = "total_tokens"
    # A dict literal is treated as the config-carrier shape iff it carries this
    # key (``{**config, "configurable": {...}}``) — the copy-not-mutate write to
    # config['configurable'] (review PAT-02). Single-sited on
    # ``_config_carrier._with_configurable``; a re-inline in BOTH twins of a pair
    # (as the di_inputs injectors once did) is the drift this pins.
    _CARRIER_DICT_MARKER = "configurable"

    # Async→sync identifier renames so a builder/log block that references an
    # async seam normalizes to its sync twin's spelling. Longest-first so a
    # short key cannot corrupt a longer identifier. Builders rarely reference
    # these, but keeping the map makes the match await-agnostic.
    _ASYNC_RENAMES = (
        ("_aauto_resume_from_divergence", "_auto_resume_from_divergence"),
        ("_averify_checkpoint_schema", "_verify_checkpoint_schema"),
        ("_ainvoke_json_with_retry", "_invoke_json_with_retry"),
        ("_aprepare_tool_loop", "_prepare_tool_loop"),
        ("_abuild_turn_prep", "_build_turn_prep"),
        ("_aparse_final_turn", "_parse_final_turn"),
        ("_acall_structured", "_call_structured"),
        ("ainvoke_structured", "invoke_structured"),
        ("arecover_dsml", "recover_dsml"),
        ("aget_tuple", "get_tuple"),
        ("ainvoke", "invoke"),
    )

    @classmethod
    def _normalize(cls, text: str) -> str:
        """Await-agnostic normalization of an unparsed AST block."""
        text = text.replace("await ", "").replace("async ", "")
        for a, s in cls._ASYNC_RENAMES:
            text = text.replace(a, s)
        return " ".join(text.split())

    @staticmethod
    def _find_func(source: str, name: str) -> ast.AST | None:
        """First def named ``name`` anywhere (top-level OR nested closure)."""
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name:
                return node
        return None

    @classmethod
    def _first_arg_is_str(cls, call: ast.Call) -> bool:
        return bool(call.args) and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)

    @classmethod
    def _builder_blocks(cls, fn: ast.AST) -> list[str]:
        """Collect normalized value-builder blocks defined DIRECTLY in ``fn``'s
        body (not in sibling/nested helper functions it merely calls)."""
        # Do not descend into nested function defs — a builder that lives in a
        # nested helper is already single-site by construction.
        skip: set[int] = set()
        for sub in ast.walk(fn):
            if sub is not fn and isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef):
                for inner in ast.walk(sub):
                    skip.add(id(inner))

        blocks: list[str] = []

        def emit(node: ast.AST) -> None:
            if id(node) in skip:
                return
            blocks.append(cls._normalize(ast.unparse(node)))

        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                func = node.func
                # X.build(...) where X is an error class
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "build"
                    and isinstance(func.value, ast.Name)
                    and func.value.id in cls._ERROR_BUILDER_CLASSES
                ):
                    emit(node)
                # direct constructor builders: CheckpointSchemaError(...), ToolMessage(...)
                elif isinstance(func, ast.Name) and func.id in cls._CONSTRUCTOR_BUILDERS:
                    emit(node)
                # logger event: <x>.info("event", ...) with a string event
                elif isinstance(func, ast.Attribute) and func.attr in cls._LOG_METHODS and cls._first_arg_is_str(node):
                    emit(node)
            elif isinstance(node, ast.Dict):
                keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
                if cls._USAGE_DICT_MARKER in keys or cls._CARRIER_DICT_MARKER in keys:
                    emit(node)
        return blocks

    def test_no_value_builder_block_is_duplicated_across_twins(self):
        violations: list[str] = []
        for filename, pairs in self.TWIN_TABLE.items():
            source = (SRC_DIR / filename).read_text()
            for sync_name, async_name in pairs:
                sync_fn = self._find_func(source, sync_name)
                async_fn = self._find_func(source, async_name)
                assert sync_fn is not None, f"{filename}: twin `{sync_name}` not found"
                assert async_fn is not None, f"{filename}: twin `{async_name}` not found"
                shared = set(self._builder_blocks(sync_fn)) & set(self._builder_blocks(async_fn))
                for block in sorted(shared):
                    violations.append(f"  {filename}: {sync_name}/{async_name} both build:\n      {block[:140]}")
        assert not violations, (
            f"\n{len(violations)} value-builder block(s) duplicated across sync/async "
            "twins — extract each into ONE shared pure helper (error/log/result "
            "builders must be single-site so a fix is applied once):\n"
            + "\n".join(violations)
            + "\n\nThe awaiting control skeleton may stay per-twin; the CONTENT "
            "(error strings, log events, usage dicts, ToolMessages) may not. See "
            "_oracle.py / _subconstruct.py for the target shape (neograph-ykun)."
        )

    # ── slip meta-tests: synthetic twins prove the guard discriminates ──

    def test_meta_positive_shared_builder_block_is_flagged(self):
        src = (
            "def f():\n"
            "    raise ExecutionError.build('boom', hint='x')\n"
            "async def af():\n"
            "    raise ExecutionError.build('boom', hint='x')\n"
        )
        sync_fn = self._find_func(src, "f")
        async_fn = self._find_func(src, "af")
        shared = set(self._builder_blocks(sync_fn)) & set(self._builder_blocks(async_fn))
        assert shared, "a byte-identical error builder in both twins must be flagged"

    def test_meta_negative_helper_routed_builder_is_not_flagged(self):
        src = "def f():\n    raise _boom_error()\nasync def af():\n    raise _boom_error()\n"
        sync_fn = self._find_func(src, "f")
        async_fn = self._find_func(src, "af")
        shared = set(self._builder_blocks(sync_fn)) & set(self._builder_blocks(async_fn))
        assert not shared, "a builder routed through a shared helper must NOT be flagged"

    def test_meta_await_normalization_matches_seam_twins(self):
        """A usage dict built after a sync vs awaited call still matches."""
        src = (
            "def f():\n"
            "    r = llm.invoke(m)\n"
            "    return {'input_tokens': 1, 'output_tokens': 2, 'total_tokens': 3}\n"
            "async def af():\n"
            "    r = await llm.ainvoke(m)\n"
            "    return {'input_tokens': 1, 'output_tokens': 2, 'total_tokens': 3}\n"
        )
        sync_fn = self._find_func(src, "f")
        async_fn = self._find_func(src, "af")
        shared = set(self._builder_blocks(sync_fn)) & set(self._builder_blocks(async_fn))
        assert shared, "the duplicated usage dict must be flagged across the twins"

    def test_meta_finds_nested_closure_twins(self):
        """The walker must reach body twins nested inside a factory function."""
        src = (
            "def make():\n"
            "    def body(s):\n"
            "        return ToolMessage(content='x', tool_call_id=1)\n"
            "    async def abody(s):\n"
            "        return ToolMessage(content='x', tool_call_id=1)\n"
            "    return body, abody\n"
        )
        assert self._find_func(src, "body") is not None
        assert self._find_func(src, "abody") is not None

    def test_meta_positive_shared_config_carrier_dict_is_flagged(self):
        """A re-inlined ``{**config, 'configurable': {...}}`` in BOTH twins (the
        PAT-02 di_inputs blind spot) must be flagged by the carrier-shape rule."""
        src = (
            "def f(config):\n"
            "    return {**config, 'configurable': {**cfg, 'k': v}}\n"
            "async def af(config):\n"
            "    return {**config, 'configurable': {**cfg, 'k': v}}\n"
        )
        sync_fn = self._find_func(src, "f")
        async_fn = self._find_func(src, "af")
        shared = set(self._builder_blocks(sync_fn)) & set(self._builder_blocks(async_fn))
        assert shared, "a duplicated config-carrier dict in both twins must be flagged"

    def test_meta_negative_helper_routed_config_carrier_is_not_flagged(self):
        """The carrier routed through ``_with_configurable`` in both twins is a
        bare call, no dict literal — so it is NOT flagged (the migrated shape)."""
        src = (
            "def f(config):\n"
            "    return _with_configurable(config, k=v)\n"
            "async def af(config):\n"
            "    return _with_configurable(config, k=v)\n"
        )
        sync_fn = self._find_func(src, "f")
        async_fn = self._find_func(src, "af")
        shared = set(self._builder_blocks(sync_fn)) & set(self._builder_blocks(async_fn))
        assert not shared, "a carrier routed through a shared helper must NOT be flagged"


class TestDiInputsInjectedAtLlmDispatchSeams:
    """di_inputs must be injected at every LLM-mode dispatch seam BEFORE that
    mode's prompt is compiled. The disease this bans (the agent-stark incident):
    a mode dispatch reaches ``_compile_prompt`` without ``_inject_di_inputs``
    having run, so a ``{FromInput}`` template placeholder ships to the model as
    the literal ``{domain}`` instead of the resolved value.

    Two structural facts pin the invariant:

    1. Every KNOWN dispatch seam still calls ``_inject_di_inputs``. Think/raw
       inject in ``_dispatch.py`` (ThinkDispatch); agent/act inject in
       ``_agent_cycle.py`` (``_turn_prep_kwargs``, the single shared pre-prep
       both sync/async turn-prep twins call). Dropping the call from either
       seam is a regression.
    2. No NEW module calls ``_compile_prompt``. The compile sites are a small
       reviewed set (``_llm.py``/``_tool_loop.py`` downstream of a seam, plus
       ``_llm_render.py`` which owns the definition + the ``render_prompt``
       inspection helper). A new compile site is a new dispatch path — it must
       be reviewed for injection and consciously allowlisted, not added
       silently.
    """

    # Modules that own a mode-dispatch seam feeding an LLM prompt compile.
    INJECTION_SEAM_MODULES = frozenset({"_dispatch.py", "_agent_cycle.py"})

    # Modules allowed to CALL _compile_prompt (downstream of a seam, or the
    # inspection helper + the definition site). Extend ONLY after confirming
    # the owning dispatch seam injects di_inputs first.
    COMPILE_SITE_MODULES = frozenset({"_llm.py", "_tool_loop.py", "_llm_render.py"})

    @staticmethod
    def _calls_named(tree: ast.AST, name: str) -> list[int]:
        """Line numbers of every ``name(...)`` call in the tree (bare Name or
        attribute-tail ``obj.name(...)``). Imports and string examples are AST
        nodes of other types, so they never count."""
        hits: list[int] = []
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            fn = n.func
            if isinstance(fn, ast.Name) and fn.id == name:
                hits.append(n.lineno)
            elif isinstance(fn, ast.Attribute) and fn.attr == name:
                hits.append(n.lineno)
        return hits

    @classmethod
    def _seams_missing_injection(cls, src_dir: pathlib.Path) -> list[str]:
        missing: list[str] = []
        for mod in sorted(cls.INJECTION_SEAM_MODULES):
            path = src_dir / mod
            if not path.exists():
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            if not cls._calls_named(tree, "_inject_di_inputs"):
                missing.append(mod)
        return missing

    @classmethod
    def _unreviewed_compile_sites(cls, src_dir: pathlib.Path) -> list[str]:
        offenders: list[str] = []
        for path in sorted(src_dir.glob("*.py")):
            if path.name in cls.COMPILE_SITE_MODULES:
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            for lineno in cls._calls_named(tree, "_compile_prompt"):
                offenders.append(f"{path.name}:{lineno}")
        return offenders

    def test_every_injection_seam_injects_di_inputs(self):
        missing = self._seams_missing_injection(SRC_DIR)
        assert missing == [], (
            f"LLM-mode dispatch seam(s) no longer call _inject_di_inputs: "
            f"{missing}. di_inputs would stop reaching that mode's prompt — a "
            f"{{FromInput}} template would ship the literal placeholder to the "
            f"model. Re-add the injection at the seam's shared pre-prep."
        )

    def test_no_unreviewed_compile_prompt_call_sites(self):
        offenders = self._unreviewed_compile_sites(SRC_DIR)
        assert offenders == [], (
            f"_compile_prompt called outside the reviewed set "
            f"{sorted(self.COMPILE_SITE_MODULES)}: {offenders}. A new compile "
            f"site is a new dispatch path — ensure its owning seam calls "
            f"_inject_di_inputs, then add the module to COMPILE_SITE_MODULES."
        )

    def test_every_injection_seam_injects_async_di_inputs(self):
        """Lockstep async twin (neograph-3q6j): every LLM-mode seam ALSO calls the
        async injector ``_ainject_di_inputs`` on its arun() path, so a FROM_RESOURCE
        template var (fetched, awaited) reaches the prompt on the async driver. Think
        injects in ``_dispatch.py`` (ThinkDispatch.aexecute); agent/act in
        ``_agent_cycle.py`` (``_abuild_turn_prep``). Dropping either async call would
        silently strand FROM_RESOURCE template vars on that mode's async path."""
        missing: list[str] = []
        for mod in sorted(self.INJECTION_SEAM_MODULES):
            path = SRC_DIR / mod
            if not path.exists():
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            if not self._calls_named(tree, "_ainject_di_inputs"):
                missing.append(mod)
        assert missing == [], (
            f"LLM-mode dispatch seam(s) no longer call _ainject_di_inputs: "
            f"{missing}. FROM_RESOURCE template vars would stop reaching that "
            f"mode's prompt on the arun() path. Re-add the async injection at the "
            f"seam's async pre-prep twin."
        )

    def test_mutation_seam_without_injection_detected(self, tmp_path: pathlib.Path):
        """A seam module that stops calling _inject_di_inputs is flagged."""
        (tmp_path / "_agent_cycle.py").write_text(
            "def _turn_prep_kwargs(node, config):\n    return {'config': config}\n"
        )
        assert "_agent_cycle.py" in self._seams_missing_injection(tmp_path)

    def test_mutation_seam_with_injection_passes(self, tmp_path: pathlib.Path):
        """A seam that keeps the injection call is NOT flagged."""
        (tmp_path / "_dispatch.py").write_text(
            "def execute(node, config):\n    config = _inject_di_inputs(node, config)\n    return config\n"
        )
        assert self._seams_missing_injection(tmp_path) == []

    def test_mutation_new_compile_site_detected(self, tmp_path: pathlib.Path):
        """A _compile_prompt call in a NEW (non-allowlisted) module is flagged."""
        (tmp_path / "_newmode.py").write_text(
            "def dispatch(runtime, node, config):\n    return _compile_prompt(runtime, node.prompt, config=config)\n"
        )
        offenders = self._unreviewed_compile_sites(tmp_path)
        assert any(o.startswith("_newmode.py:") for o in offenders), offenders

    def test_mutation_allowlisted_compile_site_passes(self, tmp_path: pathlib.Path):
        """A _compile_prompt call in an allowlisted module is NOT flagged (the
        injection happens upstream at the dispatch seam, not here)."""
        (tmp_path / "_llm.py").write_text(
            "def invoke_structured(runtime, node, config):\n"
            "    return _compile_prompt(runtime, node.prompt, config=config)\n"
        )
        assert self._unreviewed_compile_sites(tmp_path) == []

    def test_mutation_import_of_inject_is_not_a_call(self, tmp_path: pathlib.Path):
        """The would-be-missed case: a seam that merely IMPORTS _inject_di_inputs
        but never CALLS it must still be flagged as missing — an import is not
        an injection."""
        (tmp_path / "_agent_cycle.py").write_text(
            "from neograph._dispatch import _inject_di_inputs\n"
            "def _turn_prep_kwargs(node, config):\n"
            "    return {'config': config}\n"
        )
        assert "_agent_cycle.py" in self._seams_missing_injection(tmp_path)


class TestRepairJsonGuarded:
    """Every ``repair_json(...)`` call in src/ must sit inside a try/except.

    neograph-8uoot: a max_tokens-truncated 290KB response sent json_repair's
    recursive-descent parser over the stack limit; the call sat one line ABOVE
    the guarded block, so the ValueError/RecursionError escaped
    ``_invoke_json_with_retry`` uncaught and killed the run — bypassing the
    retry machinery built for exactly this malformation class. Repairer input
    is LLM-controlled text: every blowup it produces is a data malformation
    and must be converted to ExecutionError inside a guard.

    AST-walk guard (no regex-slip case): positive + negative meta-tests suffice.
    """

    @staticmethod
    def _unguarded_repair_calls(source: str) -> list[int]:
        """Line numbers of repair_json() calls not inside a try body that has
        at least one except handler (try/finally does not count)."""
        tree = ast.parse(source)
        guarded_spans: list[tuple[int, int]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Try) and node.handlers:
                start = node.body[0].lineno
                end = max(getattr(n, "end_lineno", n.lineno) or n.lineno for n in node.body)
                guarded_spans.append((start, end))
        offenders: list[int] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
            if name == "repair_json" and not any(s <= node.lineno <= e for s, e in guarded_spans):
                offenders.append(node.lineno)
        return offenders

    def test_no_unguarded_repair_json_call_in_src(self):
        offenders: dict[str, list[int]] = {}
        for path in sorted(SRC_DIR.glob("**/*.py")):
            lines = self._unguarded_repair_calls(path.read_text())
            if lines:
                offenders[path.name] = lines
        assert not offenders, (
            f"repair_json() called outside a try/except guard: {offenders}. "
            "Repairer input is LLM-controlled text — wrap the call and convert "
            "failures to ExecutionError so the retry loop handles them "
            "(neograph-8uoot)."
        )

    # ---- meta-tests: prove the guard actually discriminates ----

    def test_meta_positive_bare_call_is_detected(self):
        src = "def parse(text):\n    repaired = repair_json(text)\n    return repaired\n"
        assert self._unguarded_repair_calls(src) == [2]

    def test_meta_positive_try_finally_without_except_is_detected(self):
        src = (
            "def parse(text):\n"
            "    try:\n"
            "        repaired = repair_json(text)\n"
            "    finally:\n"
            "        pass\n"
        )
        assert self._unguarded_repair_calls(src) == [3]

    def test_meta_negative_guarded_call_is_allowed(self):
        src = (
            "def parse(text):\n"
            "    try:\n"
            "        repaired = repair_json(text)\n"
            "    except Exception as exc:\n"
            "        raise ExecutionError('repair failed') from exc\n"
            "    return repaired\n"
        )
        assert self._unguarded_repair_calls(src) == []

    def test_meta_positive_attribute_call_is_detected(self):
        # json_repair.repair_json(...) (module-qualified) must also be caught.
        src = "def parse(text):\n    return json_repair.repair_json(text)\n"
        assert self._unguarded_repair_calls(src) == [2]
