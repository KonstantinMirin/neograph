"""TDD-red regression tests for dyp3 — public `neograph.testing` fakes API.

These tests pin the AMENDED dyp3 plan (refine atom 938w.28). They MUST fail now:
nothing is extracted yet — `neograph.testing` is still the single scaffold module,
so every `from neograph.testing import FakeLLM/install_fake_llm/...` raises
ImportError, and `neograph.testing.fakes` does not exist as a submodule. The
behavioral asserts are written so that once the package exists but the FakeLLM
contract is wrong, THEY fail (not just the import).

Core Invariant (dyp3): EXACTLY ONE implementation of the fake-LLM contract, and
neograph's own suite consumes it through the PUBLIC location — so a future loop
refactor that breaks the public double breaks OUR suite first.

Coverage map (task neograph-938w.14):
  1. PUBLIC IMPORT — FakeLLM/install_fake_llm + migrated doubles importable.
  2. COMPAT GUARANTEE, TWO DRIVEN SCENARIOS (R-M1):
       (a) AGENT node end-to-end via FakeLLM(outputs) as llm_factory=
           (real AIMessage / bind_tools / empty-tool_calls final turn) — hits
           _tool_loop._get_llm.
       (b) DISTINCT Oracle-merge pipeline with node_name=='' — the ONLY path that
           exercises the merge key, via invoke_structured->_get_llm (_llm.py:191).
  3. BEHAVIORAL install_fake_llm (R-M3) — drive an AGENT through
     install_fake_llm(monkeypatch, outputs); a mistyped 2nd setattr fails HERE.
     Plus a distinct llm_factory= scenario (real _get_llm param-filtering).
  4. FakeLLM.__call__ declares node_name in its signature (so _get_llm does not
     strip it) AND routing selects per-node when node_name is passed.
  5. ANTI-DUPLICATION (R-M2, HYBRID) — tests/fakes.py re-exports the migrated
     fakes (identity with the public module) and does NOT redefine them, while it
     RETAINS the test-only registry plumbing that must stay out of the package.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Any

import pytest
from pydantic import BaseModel

from neograph import (
    Tool,
    compile,
    construct_from_functions,
    construct_from_module,
    node,
    run,
)
from tests.fakes import (
    FakeTool,
    build_fake_llm_kwargs,
    build_test_compile_kwargs,
    register_scripted,
    register_tool_factory,
)
from tests.schemas import Claims, RawText

FAKES_PATH = pathlib.Path(__file__).resolve().parent / "fakes.py"

# The doubles the AMENDED plan MIGRATES verbatim into src/neograph/testing/fakes.py.
# tests/fakes.py must re-export (NOT redefine) each of these after extraction.
_MIGRATED_FAKE_SYMBOLS = (
    "StructuredFake",
    "StructuredFakeWithRaw",
    "ReActFake",
    "StringArgsFake",
    "TextFake",
    "FakeTool",
    "GuardFake",
    "StubbornFake",
    "GatedAsyncFake",
    "_final_json_content",
)

# Test-only registry plumbing (deliberately removed from src/ in ezqz). R-M2:
# these MUST stay in tests/fakes.py and MUST NOT migrate into the public package.
_RETAINED_TEST_ONLY_SYMBOLS = (
    "register_scripted",
    "register_condition",
    "register_tool_factory",
    "build_test_compile_kwargs",
    "reset_test_registry",
    "build_fake_runtime",
    "build_fake_tool_lookup",
    "lookup_scripted",
    "lookup_condition",
)


class KResult(BaseModel, frozen=True):
    """Typed agent-node output for the compat-guarantee scenarios."""

    items: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# 1. PUBLIC IMPORT
# ─────────────────────────────────────────────────────────────────────────────


class TestPublicImportSurface:
    """The promoted API is importable from the public location."""

    def test_fakellm_and_install_helper_import_from_public_location(self) -> None:
        """RED now: neograph.testing has no FakeLLM/install_fake_llm yet."""
        from neograph.testing import FakeLLM, install_fake_llm

        assert callable(FakeLLM)
        assert callable(install_fake_llm)

    def test_migrated_doubles_import_from_public_location(self) -> None:
        """RED now: the battle-hardened doubles are not yet in neograph.testing."""
        from neograph.testing import (
            ReActFake,
            StructuredFake,
            TextFake,
        )

        assert StructuredFake is not None
        assert ReActFake is not None
        assert TextFake is not None


# ─────────────────────────────────────────────────────────────────────────────
# 2. THE COMPAT GUARANTEE — TWO DISTINCT DRIVEN SCENARIOS (R-M1)
# ─────────────────────────────────────────────────────────────────────────────


def _build_agent_graph_with_public_fakellm() -> Any:
    """Compile a single AGENT node driven by a public FakeLLM(outputs) as the
    llm_factory=. Exercises the ReAct cycle (_tool_loop._get_llm): bind_tools on
    the declared tool, a real langchain_core AIMessage flowing through message
    coercion, and the empty-tool_calls final turn parsing to KResult."""
    from neograph.testing import FakeLLM

    register_tool_factory("noop", lambda config, tool_config: FakeTool("noop"))

    @node(
        mode="agent",
        outputs=KResult,
        model="reason",
        prompt="test/explore",
        tools=[Tool(name="noop", budget=3)],
    )
    def research() -> KResult: ...

    return compile(
        construct_from_functions("public_fake_agent", [research]),
        **build_test_compile_kwargs(),
        **build_fake_llm_kwargs(FakeLLM({"research": KResult(items=["done"])})),
    )


class TestCompatGuaranteeAgentScenario:
    """Scenario (a): the tripwire — a public FakeLLM drives an AGENT node end to
    end through the ReAct cycle. If a future loop refactor breaks the public
    double's AIMessage/bind_tools/final-turn surface, THIS breaks first."""

    def test_agent_node_finalizes_through_public_fakellm(self) -> None:
        graph = _build_agent_graph_with_public_fakellm()
        result = run(graph, input={"node_id": "REQ-1"}, config={"configurable": {"thread_id": "pf-agent"}})

        assert result.get("research") == KResult(items=["done"]), (
            f"agent node did not finalize through the public FakeLLM ReAct cycle: {result!r}"
        )
        # No agent ReAct internals leak into returned state.
        assert not any(k.startswith("neo_") for k in result), sorted(result)


class TestCompatGuaranteeOracleMergeScenario:
    """Scenario (b): the ONLY path that exercises node_name==''. Oracle merge
    resolves via invoke_structured -> _get_llm (_llm.py:191) WITHOUT passing
    node_name, so the router's '' merge key must resolve. A single agent test
    cannot reach this."""

    def test_oracle_merge_resolves_through_empty_node_name_key(self) -> None:
        from neograph.testing import FakeLLM

        register_scripted("gen_variant", lambda input_data, config: Claims(items=["v1"]))

        # Scripted generators run the body; only the merge_prompt hits the LLM,
        # and it arrives with node_name=='' -> the FakeLLM '' key must resolve.
        @node(outputs=Claims, ensemble_n=2, merge_prompt="test/merge")
        def generate() -> Claims:
            return Claims(items=["v1"])

        import types as _types

        mod = _types.ModuleType("test_public_fakes_oracle_mod")
        mod.generate = generate
        pipeline = construct_from_module(mod, name="public-fake-oracle")

        graph = compile(
            pipeline,
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(FakeLLM({"": Claims(items=["merged-consensus"])})),
        )
        result = run(graph, input={"node_id": "REQ-1"})

        merged = result.get("generate")
        assert merged == Claims(items=["merged-consensus"]), (
            f"Oracle merge did not resolve through the FakeLLM '' (node_name=='') merge key: {merged!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. BEHAVIORAL install_fake_llm (R-M3) — dual-patch on the agent path
# ─────────────────────────────────────────────────────────────────────────────


class TestInstallFakeLlmBehavioral:
    """install_fake_llm's whole value over llm_factory= is patching _get_llm at
    BOTH binding sites. A mistyped second setattr (neograph._tool_loop._get_llm)
    passes a static set-equality guard but leaves the AGENT path on the real
    factory — which this test catches by driving the agent THROUGH the helper."""

    def test_agent_resolves_fake_via_install_fake_llm_on_tool_loop_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from neograph.testing import install_fake_llm

        register_tool_factory("noop", lambda config, tool_config: FakeTool("noop"))

        @node(
            mode="agent",
            outputs=KResult,
            model="reason",
            prompt="test/explore",
            tools=[Tool(name="noop", budget=3)],
        )
        def research() -> KResult: ...

        # neograph's compile() HARD-REQUIRES llm_factory + prompt_compiler for
        # LLM nodes at assembly time, so a graph with an agent node cannot be
        # built with no factory at all. Model the real install_fake_llm scenario:
        # app code compiled the graph with a REAL factory the test can't influence
        # — here a sentinel that BOOMS if ever called. install_fake_llm's dual-patch
        # must intercept _get_llm on the _tool_loop (agent) path BEFORE that factory
        # is reached; if the second setattr target is wrong, the boom fires.
        def _boom_factory(tier, **kw):
            raise AssertionError(
                "real llm_factory was called — install_fake_llm did NOT intercept _get_llm on this path"
            )

        graph = compile(
            construct_from_functions("install_fake_agent", [research]),
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(_boom_factory),
        )

        install_fake_llm(monkeypatch, {"research": KResult(items=["installed"])})

        result = run(graph, input={"node_id": "REQ-1"}, config={"configurable": {"thread_id": "pf-install"}})
        assert result.get("research") == KResult(items=["installed"]), (
            "install_fake_llm did not resolve the fake on the _tool_loop._get_llm "
            "(agent) path — the second setattr target is likely wrong/missing"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. FakeLLM.__call__ signature + per-node routing
# ─────────────────────────────────────────────────────────────────────────────


class TestFakeLlmNodeNameRouting:
    """The router MUST declare node_name so _get_llm's param-filtering
    (LlmRuntime.llm_factory_params) does not strip it, and routing must select
    the per-node double keyed on the node_name it receives."""

    def test_call_signature_declares_node_name(self) -> None:
        from neograph.testing import FakeLLM

        params = inspect.signature(FakeLLM({"x": RawText(text="x")})).parameters
        assert "node_name" in params, (
            "FakeLLM.__call__ must declare node_name explicitly (or _get_llm "
            f"param-filtering strips it); got params: {list(params)}"
        )

    def test_routing_selects_per_node_output_when_node_name_passed(self) -> None:
        from neograph.testing import FakeLLM

        # Two keys: the node's own name AND the '' fallback. Correct routing must
        # pick the node's key, not the '' default — proving node_name reached the
        # factory and was used.
        fake = FakeLLM({"alpha": Claims(items=["from-alpha"]), "": Claims(items=["fallback"])})

        @node(mode="think", outputs=Claims, model="fast", prompt="test/extract")
        def alpha() -> Claims: ...

        import types as _types

        mod = _types.ModuleType("test_public_fakes_routing_mod")
        mod.alpha = alpha
        pipeline = construct_from_module(mod, name="public-fake-routing")

        graph = compile(pipeline, **build_test_compile_kwargs(), **build_fake_llm_kwargs(fake))
        result = run(graph, input={"node_id": "REQ-1"})

        assert result.get("alpha") == Claims(items=["from-alpha"]), (
            f"node_name routing degraded — alpha resolved to the '' fallback "
            f"instead of its own key: {result.get('alpha')!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. ANTI-DUPLICATION (R-M2, HYBRID)
# ─────────────────────────────────────────────────────────────────────────────


class TestAntiDuplicationHybridShim:
    """One implementation, two consumers. tests/fakes.py re-exports the migrated
    doubles from the public module (identity), does NOT redefine them, yet KEEPS
    its test-only registry plumbing (which must stay out of the public package)."""

    def test_tests_fakes_shim_still_imports_and_is_identical_to_public(self) -> None:
        """The 122 `from tests.fakes import ...` sites keep working AND resolve to
        the SAME object as the public module (proves single implementation)."""
        from neograph.testing.fakes import StructuredFake as PublicStructuredFake
        from tests.fakes import StructuredFake as ShimStructuredFake

        assert ShimStructuredFake is PublicStructuredFake, (
            "tests/fakes.py must RE-EXPORT StructuredFake from neograph.testing.fakes "
            "(one implementation), not hold its own copy"
        )

    def test_migrated_fakes_not_redefined_in_tests_fakes(self) -> None:
        """The migrated doubles must be imported/re-exported, never redefined, in
        tests/fakes.py. RED now: they are all still ClassDef/FunctionDef there."""
        tree = ast.parse(FAKES_PATH.read_text())
        defined = {
            n.name for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        redefined = sorted(set(_MIGRATED_FAKE_SYMBOLS) & defined)
        assert not redefined, (
            f"tests/fakes.py redefines migrated fake symbols {redefined} — after "
            f"dyp3 these must be re-exported from neograph.testing.fakes, not "
            f"duplicated (Core Invariant: exactly one implementation)"
        )

    def test_test_registry_plumbing_stays_out_of_public_package(self) -> None:
        """R-M2 HYBRID: the ezqz-removed registry helpers stay in tests/fakes.py
        and must NOT be dragged back into the public package."""
        import tests.fakes as tf

        for sym in _RETAINED_TEST_ONLY_SYMBOLS:
            assert hasattr(tf, sym), f"{sym} must remain in tests/fakes.py (R-M2 hybrid)"

        # RED now: neograph.testing.fakes does not exist. Once it does, it must NOT
        # expose the test-only registry plumbing.
        from neograph.testing import fakes as public_fakes

        leaked = [s for s in _RETAINED_TEST_ONLY_SYMBOLS if hasattr(public_fakes, s)]
        assert not leaked, (
            f"test-only registry helpers leaked into the public package {leaked} — "
            f"these were deliberately removed from src/ in ezqz and must stay test-only"
        )
