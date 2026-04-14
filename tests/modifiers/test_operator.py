"""Modifier tests — Operator interrupt/resume modifier"""

from __future__ import annotations

from typing import Annotated

from neograph import (
    Construct,
    Node,
    Operator,
    compile,
    construct_from_module,
    node,
    run,
)
from neograph.factory import register_scripted
from tests.schemas import (
    Claims,
    RawText,
    ValidationResult,
)

# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: Operator — human-in-the-loop interrupt
#
# A node produces a validation result. If validation fails,
# the graph pauses via interrupt(). Resume with human input.
# This proves: Operator modifier wires interrupt() correctly,
# graph pauses and resumes.
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: Operator — human-in-the-loop interrupt
#
# A node produces a validation result. If validation fails,
# the graph pauses via interrupt(). Resume with human input.
# This proves: Operator modifier wires interrupt() correctly,
# graph pauses and resumes.
# ═══════════════════════════════════════════════════════════════════════════

class TestOperator:
    def test_graph_pauses_when_operator_condition_truthy(self):
        """Graph pauses when Operator condition is met."""
        import types as _types

        from langgraph.checkpoint.memory import MemorySaver


        mod = _types.ModuleType("test_operator_mod")

        @node(
            mode="scripted",
            outputs=ValidationResult,
            interrupt_when=lambda state: (
                {"issues": state.check_quality.issues}
                if state.check_quality and not state.check_quality.passed
                else None
            ),
        )
        def check_quality() -> ValidationResult:
            return ValidationResult(passed=False, issues=["missing stakeholder coverage"])

        mod.check_quality = check_quality

        pipeline = construct_from_module(mod, name="test-operator")
        graph = compile(pipeline, checkpointer=MemorySaver())

        # Run — with checkpointer, interrupt returns result with __interrupt__
        config = {"configurable": {"thread_id": "test-interrupt"}}
        result = run(graph, input={"node_id": "test-001"}, config=config)

        # Verify the graph paused
        assert "__interrupt__" in result
        assert result["check_quality"].passed is False


# ═══════════════════════════════════════════════════════════════════════════
# TEST 7: Raw node alongside declarative nodes
#
# A @node(mode='raw') function mixed with @node declarations in the same Construct.
# This proves: raw escape hatch works, framework wires edges around it,
# data flows through raw node like any other.
# ═══════════════════════════════════════════════════════════════════════════





# ═══════════════════════════════════════════════════════════════════════════
# TEST 7: Raw node alongside declarative nodes
#
# A @node(mode='raw') function mixed with @node declarations in the same Construct.
# This proves: raw escape hatch works, framework wires edges around it,
# data flows through raw node like any other.
# ═══════════════════════════════════════════════════════════════════════════


class TestOperatorContinues:
    """Operator condition is falsy — graph continues without interrupt."""

    def test_graph_continues_when_operator_condition_falsy(self):
        """Graph runs through Operator without pausing when condition returns None."""
        import types as _types

        from langgraph.checkpoint.memory import MemorySaver


        mod = _types.ModuleType("test_operator_continues_mod")

        @node(
            mode="scripted",
            outputs=ValidationResult,
            interrupt_when=lambda state: None,  # always falsy
        )
        def check_quality() -> ValidationResult:
            return ValidationResult(passed=True, issues=[])

        mod.check_quality = check_quality

        pipeline = construct_from_module(mod, name="test-operator-pass")
        graph = compile(pipeline, checkpointer=MemorySaver())
        result = run(graph, input={"node_id": "test-001"}, config={"configurable": {"thread_id": "pass-test"}})

        assert result["check_quality"].passed is True
        assert result.get("human_feedback") is None





class TestOperatorResume:
    """Operator interrupt + resume flow."""

    def test_graph_resumes_when_human_feedback_provided(self):
        """Graph pauses at interrupt, resumes with human feedback via run()."""
        import types as _types

        from langgraph.checkpoint.memory import MemorySaver


        mod = _types.ModuleType("test_operator_resume_mod")

        @node(
            mode="scripted",
            outputs=ValidationResult,
            name="validate-thing",
            interrupt_when=lambda state: (
                {"issues": state.validate_thing.issues}
                if state.validate_thing and not state.validate_thing.passed
                else None
            ),
        )
        def validate_thing() -> ValidationResult:
            return ValidationResult(passed=False, issues=["bad coverage"])

        mod.validate_thing = validate_thing

        pipeline = construct_from_module(mod, name="test-resume")
        graph = compile(pipeline, checkpointer=MemorySaver())

        config = {"configurable": {"thread_id": "resume-test"}}

        # First run: hits interrupt — returns with __interrupt__
        result = run(graph, input={"node_id": "test-001"}, config=config)
        assert "__interrupt__" in result

        # Resume via run()
        result = run(graph, resume={"approved": True}, config=config)

        assert result["validate_thing"].passed is False
        assert result["human_feedback"] == {"approved": True}





class TestOperatorResumeFromInput:
    """FromInput DI must resolve after Operator resume (neograph-pd8j)."""

    def test_from_input_resolves_after_resume(self):
        """Nodes running after interrupt/resume still get FromInput values."""
        import types as _types

        from langgraph.checkpoint.memory import MemorySaver
        from pydantic import BaseModel

        from neograph import FromInput

        class Result(BaseModel, frozen=True):
            text: str

        mod = _types.ModuleType("test_pd8j_mod")

        @node(mode="scripted", outputs=Result,
              interrupt_when=lambda state: {"gate": True} if state.gate_node else None)
        def gate_node() -> Result:
            return Result(text="gate")

        @node(mode="scripted", outputs=Result)
        def after_gate(
            gate_node: Result,
            pipeline_id: Annotated[str, FromInput],
        ) -> Result:
            return Result(text=f"{pipeline_id}:{gate_node.text}")

        mod.gate_node = gate_node
        mod.after_gate = after_gate

        pipeline = construct_from_module(mod, name="test-pd8j")
        graph = compile(pipeline, checkpointer=MemorySaver())

        config = {"configurable": {"thread_id": "pd8j-test"}}
        result = run(graph, input={"node_id": "pd8j", "pipeline_id": "my-run-123"}, config=config)
        assert "__interrupt__" in result

        # Resume — after_gate runs, must resolve pipeline_id from original input
        result = run(graph, resume={"approved": True}, config=config)
        assert result["after_gate"].text == "my-run-123:gate"





class TestResumeWithFreshConfig:
    """Obligation R-04: resume with fresh config missing _neo_input (neograph-n6nt)."""

    def test_resume_with_fresh_config_warns_or_degrades_gracefully(self):
        """When resume is called with a fresh config (not the one from initial run),
        _neo_input is absent. FromInput(required=False) resolves to None."""
        import types as _types

        from langgraph.checkpoint.memory import MemorySaver
        from pydantic import BaseModel

        from neograph import FromInput

        class Result(BaseModel, frozen=True):
            text: str

        mod = _types.ModuleType("test_r04_mod")

        @node(mode="scripted", outputs=Result,
              interrupt_when=lambda state: {"gate": True} if state.gate_node else None)
        def gate_node() -> Result:
            return Result(text="gate")

        @node(mode="scripted", outputs=Result)
        def after_gate(
            gate_node: Result,
            pipeline_id: Annotated[str, FromInput(required=False)],
        ) -> Result:
            return Result(text=f"{pipeline_id}:{gate_node.text}")

        mod.gate_node = gate_node
        mod.after_gate = after_gate

        pipeline = construct_from_module(mod, name="test-r04")
        graph = compile(pipeline, checkpointer=MemorySaver())

        # First run with one config
        config1 = {"configurable": {"thread_id": "r04-test"}}
        result = run(graph, input={"node_id": "r04", "pipeline_id": "original"}, config=config1)
        assert "__interrupt__" in result

        # Resume with a FRESH config (same thread_id but no _neo_input stash)
        config2 = {"configurable": {"thread_id": "r04-test"}}
        result = run(graph, resume={"approved": True}, config=config2)
        # Without _neo_input, FromInput resolves to None
        assert result["after_gate"].text == "None:gate"

    def test_resume_with_original_config_resolves_correctly(self):
        """Control: reusing the original config (which has _neo_input) works."""
        import types as _types

        from langgraph.checkpoint.memory import MemorySaver
        from pydantic import BaseModel

        from neograph import FromInput

        class Result(BaseModel, frozen=True):
            text: str

        mod = _types.ModuleType("test_r04b_mod")

        @node(mode="scripted", outputs=Result,
              interrupt_when=lambda state: {"gate": True} if state.gate_node else None)
        def gate_node() -> Result:
            return Result(text="gate")

        @node(mode="scripted", outputs=Result)
        def after_gate(
            gate_node: Result,
            pipeline_id: Annotated[str, FromInput],
        ) -> Result:
            return Result(text=f"{pipeline_id}:{gate_node.text}")

        mod.gate_node = gate_node
        mod.after_gate = after_gate

        pipeline = construct_from_module(mod, name="test-r04b")
        graph = compile(pipeline, checkpointer=MemorySaver())

        config = {"configurable": {"thread_id": "r04b-test"}}
        result = run(graph, input={"node_id": "r04b", "pipeline_id": "kept"}, config=config)
        assert "__interrupt__" in result

        # Same config object — has _neo_input from first run
        result = run(graph, resume={"approved": True}, config=config)
        assert result["after_gate"].text == "kept:gate"

    def test_neo_input_stashed_in_config_after_run(self):
        """Obligation R-SE1: config is mutated with _neo_input after run() (neograph-cqap)."""

        register_scripted("se1_fn", lambda i, c: RawText(text="ok"))

        pipeline = Construct("se1-test", nodes=[
            Node.scripted("a", fn="se1_fn", outputs=RawText),
        ])
        graph = compile(pipeline)

        config = {"configurable": {"thread_id": "se1"}}
        run(graph, input={"node_id": "se1", "custom_field": "hello"}, config=config)

        # Direct assertion: _neo_input is stashed
        assert "_neo_input" in config["configurable"]
        assert config["configurable"]["_neo_input"]["node_id"] == "se1"
        assert config["configurable"]["_neo_input"]["custom_field"] == "hello"





class TestConstructOperator:
    """Construct | Operator — check condition after sub-pipeline completes."""

    def test_parent_pauses_when_sub_construct_operator_truthy(self):
        """Sub-pipeline runs, then Operator checks and interrupts."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_condition

        register_scripted("sub_validate", lambda input_data, config: ValidationResult(
            passed=False, issues=["coverage gap"],
        ))

        register_condition("sub_failed", lambda state: (
            {"issues": state.enrich.issues}
            if hasattr(state, 'enrich') and state.enrich and not state.enrich.passed
            else None
        ))

        sub = Construct(
            "enrich",
            input=Claims,
            output=ValidationResult,
            nodes=[Node.scripted("val", fn="sub_validate", outputs=ValidationResult)],
        ) | Operator(when="sub_failed")

        register_scripted("seed", lambda input_data, config: Claims(items=["data"]))

        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="seed", outputs=Claims),
            sub,
        ])
        graph = compile(parent, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "construct-op-test"}}

        result = run(graph, input={"node_id": "test-001"}, config=config)

        # Sub-pipeline ran and produced output
        assert isinstance(result["enrich"], ValidationResult)
        assert result["enrich"].passed is False
        # Interrupted
        assert "__interrupt__" in result

    def test_parent_continues_when_sub_construct_operator_falsy(self):
        """Sub-pipeline runs, condition is falsy, graph continues."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_condition

        register_scripted("sub_ok", lambda input_data, config: ValidationResult(
            passed=True, issues=[],
        ))

        register_condition("sub_check", lambda state: None)  # always passes

        sub = Construct(
            "check",
            input=Claims,
            output=ValidationResult,
            nodes=[Node.scripted("ok", fn="sub_ok", outputs=ValidationResult)],
        ) | Operator(when="sub_check")

        register_scripted("seed2", lambda input_data, config: Claims(items=["ok"]))
        register_scripted("done", lambda input_data, config: RawText(text="complete"))

        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="seed2", outputs=Claims),
            sub,
            Node.scripted("done", fn="done", outputs=RawText),
        ])
        graph = compile(parent, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "construct-op-pass"}}
        result = run(graph, input={"node_id": "test-001"}, config=config)

        assert result["done"].text == "complete"


