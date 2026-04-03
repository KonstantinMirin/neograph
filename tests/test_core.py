"""Core tests — Node, Tool, Construct, modifiers, state compilation."""

from __future__ import annotations

from pydantic import BaseModel

from neograph import Construct, Node, Operator, Oracle, Replicate, Tool


# ── Test schemas ──────────────────────────────────────────────────────────

class InputA(BaseModel, frozen=True):
    text: str

class OutputA(BaseModel, frozen=True):
    result: str

class OutputB(BaseModel, frozen=True):
    items: list[str]


# ── Tool ──────────────────────────────────────────────────────────────────

class TestTool:
    def test_tool_creation(self):
        t = Tool(name="search", budget=5)
        assert t.name == "search"
        assert t.budget == 5
        assert t.config == {}

    def test_tool_with_config(self):
        t = Tool(name="read", budget=10, config={"max_chars": 6000})
        assert t.config["max_chars"] == 6000

    def test_tool_frozen(self):
        t = Tool(name="search", budget=5)
        try:
            t.name = "other"  # type: ignore[misc]
            assert False, "Should be frozen"
        except Exception:
            pass


# ── Node ──────────────────────────────────────────────────────────────────

class TestNode:
    def test_produce_node(self):
        n = Node(
            name="classify",
            mode="produce",
            input=InputA,
            output=OutputA,
            model="reason",
            prompt="rw/classify",
        )
        assert n.name == "classify"
        assert n.mode == "produce"
        assert n.input is InputA
        assert n.output is OutputA
        assert n.modifiers == []

    def test_gather_node_with_tools(self):
        search = Tool(name="search", budget=5)
        read = Tool(name="read", budget=10)
        n = Node(
            name="explore",
            mode="gather",
            input=InputA,
            output=OutputA,
            model="reason",
            prompt="explore",
            tools=[search, read],
        )
        assert len(n.tools) == 2
        assert n.tools[0].budget == 5
        assert n.tools[1].budget == 10

    def test_scripted_node(self):
        n = Node.scripted("build-catalog", fn="build_catalog", output=str)
        assert n.mode == "scripted"
        assert n.scripted_fn == "build_catalog"

    def test_pipe_oracle(self):
        n = Node(name="decompose", mode="produce", output=OutputA, prompt="x")
        modified = n | Oracle(n=3, merge_prompt="merge")
        assert modified.has_modifier(Oracle)
        oracle = modified.get_modifier(Oracle)
        assert oracle is not None
        assert oracle.n == 3

    def test_pipe_replicate(self):
        n = Node(name="verify", mode="gather", output=OutputA, prompt="x")
        modified = n | Replicate(over="clusters.items", key="label")
        assert modified.has_modifier(Replicate)
        rep = modified.get_modifier(Replicate)
        assert rep is not None
        assert rep.over == "clusters.items"

    def test_pipe_operator(self):
        n = Node(name="validate", mode="produce", output=OutputA, prompt="x")
        modified = n | Operator(when="has_failures")
        assert modified.has_modifier(Operator)

    def test_pipe_chaining(self):
        n = Node(name="write-si", mode="produce", output=OutputA, prompt="x")
        modified = n | Oracle(n=3) | Operator(when="has_oqs")
        assert modified.has_modifier(Oracle)
        assert modified.has_modifier(Operator)
        assert len(modified.modifiers) == 2

    def test_pipe_does_not_mutate_original(self):
        n = Node(name="test", mode="produce", output=OutputA, prompt="x")
        modified = n | Oracle(n=3)
        assert len(n.modifiers) == 0
        assert len(modified.modifiers) == 1


# ── Construct ─────────────────────────────────────────────────────────────

class TestConstruct:
    def test_basic_construct(self):
        a = Node(name="step-a", mode="produce", output=OutputA, prompt="a")
        b = Node(name="step-b", mode="produce", input=OutputA, output=OutputB, prompt="b")
        c = Construct(name="test-pipeline", description="test", nodes=[a, b])
        assert len(c.nodes) == 2
        assert c.name == "test-pipeline"

    def test_construct_with_modifiers(self):
        a = Node(name="gen", mode="produce", output=OutputA, prompt="a") | Oracle(n=3)
        b = Node(name="verify", mode="gather", output=OutputB, prompt="b") | Replicate(over="x", key="id")
        c = Construct(name="complex", nodes=[a, b])
        assert c.nodes[0].has_modifier(Oracle)
        assert c.nodes[1].has_modifier(Replicate)


# ── Tool Budget Tracker ──────────────────────────────────────────────────

class TestToolBudgetTracker:
    def test_budget_enforcement(self):
        from neograph.tool import ToolBudgetTracker

        tracker = ToolBudgetTracker([
            Tool(name="search", budget=2),
            Tool(name="read", budget=3),
        ])

        assert tracker.can_call("search")
        tracker.record_call("search")
        assert tracker.can_call("search")
        tracker.record_call("search")
        assert not tracker.can_call("search")  # budget exhausted
        assert tracker.can_call("read")  # still has budget

    def test_unlimited_budget(self):
        from neograph.tool import ToolBudgetTracker

        tracker = ToolBudgetTracker([Tool(name="search", budget=0)])
        for _ in range(100):
            assert tracker.can_call("search")
            tracker.record_call("search")

    def test_all_exhausted(self):
        from neograph.tool import ToolBudgetTracker

        tracker = ToolBudgetTracker([
            Tool(name="a", budget=1),
            Tool(name="b", budget=1),
        ])
        assert not tracker.all_exhausted()
        tracker.record_call("a")
        assert not tracker.all_exhausted()
        tracker.record_call("b")
        assert tracker.all_exhausted()

    def test_exhausted_tools(self):
        from neograph.tool import ToolBudgetTracker

        tracker = ToolBudgetTracker([
            Tool(name="search", budget=1),
            Tool(name="read", budget=2),
        ])
        tracker.record_call("search")
        assert tracker.exhausted_tools() == {"search"}


# ── State Compilation ────────────────────────────────────────────────────

class TestStateCompilation:
    def test_basic_state_generation(self):
        from neograph.state import compile_state_model

        a = Node(name="step-a", mode="produce", output=OutputA, prompt="a")
        b = Node(name="step-b", mode="produce", output=OutputB, prompt="b")
        construct = Construct(name="test", nodes=[a, b])

        StateModel = compile_state_model(construct)

        # Check generated fields
        fields = StateModel.model_fields
        assert "step_a" in fields
        assert "step_b" in fields
        assert "node_id" in fields
        assert "total_cost" in fields
        assert "completed_atoms" in fields

    def test_state_instantiation(self):
        from neograph.state import compile_state_model

        a = Node(name="classify", mode="produce", output=OutputA, prompt="a")
        construct = Construct(name="test", nodes=[a])

        StateModel = compile_state_model(construct)
        state = StateModel(node_id="test-001")

        assert state.node_id == "test-001"
        assert state.classify is None  # not yet populated
        assert state.total_cost == 0.0

    def test_replicate_generates_dict_field(self):
        from neograph.state import compile_state_model

        n = Node(name="verify", mode="gather", output=OutputA, prompt="x") | Replicate(over="items", key="id")
        construct = Construct(name="test", nodes=[n])

        StateModel = compile_state_model(construct)
        state = StateModel(node_id="test")

        assert state.verify is None  # dict field, starts None
