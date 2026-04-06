"""Renderer tests — XmlRenderer, DelimitedRenderer, JsonRenderer, describe_type,
render_input dispatch, renderer dispatch hierarchy, json_mode output schema,
render_prompt inspector, and three-surface parity.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field as PydanticField

from neograph import (
    Construct, Node, compile, run, configure_llm,
    XmlRenderer, DelimitedRenderer, JsonRenderer, Renderer,
    describe_type, render_prompt,
)
from neograph.renderers import render_input
from tests.fakes import StructuredFake, TextFake, configure_fake_llm
from tests.schemas import RawText, Claims, MatchResult


class TestXmlRenderer:
    """XmlRenderer: Pydantic AI format_as_xml style."""

    def test_flat_model(self):
        """Flat model fields become XML elements."""

        class Info(BaseModel):
            name: str
            age: int

        r = XmlRenderer()
        result = r.render(Info(name="Alice", age=30))
        assert "<name>Alice</name>" in result
        assert "<age>30</age>" in result

    def test_nested_model(self):
        """Nested BaseModel becomes nested XML."""

        class Address(BaseModel):
            city: str

        class Person(BaseModel):
            name: str
            address: Address

        r = XmlRenderer()
        result = r.render(Person(name="Bob", address=Address(city="NYC")))
        assert "<address>" in result
        assert "<city>NYC</city>" in result
        assert "</address>" in result

    def test_list_field(self):
        """List fields produce repeated <item> elements."""

        class Tags(BaseModel):
            values: list[str]

        r = XmlRenderer()
        result = r.render(Tags(values=["a", "b", "c"]))
        assert result.count("<item>") == 3
        assert "<item>a</item>" in result
        assert "<item>b</item>" in result
        assert "<item>c</item>" in result

    def test_emits_description_attr_once_when_include_field_info_once(self):
        """include_field_info='once' emits description attribute on first occurrence only."""

        class Doc(BaseModel):
            title: str = PydanticField(description="The document title")
            title2: str = PydanticField(description="Another title")

        r = XmlRenderer(include_field_info="once")
        result = r.render(Doc(title="Hello", title2="World"))
        assert 'description="The document title"' in result
        assert 'description="Another title"' in result

    def test_emits_description_attr_every_time_when_include_field_info_always(self):
        """include_field_info='always' emits description every time."""

        class Repeated(BaseModel):
            name: str = PydanticField(description="A name")

        r = XmlRenderer(include_field_info="always")
        result = r.render(Repeated(name="x"))
        assert 'description="A name"' in result

    def test_suppresses_descriptions_when_include_field_info_never(self):
        """include_field_info='never' suppresses descriptions."""

        class Described(BaseModel):
            name: str = PydanticField(description="Should not appear")

        r = XmlRenderer(include_field_info="never")
        result = r.render(Described(name="x"))
        assert "description=" not in result

    def test_preserves_real_newlines_when_rendering_prose(self):
        r"""Multi-line prose is NOT JSON-escaped (no literal backslash-n)."""

        class Article(BaseModel):
            body: str

        prose = "Line one.\nLine two.\nLine three."
        r = XmlRenderer()
        result = r.render(Article(body=prose))
        # The literal text \n should NOT appear — real newlines should
        assert "\\n" not in result
        assert "Line one.\nLine two.\nLine three." in result


class TestDelimitedRenderer:
    """DelimitedRenderer: DSPy-style [[ ## field ## ]] headers."""

    def test_flat_model(self):
        """Flat model fields get delimited headers."""

        class Info(BaseModel):
            name: str
            age: int

        r = DelimitedRenderer()
        result = r.render(Info(name="Alice", age=30))
        assert "[[ ## name ## ]]" in result
        assert "Alice" in result
        assert "[[ ## age ## ]]" in result
        assert "30" in result

    def test_nested_model(self):
        """Nested models use dotted header prefixes."""

        class Address(BaseModel):
            city: str

        class Person(BaseModel):
            name: str
            address: Address

        r = DelimitedRenderer()
        result = r.render(Person(name="Bob", address=Address(city="NYC")))
        assert "[[ ## name ## ]]" in result
        assert "[[ ## address.city ## ]]" in result

    def test_list_field(self):
        """Lists use bullet-point format."""

        class Tags(BaseModel):
            values: list[str]

        r = DelimitedRenderer()
        result = r.render(Tags(values=["a", "b"]))
        assert "[[ ## values ## ]]" in result
        assert "- a" in result
        assert "- b" in result


class TestJsonRenderer:
    """JsonRenderer: explicit opt-in backward compat."""

    def test_renders_via_model_dump_json(self):
        """BaseModel rendered via model_dump_json."""

        class Info(BaseModel):
            name: str
            age: int

        r = JsonRenderer()
        result = r.render(Info(name="Alice", age=30))
        import json as _json

        parsed = _json.loads(result)
        assert parsed == {"name": "Alice", "age": 30}

    def test_respects_custom_indent_param(self):
        """Custom indent parameter is respected."""

        class Info(BaseModel):
            name: str

        r = JsonRenderer(indent=4)
        result = r.render(Info(name="X"))
        # indent=4 produces 4-space indentation
        assert "    " in result


class TestRenderInput:
    """render_input() dispatch helper."""

    def test_returns_raw_value_when_renderer_is_none(self):
        """When renderer is None, raw value returned unchanged."""

        class Info(BaseModel):
            name: str

        obj = Info(name="raw")
        result = render_input(obj, renderer=None)
        assert result is obj

    def test_renders_each_value_independently_when_dict_input(self):
        """Dict input (fan-in) renders each value independently."""

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        r = XmlRenderer()
        result = render_input({"a": A(x="hello"), "b": B(y="world")}, renderer=r)
        assert isinstance(result, dict)
        assert "<x>hello</x>" in result["a"]
        assert "<y>world</y>" in result["b"]

    def test_model_render_for_prompt_wins_over_renderer(self):
        """Model with render_for_prompt() method takes precedence over renderer."""

        class Custom(BaseModel):
            name: str

            def render_for_prompt(self) -> str:
                return f"CUSTOM: {self.name}"

        r = XmlRenderer()
        result = render_input(Custom(name="test"), renderer=r)
        assert result == "CUSTOM: test"

    def test_custom_class_satisfies_renderer_protocol(self):
        """Any object with render(value) -> str satisfies the Renderer protocol."""

        class MyRenderer:
            def render(self, value: Any) -> str:
                return f"RENDERED: {value}"

        assert isinstance(MyRenderer(), Renderer)


# ═══════════════════════════════════════════════════════════════════════════
# TestDescribeType — TypeScript-style schema emitter
# ═══════════════════════════════════════════════════════════════════════════

class TestDescribeType:
    """Tests for describe_type() — two-pass Pydantic model walker that emits
    TypeScript-style schema notation."""

    def test_maps_primitives_to_typescript_names(self):
        """Primitive types map to their TypeScript-style names."""
        from neograph import describe_type

        class Prims(BaseModel):
            name: str
            age: int
            score: float
            active: bool

        result = describe_type(Prims, prefix="")
        assert "name: string" in result
        assert "age: int" in result
        assert "score: float" in result
        assert "active: bool" in result

    def test_list_field(self):
        """list[X] renders as [X]."""
        from neograph import describe_type

        class WithList(BaseModel):
            tags: list[str]

        result = describe_type(WithList, prefix="")
        assert "tags: [string]" in result

    def test_renders_dict_as_object_generic(self):
        """dict[K, V] renders as object<K, V>."""
        from neograph import describe_type

        class WithDict(BaseModel):
            metadata: dict[str, int]

        result = describe_type(WithDict, prefix="")
        assert "metadata: object<string, int>" in result

    def test_appends_or_null_when_optional(self):
        """Optional fields (with default None) get ' or null' suffix."""
        from typing import Optional
        from neograph import describe_type

        class WithOptional(BaseModel):
            name: str
            nickname: Optional[str] = None

        result = describe_type(WithOptional, prefix="")
        assert "nickname: string or null" in result
        # name should NOT have 'or null'
        lines = [l.strip() for l in result.splitlines()]
        name_line = [l for l in lines if l.startswith("name:")][0]
        assert "null" not in name_line

    def test_joins_union_types_with_or_splitter(self):
        """Union[A, B] renders with or_splitter."""
        from typing import Union
        from neograph import describe_type

        class WithUnion(BaseModel):
            value: Union[str, int]

        result = describe_type(WithUnion, prefix="", or_splitter=" or ")
        assert "value: string or int" in result

    def test_renders_literal_values_as_quoted_strings(self):
        """Literal types render as quoted strings joined by or_splitter."""
        from typing import Literal
        from neograph import describe_type

        class WithLiteral(BaseModel):
            status: Literal["active", "inactive", "pending"]

        result = describe_type(WithLiteral, prefix="")
        assert '"active" or "inactive" or "pending"' in result

    def test_inlines_nested_model_when_used_once(self):
        """Nested BaseModel renders inline when it appears once."""
        from neograph import describe_type

        class Inner(BaseModel):
            value: int

        class Outer(BaseModel):
            child: Inner

        result = describe_type(Outer, prefix="")
        # Inner appears only once, so it should be inlined, not hoisted
        assert "type Inner" not in result
        assert "child:" in result
        assert "value: int" in result

    def test_renders_field_description_as_inline_comment(self):
        """Field(description=...) renders as an inline // comment."""
        from pydantic import Field
        from neograph import describe_type

        class Documented(BaseModel):
            name: str = Field(description="the person name")
            age: int = Field(description="years old")

        result = describe_type(Documented, prefix="")
        assert "// the person name" in result
        assert "// years old" in result

    def test_hoists_class_when_used_twice_with_auto(self):
        """hoist_classes='auto' hoists classes that appear 2+ times."""
        from neograph import describe_type

        class Shared(BaseModel):
            x: int

        class Root(BaseModel):
            a: Shared
            b: Shared

        result = describe_type(Root, prefix="")
        assert "type Shared = {" in result
        # Both fields should reference Shared by name, not inline it
        lines = result.splitlines()
        body_lines = [l.strip() for l in lines if l.strip().startswith(("a:", "b:"))]
        for line in body_lines:
            assert "Shared" in line

    def test_hoists_every_nested_model_when_hoist_all(self):
        """hoist_classes='all' hoists every nested BaseModel."""
        from neograph import describe_type

        class Once(BaseModel):
            v: str

        class Container(BaseModel):
            only: Once

        result = describe_type(Container, prefix="", hoist_classes="all")
        assert "type Once = {" in result

    def test_hoists_only_named_classes_when_explicit_list(self):
        """hoist_classes=['Foo'] hoists only named classes."""
        from neograph import describe_type

        class Foo(BaseModel):
            a: int

        class Bar(BaseModel):
            b: str

        class Root(BaseModel):
            f: Foo
            g: Bar

        result = describe_type(Root, prefix="", hoist_classes=["Foo"])
        assert "type Foo = {" in result
        assert "type Bar" not in result

    def test_renders_enum_as_quoted_values_when_not_hoisted(self):
        """Enum types render as inlined quoted values."""
        import enum
        from neograph import describe_type

        class Color(enum.Enum):
            RED = "red"
            GREEN = "green"
            BLUE = "blue"

        class WithEnum(BaseModel):
            color: Color

        result = describe_type(
            WithEnum, prefix="", always_hoist_enums=False,
        )
        assert '"red"' in result
        assert '"green"' in result

    def test_hoists_enum_as_declaration_when_always_hoist_enums(self):
        """always_hoist_enums=True hoists Enum as 'enum Foo { ... }' declaration."""
        import enum
        from neograph import describe_type

        class Status(enum.Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        class WithStatus(BaseModel):
            status: Status

        result = describe_type(WithStatus, prefix="", always_hoist_enums=True)
        assert 'enum Status { "active", "inactive" }' in result

    def test_terminates_without_recursion_on_circular_model(self):
        """Circular model references don't cause infinite recursion."""
        from neograph import describe_type

        class TreeNode(BaseModel):
            name: str
            children: list["TreeNode"] = []

        TreeNode.model_rebuild()

        result = describe_type(TreeNode, prefix="")
        assert "name: string" in result
        # Should terminate without RecursionError

    def test_renders_empty_model_as_empty_braces(self):
        """Empty model renders as {}."""
        from neograph import describe_type

        class Empty(BaseModel):
            pass

        result = describe_type(Empty, prefix="")
        assert "{}" in result

    def test_prepends_custom_prefix(self):
        """Custom prefix appears at the start of output."""
        from neograph import describe_type

        class Simple(BaseModel):
            x: int

        result = describe_type(Simple, prefix="Return JSON:")
        assert result.startswith("Return JSON:")

    def test_output_shorter_than_json_schema(self):
        """describe_type output is significantly shorter than JSON Schema."""
        import json
        from neograph import describe_type

        class Education(BaseModel):
            school: str
            graduation_year: int

        class Resume(BaseModel):
            name: str
            education: list[Education]

        ts_output = describe_type(Resume, prefix="")
        json_schema = json.dumps(Resume.model_json_schema(), indent=2)
        assert len(ts_output) < len(json_schema)


# ═══════════════════════════════════════════════════════════════════════════
# TestRendererDispatch (neograph-46w)
#
# Tests for the 5-level renderer dispatch hierarchy:
#   1. Model method render_for_prompt() wins over any renderer
#   2. Node(renderer=...) applies
#   3. Construct(renderer=...) propagates to child nodes
#   3b. Node with own renderer beats Construct default
#   4. Global renderer (mocked — actual configure_llm integration is Phase 3)
#   5. No renderer = raw passthrough
# ═══════════════════════════════════════════════════════════════════════════

class TestRendererDispatch:
    """Renderer dispatch hierarchy: model method > node > construct > global > None."""

    def test_model_render_for_prompt_wins_over_all_renderers(self):
        """Level 1: model with render_for_prompt() method wins over any renderer."""

        class CustomModel(BaseModel):
            name: str
            value: int

            def render_for_prompt(self) -> str:
                return f"CUSTOM: {self.name}={self.value}"

        instance = CustomModel(name="test", value=42)
        xml = XmlRenderer()
        result = render_input(instance, renderer=xml)
        assert result == "CUSTOM: test=42"

    def test_node_renderer_field_is_stored(self):
        """Level 2: Node(renderer=XmlRenderer()) stores renderer on the node."""
        xml = XmlRenderer()
        n = Node("render-test", outputs=Claims, renderer=xml)
        assert n.renderer is xml

    def test_construct_propagates_renderer_to_children(self):
        """Level 3: Construct(renderer=...) propagates to child nodes."""
        xml = XmlRenderer()
        child = Node.scripted("child", fn="noop", outputs=Claims)
        assert child.renderer is None

        pipeline = Construct("prop-test", renderer=xml, nodes=[child])
        assert child.renderer is xml
        assert pipeline.renderer is xml

    def test_node_renderer_beats_construct_default(self):
        """Level 3 override: Node with own renderer beats Construct default."""
        xml = XmlRenderer()
        json_r = JsonRenderer()
        child = Node("child", mode="scripted", scripted_fn="noop", outputs=Claims, renderer=json_r)

        pipeline = Construct("override-test", renderer=xml, nodes=[child])
        # Node's own renderer should NOT be overwritten
        assert child.renderer is json_r
        assert pipeline.renderer is xml

    def test_global_renderer_applied_when_node_has_none(self):
        """Level 4: global renderer checked when node.renderer is None.

        Actual configure_llm integration is Phase 3. For now, verify that
        render_input with an explicit renderer works as the global fallback
        would use it.
        """

        class Info(BaseModel):
            name: str

        xml = XmlRenderer()
        result = render_input(Info(name="test"), renderer=xml)
        assert "<name>test</name>" in result

    def test_returns_raw_object_when_no_renderer(self):
        """Level 5: no renderer = raw passthrough (identical to pre-renderer behavior)."""

        class Info(BaseModel):
            name: str

        instance = Info(name="test")
        result = render_input(instance, renderer=None)
        assert result is instance  # exact same object, no transformation

    def test_renders_each_dict_value_independently_when_fan_in(self):
        """Fan-in dict: each value rendered independently."""

        class Alpha(BaseModel):
            text: str

        class Beta(BaseModel):
            text: str

        xml = XmlRenderer()
        fan_in = {"alpha": Alpha(text="a"), "beta": Beta(text="b")}
        result = render_input(fan_in, renderer=xml)
        assert isinstance(result, dict)
        assert "<text>a</text>" in result["alpha"]
        assert "<text>b</text>" in result["beta"]

    def test_render_for_prompt_wins_per_value_in_fan_in_dict(self):
        """Fan-in: model with render_for_prompt still wins per-value in a dict."""

        class Custom(BaseModel):
            x: int

            def render_for_prompt(self) -> str:
                return f"X={self.x}"

        class Normal(BaseModel):
            y: int

        xml = XmlRenderer()
        fan_in = {"a": Custom(x=1), "b": Normal(y=2)}
        result = render_input(fan_in, renderer=xml)
        assert result["a"] == "X=1"
        assert "<y>2</y>" in result["b"]

    def test_renders_none_as_string_when_renderer_set(self):
        """None input passes through unchanged regardless of renderer."""
        xml = XmlRenderer()
        result = render_input(None, renderer=xml)
        # None has no render_for_prompt and is not dict/BaseModel —
        # renderer.render(None) returns "None" as str
        assert result == "None"

    def test_all_builtins_satisfy_renderer_protocol(self):
        """Renderer protocol is runtime-checkable: built-ins satisfy it."""
        assert isinstance(XmlRenderer(), Renderer)
        assert isinstance(DelimitedRenderer(), Renderer)
        assert isinstance(JsonRenderer(), Renderer)

        # A class with a render(value) method also satisfies the protocol
        class CustomRenderer:
            def render(self, value: Any) -> str:
                return "custom"

        assert isinstance(CustomRenderer(), Renderer)


# ═══════════════════════════════════════════════════════════════════════════
# json_mode output_schema generation
# ═══════════════════════════════════════════════════════════════════════════


class TestJsonModeOutputSchema:
    """invoke_structured passes output_schema to prompt_compiler for json_mode."""

    def test_passes_describe_type_schema_when_json_mode(self):
        """json_mode strategy generates describe_type() output and passes as output_schema."""
        compiler_calls = []

        def tracking_compiler(template, data, **kw):
            compiler_calls.append(kw)
            return [{"role": "user", "content": "test"}]

        configure_fake_llm(
            lambda tier: TextFake('{"items": ["schema-test"]}'),
            prompt_compiler=tracking_compiler,
        )

        n = Node(
            name="extract",
            mode="produce",
            outputs=Claims,
            model="fast",
            prompt="test",
            llm_config={"output_strategy": "json_mode"},
        )
        pipeline = Construct("test-schema", nodes=[n])
        graph = compile(pipeline)
        run(graph, input={"node_id": "test"})

        assert len(compiler_calls) == 1
        schema = compiler_calls[0].get("output_schema")
        assert schema is not None
        # describe_type produces TypeScript-style notation containing the field name
        assert "items" in schema

    def test_omits_output_schema_when_structured_strategy(self):
        """structured strategy does not generate output_schema."""
        compiler_calls = []

        def tracking_compiler(template, data, **kw):
            compiler_calls.append(kw)
            return [{"role": "user", "content": "test"}]

        configure_fake_llm(
            lambda tier: StructuredFake(lambda model: model(items=["ok"])),
            prompt_compiler=tracking_compiler,
        )

        n = Node(
            name="extract",
            mode="produce",
            outputs=Claims,
            model="fast",
            prompt="test",
            llm_config={"output_strategy": "structured"},
        )
        pipeline = Construct("test-no-schema", nodes=[n])
        graph = compile(pipeline)
        run(graph, input={"node_id": "test"})

        assert len(compiler_calls) == 1
        assert compiler_calls[0].get("output_schema") is None

    def test_omits_output_schema_when_text_strategy(self):
        """text strategy does not generate output_schema."""
        compiler_calls = []

        def tracking_compiler(template, data, **kw):
            compiler_calls.append(kw)
            return [{"role": "user", "content": "test"}]

        configure_fake_llm(
            lambda tier: TextFake('{"items": ["text-test"]}'),
            prompt_compiler=tracking_compiler,
        )

        n = Node(
            name="extract",
            mode="produce",
            outputs=Claims,
            model="fast",
            prompt="test",
            llm_config={"output_strategy": "text"},
        )
        pipeline = Construct("test-text-no-schema", nodes=[n])
        graph = compile(pipeline)
        run(graph, input={"node_id": "test"})

        assert len(compiler_calls) == 1
        assert compiler_calls[0].get("output_schema") is None

    def test_old_compiler_without_output_schema_param_still_works(self):
        """Old compilers that don't accept output_schema= still work (param filtering)."""
        def old_compiler(template, data, *, node_name=None, config=None):
            return [{"role": "user", "content": "test"}]

        configure_fake_llm(
            lambda tier: TextFake('{"items": ["compat"]}'),
            prompt_compiler=old_compiler,
        )

        n = Node(
            name="extract",
            mode="produce",
            outputs=Claims,
            model="fast",
            prompt="test",
            llm_config={"output_strategy": "json_mode"},
        )
        pipeline = Construct("test-compat", nodes=[n])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test"})

        # Still works — old compiler just doesn't receive output_schema
        assert result["extract"].items == ["compat"]


# ═══════════════════════════════════════════════════════════════════════════
# render_prompt() inspector
# ═══════════════════════════════════════════════════════════════════════════


class TestRenderPromptInspector:
    """render_prompt() returns the exact prompt without making an LLM call."""

    def test_returns_formatted_messages_without_llm_call(self):
        """render_prompt returns formatted messages from the prompt compiler."""
        from neograph._llm import configure_llm, render_prompt

        configure_llm(
            llm_factory=lambda tier: None,
            prompt_compiler=lambda template, data, **kw: [
                {"role": "system", "content": f"Template: {template}"},
                {"role": "user", "content": str(data)},
            ],
        )

        n = Node(name="test-node", mode="produce", outputs=Claims, model="fast", prompt="my/template")
        result = render_prompt(n, "hello world")

        assert "[system]" in result
        assert "Template: my/template" in result
        assert "[user]" in result
        assert "hello world" in result

    def test_applies_node_renderer_before_compilation(self):
        """render_prompt applies node.renderer before compilation."""
        from neograph._llm import configure_llm, render_prompt
        from neograph.renderers import XmlRenderer

        compiled_data = []

        def capturing_compiler(template, data, **kw):
            compiled_data.append(data)
            return [{"role": "user", "content": str(data)}]

        configure_llm(
            llm_factory=lambda tier: None,
            prompt_compiler=capturing_compiler,
        )

        class MyInput(BaseModel):
            name: str
            value: int

        n = Node(
            name="test-rendered",
            mode="produce",
            outputs=Claims,
            model="fast",
            prompt="test",
            renderer=XmlRenderer(),
        )
        result = render_prompt(n, MyInput(name="Alice", value=42))

        assert len(compiled_data) == 1
        # XmlRenderer produces XML-tagged output
        rendered = compiled_data[0]
        assert isinstance(rendered, str)
        assert "<name>Alice</name>" in rendered
        assert "<value>42</value>" in rendered

    def test_passes_output_schema_to_compiler_when_json_mode(self):
        """render_prompt for json_mode node passes output_schema to compiler."""
        from neograph._llm import configure_llm, render_prompt

        compiler_kwargs = []

        def tracking_compiler(template, data, **kw):
            compiler_kwargs.append(kw)
            return [{"role": "user", "content": "test"}]

        configure_llm(
            llm_factory=lambda tier: None,
            prompt_compiler=tracking_compiler,
        )

        n = Node(
            name="json-node",
            mode="produce",
            outputs=Claims,
            model="fast",
            prompt="test",
            llm_config={"output_strategy": "json_mode"},
        )
        render_prompt(n, "some input")

        assert len(compiler_kwargs) == 1
        schema = compiler_kwargs[0].get("output_schema")
        assert schema is not None
        assert "items" in schema


# ═══════════════════════════════════════════════════════════════════════════
# RENDERER THREE-SURFACE PARITY — @node / declarative / programmatic / ForwardConstruct
# ═══════════════════════════════════════════════════════════════════════════


class TestRendererThreeSurfaces:
    """Renderer field is correctly set and propagated across all three API surfaces."""

    def test_renderer_set_via_at_node_decorator(self):
        """@node(renderer=XmlRenderer()) sets renderer on the resulting Node."""
        from neograph import node
        from neograph.renderers import XmlRenderer

        xml = XmlRenderer()

        @node(renderer=xml, outputs=Claims, prompt="test", model="fast")
        def my_produce(topic: RawText) -> Claims: ...

        assert my_produce.renderer is xml

    def test_renderer_survives_construct_assembly_when_declarative(self):
        """Node('x', renderer=XmlRenderer()) survives Construct assembly."""
        xml = XmlRenderer()

        child = Node(
            "render-decl",
            mode="produce",
            outputs=Claims,
            model="fast",
            prompt="test",
            renderer=xml,
        )
        pipeline = Construct("decl-test", nodes=[child])
        assert child.renderer is xml
        # Construct doesn't override an already-set renderer
        assert pipeline.nodes[0].renderer is xml

    def test_renderer_propagates_from_construct_to_children(self):
        """Construct(renderer=...) propagates through to child nodes without own renderer."""
        xml = XmlRenderer()
        child = Node.scripted("prog-child", fn="noop", outputs=Claims)
        assert child.renderer is None

        pipeline = Construct("prog-test", renderer=xml, nodes=[child])

        # Child should inherit from Construct
        assert child.renderer is xml
        # Verify propagation through modifier: Each on Construct level
        child2 = Node.scripted("prog-child2", fn="noop2", outputs=MatchResult)
        pipeline2 = Construct("prog-test2", renderer=xml, nodes=[child2])
        assert child2.renderer is xml

    def test_renderer_propagates_in_forward_construct(self):
        """ForwardConstruct(renderer=...) propagates renderer to traced nodes."""
        from neograph import ForwardConstruct

        xml = XmlRenderer()

        class RenderPipeline(ForwardConstruct):
            a = Node.scripted("a", fn="a_fn", outputs=RawText)
            b = Node.scripted("b", fn="b_fn", outputs=Claims)

            def forward(self, topic):
                x = self.a(topic)
                return self.b(x)

        pipeline = RenderPipeline(renderer=xml)
        assert pipeline.renderer is xml
        # Traced nodes should have renderer propagated
        for n in pipeline.nodes:
            assert n.renderer is xml

