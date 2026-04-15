"""Renderer tests — XmlRenderer, DelimitedRenderer, JsonRenderer, describe_type,
render_input dispatch, renderer dispatch hierarchy, json_mode output schema,
render_prompt inspector, and three-surface parity.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import Field as PydanticField

from neograph import (
    Construct,
    DelimitedRenderer,
    JsonRenderer,
    Node,
    Renderer,
    XmlRenderer,
    compile,
    describe_type,
    run,
)
from neograph.renderers import render_input
from tests.fakes import StructuredFake, TextFake, configure_fake_llm
from tests.schemas import Claims, MatchResult, RawText


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

    def test_returns_baml_when_renderer_is_none(self):
        """When renderer is None, Pydantic models get BAML rendering (neograph-qybn)."""

        class Info(BaseModel):
            name: str

        obj = Info(name="raw")
        result = render_input(obj, renderer=None)
        assert isinstance(result, str)
        assert "name" in result
        assert "raw" in result

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

    def test_render_for_prompt_returning_model_is_re_rendered(self):
        """render_for_prompt() returning a BaseModel gets BAML-rendered automatically.

        FEATURE neograph-bwbt: typed presentation projections.
        """

        class Presentation(BaseModel):
            summary: str
            score: float

        class FullData(BaseModel):
            raw: str
            internal_id: int

            def render_for_prompt(self) -> Presentation:
                return Presentation(summary=self.raw.upper(), score=0.95)

        r = XmlRenderer()
        result = render_input(FullData(raw="hello", internal_id=42), renderer=r)
        # Should be XML-rendered Presentation, not a raw Presentation object
        assert isinstance(result, str)
        assert "<summary>" in result
        assert "HELLO" in result
        assert "internal_id" not in result  # projection strips internal fields

    def test_exclude_true_fields_omitted_from_describe_type(self):
        """Fields with exclude=True must not appear in BAML schema.

        BUG neograph-uau8: describe_type renders exclude=True fields,
        causing LLMs to produce values for pipeline-internal fields.
        """
        from neograph import describe_type

        class Item(BaseModel):
            name: str
            internal_id: str = PydanticField(exclude=True, default="auto")

        schema = describe_type(Item, prefix="")
        assert "name" in schema
        assert "internal_id" not in schema

    def test_exclude_true_fields_omitted_from_xml_renderer(self):
        """XmlRenderer must skip exclude=True fields."""
        class Item(BaseModel):
            name: str
            internal_id: str = PydanticField(exclude=True, default="set-by-pipeline")

        r = XmlRenderer()
        result = r.render(Item(name="test", internal_id="abc"))
        assert "<name>" in result
        assert "internal_id" not in result

    def test_exclude_true_fields_omitted_from_delimited_renderer(self):
        """DelimitedRenderer must skip exclude=True fields."""
        class Item(BaseModel):
            name: str
            internal_id: str = PydanticField(exclude=True, default="set-by-pipeline")

        r = DelimitedRenderer()
        result = r.render(Item(name="test", internal_id="abc"))
        assert "name" in result
        assert "internal_id" not in result

    def test_exclude_from_output_hidden_in_schema_but_visible_in_renderer(self):
        """ExcludeFromOutput fields: hidden from describe_type, visible in renderers.

        FEATURE neograph-tpab: pipeline-internal fields should be renderable
        as input but hidden from LLM output schema.
        """
        from typing import Annotated
        from neograph import describe_type
        from neograph import ExcludeFromOutput

        class SearchResult(BaseModel):
            answer: str
            source_url: Annotated[str, ExcludeFromOutput] = ""

        # Schema (output) — should NOT include source_url
        schema = describe_type(SearchResult, prefix="")
        assert "answer" in schema
        assert "source_url" not in schema

        # XML render (input) — SHOULD include source_url
        r = XmlRenderer()
        result = r.render(SearchResult(answer="42", source_url="https://example.com"))
        assert "<answer>" in result
        assert "source_url" in result
        assert "example.com" in result

        # Delimited render (input) — SHOULD include source_url
        d = DelimitedRenderer()
        result_d = d.render(SearchResult(answer="42", source_url="https://example.com"))
        assert "answer" in result_d
        assert "source_url" in result_d

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

        from neograph import describe_type

        class WithOptional(BaseModel):
            name: str
            nickname: str | None = None

        result = describe_type(WithOptional, prefix="")
        assert "nickname: string or null" in result
        # name should NOT have 'or null'
        lines = [l.strip() for l in result.splitlines()]
        name_line = [l for l in lines if l.startswith("name:")][0]
        assert "null" not in name_line

    def test_joins_union_types_with_or_splitter(self):
        """Union[A, B] renders with or_splitter."""

        from neograph import describe_type

        class WithUnion(BaseModel):
            value: str | int

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
            children: list[TreeNode] = []

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


class TestDescribeValue:
    """describe_value renders Pydantic instances in BAML notation (neograph-q8uj).

    BAML protocol: same TypeScript-style notation as describe_type but with
    actual values instead of type names. Field descriptions as // comments.
    Strings quoted, numbers bare, booleans lowercase, None as null.
    """

    def test_flat_model_exact_output(self):
        """Flat model with Field descriptions produces exact BAML instance notation."""
        from pydantic import Field

        from neograph.describe_type import describe_value

        class SearchHit(BaseModel, frozen=True):
            node_id: str = Field(description="Graph node identifier")
            score: float = Field(description="Relevance score 0-1")

        hit = SearchHit(node_id="UC-042", score=0.9)
        result = describe_value(hit)

        expected = (
            '{\n'
            '  node_id: "UC-042"  // Graph node identifier\n'
            '  score: 0.9  // Relevance score 0-1\n'
            '}'
        )
        assert result == expected

    def test_flat_model_no_descriptions(self):
        """Fields without descriptions produce no // comment."""
        from neograph.describe_type import describe_value

        class Point(BaseModel, frozen=True):
            x: int
            y: int

        result = describe_value(Point(x=10, y=20))
        expected = '{\n  x: 10\n  y: 20\n}'
        assert result == expected

    def test_nested_model_exact_output(self):
        """Nested BaseModel renders recursively with correct indentation."""
        from neograph.describe_type import describe_value

        class Inner(BaseModel, frozen=True):
            x: int

        class Outer(BaseModel, frozen=True):
            name: str
            detail: Inner

        result = describe_value(Outer(name="test", detail=Inner(x=42)))
        expected = (
            '{\n'
            '  name: "test"\n'
            '  detail: {\n'
            '    x: 42\n'
            '  }\n'
            '}'
        )
        assert result == expected

    def test_list_of_models_exact_output(self):
        """List of BaseModel instances renders as BAML array."""
        from neograph.describe_type import describe_value

        class Item(BaseModel, frozen=True):
            label: str

        result = describe_value([Item(label="a"), Item(label="b")])
        expected = (
            '[\n'
            '  {\n'
            '    label: "a"\n'
            '  },\n'
            '  {\n'
            '    label: "b"\n'
            '  }\n'
            ']'
        )
        assert result == expected

    def test_primitive_values_protocol(self):
        """BAML value rendering: str→quoted, int/float→bare, bool→lowercase, None→null."""
        from neograph.describe_type import describe_value

        class AllTypes(BaseModel, frozen=True):
            s: str
            i: int
            f: float
            b_true: bool
            b_false: bool
            n: str | None = None

        result = describe_value(AllTypes(
            s="hello", i=42, f=3.14, b_true=True, b_false=False, n=None,
        ))
        assert '  s: "hello"' in result
        assert '  i: 42' in result
        assert '  f: 3.14' in result
        assert '  b_true: true' in result
        assert '  b_false: false' in result
        assert '  n: null' in result

    def test_prefix_prepended(self):
        """Custom prefix on first line, body follows."""
        from neograph.describe_type import describe_value

        class Simple(BaseModel, frozen=True):
            x: int

        result = describe_value(Simple(x=1), prefix="Tool result:")
        assert result == 'Tool result:\n{\n  x: 1\n}'

    def test_empty_list_renders_brackets(self):
        """Empty list renders as []."""
        from neograph.describe_type import describe_value

        assert describe_value([]) == "[]"


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
        """Level 3: Construct(renderer=...) propagates to child nodes via model_copy."""
        xml = XmlRenderer()
        child = Node.scripted("child", fn="noop", outputs=Claims)
        assert child.renderer is None

        pipeline = Construct("prop-test", renderer=xml, nodes=[child])
        # Original child is unchanged (immutable IR)
        assert child.renderer is None
        # The copy inside the construct has the propagated renderer
        assert pipeline.nodes[0].renderer is xml
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

    def test_returns_baml_when_no_renderer(self):
        """Level 5: no renderer = BAML default (symmetric with tool-result rendering, neograph-qybn)."""

        class Info(BaseModel):
            name: str

        instance = Info(name="test")
        result = render_input(instance, renderer=None)
        assert isinstance(result, str)
        assert "name" in result
        assert "test" in result

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
            mode="think",
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
        assert isinstance(schema, str), "output_schema should be a string from describe_type()"
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
            mode="think",
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
            mode="think",
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
            mode="think",
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

        n = Node(name="test-node", mode="think", outputs=Claims, model="fast", prompt="my/template")
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
            mode="think",
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
            mode="think",
            outputs=Claims,
            model="fast",
            prompt="test",
            llm_config={"output_strategy": "json_mode"},
        )
        render_prompt(n, "some input")

        assert len(compiler_kwargs) == 1
        schema = compiler_kwargs[0].get("output_schema")
        assert isinstance(schema, str), "output_schema should be a string from describe_type()"
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
            mode="think",
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
        """Construct(renderer=...) propagates via model_copy to child nodes without own renderer."""
        xml = XmlRenderer()
        child = Node.scripted("prog-child", fn="noop", outputs=Claims)
        assert child.renderer is None

        pipeline = Construct("prog-test", renderer=xml, nodes=[child])

        # Original child unchanged (immutable IR)
        assert child.renderer is None
        # Copy inside construct has the propagated renderer
        assert pipeline.nodes[0].renderer is xml
        # Verify propagation through another construct
        child2 = Node.scripted("prog-child2", fn="noop2", outputs=MatchResult)
        pipeline2 = Construct("prog-test2", renderer=xml, nodes=[child2])
        assert child2.renderer is None
        assert pipeline2.nodes[0].renderer is xml

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


class TestRenderToolResultForLlm:
    """_render_tool_result_for_llm: uses renderer when provided (neograph-w6hk)."""

    def test_renderer_used_for_pydantic_model(self):
        """When a renderer is supplied, it should be used instead of describe_value."""
        from neograph._llm import _render_tool_result_for_llm

        class Result(BaseModel):
            score: int = 42

        class FakeRenderer:
            def render(self, value: Any) -> str:
                return f"CUSTOM:{value.score}"

        result = _render_tool_result_for_llm(Result(), renderer=FakeRenderer())
        assert "CUSTOM:42" in result

    def test_renderer_used_for_list_of_models(self):
        """When a renderer is supplied for a list of models, it should be used."""
        from neograph._llm import _render_tool_result_for_llm

        class Item(BaseModel):
            label: str

        class FakeRenderer:
            def render(self, value: Any) -> str:
                return f"RENDERED:{len(value)}"

        items = [Item(label="a"), Item(label="b")]
        result = _render_tool_result_for_llm(items, renderer=FakeRenderer())
        assert "RENDERED:2" in result

    def test_fallback_to_describe_value_when_no_renderer(self):
        """Without renderer, Pydantic models still use describe_value (default)."""
        from neograph._llm import _render_tool_result_for_llm

        class Simple(BaseModel):
            x: int = 5

        result = _render_tool_result_for_llm(Simple())
        # describe_value produces BAML-style notation
        assert "x" in result
        assert "5" in result


# ═══════════════════════════════════════════════════════════════════════════
# Coverage gap tests — renderers.py
# ═══════════════════════════════════════════════════════════════════════════


class TestXmlRendererCoverageGaps:
    """Tests covering previously uncovered lines in XmlRenderer."""

    def test_renders_raw_list_at_top_level(self):
        """XmlRenderer.render() with a raw list (not inside a model field)."""
        r = XmlRenderer()
        result = r.render(["alpha", "beta"])
        assert "<item>alpha</item>" in result
        assert "<item>beta</item>" in result

    def test_renders_raw_dict_at_top_level(self):
        """XmlRenderer.render() with a raw dict (not inside a model field)."""
        r = XmlRenderer()
        result = r.render({"key1": "val1", "key2": "val2"})
        assert "<key1>val1</key1>" in result
        assert "<key2>val2</key2>" in result

    def test_dict_typed_field_in_model(self):
        """Model with a dict-typed field renders via _render_dict."""

        class WithDict(BaseModel):
            meta: dict[str, str]

        r = XmlRenderer()
        result = r.render(WithDict(meta={"k": "v"}))
        assert "<meta>" in result
        assert "<k>v</k>" in result
        assert "</meta>" in result

    def test_render_list_standalone_with_tag(self):
        """_render_list with a tag wraps items in outer tag."""
        r = XmlRenderer()
        result = r._render_list(["x", "y"], tag="items", seen=set())
        assert result.startswith("<items>")
        assert result.endswith("</items>")
        assert "<item>x</item>" in result

    def test_render_dict_standalone_with_tag(self):
        """_render_dict with a tag wraps entries in outer tag."""
        r = XmlRenderer()
        result = r._render_dict({"a": 1}, tag="data", seen=set())
        assert result.startswith("<data>")
        assert result.endswith("</data>")
        assert "<a>1</a>" in result

    def test_description_dedup_in_once_mode(self):
        """include_field_info='once' deduplicates by field name across nesting."""

        class Inner(BaseModel):
            name: str = PydanticField(description="A name field")

        class Outer(BaseModel):
            first: Inner
            second: Inner

        r = XmlRenderer(include_field_info="once")
        result = r.render(Outer(first=Inner(name="a"), second=Inner(name="b")))
        # 'description="A name field"' should appear only once
        assert result.count('description="A name field"') == 1

    def test_renders_list_of_models_at_top_level(self):
        """Rendering a list of BaseModel items wraps each model in <item> tag."""

        class Item(BaseModel):
            val: int

        r = XmlRenderer()
        result = r.render([Item(val=1), Item(val=2)])
        assert "<item>" in result
        assert "</item>" in result
        assert "<val>1</val>" in result
        assert "<val>2</val>" in result


class TestDelimitedRendererCoverageGaps:
    """Tests covering previously uncovered lines in DelimitedRenderer."""

    def test_renders_list_of_models_in_field(self):
        """List of BaseModel items in a model field uses nested model rendering."""

        class Item(BaseModel):
            label: str

        class Container(BaseModel):
            items: list[Item]

        r = DelimitedRenderer()
        result = r.render(Container(items=[Item(label="x"), Item(label="y")]))
        assert "[[ ## items ## ]]" in result
        assert "- " in result
        assert "x" in result
        assert "y" in result

    def test_renders_scalar_value_as_string(self):
        """DelimitedRenderer.render() with a plain scalar returns str()."""
        r = DelimitedRenderer()
        result = r.render(42)
        assert result == "42"

    def test_renders_standalone_list_of_scalars(self):
        """DelimitedRenderer.render() with a raw list of scalars."""
        r = DelimitedRenderer()
        result = r.render(["alpha", "beta"])
        assert "- alpha" in result
        assert "- beta" in result

    def test_renders_standalone_list_of_models(self):
        """DelimitedRenderer.render() with a raw list of BaseModel instances."""

        class Tag(BaseModel):
            name: str

        r = DelimitedRenderer()
        result = r.render([Tag(name="a"), Tag(name="b")])
        assert "- " in result
        assert "a" in result
        assert "b" in result


class TestJsonRendererCoverageGaps:
    """Tests covering previously uncovered lines in JsonRenderer."""

    def test_renders_non_basemodel_value(self):
        """JsonRenderer falls back to json.dumps for non-BaseModel values."""
        r = JsonRenderer()
        result = r.render({"key": "value", "num": 42})
        import json as _json
        parsed = _json.loads(result)
        assert parsed == {"key": "value", "num": 42}

    def test_renders_plain_list(self):
        """JsonRenderer renders a plain list via json.dumps."""
        r = JsonRenderer()
        result = r.render([1, 2, 3])
        import json as _json
        assert _json.loads(result) == [1, 2, 3]


# ═══════════════════════════════════════════════════════════════════════════
# Coverage gap tests — describe_type.py
# ═══════════════════════════════════════════════════════════════════════════


class TestDescribeTypeCoverageGaps:
    """Tests covering previously uncovered lines in describe_type."""

    def test_bare_list_no_args_renders_as_any_array(self):
        """typing.List (no subscription) has get_origin=list, get_args=()."""
        from typing import List  # noqa: UP006, UP035

        from neograph.describe_type import _render_type

        result = _render_type(
            List, indent="  ", depth=0, or_splitter=" or ",  # noqa: UP006
            hoisted=set(), visited=set(),
        )
        assert result == "[any]"

    def test_bare_dict_no_args_renders_as_object(self):
        """typing.Dict (no subscription) has get_origin=dict, get_args=()."""
        from typing import Dict  # noqa: UP006, UP035

        from neograph.describe_type import _render_type

        result = _render_type(
            Dict, indent="  ", depth=0, or_splitter=" or ",  # noqa: UP006
            hoisted=set(), visited=set(),
        )
        assert result == "object"

    def test_any_type_renders_as_any(self):
        """typing.Any renders as 'any'."""

        class WithAny(BaseModel):
            data: Any

        result = describe_type(WithAny, prefix="")
        assert "data: any" in result

    def test_unknown_type_renders_as_str(self):
        """Unknown/unrecognized types fall through to str(annotation)."""
        import datetime

        class WithDate(BaseModel):
            ts: datetime.datetime

        result = describe_type(WithDate, prefix="")
        # Should contain str() of the type
        assert "ts:" in result


class TestDescribeValueCoverageGaps:
    """Tests covering previously uncovered lines in describe_value."""

    def test_plain_primitive_hits_else_branch(self):
        """describe_value with a plain primitive hits the else branch at line 360."""
        from neograph.describe_type import describe_value

        result = describe_value(42)
        assert result == "42"

    def test_empty_model_renders_as_empty_braces(self):
        """describe_value with a model that has zero fields renders {}."""
        from neograph.describe_type import describe_value

        class Empty(BaseModel):
            pass

        result = describe_value(Empty())
        assert "{}" in result

    def test_model_with_list_field(self):
        """describe_value with a model containing a list field."""
        from neograph.describe_type import describe_value

        class WithList(BaseModel):
            tags: list[str]

        result = describe_value(WithList(tags=["a", "b"]))
        assert '"a"' in result
        assert '"b"' in result

    def test_model_with_dict_field(self):
        """describe_value with a model containing a dict field."""
        from neograph.describe_type import describe_value

        class WithDict(BaseModel):
            meta: dict[str, int]

        result = describe_value(WithDict(meta={"x": 1, "y": 2}))
        assert '"x"' in result
        assert "1" in result
        assert '"y"' in result
        assert "2" in result

    def test_render_dict_value_standalone(self):
        """_render_dict_value directly."""
        from neograph.describe_type import _render_dict_value

        result = _render_dict_value({"a": 1, "b": "two"}, indent="  ", depth=0)
        assert '"a": 1' in result
        assert '"b": "two"' in result

    def test_render_dict_value_empty(self):
        """_render_dict_value with empty dict returns {}."""
        from neograph.describe_type import _render_dict_value

        result = _render_dict_value({}, indent="  ", depth=0)
        assert result == "{}"

    def test_render_value_with_unknown_type_uses_repr(self):
        """_render_value with an unsupported type falls back to repr()."""
        from neograph.describe_type import _render_value

        result = _render_value(object(), indent="  ", depth=0)
        assert "object" in result


class TestRenderInputBAMLDefault:
    """BUG neograph-qybn: render_input must BAML-render Pydantic models by default.

    When no renderer is configured, render_input currently returns the raw
    Pydantic object. Tool results fall back to describe_value() BAML. This
    class tests that input rendering matches tool-result rendering.
    """

    def test_baml_default_for_pydantic_model(self):
        """render_input(model, renderer=None) returns BAML string, not raw model."""
        class Draft(BaseModel):
            content: str
            score: float

        instance = Draft(content="hello world", score=0.95)
        result = render_input(instance, renderer=None)

        # Must be a string (BAML notation), NOT a raw Pydantic object
        assert isinstance(result, str), (
            f"Expected BAML string, got {type(result).__name__}. "
            f"render_input must not return raw Pydantic models when renderer=None."
        )
        assert "content" in result
        assert "hello world" in result
        assert "score" in result

    def test_baml_default_for_list_of_models(self):
        """render_input([model, ...], renderer=None) returns BAML string for lists."""
        class Item(BaseModel):
            name: str

        items = [Item(name="a"), Item(name="b")]
        result = render_input(items, renderer=None)

        assert isinstance(result, str), (
            f"Expected BAML string for list of models, got {type(result).__name__}."
        )
        assert "a" in result
        assert "b" in result

    def test_baml_default_for_fan_in_dict(self):
        """render_input({k: model}, renderer=None) BAML-renders each value."""
        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: int

        result = render_input({"a": A(x="hello"), "b": B(y=42)}, renderer=None)
        assert isinstance(result, dict)
        assert isinstance(result["a"], str), "Fan-in dict values must be BAML strings"
        assert "hello" in result["a"]
        assert isinstance(result["b"], str)
        assert "42" in result["b"]

    def test_primitives_pass_through(self):
        """render_input with primitives still passes them through unchanged."""
        assert render_input("hello", renderer=None) == "hello"
        assert render_input(42, renderer=None) == 42

    def test_exclude_true_honored_in_baml_default(self):
        """Fields with exclude=True are omitted from BAML default rendering."""
        class Secret(BaseModel):
            visible: str
            hidden: str = PydanticField(exclude=True, default="secret")

        result = render_input(Secret(visible="shown"), renderer=None)
        assert isinstance(result, str)
        assert "visible" in result
        assert "hidden" not in result
        assert "secret" not in result


class TestRenderForPromptUnconditional:
    """BUG neograph-qybn: render_for_prompt() must fire regardless of renderer config.

    Currently render_for_prompt() only runs inside _render_single(), which is
    only called when a Renderer is active. With renderer=None the whole
    render_for_prompt chain is dead code.
    """

    def test_render_for_prompt_fires_without_renderer(self):
        """render_for_prompt() must be called even when renderer=None."""
        class Projected(BaseModel):
            summary: str

        class Full(BaseModel):
            raw: str
            internal_id: int

            def render_for_prompt(self) -> str:
                return f"PROJECTED: {self.raw}"

        result = render_input(Full(raw="data", internal_id=99), renderer=None)
        assert result == "PROJECTED: data", (
            "render_for_prompt() must run without a renderer configured"
        )

    def test_render_for_prompt_returning_model_baml_rendered_without_renderer(self):
        """render_for_prompt() returning BaseModel gets BAML-rendered even without renderer."""
        class Presentation(BaseModel):
            summary: str
            score: float

        class Full(BaseModel):
            raw: str
            internal_id: int

            def render_for_prompt(self) -> "Presentation":
                return Presentation(summary=self.raw.upper(), score=0.95)

        result = render_input(Full(raw="hello", internal_id=42), renderer=None)
        assert isinstance(result, str), (
            f"Expected BAML string, got {type(result).__name__}"
        )
        assert "HELLO" in result
        assert "summary" in result
        assert "internal_id" not in result  # projection strips internal fields

    def test_render_for_prompt_wins_over_baml_default(self):
        """render_for_prompt() takes precedence over automatic BAML rendering."""
        class Custom(BaseModel):
            name: str

            def render_for_prompt(self) -> str:
                return "CUSTOM_WINS"

        result = render_input(Custom(name="test"), renderer=None)
        assert result == "CUSTOM_WINS"

    def test_render_for_prompt_in_fan_in_dict_without_renderer(self):
        """render_for_prompt() fires per-value in fan-in dict without renderer."""
        class A(BaseModel):
            x: str

            def render_for_prompt(self) -> str:
                return f"A:{self.x}"

        class B(BaseModel):
            y: str

        result = render_input({"a": A(x="1"), "b": B(y="2")}, renderer=None)
        assert isinstance(result, dict)
        assert result["a"] == "A:1"  # render_for_prompt wins
        assert isinstance(result["b"], str)  # B gets BAML default
        assert "y" in result["b"]


class TestRenderingModeDispatch:
    """Rendering obligation: think and agent modes produce BAML for prompt compiler.

    When no renderer is configured, template-ref prompts go through the prompt
    compiler with BAML-rendered input data. These tests verify the full dispatch
    chain: _extract_input → _render_input → invoke_structured → _compile_prompt
    → prompt_compiler.
    """

    def test_think_mode_prompt_compiler_receives_baml_no_renderer(self):
        """Think-mode node: prompt compiler receives BAML strings, not raw models."""
        from neograph import Construct, Node, compile, run
        from neograph.factory import register_scripted
        from tests.fakes import StructuredFakeWithRaw, configure_fake_llm

        class Input(BaseModel):
            text: str
            score: float

        class Output(BaseModel):
            result: str

        captured = {}

        def capturing_compiler(template, data, **kw):
            captured[template] = data
            return [{"role": "user", "content": "test"}]

        configure_fake_llm(
            factory=lambda tier: StructuredFakeWithRaw(lambda m: m(result="ok")),
            prompt_compiler=capturing_compiler,
        )

        register_scripted("mode_test_seed", lambda _in, _cfg: Input(text="hello", score=0.5))
        parent = Construct("mode-test", nodes=[
            Node.scripted("seed", fn="mode_test_seed", outputs=Input),
            Node("think-node", prompt="analyze/input", model="default",
                 outputs=Output, inputs={"seed": Input}),
        ])
        graph = compile(parent)
        run(graph, input={"node_id": "mode-test"})

        assert "analyze/input" in captured
        data = captured["analyze/input"]
        assert isinstance(data, dict)
        seed_val = data["seed"]
        assert isinstance(seed_val, str), f"Expected BAML string, got {type(seed_val).__name__}"
        assert "text" in seed_val
        assert "hello" in seed_val

    def test_agent_mode_prompt_compiler_receives_baml_no_renderer(self):
        """Agent-mode node: prompt compiler receives BAML strings, not raw models."""
        from neograph import Construct, Node, Tool, compile, run
        from neograph.factory import register_scripted
        from neograph.factory import register_tool_factory
        from tests.fakes import ReActFake, configure_fake_llm

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            answer: str

        captured = {}

        def capturing_compiler(template, data, **kw):
            captured[template] = data
            return [{"role": "user", "content": "test"}]

        configure_fake_llm(
            factory=lambda tier: ReActFake(
                tool_calls=[],
                final=lambda m: m(answer="done"),
            ),
            prompt_compiler=capturing_compiler,
        )

        register_tool_factory("search", lambda config, tool_config: (lambda **kw: "result"))
        register_scripted("agent_test_seed", lambda _in, _cfg: Input(query="test query"))
        parent = Construct("agent-test", nodes=[
            Node.scripted("seed", fn="agent_test_seed", outputs=Input),
            Node("agent-node", prompt="search/query", model="default",
                 mode="agent", outputs=Output, inputs={"seed": Input},
                 tools=[Tool("search", budget=3)]),
        ])
        graph = compile(parent)
        run(graph, input={"node_id": "agent-test"})

        assert "search/query" in captured
        data = captured["search/query"]
        assert isinstance(data, dict)
        seed_val = data["seed"]
        assert isinstance(seed_val, str), f"Expected BAML string, got {type(seed_val).__name__}"
        assert "query" in seed_val
        assert "test query" in seed_val


class TestRenderingThreeSurfaceParity:
    """Rendering obligation: all 3 API surfaces produce identical BAML.

    The same Pydantic model rendered through @node, Node.scripted(), and
    programmatic Node() must produce the same BAML string. Tests use
    template-ref prompts (not inline) to exercise the full render_input path.
    """

    def test_declarative_node_renders_baml_for_prompt_compiler(self):
        """Declarative Node() with template-ref prompt: BAML rendering."""
        from neograph import Construct, Node, compile, run
        from neograph.factory import register_scripted
        from tests.fakes import StructuredFakeWithRaw, configure_fake_llm

        class Data(BaseModel):
            value: str

        class Result(BaseModel):
            out: str

        captured = {}

        def capturing_compiler(template, data, **kw):
            captured[template] = data
            return [{"role": "user", "content": "test"}]

        configure_fake_llm(
            factory=lambda tier: StructuredFakeWithRaw(lambda m: m(out="ok")),
            prompt_compiler=capturing_compiler,
        )

        register_scripted("surf_seed", lambda _in, _cfg: Data(value="test-val"))
        parent = Construct("surface-test", nodes=[
            Node.scripted("seed", fn="surf_seed", outputs=Data),
            Node("proc", prompt="process/data", model="default",
                 outputs=Result, inputs={"seed": Data}),
        ])
        graph = compile(parent)
        run(graph, input={"node_id": "surface-test"})

        assert "process/data" in captured
        seed_rendered = captured["process/data"]["seed"]
        assert isinstance(seed_rendered, str)
        assert "value" in seed_rendered
        assert "test-val" in seed_rendered

    def test_decorator_node_renders_baml_for_prompt_compiler(self):
        """@node with template-ref prompt: same BAML rendering as declarative."""
        from neograph import compile, node, run
        from neograph.decorators import construct_from_functions
        from tests.fakes import StructuredFakeWithRaw, configure_fake_llm

        class Data(BaseModel):
            value: str

        class Result(BaseModel):
            out: str

        captured = {}

        def capturing_compiler(template, data, **kw):
            captured[template] = data
            return [{"role": "user", "content": "test"}]

        configure_fake_llm(
            factory=lambda tier: StructuredFakeWithRaw(lambda m: m(out="ok")),
            prompt_compiler=capturing_compiler,
        )

        @node(outputs=Data)
        def seed() -> Data:
            return Data(value="test-val")

        @node(mode="think", outputs=Result, model="default", prompt="process/data")
        def proc(seed: Data) -> Result: ...

        pipeline = construct_from_functions("decorator-test", [seed, proc])
        graph = compile(pipeline)
        run(graph, input={"node_id": "decorator-test"})

        assert "process/data" in captured
        seed_rendered = captured["process/data"]["seed"]
        assert isinstance(seed_rendered, str)
        assert "value" in seed_rendered
        assert "test-val" in seed_rendered

    def test_all_surfaces_produce_same_baml(self):
        """Direct render_input call matches what the pipeline produces."""
        from neograph.describe_type import describe_value

        class Data(BaseModel):
            value: str

        instance = Data(value="test-val")
        direct_baml = describe_value(instance)
        via_render = render_input(instance, renderer=None)

        assert direct_baml == via_render, (
            "render_input(model, renderer=None) must equal describe_value(model)"
        )


class TestToolInputRenderingParity:
    """BUG neograph-qybn: tool-result and input rendering must produce same BAML.

    _render_tool_result_for_llm falls back to describe_value() for Pydantic models.
    render_input must produce the same BAML format for the same model instance.
    """

    def test_same_model_same_baml(self):
        """Same Pydantic instance → same BAML from both paths (minus prefix)."""
        from neograph._llm import _render_tool_result_for_llm
        from neograph.describe_type import describe_value

        class Result(BaseModel):
            answer: str
            confidence: float

        instance = Result(answer="yes", confidence=0.9)

        tool_result = _render_tool_result_for_llm(instance, renderer=None)
        input_result = render_input(instance, renderer=None)

        # Tool result has "Tool result:" prefix, input doesn't
        # Both must contain the same BAML body
        expected_baml = describe_value(instance)
        assert isinstance(input_result, str)
        assert expected_baml == input_result, (
            f"Input BAML must match describe_value output.\n"
            f"Got: {input_result!r}\n"
            f"Expected: {expected_baml!r}"
        )

    def test_parity_with_exclude_fields(self):
        """Both paths honor exclude=True identically."""
        from neograph._llm import _render_tool_result_for_llm
        from neograph.describe_type import describe_value

        class Data(BaseModel):
            visible: str
            hidden: str = PydanticField(exclude=True, default="nope")

        instance = Data(visible="shown")

        tool_result = _render_tool_result_for_llm(instance, renderer=None)
        input_result = render_input(instance, renderer=None)

        # Neither should contain "hidden"
        assert "hidden" not in tool_result
        assert isinstance(input_result, str)
        assert "hidden" not in input_result

    def test_parity_with_list_of_models(self):
        """Both paths handle list[BaseModel] identically."""
        from neograph._llm import _render_tool_result_for_llm
        from neograph.describe_type import describe_value

        class Item(BaseModel):
            name: str

        items = [Item(name="a"), Item(name="b")]

        tool_result = _render_tool_result_for_llm(items, renderer=None)
        input_result = render_input(items, renderer=None)

        expected_baml = describe_value(items)
        assert isinstance(input_result, str)
        # Tool result has prefix, strip it for comparison
        tool_body = tool_result.replace("Tool result:\n", "")
        assert tool_body == input_result or expected_baml == input_result

