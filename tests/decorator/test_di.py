"""@node decorator tests — DI (FromInput/FromConfig), @merge_fn DI"""

from __future__ import annotations

from typing import Annotated

import pytest
from pydantic import BaseModel

from neograph import (
    Construct,
    Node,
    Oracle,
    compile,
    construct_from_functions,
    node,
    run,
)
from tests.schemas import (
    Claims,
    RawText,
)


class TestFromInputPydanticModel:
    """neograph-6jd — Annotated[PydanticModel, FromInput] bundles multiple config fields."""

    def test_bundle_populates_when_all_fields_in_run_input(self):
        """Annotated[RunCtx, FromInput] populates each field from config['configurable']."""
        from neograph import FromInput

        class RunCtx(BaseModel):
            node_id: str
            project_root: str

        @node(outputs=RawText)
        def fipb_produce(ctx: Annotated[RunCtx, FromInput]) -> RawText:
            return RawText(text=f"{ctx.node_id}|{ctx.project_root}")

        pipeline = construct_from_functions("fipb", [fipb_produce])
        graph = compile(pipeline)
        result = run(
            graph,
            input={"node_id": "REQ-001", "project_root": "/tmp/repo"},
        )
        assert result["fipb_produce"].text == "REQ-001|/tmp/repo"

    def test_bundle_composes_when_mixed_with_upstream_param(self):
        """Annotated[PydanticModel, FromInput] composes with an upstream @node parameter."""
        from neograph import FromInput

        class RunCtx(BaseModel):
            node_id: str

        @node(outputs=Claims)
        def fipb2_source() -> Claims:
            return Claims(items=["a", "b"])

        @node(outputs=RawText)
        def fipb2_join(fipb2_source: Claims, ctx: Annotated[RunCtx, FromInput]) -> RawText:
            return RawText(text=f"{ctx.node_id}: {','.join(fipb2_source.items)}")

        pipeline = construct_from_functions("fipb2", [fipb2_source, fipb2_join])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "X-42"})
        assert result["fipb2_join"].text == "X-42: a,b"

    def test_bundle_field_none_when_missing_from_configurable(self):
        """A missing field in config['configurable'] is passed as None."""
        from neograph import FromInput

        class PartialCtx(BaseModel):
            node_id: str | None = None
            project_root: str | None = None

        @node(outputs=RawText)
        def fipbm_read(ctx: Annotated[PartialCtx, FromInput(required=False)]) -> RawText:
            return RawText(text=f"id={ctx.node_id!r},root={ctx.project_root!r}")

        pipeline = construct_from_functions("fipbm", [fipbm_read])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "only-this"})
        assert result["fipbm_read"].text == "id='only-this',root=None"

    def test_from_config_bundle_populates_when_fields_in_configurable(self):
        """Annotated[PydanticModel, FromConfig] pulls every field from configurable as well."""
        from neograph import FromConfig

        class Shared(BaseModel):
            model_config = {"arbitrary_types_allowed": True}
            tenant: str
            max_items: int

        @node(outputs=RawText)
        def fcb_read(shared: Annotated[Shared, FromConfig]) -> RawText:
            return RawText(text=f"{shared.tenant}:{shared.max_items}")

        pipeline = construct_from_functions("fcb", [fcb_read])
        graph = compile(pipeline)
        result = run(
            graph,
            input={"node_id": "x"},
            config={"configurable": {"tenant": "acme", "max_items": 7}},
        )
        assert result["fcb_read"].text == "acme:7"





class TestOracleMergeFnDI:
    """neograph-9zj — @merge_fn decorator with FromInput/FromConfig DI."""

    def test_merge_fn_receives_bundle_when_annotated_with_from_config(self):
        """@merge_fn function can receive a bundled Annotated[PydanticModel, FromConfig]
        whose fields are resolved from config['configurable'] keys."""
        from neograph import (
            FromConfig,
            merge_fn,
            register_scripted,
        )

        class SharedResources(BaseModel):
            prefix: str

        @merge_fn
        def combine_with_prefix(
            variants: list[Claims],
            shared: Annotated[SharedResources, FromConfig],
        ) -> Claims:
            # Collect all unique items, prepend the shared prefix.
            seen: list[str] = []
            for v in variants:
                for it in v.items:
                    if it not in seen:
                        seen.append(it)
            return Claims(items=[f"{shared.prefix}:{x}" for x in seen])

        # Register a scripted generator that produces a Claims variant.
        def gen_fn(input_data, config):
            return Claims(items=["alpha", "beta"])
        register_scripted("omfd_gen_fn", gen_fn)

        gen = Node.scripted("omfd-gen", fn="omfd_gen_fn", outputs=Claims) | Oracle(
            n=2, merge_fn="combine_with_prefix"
        )

        pipeline = Construct("omfd-test", nodes=[gen])
        graph = compile(pipeline)
        # Bundled form: SharedResources has a single field `prefix`, so we
        # provide it directly in configurable under that key name.
        result = run(
            graph,
            input={"node_id": "omfd-001"},
            config={"configurable": {"prefix": "tag"}},
        )

        # Both Oracle generators produce ["alpha", "beta"], merge dedups, prefixes.
        assert result["omfd_gen"].items == ["tag:alpha", "tag:beta"]

    def test_merge_fn_receives_value_when_annotated_with_from_input(self):
        """@merge_fn can also receive Annotated[T, FromInput] values from run(input=...)."""
        from neograph import (
            FromInput,
            merge_fn,
            register_scripted,
        )

        @merge_fn
        def tagged_merge(
            variants: list[Claims],
            node_id: Annotated[str, FromInput],
        ) -> Claims:
            all_items = []
            for v in variants:
                all_items.extend(v.items)
            return Claims(items=[f"{node_id}:{it}" for it in dict.fromkeys(all_items)])

        def gen_fn2(input_data, config):
            return Claims(items=["x"])
        register_scripted("omfdi_gen_fn", gen_fn2)

        gen = Node.scripted("omfdi-gen", fn="omfdi_gen_fn", outputs=Claims) | Oracle(
            n=2, merge_fn="tagged_merge"
        )

        pipeline = Construct("omfdi-test", nodes=[gen])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "REQ-99"})

        assert result["omfdi_gen"].items == ["REQ-99:x"]

    def test_plain_merge_fn_works_when_no_decorator(self):
        """Back-compat: plain (variants, config) merge_fn still works."""
        from neograph import (
            register_scripted,
        )

        def plain_merge(variants, config):
            # Old-style signature — two positional args, no decorator.
            all_items = []
            for v in variants:
                all_items.extend(v.items)
            return Claims(items=list(dict.fromkeys(all_items)))
        register_scripted("plain_merge_backcompat", plain_merge)

        def pmg_gen(input_data, config):
            return Claims(items=["one", "two"])
        register_scripted("pmg_gen_fn", pmg_gen)

        gen = Node.scripted("pmg-gen", fn="pmg_gen_fn", outputs=Claims) | Oracle(
            n=2, merge_fn="plain_merge_backcompat"
        )

        pipeline = Construct("pmg-test", nodes=[gen])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "pmg-001"})
        assert result["pmg_gen"].items == ["one", "two"]


# ═══════════════════════════════════════════════════════════════════════════
# TestNodeInputsFieldRename (neograph-kqd.1)
#
# Step 1 of the Node.inputs refactor is a pure field rename:
# Node.input → Node.inputs. Field type stays Any and keeps the same shape
# acceptance (None | type | dict). Runtime behavior is unchanged. These
# tests fail before the rename (Node has no `inputs` field) and pass after.
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# TestNodeDecoratorDictInputs (neograph-kqd.4)
#
# @node decoration now emits dict-form inputs={param_name: annotation, ...}
# for all typed upstream params. This is the metadata shift that lets
# step-2's validator catch fan-in mismatches via _check_fan_in_inputs.
# Fan-out params (Each) are stripped from inputs at construct-assembly time.
# ═══════════════════════════════════════════════════════════════════════════




class TestClassifyDiParamsEdges:
    """Edge cases in _classify_di_params (lines 263-264, 275, 306)."""

    def test_fallback_when_get_type_hints_raises(self, monkeypatch):
        """Exception from get_type_hints returns empty dict."""
        import inspect
        import typing

        from neograph.decorators import _classify_di_params

        def bad_hints(*a, **kw):
            raise NameError("unresolvable annotation")
        monkeypatch.setattr(typing, "get_type_hints", bad_hints)

        def sample(x: int) -> int:
            return x
        sig = inspect.signature(sample)
        result = _classify_di_params(sample, sig, caller_ns=None)
        assert result == {}

    def test_annotated_with_single_arg_skipped(self):
        """Annotated with < 2 args is skipped (no marker)."""
        import inspect

        from neograph.decorators import _classify_di_params

        # Annotated with just a type and no marker — the get_args will
        # have < 2 elements. We can't create a real Annotated[T] with < 2
        # args, but we can test with a non-DI marker.
        def sample(x: Annotated[str, "just_a_string"]) -> str:
            return x
        sig = inspect.signature(sample)
        result = _classify_di_params(sample, sig, caller_ns=None)
        # "just_a_string" is not FromInput/FromConfig, so kind_base is None → skip
        assert result == {}

    def test_non_di_annotated_marker_skipped(self):
        """Annotated with a non-DI marker (kind_base is None) is skipped."""
        import inspect

        from neograph.decorators import _classify_di_params

        class CustomMarker:
            pass

        def sample(x: Annotated[str, CustomMarker()]) -> str:
            return x
        sig = inspect.signature(sample)
        result = _classify_di_params(sample, sig, caller_ns=None)
        assert result == {}





class TestResolveDiValueEdges:
    """Edge cases in DIBinding.resolve() — formerly _resolve_di_value."""

    def test_from_config_scalar_resolves(self):
        """from_config kind reads from config attr-style configurable."""
        from neograph.di import DIBinding, DIKind

        class AttrConfig:
            configurable = {"my_key": "attr_val"}

        binding = DIBinding(name="my_key", kind=DIKind.FROM_CONFIG, inner_type=str, required=False)
        result = binding.resolve(AttrConfig())
        assert result == "attr_val"

    def test_from_config_scalar_resolves_from_dict(self):
        """from_config reads from dict config."""
        from neograph.di import DIBinding, DIKind
        config = {"configurable": {"rate_limit": 42}}
        binding = DIBinding(name="rate_limit", kind=DIKind.FROM_CONFIG, inner_type=int, required=False)
        result = binding.resolve(config)
        assert result == 42

    def test_required_bundled_model_missing_fields_raises(self):
        """Required bundled model with missing fields raises."""
        from neograph.di import DIBinding, DIKind
        from neograph.errors import ExecutionError

        class MyModel(BaseModel):
            a: str
            b: int

        config = {"configurable": {}}  # no fields provided
        binding = DIBinding(name="ctx", kind=DIKind.FROM_INPUT_MODEL, inner_type=MyModel, required=True, model_cls=MyModel)
        with pytest.raises(ExecutionError, match="missing fields"):
            binding.resolve(config)

    def test_required_bundled_model_construction_failure_raises(self):
        """Required bundled model construction fails raises ExecutionError."""
        from neograph.di import DIBinding, DIKind
        from neograph.errors import ExecutionError

        class StrictModel(BaseModel):
            x: int  # requires int, will fail with string

        config = {"configurable": {"x": "not_an_int_and_model_rejects"}}
        binding = DIBinding(name="ctx", kind=DIKind.FROM_INPUT_MODEL, inner_type=StrictModel, required=True, model_cls=StrictModel)
        # Pydantic will coerce "not_an_int_and_model_rejects" and fail
        with pytest.raises(ExecutionError, match="construction failed"):
            binding.resolve(config)

    def test_optional_bundled_model_construction_failure_returns_none(self):
        """Optional model construction fails returns None."""
        from neograph.di import DIBinding, DIKind

        class StrictModel(BaseModel):
            x: int

        config = {"configurable": {"x": "not_an_int_fail"}}
        binding = DIBinding(name="ctx", kind=DIKind.FROM_INPUT_MODEL, inner_type=StrictModel, required=False, model_cls=StrictModel)
        result = binding.resolve(config)
        assert result is None

    def test_constant_kind_returns_payload(self):
        """Constant kind returns the payload directly."""
        from neograph.di import DIBinding, DIKind
        binding = DIBinding(name="ignored", kind=DIKind.CONSTANT, inner_type=int, required=False, default_value=42)
        assert binding.resolve(None) == 42





class TestResolveDiArgsFiltering:
    """_resolve_di_args filtering (line 398)."""

    def test_from_state_excluded_from_di_args(self):
        """Line 398: from_state params are filtered out."""
        from neograph.decorators import _resolve_di_args
        from neograph.di import DIBinding, DIKind
        param_res = {
            "topic": DIBinding(name="topic", kind=DIKind.FROM_INPUT, inner_type=str, required=False),
            "state_val": DIBinding(name="state_val", kind=DIKind.FROM_STATE, inner_type=str, required=False),
        }
        config = {"configurable": {"topic": "hello"}}
        result = _resolve_di_args(param_res, config)
        assert result == ["hello"]


