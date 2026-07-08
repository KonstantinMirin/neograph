"""Construct validation: merge hook signature checking (neograph-mzit).

Validates that merge_pre_process, merge_post_process, merge_fallback
have correct arity and type annotations at Construct assembly time.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from neograph import Construct, ConstructError, Node, Oracle

# ── Schemas ──────────────────────────────────────────────────────────────


class ModelA(BaseModel, frozen=True):
    value: str


class ModelB(BaseModel, frozen=True):
    score: float


# ═══════════════════════════════════════════════════════════════════════════
# WRONG VARIANT TYPE — the real piarch bug (piarch-1kz2a)
# ═══════════════════════════════════════════════════════════════════════════


class TestWrongVariantType:
    """merge_pre_process annotated with wrong variant type should fail at assembly."""

    def test_pre_process_wrong_variant_type_raises(self):
        """pre_process(variants: list[ModelB]) on a node with outputs=ModelA should fail."""

        def bad_pre(variants: list[ModelB]) -> dict:
            return {"items": variants}

        n = Node("gen", mode="think", outputs=ModelA, prompt="gen", model="fast") | Oracle(
            n=2,
            merge_prompt="merge: ${variants}",
            merge_pre_process=bad_pre,
        )
        with pytest.raises(ConstructError, match="merge_pre_process"):
            Construct("test", nodes=[n])

    def test_post_process_wrong_variant_type_raises(self):
        """post_process(result, variants: list[ModelB]) with wrong variant type."""

        def bad_post(result: ModelA, variants: list[ModelB]) -> ModelA:
            return result

        n = Node("gen", mode="think", outputs=ModelA, prompt="gen", model="fast") | Oracle(
            n=2,
            merge_prompt="merge: ${variants}",
            merge_post_process=bad_post,
        )
        with pytest.raises(ConstructError, match="merge_post_process"):
            Construct("test", nodes=[n])

    def test_fallback_wrong_variant_type_raises(self):
        """fallback(variants: list[ModelB], error) with wrong variant type."""

        def bad_fb(variants: list[ModelB], error: Exception) -> ModelA:
            return ModelA(value="fb")

        n = Node("gen", mode="think", outputs=ModelA, prompt="gen", model="fast") | Oracle(
            n=2,
            merge_prompt="merge: ${variants}",
            merge_fallback=bad_fb,
        )
        with pytest.raises(ConstructError, match="merge_fallback"):
            Construct("test", nodes=[n])

    def test_post_process_variant_error_names_the_variants_param(self):
        """neograph-dyy7: for merge_post_process the VARIANTS param is index 1
        (index 0 is the result). The type-mismatch message previously
        interpolated ``param_names[0]`` — naming the RESULT param where it means
        the VARIANTS param. The message must name the second param ('variants'),
        never the first ('result')."""

        def bad_post(result: ModelA, variants: list[ModelB]) -> ModelA:
            return result

        n = Node("gen", mode="think", outputs=ModelA, prompt="gen", model="fast") | Oracle(
            n=2,
            merge_prompt="merge: ${variants}",
            merge_post_process=bad_post,
        )
        with pytest.raises(ConstructError) as excinfo:
            Construct("test", nodes=[n])
        msg = str(excinfo.value)
        assert "variants param 'variants'" in msg, (
            f"message should name the VARIANTS param ('variants'), got: {msg}"
        )
        assert "'result'" not in msg, f"message wrongly names the RESULT param: {msg}"


# ═══════════════════════════════════════════════════════════════════════════
# WRONG RETURN TYPE
# ═══════════════════════════════════════════════════════════════════════════


class TestWrongReturnType:
    """Hooks returning wrong type should fail at assembly."""

    def test_post_process_wrong_return_type_raises(self):
        """post_process returning ModelB when node outputs=ModelA."""

        def bad_post(result: ModelA, variants: list[ModelA]) -> ModelB:
            return ModelB(score=0.5)

        n = Node("gen", mode="think", outputs=ModelA, prompt="gen", model="fast") | Oracle(
            n=2,
            merge_prompt="merge: ${variants}",
            merge_post_process=bad_post,
        )
        with pytest.raises(ConstructError, match="merge_post_process"):
            Construct("test", nodes=[n])

    def test_fallback_wrong_return_type_raises(self):
        """fallback returning ModelB when node outputs=ModelA."""

        def bad_fb(variants: list[ModelA], error: Exception) -> ModelB:
            return ModelB(score=0.5)

        n = Node("gen", mode="think", outputs=ModelA, prompt="gen", model="fast") | Oracle(
            n=2,
            merge_prompt="merge: ${variants}",
            merge_fallback=bad_fb,
        )
        with pytest.raises(ConstructError, match="merge_fallback"):
            Construct("test", nodes=[n])


# ═══════════════════════════════════════════════════════════════════════════
# WRONG ARITY
# ═══════════════════════════════════════════════════════════════════════════


class TestWrongArity:
    """Hooks with wrong number of params should fail at assembly."""

    def test_pre_process_zero_params_raises(self):
        def bad() -> dict:
            return {}

        n = Node("gen", mode="think", outputs=ModelA, prompt="gen", model="fast") | Oracle(
            n=2,
            merge_prompt="merge: ${variants}",
            merge_pre_process=bad,
        )
        with pytest.raises(ConstructError, match="merge_pre_process"):
            Construct("test", nodes=[n])

    def test_post_process_one_param_raises(self):
        def bad(result: ModelA) -> ModelA:
            return result

        n = Node("gen", mode="think", outputs=ModelA, prompt="gen", model="fast") | Oracle(
            n=2,
            merge_prompt="merge: ${variants}",
            merge_post_process=bad,
        )
        with pytest.raises(ConstructError, match="merge_post_process"):
            Construct("test", nodes=[n])


# ═══════════════════════════════════════════════════════════════════════════
# HAPPY PATHS
# ═══════════════════════════════════════════════════════════════════════════


class TestHappyPaths:
    """Correct hook signatures should pass validation."""

    def test_correct_types_accepted(self):
        def ok_pre(variants: list[ModelA]) -> dict:
            return {"items": variants}

        def ok_post(result: ModelA, variants: list[ModelA]) -> ModelA:
            return result

        def ok_fb(variants: list[ModelA], error: Exception) -> ModelA:
            return ModelA(value="fb")

        n = Node("gen", mode="think", outputs=ModelA, prompt="gen", model="fast") | Oracle(
            n=2,
            merge_prompt="merge: ${variants}",
            merge_pre_process=ok_pre,
            merge_post_process=ok_post,
            merge_fallback=ok_fb,
        )
        # Should not raise
        Construct("test", nodes=[n])

    def test_any_type_accepted(self):
        """list[Any] should be accepted for any output type."""

        def pre(variants: list[Any]) -> dict:
            return {"items": variants}

        n = Node("gen", mode="think", outputs=ModelA, prompt="gen", model="fast") | Oracle(
            n=2,
            merge_prompt="merge: ${variants}",
            merge_pre_process=pre,
        )
        Construct("test", nodes=[n])

    def test_lambda_accepted(self):
        """Lambdas have no annotations — should pass (arity-only check)."""
        n = Node("gen", mode="think", outputs=ModelA, prompt="gen", model="fast") | Oracle(
            n=2,
            merge_prompt="merge: ${variants}",
            merge_pre_process=lambda v: {"items": v},
            merge_post_process=lambda r, v: r,
            merge_fallback=lambda v, e: ModelA(value="fb"),
        )
        Construct("test", nodes=[n])

    def test_oracle_gen_type_override(self):
        """When oracle_gen_type is set, variants type should match it, not outputs."""

        # oracle_gen_type=ModelB means variants are list[ModelB],
        # but the final outputs is ModelA (post-merge type).
        def pre(variants: list[ModelB]) -> dict:
            return {"items": variants}

        def post(result: ModelA, variants: list[ModelB]) -> ModelA:
            return result

        n = Node("gen", mode="think", outputs=ModelA, prompt="gen", model="fast") | Oracle(
            n=2,
            merge_prompt="merge: ${variants}",
            merge_pre_process=pre,
            merge_post_process=post,
        )
        n.oracle_gen_type = ModelB

        Construct("test", nodes=[n])

    def test_no_hooks_no_validation(self):
        """Oracle with merge_prompt but no hooks should not trigger validation."""
        n = Node("gen", mode="think", outputs=ModelA, prompt="gen", model="fast") | Oracle(
            n=2,
            merge_prompt="merge: ${variants}",
        )
        Construct("test", nodes=[n])
