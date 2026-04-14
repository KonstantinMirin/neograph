"""Unified DI resolution module.

Single module with one resolver path for all DI parameters.

Four types/helpers live here:
- ``DIKind`` — enum of 6 DI parameter kinds (no UPSTREAM)
- ``DIBinding`` — a fully resolved DI parameter binding
- ``_unwrap_loop_value`` — Loop append-list unwrap (shared by factory, compiler, DI)
- ``_unwrap_each_dict`` — Each dict-to-list unwrap (shared by factory, DI)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Union
from typing import get_args as _get_args
from typing import get_origin as _get_origin

import structlog
from pydantic import ValidationError

from neograph.errors import ExecutionError as _ExecutionError

log = structlog.get_logger(__name__)


class DIKind(Enum):
    """How a DI parameter gets its value at runtime.

    Six kinds. Upstream resolution is NOT a DI kind — it stays in
    ``_extract_input`` in the factory layer.
    """

    FROM_INPUT = "from_input"
    FROM_CONFIG = "from_config"
    FROM_INPUT_MODEL = "from_input_model"
    FROM_CONFIG_MODEL = "from_config_model"
    FROM_STATE = "from_state"  # merge_fn only — resolved from graph state
    CONSTANT = "constant"


def _get_configurable(config: Any, key: str) -> Any:
    """Read a key from config['configurable'], handling dict and attr forms."""
    cfg = config or {}
    if isinstance(cfg, dict):
        return cfg.get("configurable", {}).get(key)
    return getattr(cfg, "configurable", {}).get(key)


def _unwrap_loop_value(val: Any, expected_type: Any) -> Any:
    """Unwrap a Loop append-list to the latest value.

    Loop nodes store list[T] in state via the append reducer. Consumers
    see the latest value ([-1]), not the full list. Empty list (first
    iteration) returns None. If the consumer expects list[T], the value
    passes through unchanged.

    This is the single source of truth for Loop unwrap — used by
    _extract_input, _resolve_merge_args, and loop_router.
    """
    if val is None:
        return None
    if not isinstance(val, list):
        return val
    # Consumer wants a list → pass through
    if _get_origin(expected_type) is list:
        return val
    # Empty list = first iteration → None
    if not val:
        return None
    return val[-1]


def _unwrap_each_dict(val: Any, expected_type: Any) -> Any:
    """Unwrap a dict[str, X] Each result to list[X] when consumer expects list.

    Each fan-out produces dict[str, X] keyed by each.key. A downstream
    consumer declaring list[X] gets list(dict.values()). Non-list consumers
    and non-dict values pass through unchanged.
    """
    if val is None:
        return None
    if not isinstance(val, dict):
        return val
    if _get_origin(expected_type) is not list:
        return val
    return list(val.values())


def _isinstance_safe(val: Any, tp: type) -> bool:
    """Check if val is an instance of tp, handling Union/Optional and generic origins."""
    origin = _get_origin(tp)

    # Union / Optional — unwrap and check each branch
    if origin is Union:
        args = _get_args(tp)
        return any(_isinstance_safe(val, a) for a in args if a is not type(None))

    # Generic origins (list, dict, set, tuple, etc.) — check container type only
    if origin is not None:
        return isinstance(val, origin)

    # Plain type — direct isinstance
    try:
        return isinstance(val, tp)
    except TypeError:
        return False


@dataclass
class DIBinding:
    """A fully resolved DI parameter binding.

    Created at assembly time by ``_classify_di_params``, resolved at
    runtime by ``resolve()``.

    Typed fields per kind (no untagged payload):
    - CONSTANT: ``default_value`` holds the literal default
    - FROM_INPUT_MODEL / FROM_CONFIG_MODEL: ``model_cls`` holds the Pydantic model class
    - FROM_INPUT / FROM_CONFIG: no extra fields needed (``required`` + ``inner_type`` suffice)
    - FROM_STATE: no extra fields needed (``inner_type`` used for loop unwrap)
    """

    name: str
    kind: DIKind
    inner_type: type
    required: bool
    default_value: Any = None  # CONSTANT only: the literal default value
    model_cls: Any = None      # MODEL kinds only: the Pydantic BaseModel subclass

    def resolve(self, config: Any, *, state: Any = None) -> Any:
        """The ONE resolution path for DI parameters.

        For FROM_STATE (merge_fn only), ``state`` must be provided.
        For all other kinds, reads from ``config['configurable']``.
        """
        if self.kind in (DIKind.FROM_INPUT, DIKind.FROM_CONFIG):
            val = _get_configurable(config, self.name)
            if val is None and self.required:
                source = "input" if self.kind == DIKind.FROM_INPUT else "config"
                raise _ExecutionError(
                    f"Required DI parameter '{self.name}' (from {source}) is missing "
                    f"from config['configurable']. Provide it via "
                    f"run(input={{'{self.name}': ...}})."
                )
            if val is not None and self.inner_type is not None:
                if not _isinstance_safe(val, self.inner_type):
                    source = "input" if self.kind == DIKind.FROM_INPUT else "config"
                    raise _ExecutionError(
                        f"DI parameter '{self.name}' (from {source}) expects "
                        f"{self.inner_type.__name__}, got {type(val).__name__}. "
                        f"Provide the correct type via run(input={{'{self.name}': ...}})."
                    )
            return val

        if self.kind in (DIKind.FROM_INPUT_MODEL, DIKind.FROM_CONFIG_MODEL):
            model = self.model_cls
            field_values: dict[str, Any] = {}
            for fname in model.model_fields:
                val = _get_configurable(config, fname)
                if val is not None:
                    field_values[fname] = val
            if self.required:
                missing = [f for f in model.model_fields if f not in field_values]
                if missing:
                    source = "input" if self.kind == DIKind.FROM_INPUT_MODEL else "config"
                    raise _ExecutionError(
                        f"Required DI bundled model '{self.name}' ({model.__name__}) "
                        f"is missing fields from {source}: {sorted(missing)}. "
                        f"Provide them via run(input={{...}})."
                    )
            try:
                return model(**field_values)
            except (ValidationError, TypeError, ValueError):
                if self.required:
                    source = "input" if self.kind == DIKind.FROM_INPUT_MODEL else "config"
                    raise _ExecutionError(
                        f"Required DI bundled model '{self.name}' ({model.__name__}) "
                        f"construction failed. Provide all fields via "
                        f"run(input={{...}}) or config['configurable']."
                    ) from None
                log.warning(
                    "DI model construction failed, returning None",
                    model=model.__name__,
                    param=self.name,
                    fields=field_values,
                )
                return None

        if self.kind == DIKind.CONSTANT:
            return self.default_value

        if self.kind == DIKind.FROM_STATE:
            val = getattr(state, self.name, None) if state is not None else None
            val = _unwrap_loop_value(val, self.inner_type)
            return val

        return None
