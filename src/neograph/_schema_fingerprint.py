"""Schema fingerprints — what changed since the last checkpoint.

Extracted from ``state.py`` (neograph-3ffdg.14) as a pure file split — the
functions below are unchanged, only their home moved. ``state.py`` re-exports
them, so existing imports keep resolving.

``_type_signature`` folds one level of field detail into the hash, so a
same-qualname model whose field TYPE changed still invalidates. Both fingerprints
depend on that: without it the schema-level gate never opens and the per-node
diff is dead code.
"""

from __future__ import annotations

import hashlib
from typing import Any, get_args, get_origin

from pydantic import BaseModel

from neograph._ir_branch import _BranchNode
from neograph._normalize import _declared_output, normalize_outputs
from neograph._state_keys import StateKeys
from neograph.naming import field_name_for, output_field_name


def _type_signature(typ: Any) -> str:
    """Structural signature of a type, used by both fingerprint computations.

    Qualname alone is too coarse: two structurally-different models that share a
    ``__qualname__`` (or the same class after a field-level edit) collide into a
    false negative, so the schema/node fingerprints never change and the
    checkpoint auto-rewind never triggers (neograph-v63o / review 080726 PAT-03).

    This folds one level of field detail into the signature — the same
    ``(field_name, str(annotation))`` detail ``compute_schema_fingerprint``
    records — so a same-name field add/remove/retype changes the signature:

    - Pydantic model  -> ``module.Qualname`` + sorted ``(field, str(annotation))``
      pairs. Nested models contribute their ``str(annotation)`` (not their own
      structure) to stay cycle-safe, matching the schema fingerprint's depth.
    - Generic (``list[X]``, ``dict[K,V]``, Each's ``dict[str, X]``) -> unwrapped
      so a field change on the wrapped model ``X`` is still visible.
    - Anything else -> ``str(typ)`` (already carries module + qualname).
    """
    args = get_args(typ)
    if args:
        origin = get_origin(typ)
        origin_name = getattr(origin, "__qualname__", str(origin))
        return f"{origin_name}[{','.join(_type_signature(a) for a in args)}]"
    if isinstance(typ, type) and issubclass(typ, BaseModel):
        fields = sorted((fname, str(finfo.annotation)) for fname, finfo in typ.model_fields.items())
        return f"{typ.__module__}.{typ.__qualname__}{fields!r}"
    return str(typ)


def compute_node_fingerprints(construct: Any) -> dict[str, str]:
    """Compute per-node output type fingerprints for checkpoint invalidation.

    Returns {field_name: sha256_prefix} for each node in the construct.
    Used to identify which specific nodes changed between runs.
    """

    def _fp(name: str, typ: Any) -> str:
        # The fingerprint contract: sha256('{name}:{type_signature}')[:12]. The
        # :12 width and '{name}:{sig}' layout are load-bearing — schema and node
        # fingerprints move in lockstep, neograph-v63o, so the two branches
        # (dict-form per-key + singular) MUST share one definition, neograph-2yi7q.
        return hashlib.sha256(f"{name}:{_type_signature(typ)}".encode()).hexdigest()[:12]

    result: dict[str, str] = {}

    def _fingerprint_item(item: Any) -> None:
        """Fingerprint one Node (per output key) or Construct (its output).

        Shared between top-level items and branch-arm items so an arm node's
        output type is invalidated on change exactly like a top-level node's.
        Kept as its own walk rather than routed through ``iter_nodes`` to
        preserve the top-level-only granularity: a sub-construct is
        fingerprinted by its declared output, not by its internal nodes.
        """
        # _declared_output abstracts the Node.outputs (plural) / Construct.output
        # (singular) split — Node dict-form is fingerprinted per key, a Construct's
        # single declared output as one field. No hand-rolled hasattr discrimination.
        declared = _declared_output(item)
        if declared is None:
            return
        fname = field_name_for(item.name)
        no = normalize_outputs(declared)
        if no.is_dict_form:
            # Dict-form outputs: fingerprint each key
            for key, typ in no.all_keys.items():
                full_name = output_field_name(fname, key)
                result[full_name] = _fp(full_name, typ)
        else:
            typ = no.primary
            result[fname] = _fp(fname, typ)

    for item in construct.nodes:
        if isinstance(item, _BranchNode):
            meta = item._neo_branch_meta
            for arm_item in meta.true_arm_nodes + meta.false_arm_nodes:
                _fingerprint_item(arm_item)
        else:
            _fingerprint_item(item)
    return result


def compute_schema_fingerprint(state_model: type[BaseModel]) -> str:
    """Compute a stable fingerprint from the state model's non-framework fields.

    The fingerprint changes when node output types change (field added/removed,
    type changed, class renamed). Framework fields (neo_*, node_id, project_root,
    human_feedback) are excluded — they change with modifier config, not schema.
    """

    _FRAMEWORK_PREFIXES = (
        StateKeys.FRAMEWORK_PREFIX,
        StateKeys.NODE_ID,
        StateKeys.PROJECT_ROOT,
        StateKeys.HUMAN_FEEDBACK,
    )
    items = []
    for fname, finfo in state_model.model_fields.items():
        if any(fname.startswith(p) or fname == p for p in _FRAMEWORK_PREFIXES):
            continue
        # _type_signature (not bare str(annotation)) so a same-qualname field
        # change opens the gate -- otherwise the enriched node fingerprint below
        # is never reached; see neograph-v63o. Keeps both fingerprints in lockstep.
        items.append((fname, _type_signature(finfo.annotation)))
    items.sort()
    raw = repr(items).encode()
    return hashlib.sha256(raw).hexdigest()[:16]
