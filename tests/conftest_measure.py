"""Count multi-match single-type resolutions across the real suite corpus."""
import neograph._ir_normalize as m
from neograph._normalize import normalize_inputs
from neograph._validation_types import _types_compatible

stats = {"total": 0, "multi": 0, "sites": []}
orig = m.resolve_single_type_source

def patched(node, preceding, construct_input=None):
    ni = normalize_inputs(node.inputs)
    if not (ni.is_dict_form or ni.is_none) and ni.single_type is not None:
        matches = [f for f, t, _p in preceding if t is not None and _types_compatible(t, ni.single_type)]
        stats["total"] += 1
        if len(matches) > 1:
            stats["multi"] += 1
            stats["sites"].append((node.name, tuple(matches)))
    return orig(node, preceding, construct_input)

m.resolve_single_type_source = patched
