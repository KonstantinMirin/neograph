"""Central naming utilities for node/construct → state-field conversion."""

_HYPHEN = "-"
_UNDERSCORE = "_"


def field_name_for(name: str) -> str:
    """Convert a node/construct name to a state field name."""
    return name.replace(_HYPHEN, _UNDERSCORE)


def output_field_name(base_field: str, output_key: str) -> str:
    """State-field name for one key of a dict-form ``Node.outputs``.

    Single source of the per-output-key naming convention used by both the
    validator's producer registration and the IR normalizer's peer-field set,
    so the two cannot drift. See neograph-bcct. ``base_field`` is the node's
    ``field_name_for(name)``.
    """
    return f"{base_field}{_UNDERSCORE}{output_key}"
