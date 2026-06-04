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


def split_output_field(state_field: str, base_field: str) -> str | None:
    """Recover the output key from a per-key state field, or ``None``.

    Inverse of :func:`output_field_name`: given ``state_field`` and the owning
    node's ``base_field``, return the ``output_key`` when ``state_field`` has
    the form ``{base_field}_{output_key}``, else ``None``. Single source of the
    PARSE side of the per-output-key convention, so it cannot drift from the
    build side. See neograph-bcct (build) and neograph-7s2n (parse).
    """
    prefix = f"{base_field}{_UNDERSCORE}"
    if not state_field.startswith(prefix):
        return None
    return state_field[len(prefix):]
