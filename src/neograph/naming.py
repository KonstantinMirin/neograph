"""Central naming utilities for node/construct → state-field conversion."""

_HYPHEN = "-"
_UNDERSCORE = "_"


def field_name_for(name: str) -> str:
    """Convert a node/construct name to a state field name."""
    return name.replace(_HYPHEN, _UNDERSCORE)
