"""The ``Modifier`` base class.

Extracted from ``modifiers.py`` (neograph-3ffdg.5) into a neutral module so the
per-modifier files (``_portal.py``, ``_each.py``) can subclass it without
importing ``modifiers.py``, which re-exports them. Unchanged apart from its home.
"""

from __future__ import annotations

from pydantic import BaseModel


class Modifier(BaseModel, frozen=True):
    """Base class for node modifiers. Applied via Node.__or__."""
