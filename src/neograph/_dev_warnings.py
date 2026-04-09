"""Dev-mode warnings for ambiguous patterns. Enabled via NEOGRAPH_DEV=1."""

from __future__ import annotations

import os
import warnings

DEV_MODE = os.environ.get("NEOGRAPH_DEV", "") == "1"


def dev_warn(message: str, category: type = UserWarning) -> None:
    """Emit a warning only in dev mode."""
    if DEV_MODE:
        warnings.warn(f"[neograph-dev] {message}", category, stacklevel=3)
