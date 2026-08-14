"""``ToolLedger`` — a pure read-time selector view over an existing
``list[ToolInteraction]`` (neograph-ftnxl.5).

Zero state-bus schema change, no IR field, no normalizer, no Command/monopoly
widening. Lives in its own module (rather than ``tool.py``, which has no
file-size allowlist entry) because the ledger is a read-time CONSUMER concern,
not part of the Tool runtime model itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from neograph.tool import ToolInteraction


class ToolLedger:
    """A read-only, indexed view over a ``list[ToolInteraction]`` (typically a
    node's ``tool_log`` output). Construction does the indexing once; every
    selector below is then O(1) or a cheap slice.

    Absence is ``None``/``[]`` (a tool that was never called is not a caller
    bug). ``by_key`` is the one exception: a ``None``/empty key argument IS a
    caller bug (e.g. passing an unaddressable record's ``.key`` straight
    through) and raises ``ValueError`` rather than silently returning nothing.
    """

    def __init__(self, interactions: Iterable[ToolInteraction]) -> None:
        self._interactions: list[ToolInteraction] = list(interactions)
        self._by_name: dict[str, list[ToolInteraction]] = {}
        self._by_key: dict[str, ToolInteraction] = {}
        for interaction in self._interactions:
            self._by_name.setdefault(interaction.tool_name, []).append(interaction)
            if interaction.key is not None:
                self._by_key[interaction.key] = interaction

    def first(self, tool_name: str) -> ToolInteraction | None:
        """The earliest recorded call to ``tool_name``, or ``None`` if it was
        never called."""
        records = self._by_name.get(tool_name)
        return records[0] if records else None

    def last(self, tool_name: str) -> ToolInteraction | None:
        """The most recent recorded call to ``tool_name``, or ``None`` if it
        was never called."""
        records = self._by_name.get(tool_name)
        return records[-1] if records else None

    def all(self, tool_name: str) -> list[ToolInteraction]:
        """Every recorded call to ``tool_name``, in ledger order. ``[]`` if it
        was never called."""
        return list(self._by_name.get(tool_name, []))

    def grouped(self) -> dict[str, list[ToolInteraction]]:
        """Every recorded call, grouped by ``tool_name``, each group in ledger
        order."""
        return {name: list(records) for name, records in self._by_name.items()}

    def by_key(self, key: str | None) -> ToolInteraction | None:
        """Resolve a canonical ``f"{tool_name}#{ordinal}"`` address back to its
        record. ``None`` for a well-formed key with no match. Raises
        ``ValueError`` for a ``None``/empty ``key`` -- an unaddressable
        (ordinal-0) record's ``.key`` is itself ``None``, and a caller that
        passes that straight through has a bug, not an empty lookup."""
        if not key:
            raise ValueError("ToolLedger.by_key() requires a non-empty key (an unaddressable record has key=None)")
        return self._by_key.get(key)

    def __iter__(self) -> Iterator[ToolInteraction]:
        return iter(self._interactions)

    def __len__(self) -> int:
        return len(self._interactions)
