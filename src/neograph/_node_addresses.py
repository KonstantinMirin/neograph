"""Read-only views over a Node's address table.

``neograph-9axw6.10`` collapsed four IR fields -- ``fan_out_param``,
``handoff_param``, ``handoff_channel`` and ``input_source_field`` -- into one
``Node.input_sources`` table. Each had STORED one answer to "which value is meant",
and four independent stores of one answer can drift from it and from each other.

They survive here as derived properties, so every existing reader keeps working
while the driftable storage is gone: ``Node.model_fields`` contains none of the
four, and none of them has a setter.

Separate module because ``node.py`` sits against its 500-line ceiling and these are
a cohesive cluster -- four views over one field -- rather than four unrelated
accessors. Splitting was preferred to widening the size allowlist.
"""

from __future__ import annotations

from neograph._ir_source import EachItem, HandoffChannel, Peer, Port, Source
from neograph._state_keys import StateKeys


def _key_for(table: dict[str, Source] | None, kind: type) -> str | None:
    """The inputs key whose Source is of ``kind``, or ``None``.

    One lookup shared by the views, so "which key reads the fanned item" and "which
    key reads the mesh payload" cannot answer in different shapes.
    """
    for key, src in (table or {}).items():
        if isinstance(src, kind):
            return key
    return None


class AddressViews:
    """Mixin supplying the four derived views. Expects ``self.input_sources``."""

    input_sources: dict[str, Source] | None

    @property
    def fan_out_param(self) -> str | None:
        """Which inputs key reads the fanned-out item. Derived view over
        ``input_sources`` -- was a stored field until neograph-9axw6.10."""
        return _key_for(self.input_sources, EachItem)

    @property
    def handoff_param(self) -> str | None:
        """Which inputs key reads the Portal mesh payload. Derived view."""
        return _key_for(self.input_sources, HandoffChannel)

    @property
    def handoff_channel(self) -> str | None:
        """The entry-keyed mesh-channel field a Portal member reads. Derived view:
        the channel now lives INSIDE the address that names it, so the key and the
        channel cannot disagree the way two fields could."""
        for src in (self.input_sources or {}).values():
            if isinstance(src, HandoffChannel):
                return src.channel
        return None

    @property
    def input_source_field(self) -> str | None:
        """The state field satisfying a single-type ``inputs=X``. Derived view.

        ``None`` still means "nothing to resolve", never "ambiguous" -- two
        eligible producers raise at assembly, so ambiguity cannot reach the runtime,
        and a None here must not fall back to a type scan.
        """
        src = (self.input_sources or {}).get(StateKeys.SINGLE_INPUT)
        if isinstance(src, Peer):
            return src.ref.field
        if isinstance(src, Port):
            return StateKeys.SUBGRAPH_INPUT
        return None

