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

from neograph._ir_source import EachItem, HandoffChannel, LastPresent, LoopCarry, Peer, Port, Source
from neograph._state_keys import StateKeys


def _key_for(table: dict[str, Source] | None, kind: type, *, allow_framework: bool = False) -> str | None:
    """The author's inputs key whose Source is of ``kind``, or ``None``.

    One lookup shared by the views, so "which key reads the fanned item" and "which
    key reads the mesh payload" cannot answer in different shapes.

    FRAMEWORK keys are skipped, and that is what keeps these views honest once a
    CHANNEL can also be stamped on the single-type sentinel. Both views answer
    "which PARAMETER reads this channel", and their readers -- ``_fan_agent``, the
    Agent Spec node lowering, the dict-form extractor -- pass the answer on as an
    author's parameter name. ``neo_single_input`` is not one: it is the sentinel a
    single-type read is stamped under, so an Each-modified node with
    ``inputs=Token`` would otherwise make ``fan_out_param`` return it.

    Tested against the ``neo_`` prefix rather than against that one sentinel's
    spelling, so the NEXT framework key cannot leak into a view the way this one
    would have.

    ``allow_framework`` is for the one view whose answer is a KEY INTO THE INPUTS
    DICT rather than a name handed onward: ``carry_param``. A ``@node`` port param is
    rewritten to the ``neo_subgraph_input`` key, which makes it a real declared input
    key and therefore a legitimate carry destination -- filtering it out told the
    runtime a Loop node had no carry, and the loop re-read its seed until
    ``max_iterations``. Two questions, one lookup, and the difference is which one the
    caller is asking.
    """
    for key, src in (table or {}).items():
        if (allow_framework or not key.startswith(StateKeys.FRAMEWORK_PREFIX)) and isinstance(src, kind):
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
    def carry_param(self) -> str | None:
        """Which dict-form inputs key receives this Loop's own fed-back output.

        Derived view, like the two beside it. Before it, three sites computed the
        destination from the node's declarations with two different predicates, so
        validation could approve one slot while the run bound another.
        """
        return _key_for(self.input_sources, LoopCarry, allow_framework=True)

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

        A ``LastPresent`` answers with its FIRST rung, which for a Loop read is the
        SEED. That is the field an author names and the edge the Agent Spec export
        draws: the carry is a self-edge the Loop lowering emits separately, so
        answering with it here would change every exported Loop. The view is
        deliberately lossy and the runtime does not use it for a multi-rung read.
        """
        src = (self.input_sources or {}).get(StateKeys.SINGLE_INPUT)
        if isinstance(src, LastPresent):
            src = src.rungs[0] if src.rungs else None
        if isinstance(src, Peer):
            return src.ref.field
        if isinstance(src, Port):
            return StateKeys.SUBGRAPH_INPUT
        return None
