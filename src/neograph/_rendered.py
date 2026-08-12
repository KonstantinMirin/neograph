"""``Rendered`` — prompt-ready text, and the type of everything a prompt_compiler sees.

Single-responsibility: the TYPE that carries "this value has already been through
the one rendering rule". The rule itself lives in ``renderers``; this module holds
only the marker and its loudness, so every layer (including ``errors`` consumers
and the lint layer) can import it without pulling the renderer in.

Why a ``str`` subclass rather than a wrapper object: a rendered value must behave
exactly like text everywhere downstream — ``str()``, f-strings, ``+``, ``join``,
slicing and ``json.dumps`` all yield a plain ``str``, so nothing reaches LangChain
messages, Langfuse payloads or checkpoint state as a ``Rendered``. The subclass is
invisible except at the one place it needs to be loud.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from neograph.errors import PromptInputError


class Rendered(str):
    """Prompt-ready text. Already rendered — it has no fields.

    Attribute access for a name ``str`` does not define raises
    :class:`PromptInputError`, which is NOT an ``AttributeError``. That
    distinction is the whole point: ``getattr(value, "text", "")`` and
    ``hasattr(value, "model_dump")`` swallow only ``AttributeError``, so a
    compiler written for the raw model fails LOUDLY here instead of silently
    yielding an empty payload and letting the model answer about nothing. See neograph-l2a7w.

    Dunder names fall back to ordinary ``AttributeError`` so that ``copy``,
    ``pickle``, ``deepcopy`` and any protocol probe behave exactly as they do for
    ``str``. Only user-facing attribute names are loud.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        # Dunders: behave like str. A protocol probe (__deepcopy__, __reduce_ex__,
        # __getstate__, ...) must get AttributeError or copy/pickle break.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise PromptInputError.build(
            f"{name!r} on already-rendered prompt text",
            hint=(
                "Values in a prompt_compiler's input_data are rendered text, not models. "
                "Use the value directly, or declare a raw_inputs= parameter on your "
                "compiler to receive the underlying objects."
            ),
        )


# What a prompt_compiler receives: a total mapping of name -> prompt-ready text.
# Total is deliberate — a `Rendered | Any` union would be `Any` wearing a hat and
# would re-admit the "check the type before using it" branching this type exists
# to remove. Compilers that genuinely need structure declare `raw_inputs=`.
PromptInput = Mapping[str, Rendered]


def assert_prompt_input_total(
    prompt_input: Mapping[str, Any],
    *,
    node_name: str,
    template: str,
) -> None:
    """Fail loud if anything but prompt-ready text is about to reach a compiler.

    Always on, not a test-only guard: this fires in every consumer's process,
    which is where the defect actually bit. A structural guard over this repo's
    own source cannot see a new call site added downstream, and the one it would
    have pinned (`prompt_compiler` is invoked from exactly one place) was already
    true and never the failure mode -- the failure mode was call sites deciding
    the SHAPE, not call sites invoking (design section 6 / A3).
    """
    offenders = {k: type(v).__name__ for k, v in prompt_input.items() if not isinstance(v, Rendered)}
    if not offenders:
        return
    raise PromptInputError.build(
        f"prompt_compiler would receive un-rendered value(s) {sorted(offenders)}",
        node=node_name or None,
        hint=(
            f"template {template!r} — every value handed to a prompt_compiler must go "
            f"through renderers.to_prompt_input. Saw: {offenders}. This is a neograph "
            f"bug, not a user error: some channel reached the seam without normalizing."
        ),
    )


__all__ = ["PromptInput", "Rendered", "assert_prompt_input_total"]
