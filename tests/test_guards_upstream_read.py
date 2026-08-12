"""Structural guard: ONE reader of a peer node's output field.

neograph-13k4i. Reading an upstream's value off the state bus is not one line, it
is a RULE: resolve the field name, read it, then unwrap the Loop append-list and
the Each result dict against the type the *consumer* declared. ``_extract_fan_in_dict``
knew the whole rule. ``_oracle._build_upstream_context`` re-derived the read and
knew only the first half, so an Each-produced upstream reached the node's own
prompt as the declared ``list[X]`` and the merge prompt as the raw Each dict —
``map_key`` s leaking into an LLM prompt as if they were data the author asked for.

The give-away is in that function's own docstring: it called itself "the SINGLE
source of the upstream-context rule", and it was — of its SIBLING. Two paths
agreeing with each other is not one rule. That is why this guard scans for the
SHAPE (a ``field_name_for(...)`` result handed to a bus read) rather than for the
missing unwrap: a reader that skips the unwrap is the symptom, a reader that
exists at all is the disease.

AST-based, so there is no regex-slip case to meta-test — a call is a call whatever
the surrounding text looks like.
"""

from __future__ import annotations

import ast
import pathlib

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

READER_MODULE = "di.py"
READER_FN = "read_upstream"
FIELD_NAME_FN = "field_name_for"

# What counts as a STATE-BUS read, as opposed to any old mapping lookup that
# happens to be keyed by a field name. Both halves are needed:
#   * ``get_required`` is StateBus's own verb -- no plain dict has it, so the
#     receiver does not matter.
#   * ``get`` is Mapping's verb too, so it only counts on a state-bus receiver.
# Without the second half the guard fires on ``decorated.get(field_name_for(...))``
# in _construct_builder and ``context_types.get(...)`` in state.py -- both
# compile-time dict lookups over local mappings, with no bus and no upstream
# VALUE in sight. Allowlisting those would have been the wrong repair: they are
# not exemptions from the rule, they are not instances of it.
BUS_ONLY_READ = "get_required"
MAPPING_READ = "get"
BUS_RECEIVERS = frozenset({"bus", "state", "self"})

# Modules that may read a peer field WITHOUT going through the reader. Each entry
# is a standing exemption with a structural reason -- not a to-do list. This
# allowlist may only SHRINK.
MODULE_ALLOWLIST: dict[str, str] = {
    # FORWARDING, not consumption: copies parent state into a sub-graph under the
    # same field name so the sub-node's own read can consume it. Unwrapping here
    # would pre-empt and double-apply that inner reader. Also holds the loop
    # self-field read, which reads the node's OWN output, never a peer's.
    "_subconstruct.py": "forwards parent context into an isolated sub-state; the inner reader consumes",
}

# KNOWN BOUNDARY, stated rather than papered over: the scan is per-function, so a
# read whose field name arrives as a PARAMETER is invisible to it (_wiring's loop
# routers are written that way -- the caller computes the name). _wiring.py was
# allowlisted for exactly that shape until the non-vacuity meta-test proved the
# entry inert and it was deleted. Those two routers read a node's OWN output and
# already unwrap, so nothing is being hidden today; widening the scan to follow
# names across call boundaries is the honest next ratchet if a peer read ever
# hides there.

# Functions that predate the extraction and still compose their own read. Temporary
# by construction -- each names the ticket that deletes it. EMPTY, and the emptying
# was the point: _extract_context was the sole entry and neograph-ufqr7 removed it on
# schedule. A ratchet that never reaches zero is a backlog wearing a guard's clothes.
FUNCTION_ALLOWLIST: dict[str, str] = {}


def _reads_a_peer_field(fn: ast.AST) -> bool:
    """True when *fn* hands a ``field_name_for(...)`` result to a bus read.

    Covers both spellings -- the read composed inline
    (``bus.get(field_name_for(k))``) and the two-line form that assigns the name
    to a local first, which an inline-only scan cannot see.
    """
    field_locals: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and _calls(node.value, FIELD_NAME_FN):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    field_locals.add(target.id)

    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if not node.args or not _is_bus_read(node.func):
            continue
        arg = node.args[0]
        if _calls(arg, FIELD_NAME_FN) or (isinstance(arg, ast.Name) and arg.id in field_locals):
            return True
    return False


def _is_bus_read(func: ast.Attribute) -> bool:
    """True for a StateBus read, false for a plain mapping lookup. See BUS_ONLY_READ."""
    if func.attr == BUS_ONLY_READ:
        return True
    receiver = func.value
    return func.attr == MAPPING_READ and isinstance(receiver, ast.Name) and receiver.id in BUS_RECEIVERS


def _calls(expr: ast.AST | None, name: str) -> bool:
    """True when *expr* is a call to *name* (bare or module-qualified)."""
    if not isinstance(expr, ast.Call):
        return False
    func = expr.func
    return (isinstance(func, ast.Name) and func.id == name) or (isinstance(func, ast.Attribute) and func.attr == name)


def _offenders(root: pathlib.Path) -> list[str]:
    """Every function that reads a peer field without being the one reader."""
    out: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if path.name in MODULE_ALLOWLIST:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - a broken source file fails elsewhere
            continue
        in_reader_module = path.name == READER_MODULE
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if in_reader_module and (fn.name == READER_FN or fn.name in FUNCTION_ALLOWLIST):
                continue
            if not in_reader_module and fn.name in FUNCTION_ALLOWLIST:
                continue
            if _reads_a_peer_field(fn):
                out.append(f"{path.relative_to(root).as_posix()}::{fn.name}")
    return out


class TestOneUpstreamReader:
    """A peer node's output field is read in exactly one place."""

    def test_peer_field_reads_go_through_the_one_reader(self):
        offenders = _offenders(SRC_DIR)
        assert offenders == [], (
            f"\n{len(offenders)} function(s) read a peer node's output field directly:\n"
            + "\n".join(f"  {o}" for o in offenders)
            + f"\n\nReading an upstream is a RULE (resolve the field, read it, unwrap the Loop "
            f"append-list and the Each dict against the CONSUMER's declared type), not a line. "
            f"neograph-13k4i happened because a second reader knew only half of it. Call "
            f"{READER_MODULE.removesuffix('.py')}.{READER_FN}(bus, key, expected_type, "
            f"required=..., node_label=...) instead."
        )

    # -- meta-tests: the scanner actually detects / actually accepts ------------

    def test_meta_detects_an_inline_read(self, tmp_path: pathlib.Path):
        """POSITIVE: the one-liner form ``bus.get(field_name_for(k))``."""
        (tmp_path / "_sneaky_inline.py").write_text(
            "from neograph.naming import field_name_for\n\n"
            "def build_ctx(bus, keys):\n"
            "    return {k: bus.get(field_name_for(k)) for k in keys}\n"
        )
        assert any("_sneaky_inline.py" in o for o in _offenders(tmp_path))

    def test_meta_detects_the_assign_then_read_form(self, tmp_path: pathlib.Path):
        """POSITIVE: the two-line form. This is the spelling the original inline-only
        grep missed, and it is how _extract_fan_in_dict itself is written -- so a
        scanner blind to it would have called the codebase clean."""
        (tmp_path / "_sneaky_two_line.py").write_text(
            "from neograph.naming import field_name_for\n\n"
            "def build_ctx(state, keys):\n"
            "    out = {}\n"
            "    for k in keys:\n"
            "        state_key = field_name_for(k)\n"
            "        out[k] = state.get_required(state_key)\n"
            "    return out\n"
        )
        assert any("_sneaky_two_line.py" in o for o in _offenders(tmp_path))

    def test_meta_ignores_a_plain_mapping_lookup(self, tmp_path: pathlib.Path):
        """NEGATIVE: a local dict keyed by a field name is not a bus read.

        Pinned because the first draft of this guard DID flag two of them
        (_construct_builder, state.py) and the tempting repair was to allowlist
        the modules -- which would have granted two real modules a standing
        exemption from a rule they never violated.
        """
        (tmp_path / "_plain_dict.py").write_text(
            "from neograph.naming import field_name_for\n\n"
            "def pick(decorated, items):\n"
            "    return [decorated.get(field_name_for(i.name), i) for i in items]\n"
        )
        assert _offenders(tmp_path) == []

    def test_meta_accepts_a_delegating_call_site(self, tmp_path: pathlib.Path):
        """NEGATIVE: delegating to the reader is the CURE and must pass -- a guard
        that flagged it would ban the fix along with the bug."""
        (tmp_path / "_good_citizen.py").write_text(
            "from neograph._input_shape import read_upstream\n\n"
            "def build_ctx(bus, node_inputs):\n"
            "    return {k: read_upstream(bus, k, t, required=False) for k, t in node_inputs.items()}\n"
        )
        assert _offenders(tmp_path) == []

    def test_meta_allowlist_entries_are_not_vacuous(self):
        """An allowlist entry naming a module that does not exist, or one that no
        longer reads a peer field, is dead weight that hides drift."""
        missing = sorted(name for name in MODULE_ALLOWLIST if not (SRC_DIR / name).exists())
        assert missing == [], f"allowlist names non-existent module(s): {missing}"

        # Non-vacuity: each allowlisted module must still contain the shape it is
        # exempted for -- otherwise the entry should be deleted, not carried.
        inert = []
        for name in MODULE_ALLOWLIST:
            tree = ast.parse((SRC_DIR / name).read_text())
            fns = (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef))
            if not any(_reads_a_peer_field(fn) for fn in fns):
                inert.append(name)
        assert inert == [], f"allowlist entries no longer needed -- delete them: {inert}"
