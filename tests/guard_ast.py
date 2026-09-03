"""The ONE test-side AST detector for capability-monopoly guards.

Core Invariant: a guard that confines a capability to named files must have
exactly ONE definition of "what does using that capability look like in source
text" -- otherwise the guards themselves become an instance of the
duplicated-authority disease they exist to prevent.

Why this module exists. ``TestCommandConstructionMonopoly`` (G1,
``test_guards_assembly.py``) hand-rolled an ``ast.Call -> ast.Name`` walk to
confine ``Command(...)`` to two files. ``neograph-9axw6.1`` (step 0 of the
port-addressed data-flow epic) needs the same shape to confine ``Source`` /
``PortRef`` construction to ``_ir_normalize.py``. Copying G1's walk would have added
another hand-rolled detector, so G1 is RE-POINTED here instead.

CLAIM CORRECTED, measured after the fact (``neograph-7277f``): this does NOT make
the count one. 17 test modules hand-roll an ``isinstance(node.func, ast.Name)``
walk, and THREE of them ask G1's exact question -- is ``Command`` constructed here
-- in ``test_guards_agent_spec_execute_independence.py`` (:73, :145, :156). So the
Command detector count was 4 and is now 3 plus this helper. The honest statement is
that this module is where such a walk BELONGS, not that it is the only one; the
consolidation is filed rather than claimed.

Two scan-scope defects of the guards this replaces are deliberately NOT inherited:

- ``glob("*.py")`` is TOP-LEVEL ONLY, so ``src/neograph/testing/`` (4 modules) and
  ``src/neograph/schemas/`` are invisible to G1 and G3 today. ``iter_py_files``
  uses ``rglob``, and ``TestSourceConstructionMonopoly`` carries a SCOPE control
  asserting a known ``testing/`` module is in the scanned set -- the only test that
  would catch a regression back to ``glob``.
- A ``.name`` (basename) allowlist is safe only while the scan is one flat
  directory. Under ``rglob`` it silently permits ``<anysubpackage>/_ir_normalize.py``,
  so allowlists keyed for these helpers use the POSIX RELATIVE PATH.
"""

from __future__ import annotations

import ast
import pathlib
from collections.abc import Iterable, Iterator


def iter_py_files(root: pathlib.Path) -> Iterator[pathlib.Path]:
    """Every ``.py`` file under ``root``, RECURSIVELY, in stable order.

    ``rglob``, never ``glob``: see the module docstring's scan-scope note.
    """
    yield from sorted(root.rglob("*.py"))


def rel_posix(path: pathlib.Path, root: pathlib.Path) -> str:
    """``path`` relative to ``root`` as a POSIX string -- the allowlist key.

    A basename is not a safe key under a recursive scan.
    """
    return path.resolve().relative_to(root.resolve()).as_posix()


def _parse(path: pathlib.Path) -> ast.AST | None:
    try:
        return ast.parse(path.read_text(), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return None


def construction_call_lines(path: pathlib.Path, names: Iterable[str]) -> list[int]:
    """Line numbers where ``path`` CONSTRUCTS one of ``names``.

    Detects ``Name(...)`` calls -- the direct construction form. AST-level, so a
    docstring or comment mentioning the constructor is never a hit.

    Deliberately NOT detected: ``mod.Name(...)`` attribute calls. Both callers
    import these symbols by bare name, and widening to attribute calls would flag
    unrelated same-named APIs. If a dotted construction ever needs banning, extend
    HERE -- do not add a second walk.
    """
    tree = _parse(path)
    if tree is None:
        return []
    wanted = frozenset(names)
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in wanted
    )


def subclass_site_lines(path: pathlib.Path, names: Iterable[str]) -> list[int]:
    """Line numbers where ``path`` SUBCLASSES one of ``names``.

    The half a ``__init_subclass__`` runtime tripwire cannot enforce: a class body
    can spoof ``__module__`` in one line and pass the runtime check. Source text
    cannot be spoofed, so this is the authoritative sealing check.

    Matches a bare-name base (``class X(Peer)``) and a dotted base whose final
    attribute matches (``class X(mod.Peer)``) -- the dotted form is included here
    because a subclass declaration is the thing being sealed, and an author reaching
    for it via a module alias is doing exactly what is banned.
    """
    tree = _parse(path)
    if tree is None:
        return []
    wanted = frozenset(names)
    hits: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in wanted:
                hits.append(node.lineno)
            elif isinstance(base, ast.Attribute) and base.attr in wanted:
                hits.append(node.lineno)
    return sorted(hits)
