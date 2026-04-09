"""neograph CLI — validate pipelines without executing them.

    neograph check my_pipeline.py
    neograph check my_package.pipelines --config '{"node_id": "test"}'
    neograph check my_pipeline.py --setup my_config.py
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from neograph.construct import Construct


def _import_module(target: str) -> Any:
    """Import a module from a file path or dotted module name."""
    if target.endswith(".py") or "/" in target or "\\" in target:
        path = Path(target).resolve()
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(2)
        # Add parent to sys.path so relative imports within the module work
        parent = str(path.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        mod_name = path.stem
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        # Register so sys.modules[__name__] works inside the module
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod
    else:
        return importlib.import_module(target)


def _discover_constructs(mod: Any) -> list[tuple[str, Construct]]:
    """Find all Construct instances in a module's top-level namespace."""
    found = []
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, Construct):
            found.append((name, obj))
    return found


def _load_config(args: argparse.Namespace) -> dict[str, Any] | None:
    """Load config from --config JSON or --setup module."""
    if args.setup:
        mod = _import_module(args.setup)
        fn = getattr(mod, "get_check_config", None)
        if fn is None:
            print(
                f"Error: --setup module '{args.setup}' must export "
                f"a get_check_config() function",
                file=sys.stderr,
            )
            sys.exit(2)
        return fn()
    if args.config:
        try:
            return json.loads(args.config)
        except json.JSONDecodeError as exc:
            print(f"Error: --config is not valid JSON: {exc}", file=sys.stderr)
            sys.exit(2)
    return None


def cmd_check(args: argparse.Namespace) -> int:
    """Run compile() + lint() on all constructs in the target module."""
    from neograph.compiler import compile
    from neograph.errors import CompileError, ConstructError
    from neograph.lint import lint

    mod = _import_module(args.target)
    constructs = _discover_constructs(mod)

    if not constructs:
        print(f"No Construct objects found in {args.target}")
        return 1

    config = _load_config(args)
    failed = 0

    for var_name, construct in constructs:
        label = f"{construct.name} ({var_name})"
        errors = []

        # 1. Compile
        try:
            compile(construct)
        except (CompileError, ConstructError) as exc:
            errors.append(f"compile: {exc}")

        # 2. Lint
        issues = lint(construct, config=config)
        for issue in issues:
            severity = "ERROR" if issue.required else "WARN"
            errors.append(f"lint [{severity}]: {issue.message}")

        if errors:
            print(f"FAIL  {label}")
            for err in errors:
                print(f"      {err}")
            failed += 1
        else:
            print(f"OK    {label}")

    n = len(constructs)
    print()
    print(f"{n} construct(s) checked, {n - failed} passed, {failed} failed.")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(
        prog="neograph",
        description="neograph CLI — declarative LLM pipeline compiler",
    )
    sub = parser.add_subparsers(dest="command")

    check_p = sub.add_parser(
        "check",
        help="Validate pipelines without executing them",
    )
    check_p.add_argument(
        "target",
        help="Python file or module to check (e.g., my_pipeline.py or my_package.pipelines)",
    )
    check_p.add_argument(
        "--config",
        help="JSON string with config values for lint (e.g., '{\"node_id\": \"test\"}')",
    )
    check_p.add_argument(
        "--setup",
        help="Python module exporting get_check_config() for lint with real objects",
    )

    args = parser.parse_args()

    if args.command == "check":
        sys.exit(cmd_check(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
