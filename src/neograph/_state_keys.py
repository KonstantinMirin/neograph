"""Named constants and builders for `neo_*` state-bus keys.

Single source of truth for framework-internal state-dict field names.

Why: LangGraph's state-bus returns `None` for unknown keys with no error,
so a typo in any read site is a silent runtime miss. Centralizing the
keys here makes every site refer to the same symbol and lets the
structural guard (`TestNeoStateKeysCentralized`) prevent regression.

No code outside this module should reference `neo_*` as a string literal.

Fixed names are class attributes; templated names (parameterized on a
node's `field_name`) are static methods.
"""

from __future__ import annotations


class StateKeys:
    """Named `neo_*` state-bus keys.

    Use the class attributes for fixed names and the static methods for
    templated names. All names produced here include the `neo_` framework
    prefix; values are exactly the strings that previously appeared as
    literals.
    """

    # Framework prefix — distinguishes framework plumbing keys from user
    # node outputs on the state dict.
    FRAMEWORK_PREFIX = "neo_"

    # Schema-aware checkpoint resume.
    SCHEMA_FINGERPRINT = "neo_schema_fingerprint"
    NODE_FINGERPRINTS = "neo_node_fingerprints"

    # Oracle fan-out plumbing.
    ORACLE_GEN_ID = "neo_oracle_gen_id"
    ORACLE_MODEL = "neo_oracle_model"

    # Each modifier per-Send item.
    EACH_ITEM = "neo_each_item"

    # Sub-construct boundary port.
    SUBGRAPH_INPUT = "neo_subgraph_input"

    @staticmethod
    def loop_count(field_name: str) -> str:
        """Per-loop iteration counter field name."""
        return f"neo_loop_count_{field_name}"

    @staticmethod
    def loop_history(field_name: str) -> str:
        """Per-loop state-history field name (skip_when bookkeeping)."""
        return f"neo_loop_history_{field_name}"

    @staticmethod
    def oracle_collector(field_name: str) -> str:
        """Oracle barrier/collector field name for a given producer field."""
        return f"neo_oracle_{field_name}"

    @staticmethod
    def eachoracle_collector(field_name: str) -> str:
        """Each+Oracle composed barrier/collector field name."""
        return f"neo_eachoracle_{field_name}"
