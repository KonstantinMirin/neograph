# CHECK_ERROR: entry-only
# neograph-2j208: PortalSpec's entry-only knobs (max_hops/on_exhaust) must fail
# loud when set on a YAML-defined non-entry mesh member, matching the
# programmatic form's validation -- pins the model_fields_set-forwarding rule
# (unconditional default forwarding would silently break this).
from pydantic import BaseModel

from neograph.loader import load_spec
from neograph.spec_types import register_type
from tests.fakes import register_scripted


class Handoff(BaseModel, frozen=True):
    goto: str


register_type("Handoff", Handoff)
register_scripted("spec_portal_bad_entry", lambda i, c: Handoff(goto="peer"))
register_scripted("spec_portal_bad_peer", lambda i, c: Handoff(goto="__end__"))

pipeline = load_spec(
    {
        "name": "spec-portal-non-entry-max-hops",
        "nodes": [
            {
                "name": "entry",
                "mode": "scripted",
                "scripted_fn": "spec_portal_bad_entry",
                "outputs": "Handoff",
                "portal": {"to": ["peer"]},
            },
            {
                "name": "peer",
                "mode": "scripted",
                "scripted_fn": "spec_portal_bad_peer",
                "outputs": "Handoff",
                "inputs": {"handoff": "Handoff"},
                "portal": {"to": ["entry"], "max_hops": 6},
            },
        ],
        "pipeline": {"nodes": ["entry", "peer"]},
    }
)
